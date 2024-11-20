import asyncio
import aiofiles
import websockets
import json
import logging
import os
import base64
from typing import Dict
import wave
import numpy as np
import simpleaudio as sa  # For playing audio locally
from asyncio import Queue
from collections import defaultdict
import datetime
import srt
import errno
from contextlib import suppress
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
import subprocess

class OpenAIClient:
    def __init__(self, api_key: str, translated_audio_pipe: str):
        self.api_key = api_key
        self.ws = None
        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG,  # Set to DEBUG for detailed logs
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("output/logs/app.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Logging is configured.")

        self.translated_audio_pipe = translated_audio_pipe
        self.websocket_clients: Dict[int, websockets.WebSocketServerProtocol] = {}
        self.rtmp_link = 'rtmp://sNVi5-egEGF.bintu-vtrans.nanocosmos.de/live'
        self.stream_url = 'https://bintu-play.nanocosmos.de/h5live/http/stream.mp4?url=rtmp://localhost/play&stream=sNVi5-kYN1t'

        # Create directories for output if they don't exist
        os.makedirs('output/transcripts', exist_ok=True)
        os.makedirs('output/audio/input', exist_ok=True)
        os.makedirs('output/audio/output', exist_ok=True)
        os.makedirs('output/subtitles', exist_ok=True)
        os.makedirs('output/logs', exist_ok=True)
        os.makedirs('output/video', exist_ok=True)
        os.makedirs('output/images', exist_ok=True)  # For static image

        # Create named pipes if they don't exist
        self.create_named_pipe('input_audio_pipe')
        self.create_named_pipe('translated_audio_pipe')

        # Open transcript file
        self.transcript_file = open('output/transcripts/transcript.txt', 'w', encoding='utf-8')
        self.audio_counter = 0

        # Initialize queues for playback
        self.playback_queue = Queue()

        # Initialize playback buffer
        self.playback_buffer = defaultdict(bytes)  # Buffer to store out-of-order chunks
        self.playback_sequence = 0  # Expected sequence number for playback
        self.playback_event = asyncio.Event()  # Event to signal available audio

        # Initialize a separate sequence counter for audio chunks
        self.audio_sequence = 0  # Tracks the next expected audio sequence number

        # Create task for playback
        self.playback_task = asyncio.create_task(self.audio_playback_handler())

        # Add a lock for reconnecting to prevent race conditions
        self.is_reconnecting = False
        self.reconnect_lock = asyncio.Lock()

        # Initialize WAV file for output
        try:
            self.output_wav = wave.open('output/audio/output_audio.wav', 'wb')
            self.output_wav.setnchannels(1)
            self.output_wav.setsampwidth(2)  # 16-bit PCM
            self.output_wav.setframerate(24000)
            self.logger.info("Initialized output WAV file.")
        except Exception as e:
            self.logger.error(f"Failed to initialize output WAV file: {e}", exc_info=True)
            self.output_wav = None  # Ensure it's set to None on failure

        # Initialize subtitle variables
        self.subtitle_index = 1  # Subtitle entry index
        self.current_subtitle = ""
        self.speech_start_frame = None  # Frame where current speech starts

        self.subtitle_file_path = 'output/subtitles/subtitles.srt'  # Define the path

        # Initialize the send queue
        self.send_queue = Queue()
        self.send_task = asyncio.create_task(self.send_messages())

        # Initialize FFmpeg process
        self.ffmpeg_process = None  # Will be set when FFmpeg is started

        # Initialize video start time
        self.video_start_time = None

        # Initialize latest input audio for gender detection
        self.latest_input_audio = b''

        # Define voice mappings
        self.gender_voice_map = {
            'male': ['alloy', 'ash', 'coral'],
            'female': ['echo', 'shimmer', 'ballad', 'sage', 'verse']
        }

        # To keep track of the current voice for each gender
        self.current_voice = {
            'male': 0,
            'female': 0
        }

        # Initialize total frames
        self.total_frames = 0  # Total number of audio frames processed

    def create_named_pipe(self, pipe_name):
        if not os.path.exists(pipe_name):
            os.mkfifo(pipe_name)
            self.logger.info(f"Created named pipe: {pipe_name}")
        else:
            self.logger.info(f"Named pipe already exists: {pipe_name}")

    def detect_gender(self, audio_data: bytes) -> str:
        """
        Detects the gender of the speaker from the given audio data.
        Returns 'male' or 'female'.
        """
        try:
            # Save the audio data to a temporary WAV file
            temp_audio_path = 'output/audio/input/temp_audio.wav'
            with wave.open(temp_audio_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit PCM
                wf.setframerate(24000)
                wf.writeframes(audio_data)

            # Read the audio file
            [Fs, x] = audioBasicIO.read_audio_file(temp_audio_path)
            x = audioBasicIO.stereo_to_mono(x)
            
            # Extract short-term features
            F, _ = ShortTermFeatures.feature_extraction(x, Fs, 0.050*Fs, 0.025*Fs)
            
            # Simple heuristic: average pitch or energy can be used for gender detection
            # Here, we'll use zero crossing rate as a proxy (not highly accurate)
            zcr = F[1, :]  # Zero Crossing Rate
            avg_zcr = np.mean(zcr)
            
            # Threshold based on empirical data
            gender = 'female' if avg_zcr > 0.05 else 'male'
            self.logger.debug(f"Detected gender: {gender} with avg ZCR: {avg_zcr}")
            return gender
        except Exception as e:
            self.logger.error(f"Error in gender detection: {e}", exc_info=True)
            return 'male'  # Default to 'male' if detection fails

    def get_voice_for_gender(self, gender: str) -> str:
        """
        Selects the next voice for the given gender.
        """
        voices = self.gender_voice_map.get(gender, ['alloy'])  # Default to 'alloy' if gender not found
        index = self.current_voice.get(gender, 0)
        selected_voice = voices[index % len(voices)]
        self.current_voice[gender] = (index + 1) % len(voices)
        self.logger.debug(f"Selected voice '{selected_voice}' for gender '{gender}'")
        return selected_voice

    async def send_messages(self):
        """Dedicated coroutine to send messages from the send_queue."""
        while True:
            message = await self.send_queue.get()
            if message is None:
                self.logger.info("Send queue received shutdown signal.")
                break  # Allows graceful shutdown

            try:
                await self.safe_send(message)
                # self.logger.debug(f"Sent message: {message}")
            except Exception as e:
                self.logger.error(f"Failed to send message: {e}")
                # Optionally, re-enqueue the message for retry
                await self.send_queue.put(message)
                await asyncio.sleep(5)  # Wait before retrying
            finally:
                self.send_queue.task_done()

    async def enqueue_message(self, message: str):
        """Enqueue a message to be sent over the WebSocket."""
        await self.send_queue.put(message)

    async def read_input_audio(self):
        """Coroutine to read audio from input_audio_pipe and send to Realtime API"""
        self.logger.info("Starting to read from input_audio_pipe")
        while True:
            try:
                async with aiofiles.open('input_audio_pipe', 'rb') as pipe:
                    while True:
                        data = await pipe.read(32768)  # Read 32KB
                        if not data:
                            self.logger.debug('No audio data available')
                            await asyncio.sleep(0.1)
                            continue
                        self.latest_input_audio = data  # Store for gender detection
                        base64_audio = base64.b64encode(data).decode('utf-8')
                        append_event = {
                            "type": "input_audio_buffer.append",
                            "audio": base64_audio
                        }
                        await self.enqueue_message(json.dumps(append_event))
                        self.logger.info(f"Enqueued audio chunk of size: {len(base64_audio)} bytes")
                        await asyncio.sleep(0.1)  # Prevent flooding
            except Exception as e:
                self.logger.error(f"Error in read_input_audio: {e}", exc_info=True)
                # Continue the loop to keep reading audio data
                await asyncio.sleep(1)  # Prevent tight loop on error

    async def audio_playback_handler(self):
        """Handle ordered playback of translated audio chunks from playback_queue"""
        while True:
            try:
                await self.playback_event.wait()
                while not self.playback_queue.empty():
                    item = await self.playback_queue.get()
                    if item is None:
                        self.logger.info("Playback handler received shutdown signal.")
                        continue
                    sequence, audio_data = item
                    self.logger.debug(f"Processing audio chunk: {sequence}")
                    if sequence == self.playback_sequence:
                        await self.process_playback_chunk(sequence, audio_data)
                        self.playback_sequence += 1
                        self.logger.debug(f"Processed audio chunk: {sequence}")
                        # Check if the next sequences are already in the buffer
                        while self.playback_sequence in self.playback_buffer:
                            buffered_data = self.playback_buffer.pop(self.playback_sequence)
                            self.logger.debug(f"Processing buffered audio chunk: {self.playback_sequence}")
                            await self.process_playback_chunk(self.playback_sequence, buffered_data)
                            self.playback_sequence += 1
                    elif sequence > self.playback_sequence:
                        # Future chunk, store in buffer
                        self.playback_buffer[sequence] = audio_data
                        self.logger.debug(f"Buffered out-of-order playback audio chunk: {sequence}")
                    else:
                        # Duplicate or old chunk, ignore
                        self.logger.warning(f"Ignoring duplicate or old playback audio chunk: {sequence}")
                    self.playback_queue.task_done()
                self.playback_event.clear()
            except Exception as e:
                self.logger.error(f"Error in audio_playback_handler: {e}", exc_info=True)
                await asyncio.sleep(1)
                continue

    async def process_playback_chunk(self, sequence: int, audio_data: bytes):
        """Process and play a single translated audio chunk for playback and save to WAV"""
        try:
            if not self.output_wav:
                self.logger.warning(f"Output WAV file is not open. Skipping audio chunk {sequence}.")
                return

            # Write to the output WAV file
            self.output_wav.writeframes(audio_data)
            self.logger.debug(f"Written translated audio chunk {sequence} to WAV file")

            # Update total frames
            frames_written = len(audio_data) // 2  # 2 bytes per sample (16-bit PCM)
            self.total_frames += frames_written
            self.logger.debug(f"Total frames updated to: {self.total_frames}")

            # Play the audio using simpleaudio
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            play_obj = sa.play_buffer(audio_array, 1, 2, 24000)
            await asyncio.to_thread(play_obj.wait_done)  # Wait until playback is finished
            self.logger.debug(f"Played translated audio chunk: {sequence}")

            # Write the audio data to the translated audio pipe
            try:
                fd = os.open(self.translated_audio_pipe, os.O_WRONLY | os.O_NONBLOCK)
                os.write(fd, audio_data)
                os.close(fd)
                self.logger.debug(f"Wrote translated audio chunk {sequence} to pipe")
            except BlockingIOError:
                # No reader is connected; skip writing
                self.logger.warning(f"No reader connected to translated_audio_pipe, skipping write.")
            except OSError as e:
                if e.errno == errno.ENXIO:
                    # No reader connected
                    self.logger.warning(f"No reader connected to translated_audio_pipe, skipping write.")
                else:
                    self.logger.error(f"Error writing to translated_audio_pipe: {e}")
            except Exception as e:
                self.logger.error(f"Error writing to translated_audio_pipe: {e}")

            # Save a sample audio data for verification
            sample_path = f'output/audio/output/sample_audio_{sequence}.pcm'
            try:
                async with aiofiles.open(sample_path, 'wb') as f:
                    await f.write(audio_data)
                self.logger.debug(f"Saved sample audio chunk {sequence} to {sample_path}")
            except Exception as e:
                self.logger.error(f"Error saving translated audio chunk {sequence}: {e}")

        except Exception as e:
            self.logger.error(f"Error processing playback chunk {sequence}: {e}", exc_info=True)

    async def connect(self):
        """Connect to OpenAI's Realtime API and initialize session"""
        url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        try:
            self.ws = await websockets.connect(url, extra_headers=headers)
            self.logger.info("Connected to OpenAI Realtime API")
        except Exception as e:
            self.logger.error(f"Failed to connect to OpenAI Realtime API: {e}")
            raise

        # Configure session with translator instructions
        session_update = {
            "type": "session.update",
            "session": {
                "model": "gpt-4o-realtime-preview-2024-10-01",
                "modalities": ["text", "audio"],
                "instructions": "You are a realtime translator. Please use the audio you receive and translate it into German. Try to match the tone, emotion, and duration of the original audio.",
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 200
                },
                "temperature": 0.7
            }
        }
        await self.enqueue_message(json.dumps(session_update))
        self.logger.info("Enqueued session.update message")

    async def safe_send(self, data: str):
        """Send data over WebSocket with error handling and reconnection"""
        try:
            if self.ws and self.ws.open:
                await self.ws.send(data)
            else:
                self.logger.warning("WebSocket is closed. Attempting to reconnect...")
                await self.reconnect()
                if self.ws and self.ws.open:
                    await self.ws.send(data)
                else:
                    self.logger.error("Failed to reconnect. Cannot send data.")
        except (websockets.exceptions.ConnectionClosed, websockets.exceptions.ConnectionClosedError) as e:
            self.logger.error(f"WebSocket connection closed during send: {e}")
            await self.reconnect()
            if self.ws and self.ws.open:
                await self.ws.send(data)
            else:
                self.logger.error("Failed to reconnect. Cannot send data.")
        except Exception as e:
            self.logger.error(f"Exception during WebSocket send: {e}", exc_info=True)

    async def reconnect(self):
        """Reconnect to the WebSocket server"""
        async with self.reconnect_lock:
            if self.is_reconnecting:
                self.logger.debug("Already reconnecting. Skipping additional reconnect attempts.")
                return
            self.is_reconnecting = True
            self.logger.info("Reconnecting to OpenAI Realtime API...")
            await self.disconnect(shutdown=False)
            try:
                await self.connect()
                self.logger.info("Reconnected to OpenAI Realtime API.")
            except Exception as e:
                self.logger.error(f"Failed to reconnect to OpenAI Realtime API: {e}", exc_info=True)
            self.is_reconnecting = False

    async def commit_audio_buffer(self):
        """Send input_audio_buffer.commit message to indicate end of audio input."""
        commit_event = {
            "type": "input_audio_buffer.commit"
        }
        await self.enqueue_message(json.dumps(commit_event))
        self.logger.info("Enqueued input_audio_buffer.commit message")

    async def create_response(self, selected_voice: str):
        """Send response.create message to request response generation with the selected voice."""
        response_create_event = {
            "type": "response.create",
            "response": {
                "modalities": ["text", "audio"],
                "instructions": (
                    "Please translate the audio you receive into German. "
                    "Try to keep the original tone and emotion. "
                    "Adjust the voice to match the original speaker's gender."
                ),
                "voice": selected_voice
            }
        }
        await self.enqueue_message(json.dumps(response_create_event))
        self.logger.info(f"Enqueued response.create message with voice '{selected_voice}'")

    async def handle_responses(self):
        """Handle translation responses from OpenAI"""
        while True:
            try:
                try:
                    response = await self.ws.recv()
                except (websockets.exceptions.ConnectionClosed, websockets.exceptions.ConnectionClosedError) as e:
                    self.logger.error(f"WebSocket connection closed during recv: {e}")
                    await self.reconnect()
                    continue
                except Exception as e:
                    self.logger.error(f"Exception during WebSocket recv: {e}", exc_info=True)
                    await self.reconnect()
                    continue

                event = json.loads(response)
                event_type = event.get("type")

                self.logger.debug(f"Received event type: {event_type}")

                if event_type == "input_audio_buffer.speech_started":
                    self.logger.info("Speech started")
                    self.speech_start_frame = self.total_frames  # Mark the start frame of speech
                    self.logger.debug(f"Speech started at frame: {self.speech_start_frame}")
                    # Removed reopening of self.output_wav here to prevent overwriting

                elif event_type == "input_audio_buffer.speech_stopped":
                    self.logger.info("Speech stopped")
                    if self.current_subtitle.strip() and self.speech_start_frame is not None:
                        self.logger.debug("Current subtitle is not empty, writing subtitle.")
                        
                        # Calculate start and end times in seconds
                        start_sec = self.speech_start_frame / 24000
                        end_sec = self.total_frames / 24000

                        await self.write_subtitle(
                            self.subtitle_index,
                            start_sec,
                            end_sec,
                            self.current_subtitle.strip()
                        )
                        self.logger.debug(f"Subtitle written: {self.subtitle_index}")
                        self.subtitle_index += 1
                        self.current_subtitle = ""
                        self.speech_start_frame = None
                    else:
                        self.logger.debug("Current subtitle is empty or speech_start_frame is None, skipping write_subtitle.")

                    # Commit the audio buffer and create a response
                    await self.commit_audio_buffer()
                    self.logger.debug("Audio buffer committed.")

                    # Detect gender and create response with appropriate voice
                    gender = self.detect_gender(self.latest_input_audio)
                    self.logger.debug(f"Detected gender: {gender}")
                    selected_voice = self.get_voice_for_gender(gender)
                    self.logger.debug(f"Selected voice: {selected_voice}")
                    await self.create_response(selected_voice)

                    # Removed closing of self.output_wav to keep it open for writing

                    # Introduce a small delay to ensure all data is written
                    await asyncio.sleep(0.5)  # 500 milliseconds

                    # Start FFmpeg process after speech stopped and data is written
                    await self.start_ffmpeg_process()
                    self.logger.info("FFmpeg video creation initiated after speech stopped.")

                elif event_type == "response.audio.delta":
                    self.logger.info("Received audio delta")
                    audio_data = event.get("delta", "")
                    if audio_data:
                        try:
                            decoded_audio = base64.b64decode(audio_data)
                            sequence = self.audio_sequence  # Assign current sequence number
                            await self.enqueue_audio(sequence, decoded_audio)
                            self.audio_sequence += 1  # Increment for the next chunk
                            self.logger.debug(f"Processed translated audio chunk: {sequence}")
                        except Exception as e:
                            self.logger.error(f"Error handling audio data: {e}")

                elif event_type == "response.audio_transcript.delta":
                    text = event.get("delta", "")
                    if text.strip():
                        self.logger.info(f"Translated text: {text}")
                        await self.broadcast_translation(text)
                        # Accumulate subtitle text
                        self.current_subtitle += text
                        self.logger.debug(f"Accumulated subtitle text: {self.current_subtitle}")

                elif event_type in [
                    "response.audio.done",
                    "response.audio_transcript.done",
                    "response.content_part.done",
                    "response.output_item.done",
                    "response.done"
                ]:
                    self.logger.debug(f"Ignored handled event type: {event_type}")
                    # These events are handled implicitly; no action needed

                else:
                    self.logger.warning(f"Unhandled event type: {event_type}")
            except Exception as e:
                self.logger.error(f"Error in handle_responses: {e}", exc_info=True)
                await self.reconnect()
                continue

    async def write_subtitle(self, index, start_sec, end_sec, text):
        """Write a single subtitle entry to the SRT file using the srt library"""
        try:
            # Convert seconds to timedelta
            start_td = datetime.timedelta(seconds=start_sec)
            end_td = datetime.timedelta(seconds=end_sec)

            # Debugging: Log the timing details
            self.logger.debug(f"Writing subtitle {index}: Start={start_td}, End={end_td}, Text='{text}'")

            # Check if start time is before end time
            if start_td >= end_td:
                self.logger.warning(f"Subtitle {index} skipped: Start time >= End time")
                return

            subtitle = srt.Subtitle(index=index, start=start_td, end=end_td, content=text)

            # Open the SRT file in append mode and write the subtitle
            async with aiofiles.open(self.subtitle_file_path, 'a', encoding='utf-8') as f:
                await f.write(srt.compose([subtitle]))
                await f.flush()

            self.logger.info(f"Written subtitle entry {index}: '{text}'")
        except Exception as e:
            self.logger.error(f"Error writing subtitle entry {index}: {e}", exc_info=True)

    async def enqueue_audio(self, sequence: int, audio_data: bytes):
        """Enqueue translated audio data with its sequence number into playback queue"""
        # Validate sequence number
        if sequence != self.audio_sequence:
            self.logger.error(f"Audio sequence mismatch: expected {self.audio_sequence}, got {sequence}")
            # Buffer the out-of-order chunk
            self.playback_buffer[sequence] = audio_data
            self.logger.debug(f"Buffered out-of-order audio chunk: {sequence}")
            return

        self.logger.debug(f"Enqueuing audio chunk {sequence}")
        await self.playback_queue.put((sequence, audio_data))
        self.logger.debug(f"Enqueued audio chunk: {sequence} into playback queue")
        self.playback_event.set()

        # Save the audio data to a separate PCM file for verification
        audio_output_path = f'output/audio/output/translated_audio_{sequence}.pcm'
        try:
            async with aiofiles.open(audio_output_path, 'wb') as f:
                await f.write(audio_data)
            self.logger.debug(f"Saved translated audio chunk {sequence} to {audio_output_path}")
        except Exception as e:
            self.logger.error(f"Error saving translated audio chunk {sequence}: {e}")

    async def heartbeat(self):
        """Send periodic heartbeat pings to keep the WebSocket connection alive."""
        while True:
            try:
                if self.ws and self.ws.open:
                    await self.ws.ping()
                    self.logger.debug("Sent heartbeat ping")
                else:
                    self.logger.warning("WebSocket is closed. Attempting to reconnect...")
                    await self.reconnect()
            except Exception as e:
                self.logger.error(f"Error in heartbeat: {e}", exc_info=True)
                await self.reconnect()
            await asyncio.sleep(30)  # Send a ping every 30 seconds

    async def register_websocket(self, websocket: websockets.WebSocketServerProtocol) -> int:
        """Register a new WebSocket client"""
        client_id = id(websocket)
        self.websocket_clients[client_id] = websocket
        self.logger.debug(f"Registered WebSocket client: {client_id}")
        return client_id

    async def unregister_websocket(self, client_id: int):
        """Unregister a WebSocket client"""
        self.websocket_clients.pop(client_id, None)
        self.logger.debug(f"Unregistered WebSocket client: {client_id}")

    async def broadcast_translation(self, text: str):
        """Broadcast translation to all connected WebSocket clients and handle subtitles"""
        message = json.dumps({"type": "translation", "text": text})
        disconnected_clients = []

        for client_id, websocket in self.websocket_clients.items():
            try:
                await websocket.send(message)
            except Exception as e:
                self.logger.error(f"Error broadcasting to client {client_id}: {e}")
                disconnected_clients.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.unregister_websocket(client_id)

    async def disconnect(self, shutdown=False):
        """Disconnect from OpenAI and clean up resources"""
        if self.ws:
            try:
                await self.ws.close()
                self.logger.info("Disconnected from OpenAI Realtime API")
            except Exception as e:
                self.logger.error(f"Error disconnecting from OpenAI: {e}")
            self.ws = None  # Reset the WebSocket connection

        if shutdown:
            # Shutdown queues
            if self.playback_queue:
                await self.playback_queue.put(None)
            if self.playback_task:
                self.playback_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self.playback_task
            if self.send_task:
                await self.send_queue.put(None)
                self.send_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self.send_task

            # Close WAV file
            if self.output_wav:
                try:
                    self.output_wav.close()
                    self.logger.info("Output WAV file closed")
                except Exception as e:
                    self.logger.error(f"Error closing output WAV file: {e}")

            # Terminate FFmpeg process if it were running
            if self.ffmpeg_process:
                try:
                    self.ffmpeg_process.terminate()
                    self.logger.info("Terminated FFmpeg process")
                except Exception as e:
                    self.logger.error(f"Error terminating FFmpeg process: {e}")

    async def run(self):
        while True:
            try:
                await self.connect()
                # Start handling responses
                handle_responses_task = asyncio.create_task(self.handle_responses())
                # Start reading and sending audio
                read_input_audio_task = asyncio.create_task(self.read_input_audio())
                # Start heartbeat
                heartbeat_task = asyncio.create_task(self.heartbeat())
                # Start FFmpeg process after a short delay to ensure pipes and files are ready
                await asyncio.sleep(2)
                # FFmpeg is started after speech stops
                # Wait for tasks to complete
                await asyncio.gather(
                    handle_responses_task,
                    read_input_audio_task,
                    heartbeat_task,
                    return_exceptions=True
                )
            except Exception as e:
                self.logger.error(f"Error in OpenAIClient run: {e}", exc_info=True)
                await self.disconnect(shutdown=False)
                self.logger.info("Restarting tasks in 5 seconds...")
                await asyncio.sleep(5)
                continue  # Restart the loop to reconnect and restart tasks

    async def start_ffmpeg_process(self):
        """Start the FFmpeg process to create video with subtitles."""
        if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
            self.logger.warning("FFmpeg process is already running.")
            return
        try:
            # Get absolute paths
            audio_path = os.path.abspath('output/audio/output_audio.wav')
            subtitle_path = os.path.abspath('output/subtitles/subtitles.srt')
            video_output_path = os.path.abspath('output/video/output_video.mp4')

            # Verify that input files exist
            if not os.path.isfile(audio_path):
                self.logger.error(f"Audio file not found: {audio_path}")
                return
            if not os.path.isfile(subtitle_path):
                self.logger.error(f"Subtitle file not found: {subtitle_path}")
                return

            # Define FFmpeg command
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',  # Overwrite output files without asking
                '-f', 'lavfi',  # Use libavfilter
                '-i', 'color=c=black:s=1280x720:d=10',  # Black video of 10 seconds
                '-f', 's16le',  # PCM signed 16-bit little-endian
                '-ar', '24000',  # Sampling rate
                '-ac', '1',  # Mono audio
                '-i', audio_path,  # Input audio file
                '-vf', f"subtitles={subtitle_path}",  # Subtitles
                '-c:v', 'libx264',  # Video codec
                '-c:a', 'aac',  # Audio codec
                '-b:a', '192k',  # Audio bitrate
                '-shortest',  # Finish encoding when the shortest input ends
                video_output_path  # Output video file
            ]

            self.logger.info(f"Starting FFmpeg with command: {' '.join(ffmpeg_cmd)}")
            self.ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.logger.info("FFmpeg process started.")

            # Start monitoring FFmpeg's stderr
            asyncio.create_task(self.monitor_ffmpeg())
        except Exception as e:
            self.logger.error(f"Failed to start FFmpeg process: {e}", exc_info=True)

    async def monitor_ffmpeg(self):
        """Monitor FFmpeg process and log output."""
        if self.ffmpeg_process:
            try:
                while True:
                    line = await asyncio.to_thread(self.ffmpeg_process.stderr.readline)
                    if not line:
                        break
                    self.logger.debug(f"FFmpeg stderr: {line.decode('utf-8').strip()}")
            except Exception as e:
                self.logger.error(f"Error monitoring FFmpeg process: {e}", exc_info=True)
