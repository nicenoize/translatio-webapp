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
import time

class OpenAIClient:
    def __init__(self, api_key: str, translated_audio_pipe: str):
        self.api_key = api_key
        self.ws = None
        self.translated_audio_pipe = translated_audio_pipe

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

        # Initialize timestamps and delays
        self.input_audio_timestamps = []
        self.video_start_time = time.time()
        self.last_audio_sent_time = time.time()
        self.last_translated_audio_received_time = None
        self.processing_delays = []
        self.average_processing_delay = 0.0

        # Initialize WebSocket clients (if needed)
        self.websocket_clients: Dict[int, websockets.WebSocketServerProtocol] = {}
        self.rtmp_link = 'rtmp://sNVi5-egEGF.bintu-vtrans.nanocosmos.de/live'
        self.stream_url = 'https://bintu-play.nanocosmos.de/h5live/http/stream.mp4?url=rtmp://localhost/play&stream=sNVi5-kYN1t'

        # Create directories for output if they don't exist
        self.create_directories()

        # Create named pipes if they don't exist
        self.create_named_pipe('input_audio_pipe')
        self.create_named_pipe('translated_audio_pipe')

        # Open transcript file
        self.transcript_file = open('output/transcripts/transcript.txt', 'w', encoding='utf-8')
        self.audio_counter = 0

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

        # Initialize WAV file for output (for verification)
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
            'female': 1
        }

        # Initialize total frames
        self.total_frames = 0  # Total number of audio frames processed

    def create_named_pipe(self, pipe_name):
        """
        Creates a named pipe if it doesn't already exist.
        """
        if not os.path.exists(pipe_name):
            os.mkfifo(pipe_name)
            self.logger.info(f"Created named pipe: {pipe_name}")
        else:
            self.logger.info(f"Named pipe already exists: {pipe_name}")

    def create_directories(self):
        """Create necessary directories for outputs."""
        directories = [
            'output/transcripts',
            'output/audio/input',
            'output/audio/output',
            'output/subtitles',
            'output/logs',
            'output/video',
            'output/images'  # For static image
        ]
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
                self.logger.debug(f"Ensured directory exists: {directory}")
            except Exception as e:
                self.logger.error(f"Failed to create directory {directory}: {e}", exc_info=True)

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
            F, _ = ShortTermFeatures.feature_extraction(x, Fs, 0.050 * Fs, 0.025 * Fs)

            # Simple heuristic: average zero crossing rate (ZCR) for gender detection
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

                        # Record the timestamp when the audio chunk is read
                        timestamp = time.time()
                        self.input_audio_timestamps.append(timestamp)

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
                await asyncio.sleep(1)  # Prevent tight loop on error

    async def audio_playback_handler(self):
        """Handles playback of translated audio chunks."""
        while True:
            try:
                await self.playback_event.wait()
                while self.playback_sequence in self.playback_buffer:
                    audio_data = self.playback_buffer.pop(self.playback_sequence)
                    self.logger.debug(f"Processing audio chunk: {self.playback_sequence}")
                    await self.process_playback_chunk(self.playback_sequence, audio_data)
                    self.playback_sequence += 1
                    self.logger.debug(f"Processed audio chunk: {self.playback_sequence - 1}")
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

            # Write the audio data to the translated audio pipe if it's open
            if self.translated_audio_fd is None:
                try:
                    self.translated_audio_fd = os.open(self.translated_audio_pipe, os.O_WRONLY)
                    self.logger.info(f"Opened translated_audio_pipe for writing: {self.translated_audio_pipe}")
                except OSError as e:
                    self.logger.error(f"Error opening translated_audio_pipe for writing: {e}")
                    return  # Exit the function if unable to open the pipe

            try:
                os.write(self.translated_audio_fd, audio_data)
                self.logger.debug(f"Wrote translated audio chunk {sequence} to pipe")
            except OSError as e:
                if e.errno == errno.EPIPE:
                    self.logger.warning(f"Pipe closed. Unable to write translated audio chunk {sequence}.")
                    self.translated_audio_fd = None  # Reset the file descriptor
                else:
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
                response = await self.ws.recv()
                event = json.loads(response)
                event_type = event.get("type")

                if event_type == "input_audio_buffer.speech_started":
                    self.logger.info("Speech started")
                    self.speech_start_time = time.time()  # Use actual time
                    self.logger.debug(f"Speech started at time: {self.speech_start_time}")

                elif event_type == "input_audio_buffer.speech_stopped":
                    self.logger.info("Speech stopped")
                    speech_end_time = time.time()

                    # Ensure that there is subtitle text and start time before writing
                    if self.current_subtitle.strip() and self.speech_start_time is not None:
                        self.logger.debug("Current subtitle is not empty, writing subtitle.")

                        await self.write_subtitle(
                            self.subtitle_index,
                            self.speech_start_time,
                            speech_end_time,
                            self.current_subtitle.strip()
                        )
                        self.logger.debug(f"Subtitle written: {self.subtitle_index}")
                        self.subtitle_index += 1
                        self.current_subtitle = ""
                        self.speech_start_time = None
                    else:
                        self.logger.debug("Current subtitle is empty or speech_start_time is None, skipping write_subtitle.")

                    # Commit the audio buffer and create a response
                    await self.commit_audio_buffer()
                    self.logger.debug("Audio buffer committed.")
                    self.last_audio_sent_time = time.time()

                    # Detect gender and create response with appropriate voice
                    gender = self.detect_gender(self.latest_input_audio)
                    self.logger.debug(f"Detected gender: {gender}")
                    selected_voice = self.get_voice_for_gender(gender)
                    self.logger.debug(f"Selected voice: {selected_voice}")
                    await self.create_response(selected_voice)

                    # Introduce a small delay to ensure all data is written
                    await asyncio.sleep(0.5)  # 500 milliseconds

                elif event_type == "response.audio.delta":
                    audio_data = event.get("delta", "")
                    if audio_data:
                        self.logger.info("Received audio delta")
                        self.last_translated_audio_received_time = time.time()

                        # Ensure last_audio_sent_time is set before calculating delay
                        if self.last_audio_sent_time is not None:
                            processing_delay = self.last_translated_audio_received_time - self.last_audio_sent_time
                            self.processing_delays.append(processing_delay)
                            self.logger.debug(f"Processing delay recorded: {processing_delay} seconds")

                            # Update average processing delay
                            if self.processing_delays:
                                self.average_processing_delay = sum(self.processing_delays) / len(self.processing_delays)
                                self.logger.debug(f"Updated average processing delay: {self.average_processing_delay} seconds")
                        else:
                            self.logger.warning("last_audio_sent_time is None. Cannot calculate processing delay.")

                        try:
                            decoded_audio = base64.b64decode(audio_data)
                            sequence = self.audio_sequence  # Assign current sequence number
                            await self.enqueue_audio(sequence, decoded_audio)
                            self.audio_sequence += 1  # Increment for the next chunk
                            self.logger.debug(f"Processed translated audio chunk: {sequence}")
                        except Exception as e:
                            self.logger.error(f"Error handling audio data: {e}", exc_info=True)
                    else:
                        self.logger.debug("Received empty audio delta.")

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

    async def write_subtitle(self, index, start_time, end_time, text):
        """Write a single subtitle entry to the SRT file using the srt library"""
        try:
            # Ensure average_processing_delay is calculated
            if not self.processing_delays:
                self.average_processing_delay = 0.0

            # Adjust times based on average processing delay
            adjusted_start_time = start_time - self.video_start_time - self.average_processing_delay
            adjusted_end_time = end_time - self.video_start_time - self.average_processing_delay

            # Convert to timedelta
            start_td = datetime.timedelta(seconds=adjusted_start_time)
            end_td = datetime.timedelta(seconds=adjusted_end_time)

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
        """Enqueue translated audio data with its sequence number into playback buffer"""

        self.logger.debug(f"Received audio chunk {sequence}, buffering.")
        self.playback_buffer[sequence] = audio_data

        # Save the audio data to a separate PCM file for verification
        audio_output_path = f'output/audio/output/translated_audio_{sequence}.pcm'
        try:
            async with aiofiles.open(audio_output_path, 'wb') as f:
                await f.write(audio_data)
            self.logger.debug(f"Saved translated audio chunk {sequence} to {audio_output_path}")
        except Exception as e:
            self.logger.error(f"Error saving translated audio chunk {sequence}: {e}")

        # Signal the playback handler
        self.playback_event.set()

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

        if hasattr(self, 'translated_audio_fd') and self.translated_audio_fd:
            try:
                os.close(self.translated_audio_fd)
                self.logger.info(f"Closed translated_audio_pipe: {self.translated_audio_pipe}")
            except Exception as e:
                self.logger.error(f"Error closing translated_audio_pipe: {e}")
            self.translated_audio_fd = None

        if shutdown:
            # Shutdown queues
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

    async def run_video_processing(self):
        """Run video processing within the asyncio event loop."""
        await asyncio.to_thread(self.start_video_processing)

    def start_video_processing(self):
        """Start video processing with OpenCV and dynamic subtitle overlay."""
        try:
            self.logger.info("Starting video processing with OpenCV.")

            # Open video capture
            cap = cv2.VideoCapture(self.stream_url)

            if not cap.isOpened():
                self.logger.error("Cannot open video stream.")
                self.running = False
                return

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or fps is None:
                fps = 25.0  # Default FPS if unable to get from stream
                self.logger.warning(f"Unable to get FPS from stream. Defaulting to {fps} FPS.")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

            self.logger.info(f"Video properties - FPS: {fps}, Width: {width}, Height: {height}")

            # Initialize variables for video segmentation
            segment_frames = int(fps * 5)  # 5-second segments
            frame_count = 0
            self.segment_start_time = time.time()

            while True:
                # Define video writer for each segment
                segment_output_path = f'output/video/output_video_segment_{self.segment_index}.mp4'
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for MP4
                out = cv2.VideoWriter(segment_output_path, fourcc, fps, (width, height))

                if not out.isOpened():
                    self.logger.error(f"Failed to open VideoWriter for segment {self.segment_index}.")
                    self.running = False
                    break

                self.logger.info(f"Started recording segment {self.segment_index} to {segment_output_path}")

                # Initialize the corresponding audio segment file
                audio_segment_path = f'output/audio/output_audio_segment_{self.segment_index}.wav'
                try:
                    wf = wave.open(audio_segment_path, 'wb')
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(24000)
                    self.segment_audio_writers[self.segment_index] = wf
                    self.logger.debug(f"Stored Wave_write object for segment {self.segment_index}")
                except Exception as e:
                    self.logger.error(f"Failed to initialize audio segment file {audio_segment_path}: {e}", exc_info=True)
                    continue  # Proceed to next segment

                while frame_count < segment_frames and self.running:
                    ret, frame = cap.read()
                    if not ret:
                        self.logger.warning("Failed to read frame from video stream.")
                        break

                    # Get current time relative to segment start
                    current_time = time.time() - self.segment_start_time

                    # Overlay subtitles
                    self.overlay_subtitles(frame, current_time)

                    # Write frame to output video
                    out.write(frame)
                    frame_count += 1

                # Release the current segment video writer
                out.release()
                self.logger.info(f"Segment {self.segment_index} saved to {segment_output_path}.")

                # Reset for next segment
                self.segment_index += 1
                frame_count = 0
                self.segment_start_time = time.time()

                # Schedule muxing of video and audio in the asyncio loop
                asyncio.run_coroutine_threadsafe(
                    self.mux_video_audio(self.segment_index - 1),
                    asyncio.get_event_loop()
                )

        except Exception as e:
            self.logger.error(f"Error in video processing: {e}", exc_info=True)

    async def mux_video_audio(self, segment_index: int):
        """Mux video segment with corresponding audio segment using FFmpeg"""
        video_path = f'output/video/output_video_segment_{segment_index}.mp4'
        audio_path = f'output/audio/output_audio_segment_{segment_index}.wav'
        subtitles_path = f'output/subtitles/subtitles_segment_{segment_index}.srt'
        final_output_path = f'output/final/final_output_video_segment_{segment_index}.mp4'

        # Check if video and audio files exist
        if not os.path.exists(video_path):
            self.logger.error(f"Video segment {video_path} does not exist. Cannot mux audio.")
            return
        if not os.path.exists(audio_path):
            self.logger.error(f"Audio segment {audio_path} does not exist. Cannot mux audio.")
            return

        # FFmpeg command to mux video and audio
        ffmpeg_command = [
            'ffmpeg',
            '-y',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',  # Copy video stream without re-encoding
            '-c:a', 'aac',   # Encode audio to AAC
            '-strict', 'experimental',
            final_output_path
        ]

        try:
            self.logger.info(f"Muxing video and audio for segment {segment_index}...")
            process = await asyncio.create_subprocess_exec(
                *ffmpeg_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode == 0:
                self.logger.info(f"Successfully muxed video and audio into {final_output_path}")
            else:
                self.logger.error(f"FFmpeg muxing failed for segment {segment_index}.")
                self.logger.error(f"FFmpeg stderr: {stderr.decode()}")
        except Exception as e:
            self.logger.error(f"Error during FFmpeg muxing: {e}", exc_info=True)

    async def run(self):
        """Run the OpenAIClient."""
        while True:
            try:
                await self.connect()
                # Start handling responses
                handle_responses_task = asyncio.create_task(self.handle_responses())
                # Start reading and sending audio
                read_input_audio_task = asyncio.create_task(self.read_input_audio())
                # Start heartbeat
                heartbeat_task = asyncio.create_task(self.heartbeat())
                # Start video processing
                run_video_processing_task = asyncio.create_task(self.run_video_processing())

                # Wait for tasks to complete or fail
                done, pending = await asyncio.wait(
                    [
                        handle_responses_task,
                        read_input_audio_task,
                        heartbeat_task,
                        run_video_processing_task
                    ],
                    return_when=asyncio.FIRST_EXCEPTION
                )

                for task in pending:
                    task.cancel()

            except Exception as e:
                self.logger.error(f"Error in OpenAIClient run: {e}", exc_info=True)
                await self.disconnect(shutdown=False)
                self.logger.info("Restarting tasks in 5 seconds...")
                await asyncio.sleep(5)
                continue  # Restart the loop to reconnect and restart tasks
