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
import threading
import cv2  # Import OpenCV


class OpenAIClient:
    def __init__(self, api_key: str):
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
        self.input_audio_timestamps = []
        self.video_start_time = None  # Will be set when video processing starts
        self.last_audio_sent_time = None
        self.last_translated_audio_received_time = None
        self.processing_delays = []
        self.average_processing_delay = 0.0

        # Remove translated_audio_pipe references
        # self.translated_audio_pipe = translated_audio_pipe
        # self.translated_audio_fd = None

        self.websocket_clients: Dict[int, websockets.WebSocketServerProtocol] = {}
        self.stream_url = 'https://bintu-play.nanocosmos.de/h5live/http/stream.mp4?url=rtmp://localhost/play&stream=sNVi5-kYN1t'  # Replace with your video stream URL

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
        # self.create_named_pipe('translated_audio_pipe')  # Not needed

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
        self.speech_start_time = None  # Time when current speech starts

        self.subtitle_data_list = []  # Store all subtitles with timing

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

        # Subtitle queue for OpenCV
        self.subtitle_queue = Queue()

        # Set segment duration for video splitting (e.g., 30 seconds)
        self.segment_duration = 5  # in seconds
        self.segment_index = 1  # To keep track of video segments
        self.segment_start_time = None  # Start time of the current segment

        # Start video processing in a separate thread
        self.video_processing_thread = threading.Thread(target=self.start_video_processing, daemon=True)
        self.video_processing_thread.start()

    def create_named_pipe(self, pipe_name):
        """
        Creates a named pipe if it doesn't already exist.
        """
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
            F, _ = ShortTermFeatures.feature_extraction(x, Fs, 0.050 * Fs, 0.025 * Fs)

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
                    audio_data, start_time = self.playback_buffer.pop(self.playback_sequence)
                    self.logger.debug(f"Processing audio chunk: {self.playback_sequence}")
                    await self.process_playback_chunk(self.playback_sequence, audio_data, start_time)
                    self.playback_sequence += 1
                    self.logger.debug(f"Processed audio chunk: {self.playback_sequence - 1}")
                self.playback_event.clear()
            except Exception as e:
                self.logger.error(f"Error in audio_playback_handler: {e}", exc_info=True)
                await asyncio.sleep(1)
                continue

    async def process_playback_chunk(self, sequence: int, audio_data: bytes, start_time: float):
        """Process and play a single translated audio chunk for playback and save to WAV"""
        try:
            # Calculate delay to synchronize with video start time
            if self.video_start_time is None:
                self.logger.warning("Video start time is not initialized.")
                return

            delay = (start_time - self.video_start_time) - (time.time() - self.video_start_time)
            if delay > 0:
                self.logger.debug(f"Delaying audio playback for {delay} seconds to synchronize with video.")
                await asyncio.sleep(delay)

            # Play the audio using simpleaudio
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            play_obj = sa.play_buffer(audio_array, 1, 2, 24000)
            await asyncio.to_thread(play_obj.wait_done)
            self.logger.debug(f"Played translated audio chunk: {sequence}")

            # Write to the output WAV file
            if self.output_wav:
                self.output_wav.writeframes(audio_data)
                self.logger.debug(f"Written translated audio chunk {sequence} to WAV file")

            # Removed translated_audio_pipe writing as it's not needed

        except Exception as e:
            self.logger.error(f"Error processing playback chunk {sequence}: {e}", exc_info=True)

    async def enqueue_audio(self, sequence: int, audio_data: bytes, start_time: float):
        """Enqueue translated audio data with its sequence number and start time into playback buffer"""
        self.logger.debug(f"Received audio chunk {sequence}, buffering.")
        self.playback_buffer[sequence] = (audio_data, start_time)

        # Signal the playback handler
        self.playback_event.set()

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
                    self.speech_start_time = time.time()
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
                            # Calculate the start time for this audio chunk
                            start_time = time.time()
                            await self.enqueue_audio(sequence, decoded_audio, start_time)
                            self.audio_sequence += 1  # Increment for the next chunk
                            self.logger.debug(f"Processed translated audio chunk: {sequence}")
                        except Exception as e:
                            self.logger.error(f"Error handling audio data: {e}", exc_info=True)

                elif event_type == "response.audio_transcript.delta":
                    text = event.get("delta", "")
                    if text.strip():
                        self.logger.info(f"Translated text: {text}")
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
                    self.logger.debug(f"Handled event type: {event_type}")
                    # No additional action needed

                else:
                    self.logger.warning(f"Unhandled event type: {event_type}")
            except Exception as e:
                self.logger.error(f"Error in handle_responses: {e}", exc_info=True)
                await self.reconnect()
                continue

    async def write_subtitle(self, index, start_time, end_time, text):
        """Queue subtitle data for overlay and save it for the current video segment."""
        try:
            # Calculate durations relative to segment start time
            if self.segment_start_time is None:
                self.logger.warning("Segment start time is not initialized.")
                return

            adjusted_start_time = start_time - self.segment_start_time
            adjusted_end_time = end_time - self.segment_start_time

            if adjusted_end_time <= adjusted_start_time:
                self.logger.warning(f"Invalid subtitle duration for index {index}")
                return

            subtitle_data = {
                'text': text,
                'start_time': adjusted_start_time,
                'end_time': adjusted_end_time
            }
            self.subtitle_queue.put_nowait(subtitle_data)
            self.logger.info(f"Subtitle queued: '{text}' from {adjusted_start_time} to {adjusted_end_time}")

        except Exception as e:
            self.logger.error(f"Error queuing subtitle: {e}", exc_info=True)

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

    async def disconnect(self, shutdown=False):
        """Disconnect from OpenAI and clean up resources"""
        if self.ws:
            try:
                await self.ws.close()
                self.logger.info("Disconnected from OpenAI Realtime API")
            except Exception as e:
                self.logger.error(f"Error disconnecting from OpenAI: {e}")
            self.ws = None  # Reset the WebSocket connection

        # Removed translated_audio_pipe closing as it's not used

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

    def start_video_processing(self):
        """Start video processing with OpenCV and dynamic subtitle overlay."""
        try:
            self.logger.info("Starting video processing with OpenCV.")

            # Open video capture
            cap = cv2.VideoCapture(self.stream_url)

            if not cap.isOpened():
                self.logger.error("Cannot open video stream.")
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
            segment_frames = int(fps * self.segment_duration)
            frame_count = 0
            self.segment_start_time = time.time()

            # Initialize variables for subtitle overlay
            self.subtitle_data_list = []

            while True:
                # Define video writer for each segment
                segment_output_path = f'output/video/output_video_segment_{self.segment_index}.mp4'
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for MP4
                out = cv2.VideoWriter(segment_output_path, fourcc, fps, (width, height))

                if not out.isOpened():
                    self.logger.error(f"Failed to open VideoWriter for segment {self.segment_index}.")
                    break

                self.logger.info(f"Started recording segment {self.segment_index} to {segment_output_path}")

                while frame_count < segment_frames:
                    ret, frame = cap.read()
                    if not ret:
                        self.logger.warning("Failed to read frame from video stream.")
                        break

                    # Get current time relative to segment start
                    current_time = time.time() - self.segment_start_time

                    # Check for new subtitles
                    self.check_and_update_subtitles()

                    # Overlay subtitles
                    self.overlay_subtitles(frame, current_time)

                    # Removed cv2.imshow and cv2.waitKey to prevent errors in non-GUI environments

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
                self.subtitle_data_list = []

                # Save subtitles for the completed segment
                self.save_subtitles_to_srt(self.segment_index - 1)

                # Optional: To prevent accumulating too many subtitles in memory, clear the queue
                while not self.subtitle_queue.empty():
                    try:
                        self.subtitle_queue.get_nowait()
                        self.subtitle_queue.task_done()
                    except Exception:
                        break

                # Continue to the next segment

            # Release resources
            cap.release()
            cv2.destroyAllWindows()
            self.logger.info("Video processing completed.")

        except Exception as e:
            self.logger.error(f"Error in video processing: {e}", exc_info=True)

    def check_and_update_subtitles(self):
        """Check for new subtitles and update the list."""
        while not self.subtitle_queue.empty():
            subtitle_data = self.subtitle_queue.get_nowait()
            self.subtitle_data_list.append(subtitle_data)

    def overlay_subtitles(self, frame, current_time):
        """Overlay subtitles onto the frame."""
        # Determine which subtitles to display
        display_subtitles = [
            s for s in self.subtitle_data_list
            if s['start_time'] <= current_time <= s['end_time']
        ]

        for subtitle in display_subtitles:
            text = subtitle['text']
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            color = (255, 255, 255)  # White color
            thickness = 2

            # Calculate text size
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            x = int((frame.shape[1] - text_width) / 2)
            y = frame.shape[0] - 50  # Position at the bottom

            # Add background rectangle for better visibility
            box_coords = (
                (x - 10, y - text_height - 10),
                (x + text_width + 10, y + 10)
            )
            cv2.rectangle(frame, box_coords[0], box_coords[1], (0, 0, 0), cv2.FILLED)

            # Put the text on top of the rectangle
            cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    def save_subtitles_to_srt(self, segment_index):
        """Save accumulated subtitles to an SRT file for the segment."""
        try:
            subtitles = []
            for idx, subtitle in enumerate(self.subtitle_data_list, 1):
                start_td = datetime.timedelta(seconds=subtitle['start_time'])
                end_td = datetime.timedelta(seconds=subtitle['end_time'])
                subtitles.append(srt.Subtitle(index=idx, start=start_td, end=end_td, content=subtitle['text']))

            srt_content = srt.compose(subtitles)
            srt_output_path = f'output/subtitles/subtitles_segment_{segment_index}.srt'
            with open(srt_output_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            self.logger.info(f"Subtitles for segment {segment_index} saved to {srt_output_path}")
        except Exception as e:
            self.logger.error(f"Error saving subtitles for segment {segment_index}: {e}", exc_info=True)
