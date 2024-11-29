import asyncio
import aiofiles
import websockets
import json
import logging
import os
import base64
from typing import Optional
import wave
import numpy as np
import simpleaudio as sa  # For playing audio locally
from asyncio import Queue
from collections import deque
import datetime
import srt
from contextlib import suppress
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
import time
import cv2  # Import OpenCV
from logging.handlers import RotatingFileHandler
import subprocess
import signal
import uuid


from .event_tracker import EventTracker
from .subtitle_manager import SubtitleManager
from .audio_manager import AudioManager



class OpenAIClient:
    def __init__(self, api_key: str, loop: asyncio.AbstractEventLoop):
        self.api_key = api_key
        self.ws = None
        self.loop = loop
        self.setup_logging()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize managers
        self.event_tracker = EventTracker()
        self.subtitle_manager = SubtitleManager()
        self.audio_manager = AudioManager()
        
        self.running = True
        self.video_start_time = None
        self.segment_index = 1
        self.segment_duration = 5  # 5 seconds per segment
        
        # Setup queues and buffers
        self.send_queue = Queue()
        self.send_task = asyncio.create_task(self.send_messages())
        
        # Initialize other components
        self.setup_directories_and_pipes()
        self.initialize_gender_detection()

        # Initialize WAV file for output (for verification)
        try:
            os.makedirs('output/audio/output', exist_ok=True)
            self.output_wav_path = 'output/audio/output_audio.wav'
            self.output_wav = wave.open(self.output_wav_path, 'wb')
            self.output_wav.setnchannels(1)
            self.output_wav.setsampwidth(2)  # 16-bit PCM
            self.output_wav.setframerate(24000)
            self.logger.info(f"Initialized output WAV file at {self.output_wav_path}.")
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

        # Set segment duration for video splitting (e.g., 5 seconds)
        self.segment_duration = 5  # in seconds
        self.segment_index = 1  # To keep track of video segments
        self.segment_start_time = None  # Start time of the current segment

        # Dictionary to store Wave_write objects for each segment
        self.segment_audio_writers: Dict[int, wave.Wave_write] = {}
        self.segment_audio_lock = asyncio.Lock()

        # Initialize audio buffer (simple queue for sequential processing)
        self.audio_buffer = deque(maxlen=100)  # Adjust maxlen as needed

    def setup_logging(self):
        """Setup logging with RotatingFileHandler to prevent log files from growing too large."""
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        # Create rotating file handler for the main app log
        os.makedirs('output/logs', exist_ok=True)
        app_handler = RotatingFileHandler("output/logs/app.log", maxBytes=5*1024*1024, backupCount=5)
        app_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        app_handler.setFormatter(app_formatter)
        logger.addHandler(app_handler)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    def create_directories(self):
        """Create necessary directories for outputs."""
        directories = [
            'output/transcripts',
            'output/audio/input',
            'output/audio/output',
            'output/subtitles',
            'output/logs',
            'output/video',
            'output/final',  # New folder for final muxed videos
            'output/images'  # For static image
        ]
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
                self.logger.debug(f"Ensured directory exists: {directory}")
            except Exception as e:
                self.logger.error(f"Failed to create directory {directory}: {e}", exc_info=True)

    def create_named_pipe(self, pipe_name):
        """
        Creates a named pipe if it doesn't already exist.
        """
        try:
            if not os.path.exists(pipe_name):
                os.mkfifo(pipe_name)
                self.logger.info(f"Created named pipe: {pipe_name}")
            else:
                self.logger.info(f"Named pipe already exists: {pipe_name}")
        except Exception as e:
            self.logger.error(f"Error creating named pipe {pipe_name}: {e}", exc_info=True)
            raise

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
        while self.running:
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
        while self.running:
            try:
                # Open the named pipe in binary read mode
                async with aiofiles.open(self.input_audio_pipe, 'rb') as pipe:
                    while self.running:
                        data = await pipe.read(65536)  # Increased buffer size to 64KB
                        if not data:
                            self.logger.debug('No audio data available')
                            await asyncio.sleep(0.05)  # Further reduced sleep
                            continue
                        self.latest_input_audio = data  # Store for gender detection

                        # Record the timestamp when the audio chunk is read
                        timestamp = time.perf_counter()
                        self.audio_timestamps.append(timestamp)

                        await self.send_input_audio_buffer_append(data)
                        self.logger.info(f"Enqueued audio chunk of size: {len(data)} bytes")

                        await asyncio.sleep(0.05)  # Reduced sleep to minimize latency
            except Exception as e:
                self.logger.error(f"Error in read_input_audio: {e}", exc_info=True)
                await asyncio.sleep(1)  # Prevent tight loop on error

    async def audio_playback_handler(self):
        """Handles playback of translated audio chunks."""
        while self.running:
            try:
                await self.playback_event.wait()
                while self.audio_buffer:
                    audio_data, start_time = self.audio_buffer.popleft()
                    self.logger.debug(f"Processing audio chunk.")
                    await self.process_playback_chunk(audio_data, start_time)
                self.playback_event.clear()
            except Exception as e:
                self.logger.error(f"Error in audio_playback_handler: {e}", exc_info=True)
                await asyncio.sleep(1)
                continue

    async def process_playback_chunk(self, audio_data: bytes, start_time: float):
        """Process and play a single translated audio chunk for playback and save to WAV"""
        try:
            # Calculate delay to synchronize with video start time
            if self.video_start_time is None:
                self.logger.warning("Video start time is not initialized.")
                return

            # Calculate the expected playback time relative to video start
            elapsed_time = start_time - self.video_start_time
            expected_playback_time = self.video_start_time + elapsed_time
            current_time = time.perf_counter()
            delay = expected_playback_time - current_time
            if delay > 0:
                self.logger.debug(f"Delaying audio playback for {delay:.3f} seconds to synchronize with video.")
                await asyncio.sleep(delay)

            # Play the audio using simpleaudio
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            play_obj = sa.play_buffer(audio_array, 1, 2, 24000)
            await asyncio.to_thread(play_obj.wait_done)
            self.logger.debug("Played translated audio chunk.")

            # Write to the output WAV file
            if self.output_wav:
                self.output_wav.writeframes(audio_data)
                self.logger.debug("Written translated audio chunk to WAV file.")

            # Write to the corresponding per-segment audio WAV file
            async with self.segment_audio_lock:
                wf = self.segment_audio_writers.get(self.segment_index)
                if wf:
                    wf.writeframes(audio_data)
                    self.logger.debug(f"Written translated audio chunk to segment {self.segment_index} WAV file.")
                else:
                    self.logger.error(f"No Wave_write object found for segment {self.segment_index}")

        except Exception as e:
            self.logger.error(f"Error processing playback chunk: {e}", exc_info=True)

    async def enqueue_audio(self, audio_data: bytes, start_time: float):
        """Enqueue translated audio data with its start time into playback buffer"""
        self.logger.debug("Received audio chunk, buffering.")
        self.audio_buffer.append((audio_data, start_time))

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
        await self.send_session_update()
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
        """Reconnect to the WebSocket server with exponential backoff"""
        async with self.reconnect_lock:
            if self.is_reconnecting:
                self.logger.debug("Already reconnecting. Skipping additional reconnect attempts.")
                return
            self.is_reconnecting = True
            backoff = 1  # Start with 1 second
            max_backoff = 60  # Maximum backoff time
            while not self.ws or not self.ws.open:
                self.logger.info(f"Attempting to reconnect in {backoff} seconds...")
                await asyncio.sleep(backoff)
                try:
                    await self.connect()
                    self.logger.info("Reconnected to OpenAI Realtime API.")
                    self.is_reconnecting = False
                    return
                except Exception as e:
                    self.logger.error(f"Reconnect attempt failed: {e}", exc_info=True)
                    backoff = min(backoff * 2, max_backoff)  # Exponential backoff

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
        while self.running:
            try:
                response = await self.ws.recv()
                event = json.loads(response)
                event_type = event.get("type")
                event_id = event.get("event_id")
                
                if not event_id:
                    continue
                    
                if event_type == "response.audio.delta":
                    await self.handle_audio_delta(event)
                elif event_type == "response.audio_transcript.delta":
                    await self.handle_transcript_delta(event)
                elif event_type == "response.done":
                    await self.handle_response_done(event_id)
            except Exception as e:
                self.logger.error(f"Error in handle_responses: {e}")
                await self.reconnect()

    async def mux_segment(self, segment_index, video_path, audio_path):
        """Enhanced muxing with improved synchronization"""
        subtitles_path = f'output/subtitles/subtitles_segment_{segment_index}.srt'
        final_path = f'output/final/output_final_video_segment_{segment_index}.mp4'
        
        try:
            # First overlay subtitles
            overlay_result = await self.overlay_subtitles_via_ffmpeg(
                video_path,
                subtitles_path
            )
            
            if not overlay_result:
                return
                
            # Then mux with audio
            await self.mux_video_audio_ffmpeg(
                overlay_result,
                audio_path,
                final_path,
                self.segment_duration
            )
            
            # Cleanup temporary files
            await self.cleanup_temp_files(segment_index)
            
        except Exception as e:
            self.logger.error(f"Error in muxing segment {segment_index}: {e}")

    def calculate_segment_index(self, timestamp):
        """Calculate segment index based on timestamp"""
        if self.video_start_time is None:
            return 1
        return int((timestamp - self.video_start_time) / self.segment_duration) + 1

    async def handle_audio_delta(self, event):
        event_id = event.get("event_id")
        audio_data = event.get("delta", "")
        
        if not audio_data:
            return
            
        try:
            decoded_audio = base64.b64decode(audio_data)
            event_data = self.event_tracker.get_event_data(event_id)
            
            if event_data:
                segment_index = self.calculate_segment_index(event_data.timestamp)
                await self.audio_manager.write_audio(segment_index, decoded_audio)
                
                self.event_tracker.update_event(
                    event_id,
                    translated_audio=decoded_audio
                )
        except Exception as e:
            self.logger.error(f"Error handling audio delta: {e}")

    async def handle_transcript_delta(self, event):
        event_id = event.get("event_id")
        text = event.get("delta", "")
        
        if text.strip():
            event_data = self.event_tracker.get_event_data(event_id)
            if event_data:
                segment_index = self.calculate_segment_index(event_data.timestamp)
                await self.subtitle_manager.add_subtitle(
                    text=text,
                    start_time=event_data.timestamp - self.video_start_time,
                    end_time=event_data.timestamp - self.video_start_time + len(text) * 0.06,
                    segment_index=segment_index
                )
                
                self.event_tracker.update_event(
                    event_id,
                    translated_text=text
                )

    async def handle_response_done(self, event_id):
        self.event_tracker.update_event(
            event_id,
            processing_complete=True
        )
        
        # Process any pending events
        for event_id in self.event_tracker.get_pending_events():
            if self.event_tracker.is_event_complete(event_id):
                event_data = self.event_tracker.get_event_data(event_id)
                segment_index = self.calculate_segment_index(event_data.timestamp)
                
                # Trigger segment processing if we have all components
                await self.process_segment(segment_index)

    async def process_segment(self, segment_index):
        """Process a completed segment with video, audio, and subtitles."""
        try:
            # Ensure audio writer is closed
            audio_path = await self.audio_manager.close_segment(segment_index)
            
            if not audio_path:
                return
                
            video_path = f'output/video/output_video_segment_{segment_index}.mp4'
            subtitles = self.subtitle_manager.get_subtitles_for_segment(segment_index)
            
            if not os.path.exists(video_path) or not subtitles:
                return
                
            await self.mux_segment(
                segment_index,
                video_path,
                audio_path
            )
        except Exception as e:
            self.logger.error(f"Error processing segment {segment_index}: {e}")

    async def write_subtitle(self, index: int, start_time: float, end_time: float, text: str):
        """Write subtitle data directly to an SRT file for the current segment."""
        try:
            self.logger.debug(f"Writing subtitle {index}: '{text}' from {start_time} to {end_time}")

            # Calculate durations relative to video start time
            if self.video_start_time is None:
                self.logger.warning("Video start time is not initialized.")
                return

            adjusted_start_time = start_time - self.video_start_time
            adjusted_end_time = end_time - self.video_start_time

            if adjusted_end_time <= adjusted_start_time:
                self.logger.warning(f"Subtitle {index} skipped: Start time >= End time")
                return

            # Define maximum characters per line
            max_chars = 42

            # Split text into lines not exceeding max_chars
            lines = self.split_text_into_lines(text, max_chars)

            # Calculate duration per subtitle based on number of lines
            total_duration = adjusted_end_time - adjusted_start_time
            duration_per_subtitle = total_duration / len(lines) if len(lines) > 0 else total_duration

            subtitles = []
            for i, line in enumerate(lines):
                sub_start = datetime.timedelta(seconds=adjusted_start_time + i * duration_per_subtitle)
                sub_end = datetime.timedelta(seconds=adjusted_start_time + (i + 1) * duration_per_subtitle)
                subtitles.append(srt.Subtitle(index=index + i, start=sub_start, end=sub_end, content=line))
                # Append to subtitle_data_list for overlay
                self.subtitle_data_list.append({
                    'start_time': adjusted_start_time + i * duration_per_subtitle,
                    'end_time': adjusted_start_time + (i + 1) * duration_per_subtitle,
                    'text': line
                })

            srt_content = srt.compose(subtitles)
            srt_output_path = f'output/subtitles/subtitles_segment_{self.segment_index}.srt'
            async with aiofiles.open(srt_output_path, 'a', encoding='utf-8') as f:
                await f.write(srt_content)
            self.logger.info(f"Subtitle {index} for segment {self.segment_index} saved to {srt_output_path}")

            # Increment subtitle index for each split line
            self.subtitle_index += len(lines)

        except Exception as e:
            self.logger.error(f"Error writing subtitle {index}: {e}", exc_info=True)

    def split_text_into_lines(self, text: str, max_chars: int) -> list:
        """
        Splits a long text into multiple lines, each not exceeding max_chars.
        Attempts to split at word boundaries.
        """
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            if len(current_line) + len(word) + 1 <= max_chars:
                current_line += (" " if current_line else "") + word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines

    async def overlay_subtitles_via_ffmpeg(self, video_path: str, subtitles_path: str) -> Optional[str]:
        """Overlay subtitles onto the video using FFmpeg and save as a new file."""
        try:
            # Define the path for the temporary video with subtitles
            temp_video_with_subs = video_path.replace('.mp4', '_with_subs.mp4')

            ffmpeg_command = [
                'ffmpeg',
                '-y',
                '-i', video_path,
                '-vf', f"subtitles={subtitles_path}:force_style='FontSize=24,PrimaryColour=&HFFFFFF&'",
                '-c:v', 'libx264',  # Re-encode video to ensure compatibility
                '-c:a', 'copy',      # Copy the audio stream without re-encoding (no audio in video)
                temp_video_with_subs
            ]

            self.logger.info(f"Overlaying subtitles onto video {video_path}...")
            process = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if process.returncode == 0:
                self.logger.info(f"Successfully overlaid subtitles onto {temp_video_with_subs}")
                return temp_video_with_subs
            else:
                self.logger.error(f"FFmpeg subtitle overlay failed for {video_path}.")
                self.logger.error(f"FFmpeg stderr: {process.stderr}")
                return None
        except Exception as e:
            self.logger.error(f"Error during FFmpeg subtitle overlay: {e}", exc_info=True)
            return None

    async def manage_final_output_folder(self):
        """Ensure that the final output folder contains only the latest 10 videos."""
        final_output_dir = 'output/final'
        max_videos = 10

        try:
            videos = sorted(
                [f for f in os.listdir(final_output_dir) if f.endswith('.mp4')],
                key=lambda x: os.path.getmtime(os.path.join(final_output_dir, x))
            )
            while len(videos) > max_videos:
                oldest_video = videos.pop(0)
                oldest_video_path = os.path.join(final_output_dir, oldest_video)
                os.remove(oldest_video_path)
                self.logger.info(f"Removed oldest video: {oldest_video_path}")
        except Exception as e:
            self.logger.error(f"Error managing final output folder: {e}", exc_info=True)

    async def mux_video_audio(self, segment_index: int):
        """Mux video segment with corresponding audio segment and subtitles using FFmpeg"""
        video_path = f'output/video/output_video_segment_{segment_index}.mp4'
        audio_path = f'output/audio/output_audio_segment_{segment_index}.wav'
        subtitles_path = f'output/subtitles/subtitles_segment_{segment_index}.srt'
        final_output_path = f'output/final/output_final_video_segment_{segment_index}.mp4'  # Save in output/final

        # Check if video, audio, and subtitles files exist
        missing_files = []
        for path in [video_path, audio_path, subtitles_path]:
            if not os.path.exists(path):
                missing_files.append(path)
        if missing_files:
            self.logger.error(f"Missing files for muxing: {missing_files}. Cannot mux.")
            return

        # **Critical Fix:** Close the Wave_write object before muxing
        async with self.segment_audio_lock:
            wf = self.segment_audio_writers.pop(segment_index, None)
            if wf:
                wf.close()
                self.logger.debug(f"Closed Wave_write object for segment {segment_index}")
            else:
                self.logger.error(f"No Wave_write object found for segment {segment_index}")

        # Define a temporary video file with subtitles
        temp_video_with_subs = f'output/video/output_video_segment_{segment_index}_temp.mp4'

        # FFmpeg command to overlay subtitles
        ffmpeg_subtitles_command = [
            'ffmpeg',
            '-y',
            '-i', video_path,
            '-vf', f"subtitles={subtitles_path}:force_style='FontSize=24,PrimaryColour=&HFFFFFF&'",
            '-c:v', 'libx264',  # Re-encode video to ensure compatibility
            '-c:a', 'copy',      # No audio in video
            temp_video_with_subs
        ]

        try:
            self.logger.info(f"Overlaying subtitles onto video segment {segment_index}...")
            process_subs = subprocess.run(ffmpeg_subtitles_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if process_subs.returncode != 0:
                self.logger.error(f"FFmpeg subtitle overlay failed for segment {segment_index}.")
                self.logger.error(f"FFmpeg stderr: {process_subs.stderr}")
                return

            # **New Step:** Calculate padding duration and pad audio if necessary
            # Open the audio WAV file to get its duration
            with wave.open(audio_path, 'rb') as wf_audio:
                num_frames = wf_audio.getnframes()
                framerate = wf_audio.getframerate()
                audio_duration = num_frames / float(framerate)

            required_duration = self.segment_duration
            pad_duration = required_duration - audio_duration

            if pad_duration > 0:
                self.logger.debug(f"Padding audio with {pad_duration:.3f} seconds of silence.")

                # Generate silence audio using FFmpeg
                silence_path = f'output/audio/silence_segment_{segment_index}.wav'
                ffmpeg_silence_command = [
                    'ffmpeg',
                    '-y',
                    '-f', 'lavfi',
                    '-i', f'anullsrc=r={framerate}:cl=mono',
                    '-t', f'{pad_duration:.3f}',
                    '-c:a', 'pcm_s16le',
                    silence_path
                ]
                process_silence = subprocess.run(ffmpeg_silence_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                if process_silence.returncode != 0:
                    self.logger.error(f"FFmpeg failed to generate silence for segment {segment_index}.")
                    self.logger.error(f"FFmpeg stderr: {process_silence.stderr}")
                    return

                # Concatenate original audio with silence
                padded_audio_path = f'output/audio/output_audio_segment_{segment_index}_padded.wav'
                ffmpeg_concat_command = [
                    'ffmpeg',
                    '-y',
                    '-i', audio_path,
                    '-i', silence_path,
                    '-filter_complex', '[0:a][1:a]concat=n=2:v=0:a=1[a]',
                    '-map', '[a]',
                    '-c:a', 'pcm_s16le',
                    padded_audio_path
                ]
                process_concat = subprocess.run(ffmpeg_concat_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                if process_concat.returncode != 0:
                    self.logger.error(f"FFmpeg failed to concatenate audio and silence for segment {segment_index}.")
                    self.logger.error(f"FFmpeg stderr: {process_concat.stderr}")
                    return
            else:
                padded_audio_path = audio_path  # No padding needed

            # FFmpeg command to mux video with padded audio
            ffmpeg_mux_command = [
                'ffmpeg',
                '-y',
                '-i', temp_video_with_subs,
                '-i', padded_audio_path,
                '-c:v', 'copy',  # Copy video stream without re-encoding
                '-c:a', 'aac',    # Encode audio to AAC
                '-strict', 'experimental',
                final_output_path
            ]

            self.logger.info(f"Muxing video, audio, and subtitles for segment {segment_index}...")
            process_mux = subprocess.run(ffmpeg_mux_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if process_mux.returncode == 0:
                self.logger.info(f"Successfully muxed video, audio, and subtitles into {final_output_path}")
                # Optionally, remove the temporary video with subtitles and padded audio
                os.remove(temp_video_with_subs)
                self.logger.debug(f"Removed temporary video file: {temp_video_with_subs}")
                if pad_duration > 0:
                    os.remove(silence_path)
                    self.logger.debug(f"Removed temporary silence file: {silence_path}")
                    os.remove(padded_audio_path)
                    self.logger.debug(f"Removed temporary padded audio file: {padded_audio_path}")
                # Manage the final output folder
                await self.manage_final_output_folder()
            else:
                self.logger.error(f"FFmpeg muxing failed for segment {segment_index}.")
                self.logger.error(f"FFmpeg stderr: {process_mux.stderr}")
        except Exception as e:
            self.logger.error(f"Error during FFmpeg muxing: {e}", exc_info=True)
        finally:
            # Clear subtitle_data_list for the segment
            self.subtitle_data_list.clear()

    async def heartbeat(self):
        """Send periodic heartbeat pings to keep the WebSocket connection alive."""
        while self.running:
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

        if shutdown:
            self.running = False  # Stop all running loops and tasks
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

            # Close all segment Wave_write objects
            async with self.segment_audio_lock:
                for segment_index, wf in self.segment_audio_writers.items():
                    try:
                        wf.close()
                        self.logger.debug(f"Closed Wave_write object for segment {segment_index}")
                    except Exception as e:
                        self.logger.error(f"Error closing Wave_write object for segment {segment_index}: {e}")
                self.segment_audio_writers.clear()

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
            segment_frames = int(fps * self.segment_duration)
            frame_count = 0
            self.segment_start_time = time.perf_counter()
            if self.video_start_time is None:
                self.video_start_time = self.segment_start_time  # Set video_start_time for synchronization

            while self.running:
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
                    wf.setsampwidth(2)  # 16-bit PCM
                    wf.setframerate(24000)
                    self.segment_audio_writers[self.segment_index] = wf
                    self.logger.debug(f"Initialized audio segment file: {audio_segment_path}")
                except Exception as e:
                    self.logger.error(f"Failed to initialize audio segment file {audio_segment_path}: {e}", exc_info=True)

                # Reset subtitle data for the new segment
                self.subtitle_data_list.clear()

                while frame_count < segment_frames and self.running:
                    ret, frame = cap.read()
                    if not ret:
                        self.logger.warning("Failed to read frame from video stream.")
                        break

                    # Write frame to output video
                    out.write(frame)
                    frame_count += 1

                # Release the current segment video writer
                out.release()
                self.logger.info(f"Segment {self.segment_index} saved to {segment_output_path}.")

                # Schedule muxing of video and audio in the asyncio loop
                asyncio.run_coroutine_threadsafe(
                    self.mux_video_audio(self.segment_index),
                    self.loop
                )

                # Reset for next segment
                self.segment_index += 1
                frame_count = 0
                self.segment_start_time = time.perf_counter()

        except Exception as e:
            self.logger.error(f"Error in start_video_processing: {e}", exc_info=True)
            self.running = False

    async def run(self):
        """Run the OpenAIClient."""
        # Handle graceful shutdown on signals
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self.loop.add_signal_handler(sig, lambda sig=sig: asyncio.create_task(self.shutdown(sig)))
            except NotImplementedError:
                # Signal handling might not be implemented on some platforms (e.g., Windows)
                self.logger.warning(f"Signal handling not supported on this platform.")

        # Start video processing
        video_processing_task = asyncio.create_task(self.run_video_processing())

        # Attempt to connect initially
        try:
            await self.connect()
        except Exception as e:
            self.logger.error(f"Initial connection failed: {e}")
            await self.reconnect()

        # Start handling responses
        handle_responses_task = asyncio.create_task(self.handle_responses())
        # Start reading and sending audio
        read_input_audio_task = asyncio.create_task(self.read_input_audio())
        # Start heartbeat
        heartbeat_task = asyncio.create_task(self.heartbeat())

        # Wait for tasks to complete
        done, pending = await asyncio.wait(
            [handle_responses_task, read_input_audio_task, heartbeat_task, video_processing_task],
            return_when=asyncio.FIRST_EXCEPTION
        )

        for task in pending:
            task.cancel()

    async def shutdown(self, sig):
        """Cleanup tasks tied to the service's shutdown."""
        self.logger.info(f"Received exit signal {sig.name}...")
        await self.disconnect(shutdown=True)
        self.logger.info("Shutdown complete.")
        self.loop.stop()

    async def muxing_worker(self):
        """Placeholder for any future muxing tasks or workers."""
        while self.running:
            await asyncio.sleep(1)  # Currently no tasks, keep the coroutine alive.

    async def send_input_audio_buffer_append(self, audio_data: bytes):
        """Send audio with event tracking"""
        event_id = str(uuid.uuid4())
        timestamp = time.perf_counter()
        
        self.event_tracker.register_input_event(event_id, timestamp, audio_data)
        
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        event = {
            "event_id": event_id,
            "type": "input_audio_buffer.append",
            "audio": audio_base64
        }
        await self.enqueue_message(json.dumps(event))
    

    def setup_directories_and_pipes(self):
        self.create_directories()
        self.input_audio_pipe = os.path.abspath('input_audio_pipe')
        self.create_named_pipe(self.input_audio_pipe)

    def initialize_gender_detection(self):
        self.gender_voice_map = {
            'male': ['alloy', 'ash', 'coral'],
            'female': ['echo', 'shimmer', 'ballad', 'sage', 'verse']
        }

    self.current_voice = {'male': 0, 'female': 1}
    self.latest_input_audio = b''
    async def send_session_update_with_event_id(self, event_id: str):
        """Send a session.update event with a specific event_id."""
        session_update_event = {
            "event_id": event_id,
            "type": "session.update",
            "session": {
                "model": "gpt-4o-realtime-preview-2024-10-01",
                "modalities": ["text", "audio"],
                "instructions": (
                    "You are a realtime translator. "
                    "Please use the audio you receive and translate it into German. "
                    "Do not provide any additional responses or engage in conversations."
                ),
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500
                },
                "temperature": 0.7
            }
        }
        await self.enqueue_message(json.dumps(session_update_event))
        self.logger.debug(f"Enqueued session.update event with event_id: {event_id}")
        # Note: No mapping since server event_ids are different

    def generate_event_id(self) -> str:
        """Generate a unique event ID."""
        return str(uuid.uuid4())
