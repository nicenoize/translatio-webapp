import asyncio
import aiofiles
import websockets
import json
import logging
import os
import base64
import wave
import numpy as np
import simpleaudio as sa
from asyncio import Queue
from collections import deque
import datetime
from contextlib import suppress
import time
import cv2
from logging.handlers import RotatingFileHandler
import subprocess
import signal
import uuid
import csv
import statistics
from aiohttp import web
from typing import Optional

# Constants
API_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
AUDIO_SAMPLE_RATE = 24000  # 24kHz
AUDIO_CHANNELS = 1         # Mono
AUDIO_SAMPLE_WIDTH = 2     # 16-bit PCM
SUBTITLE_PATH = 'output/subtitles/subtitles.vtt'
OUTPUT_WAV_PATH = 'output/audio/output_audio.wav'
STREAM_URL = 'https://bintu-play.nanocosmos.de/h5live/http/stream.mp4?url=rtmp://localhost/play&stream=sNVi5-kYN1t'
MUXING_QUEUE_MAXSIZE = 100
DASHBOARD_PORT = 8080
INPUT_AUDIO_PIPE = os.path.abspath('input_audio_pipe')  # Absolute path for the named pipe

class OpenAIClient:
    def __init__(self, api_key: str, loop: asyncio.AbstractEventLoop, session_id: Optional[str] = None):
        """
        Initialize the OpenAIClient.

        :param api_key: OpenAI API key.
        :param loop: The asyncio event loop.
        :param session_id: Optional session identifier for managing multiple sessions.
        """
        self.api_key = api_key
        self.loop = loop
        self.ws = None
        self.running = True
        self.session_id = session_id or str(uuid.uuid4())

        # Setup logging
        self.setup_logging()

        # Queues for handling messages and muxing
        self.send_queue = Queue()
        self.translated_audio_queue = Queue()
        self.muxing_queue = Queue(maxsize=MUXING_QUEUE_MAXSIZE)

        # Playback buffer
        self.audio_buffer = deque(maxlen=100)
        self.playback_event = asyncio.Event()

        # Timestamps
        self.video_start_time = None

        # Subtitle management
        self.subtitle_index = 1

        # Initialize output WAV file
        self.init_output_wav()

        # Initialize WebVTT file
        asyncio.create_task(self.initialize_vtt_file())

        # Setup directories
        self.setup_directories()

        # Initialize playback task
        self.playback_task = asyncio.create_task(self.audio_playback_handler())

        # Initialize muxing task
        self.muxing_task = asyncio.create_task(self.mux_audio_video_subtitles())

        # Reconnect handling
        self.is_reconnecting = False
        self.reconnect_lock = asyncio.Lock()

        # Delay benchmarking
        self.processing_delays = deque(maxlen=100)
        self.average_processing_delay = 0.0
        self.delay_benchmark_file = f'output/logs/delay_benchmark_{self.session_id}.csv'
        self.setup_delay_benchmarking()

        # Initialize the monitoring dashboard
        self.metrics = {
            "processing_delays": deque(maxlen=100),
            "average_processing_delay": 0.0,
            "min_processing_delay": 0.0,
            "max_processing_delay": 0.0,
            "stddev_processing_delay": 0.0,
            "buffer_status": 0,
            "audio_queue_size": 0,
            "muxing_queue_size": 0
        }
        self.dashboard_server = None  # Will be initialized later

        # Queue to track sent audio timestamps
        self.sent_audio_timestamps = deque()

        # Video processing task
        self.video_processing_task = None

        # Reference to current audio segment WAV file for writing translated audio
        self.current_audio_segment_wf = None

    def setup_logging(self):
        """Setup logging with RotatingFileHandler."""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.DEBUG)

        # File handler
        os.makedirs('output/logs', exist_ok=True)
        file_handler = RotatingFileHandler(f"output/logs/app_{self.session_id}.log", maxBytes=10*1024*1024, backupCount=5)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        self.logger = logger
        self.logger.info("Logging initialized.")

    def setup_directories(self):
        """Create necessary directories."""
        directories = [
            'output/transcripts',
            'output/audio/input',
            'output/audio/output',
            'output/audio/responses',
            'output/audio/processed',
            'output/subtitles',
            'output/logs',
            'output/video',
            'output/final',
            'output/images'
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            self.logger.debug(f"Ensured directory exists: {directory}")

    def init_output_wav(self):
        """Initialize the output WAV file."""
        try:
            os.makedirs('output/audio/output', exist_ok=True)
            self.output_wav = wave.open(OUTPUT_WAV_PATH, 'wb')
            self.output_wav.setnchannels(AUDIO_CHANNELS)
            self.output_wav.setsampwidth(AUDIO_SAMPLE_WIDTH)
            self.output_wav.setframerate(AUDIO_SAMPLE_RATE)
            self.logger.info(f"Output WAV file initialized at {OUTPUT_WAV_PATH}.")
        except Exception as e:
            self.logger.error(f"Failed to initialize output WAV file: {e}")
            self.output_wav = None

    async def initialize_vtt_file(self):
        """Initialize the WebVTT file by writing the header."""
        try:
            async with aiofiles.open(SUBTITLE_PATH, 'w', encoding='utf-8') as f:
                await f.write("WEBVTT\n\n")
            self.logger.info(f"WebVTT file initialized at {SUBTITLE_PATH}")
        except Exception as e:
            self.logger.error(f"Failed to initialize WebVTT file: {e}")

    def setup_delay_benchmarking(self):
        """Setup CSV for delay benchmarking."""
        try:
            if not os.path.exists(self.delay_benchmark_file):
                with open(self.delay_benchmark_file, 'w', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow([
                        'Timestamp',
                        'ProcessingDelay',
                        'AverageProcessingDelay',
                        'MinProcessingDelay',
                        'MaxProcessingDelay',
                        'StdDevProcessingDelay'
                    ])
                self.logger.info(f"Delay benchmark CSV initialized at {self.delay_benchmark_file}")
            else:
                self.logger.info(f"Delay benchmark CSV already exists at {self.delay_benchmark_file}")
        except Exception as e:
            self.logger.error(f"Failed to setup delay benchmarking: {e}")

    def log_delay_metrics(self):
        """Log delay metrics to CSV and update monitoring metrics."""
        if not self.processing_delays:
            self.logger.debug("No processing delays to log.")
            return

        current_time = datetime.datetime.utcnow().isoformat()
        processing_delay = self.processing_delays[-1]
        average_delay = statistics.mean(self.processing_delays)
        min_delay = min(self.processing_delays)
        max_delay = max(self.processing_delays)
        stddev_delay = statistics.stdev(self.processing_delays) if len(self.processing_delays) > 1 else 0.0

        # Update monitoring metrics
        self.metrics["processing_delays"].append(processing_delay)
        self.metrics["average_processing_delay"] = average_delay
        self.metrics["min_processing_delay"] = min_delay
        self.metrics["max_processing_delay"] = max_delay
        self.metrics["stddev_processing_delay"] = stddev_delay
        self.metrics["buffer_status"] = len(self.audio_buffer)
        self.metrics["audio_queue_size"] = self.translated_audio_queue.qsize()
        self.metrics["muxing_queue_size"] = self.muxing_queue.qsize()

        try:
            with open(self.delay_benchmark_file, 'a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([
                    current_time,
                    f"{processing_delay:.6f}",
                    f"{average_delay:.6f}",
                    f"{min_delay:.6f}",
                    f"{max_delay:.6f}",
                    f"{stddev_delay:.6f}"
                ])
            self.logger.debug("Delay metrics logged.")
        except Exception as e:
            self.logger.error(f"Failed to log delay metrics: {e}")

    async def connect(self):
        """Establish WebSocket connection to Realtime API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        try:
            self.ws = await websockets.connect(API_URL, extra_headers=headers)
            self.logger.info("Connected to OpenAI Realtime API.")
            await self.initialize_session()
        except Exception as e:
            self.logger.error(f"Failed to connect to Realtime API: {e}")
            raise

    async def initialize_session(self):
        """Initialize session with desired configurations."""
        session_update_event = {
            "type": "session.update",
            "session": {
                "instructions": (
                    "You are a real-time translator. Translate the audio you receive into German without performing Voice Activity Detection (VAD). "
                    "Ensure that the translated audio matches the input audio's duration and timing exactly to facilitate synchronization with video. "
                    "Provide detailed and comprehensive translations without truncating sentences. "
                    "Do not respond conversationally."
                ),
                "modalities": ["text", "audio"],
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "temperature": 0.8,
                "tools": []  # Add tool definitions here if needed
            }
        }
        await self.enqueue_message(session_update_event)
        self.logger.info("Session update event enqueued.")

    async def enqueue_message(self, message: dict):
        """Enqueue a message to be sent over WebSocket."""
        await self.send_queue.put(message)
        self.logger.debug(f"Message enqueued: {message['type']}")

    async def enqueue_muxing_job(self, job: dict):
        """
        Enqueue a muxing job to the muxing_queue.

        :param job: A dictionary containing paths for video, audio, subtitles, and output.
        """
        try:
            await self.muxing_queue.put(job)
            self.logger.debug(f"Enqueued muxing job for segment {job.get('segment_index')}.")
        except asyncio.QueueFull:
            self.logger.error("Muxing queue is full. Failed to enqueue muxing job.")
        except Exception as e:
            self.logger.error(f"Failed to enqueue muxing job: {e}")

    async def send_messages(self):
        """Coroutine to send messages from the send_queue."""
        while self.running:
            message = await self.send_queue.get()
            if message is None:
                self.logger.info("Send queue received shutdown signal.")
                break
            try:
                await self.safe_send(json.dumps(message))
            except Exception as e:
                self.logger.error(f"Failed to send message: {e}")
                await asyncio.sleep(5)  # Wait before retrying
            finally:
                self.send_queue.task_done()

    async def safe_send(self, data: str):
        """Send data over WebSocket with error handling and reconnection."""
        try:
            if self.ws and self.ws.open:
                await self.ws.send(data)
                self.logger.debug(f"Sent message: {data[:50]}...")
            else:
                self.logger.warning("WebSocket is closed. Attempting to reconnect...")
                await self.reconnect()
                if self.ws and self.ws.open:
                    await self.ws.send(data)
                    self.logger.debug(f"Sent message after reconnection: {data[:50]}...")
                else:
                    self.logger.error("Failed to reconnect. Cannot send data.")
        except (websockets.exceptions.ConnectionClosed, websockets.exceptions.ConnectionClosedError) as e:
            self.logger.error(f"WebSocket connection closed during send: {e}")
            await self.reconnect()
            if self.ws and self.ws.open:
                await self.ws.send(data)
                self.logger.debug(f"Sent message after reconnection: {data[:50]}...")
            else:
                self.logger.error("Failed to reconnect. Cannot send data.")
        except Exception as e:
            self.logger.error(f"Exception during WebSocket send: {e}")

    async def reconnect(self):
        """Reconnect to the WebSocket server with exponential backoff."""
        async with self.reconnect_lock:
            if self.is_reconnecting:
                self.logger.debug("Already reconnecting. Skipping additional attempts.")
                return
            self.is_reconnecting = True
            backoff = 1
            max_backoff = 60
            while not self.ws or not self.ws.open:
                self.logger.info(f"Reconnecting in {backoff} seconds...")
                await asyncio.sleep(backoff)
                try:
                    await self.connect()
                    self.logger.info("Reconnected to OpenAI Realtime API.")
                    self.is_reconnecting = False
                    return
                except Exception as e:
                    self.logger.error(f"Reconnect attempt failed: {e}")
                    backoff = min(backoff * 2, max_backoff)
            self.is_reconnecting = False

    async def handle_responses(self):
        """Handle incoming messages from the WebSocket."""
        while self.running:
            try:
                response = await self.ws.recv()
                event = json.loads(response)
                self.logger.debug(f"Received event: {event}")  # Log entire event for debugging
                await self.process_event(event)
            except websockets.exceptions.ConnectionClosedOK:
                self.logger.warning("WebSocket connection closed normally.")
                await self.reconnect()
            except websockets.exceptions.ConnectionClosedError as e:
                self.logger.error(f"WebSocket connection closed with error: {e}")
                await self.reconnect()
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decode error: {e} - Message: {response}")
            except Exception as e:
                self.logger.error(f"Unexpected error in handle_responses: {e}")
                await self.reconnect()

    async def process_event(self, event: dict):
        """Process a single event from the WebSocket."""
        event_type = event.get("type")
        self.logger.debug(f"Processing event: {event_type}")

        if event_type == "input_audio_buffer.speech_started":
            self.logger.info("Speech started detected by server.")
            # No action needed as server handles VAD

        elif event_type == "input_audio_buffer.speech_stopped":
            self.logger.info("Speech stopped detected by server.")
            await self.commit_audio_buffer()

        elif event_type == "conversation.item.created":
            item = event.get("item", {})
            if item.get("type") == "message" and item.get("role") == "assistant":
                self.logger.info("Assistant message received.")
                # Handle assistant messages if needed

            elif item.get("type") == "function_call":
                self.logger.info("Function call detected.")
                await self.handle_function_call(item)

        elif event_type == "response.audio.delta":
            audio_data = event.get("delta", "")
            if audio_data:
                await self.handle_audio_delta(audio_data)
            else:
                self.logger.debug("Received empty audio delta.")

        elif event_type == "response.audio_transcript.done":
            transcript = event.get("transcript", "")
            if self.sent_audio_timestamps:
                sent_time = self.sent_audio_timestamps.popleft()
                current_time = time.perf_counter()
                processing_delay = current_time - sent_time
                # Calculate audio duration based on chunk size
                audio_data = event.get("audio", "")
                audio_chunk_size = len(base64.b64decode(audio_data)) if audio_data else 0
                audio_duration = audio_chunk_size / (AUDIO_SAMPLE_RATE * AUDIO_CHANNELS * AUDIO_SAMPLE_WIDTH) if audio_chunk_size else 0.0
                # Calculate start and end times relative to video_start_time
                if self.video_start_time:
                    start_time = sent_time - self.video_start_time
                    end_time = start_time + audio_duration + processing_delay
                else:
                    # If video_start_time is not set, default to 0
                    start_time = 0.0
                    end_time = audio_duration + processing_delay

                self.logger.info(f"Received transcript: {transcript}")
                self.logger.info(f"Subtitle timing - Start: {start_time}, End: {end_time}")

                # Append processing_delay
                self.processing_delays.append(processing_delay)
                self.log_delay_metrics()

                await self.write_vtt_subtitle(self.subtitle_index, start_time, end_time, transcript)
                self.subtitle_index += 1
            else:
                self.logger.warning("No sent audio timestamp available for transcript.")
                # Optionally, assign default timings or skip

        elif event_type == "response.content_part.done":
            content_part = event.get("content_part", "")
            # Handle content part if necessary
            self.logger.info(f"Received content part: {content_part}")
            # Depending on content, may need to process further

        elif event_type == "response.output_item.done":
            output_item = event.get("output_item", "")
            # Handle output item if necessary
            self.logger.info(f"Received output item: {output_item}")
            # Depending on content, may need to process further

        elif event_type == "response.done":
            self.logger.info("Response processing completed.")
            # Possibly perform any finalization or cleanup

        elif event_type == "error":
            error = event.get("error", {})
            self.logger.error(f"Error received: {error.get('message')}, Code: {error.get('code')}")

        else:
            self.logger.warning(f"Unhandled event type: {event_type}")

    async def handle_audio_delta(self, audio_data: str):
        """Handle incoming audio delta from the server."""
        try:
            decoded_audio = base64.b64decode(audio_data)
            if not decoded_audio:
                self.logger.warning("Decoded audio data is empty.")
                return

            # Save raw audio for verification
            response_audio_filename = f"{uuid.uuid4()}.wav"
            response_audio_path = os.path.join('output/audio/responses', response_audio_filename)
            with wave.open(response_audio_path, 'wb') as wf_response:
                wf_response.setnchannels(AUDIO_CHANNELS)
                wf_response.setsampwidth(AUDIO_SAMPLE_WIDTH)
                wf_response.setframerate(AUDIO_SAMPLE_RATE)
                wf_response.writeframes(decoded_audio)
            self.logger.info(f"Saved raw audio response to {response_audio_path}")

            # Enqueue audio for playback only once
            await self.translated_audio_queue.put(decoded_audio)
            self.logger.debug("Enqueued translated audio for playback.")

            # No processing_delay calculation here

        except Exception as e:
            self.logger.error(f"Error handling audio delta: {e}")

    async def handle_function_call(self, item: dict):
        """
        Handle function call events from the server.

        :param item: The function call item.
        """
        try:
            function_name = item.get("name")
            call_id = item.get("call_id")
            arguments = item.get("arguments", "")

            self.logger.info(f"Handling function call: {function_name} with call_id: {call_id}")

            # Parse arguments
            args = json.loads(arguments) if arguments else {}

            # Execute the function (define your functions here)
            result = await self.execute_function(function_name, args)

            # Send function_call_output
            function_output_event = {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(result)
                }
            }
            await self.enqueue_message(function_output_event)
            self.logger.info(f"Sent function_call_output for call_id: {call_id}")

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error in function_call arguments: {e}")
        except Exception as e:
            self.logger.error(f"Error handling function call: {e}")

    async def execute_function(self, function_name: str, args: dict) -> dict:
        """
        Execute a function based on its name and arguments.

        :param function_name: The name of the function to execute.
        :param args: Arguments for the function.
        :return: Result of the function execution.
        """
        # Define your custom functions here
        if function_name == "get_weather":
            location = args.get("location", "unknown")
            # Simulate fetching weather data
            weather_info = {
                "location": location,
                "temperature": "20°C",
                "condition": "Sunny"
            }
            self.logger.info(f"Executed get_weather for location: {location}")
            return weather_info

        # Add more functions as needed

        else:
            self.logger.warning(f"Unknown function: {function_name}")
            return {"error": f"Unknown function: {function_name}"}

    async def commit_audio_buffer(self):
        """Commit the audio buffer to signal end of speech."""
        commit_event = {
            "type": "input_audio_buffer.commit"
        }
        await self.enqueue_message(commit_event)
        self.logger.info("Committed audio buffer.")
        # After committing, request a response
        await self.create_response()

    async def create_response(self):
        """Create a response to trigger audio generation."""
        response_event = {
            "type": "response.create",
            "response": {
                "modalities": ["text", "audio"],
                "instructions": (
                    "You are a real-time translator. Translate the audio you receive into German without performing Voice Activity Detection (VAD). "
                    "Ensure that the translated audio matches the input audio's duration and timing exactly to facilitate synchronization with video. "
                    "Provide detailed and comprehensive translations without truncating sentences. "
                    "Do not respond conversationally."
                ),
                "voice": "alloy",
                "tools": []  # Add tool definitions here if needed
            }
        }
        await self.enqueue_message(response_event)
        self.logger.info("Created response event.")

    async def read_input_audio(self):
        """Read audio from a named pipe and send to Realtime API."""
        self.create_named_pipe(INPUT_AUDIO_PIPE)
        self.logger.info("Starting to read from input_audio_pipe.")
        while self.running:
            try:
                async with aiofiles.open(INPUT_AUDIO_PIPE, 'rb') as pipe:
                    while self.running:
                        data = await pipe.read(131072)  # 128KB
                        if not data:
                            await asyncio.sleep(0.05)
                            continue
                        await self.send_input_audio(data)
                        self.logger.info(f"Enqueued audio chunk of size: {len(data)} bytes.")
            except Exception as e:
                self.logger.error(f"Error reading input audio: {e}")
                await asyncio.sleep(1)  # Prevent tight loop on error

    def create_named_pipe(self, pipe_name: str):
        """Create a named pipe if it doesn't exist."""
        try:
            if not os.path.exists(pipe_name):
                os.mkfifo(pipe_name)
                self.logger.info(f"Created named pipe: {pipe_name}")
            else:
                self.logger.info(f"Named pipe already exists: {pipe_name}")
        except Exception as e:
            self.logger.error(f"Error creating named pipe {pipe_name}: {e}")
            raise

    async def send_input_audio(self, audio_data: bytes):
        """Send input audio to the Realtime API and record the send time."""
        try:
            pcm_base64 = base64.b64encode(audio_data).decode('utf-8')
            audio_event = {
                "type": "input_audio_buffer.append",
                "audio": pcm_base64
            }
            await self.enqueue_message(audio_event)
            sent_time = time.perf_counter()
            self.sent_audio_timestamps.append(sent_time)
            self.logger.debug(f"Sent input audio buffer append event at {sent_time}.")
        except Exception as e:
            self.logger.error(f"Failed to send input audio: {e}")

    async def audio_playback_handler(self):
        """Handle playback of translated audio."""
        while self.running:
            try:
                audio_data = await self.translated_audio_queue.get()
                await self.play_audio(audio_data)
                self.translated_audio_queue.task_done()
            except Exception as e:
                self.logger.error(f"Error in audio_playback_handler: {e}")

    async def play_audio(self, audio_data: bytes):
        """Play audio using simpleaudio and write to both output WAV and current audio segment WAV."""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            play_obj = sa.play_buffer(audio_array, AUDIO_CHANNELS, AUDIO_SAMPLE_WIDTH, AUDIO_SAMPLE_RATE)
            await asyncio.to_thread(play_obj.wait_done)
            self.logger.debug("Played translated audio chunk.")

            # Write to output WAV
            if self.output_wav:
                self.output_wav.writeframes(audio_data)
                self.logger.debug("Written translated audio chunk to output WAV file.")

            # Write to current audio segment WAV
            if self.current_audio_segment_wf:
                self.current_audio_segment_wf.writeframes(audio_data)
                self.logger.debug("Written translated audio chunk to current audio segment WAV file.")

        except Exception as e:
            self.logger.error(f"Error playing audio: {e}")

    async def mux_audio_video_subtitles(self):
        """
        Asynchronous task to mux audio, video, and subtitles using FFmpeg.
        """
        while self.running:
            try:
                # Wait for a muxing job
                job = await self.muxing_queue.get()
                if job is None:
                    self.logger.info("Muxing queue received shutdown signal.")
                    break

                video_path = job['video']
                audio_path = job['audio']
                subtitles_path = job['subtitles']
                final_output_path = job['output']
                segment_index = job['segment_index']

                self.logger.info(f"Muxing audio and video for segment {segment_index}.")

                # Check if video and audio files exist
                if not os.path.exists(video_path):
                    self.logger.error(f"Video file does not exist: {video_path}")
                    self.muxing_queue.task_done()
                    continue
                if not os.path.exists(audio_path):
                    self.logger.error(f"Audio file does not exist: {audio_path}")
                    self.muxing_queue.task_done()
                    continue

                # Check if video file has valid data using ffprobe
                probe = subprocess.run(
                    ['ffprobe', '-v', 'error', '-show_format', '-show_streams', video_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                if probe.returncode != 0 or 'moov' not in probe.stdout:
                    self.logger.error(f"Invalid video file: {video_path}")
                    self.muxing_queue.task_done()
                    continue

                # FFmpeg command to mux audio, video, and subtitles
                ffmpeg_command = [
                    'ffmpeg',
                    '-y',  # Overwrite output files without asking
                    '-i', video_path,
                    '-i', audio_path,
                    '-vf', f"subtitles={subtitles_path}:force_style='FontSize=24,PrimaryColour=&HFFFFFF&'",
                    '-c:v', 'libx264',  # Re-encode video to ensure compatibility
                    '-c:a', 'aac',       # Encode audio to AAC for better compatibility
                    '-strict', 'experimental',
                    final_output_path
                ]

                self.logger.debug(f"Running FFmpeg command: {' '.join(ffmpeg_command)}")
                process = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                if process.returncode == 0:
                    self.logger.info(f"Successfully muxed video to {final_output_path}")
                else:
                    self.logger.error(f"FFmpeg muxing failed for segment {segment_index}.")
                    self.logger.error(f"FFmpeg stderr: {process.stderr}")
                    self.logger.error(f"FFmpeg stdout: {process.stdout}")

                self.muxing_queue.task_done()

            except Exception as e:
                self.logger.error(f"Error in mux_audio_video_subtitles: {e}")

    async def write_vtt_subtitle(self, index: int, start_time: float, end_time: float, text: str):
        """
        Write a single subtitle entry to the WebVTT file.

        :param index: Subtitle index.
        :param start_time: Start time in seconds.
        :param end_time: End time in seconds.
        :param text: Subtitle text.
        """
        try:
            # Convert seconds to WebVTT timestamp format
            start_vtt = self.format_timestamp_vtt(start_time)
            end_vtt = self.format_timestamp_vtt(end_time)

            # Prepare subtitle entry
            subtitle = f"{index}\n{start_vtt} --> {end_vtt}\n{text}\n\n"

            # Append to the WebVTT file
            async with aiofiles.open(SUBTITLE_PATH, 'a', encoding='utf-8') as f:
                await f.write(subtitle)
            self.logger.debug(f"Written subtitle {index} to WebVTT.")
        except Exception as e:
            self.logger.error(f"Error writing WebVTT subtitle {index}: {e}")

    def format_timestamp_vtt(self, seconds: float) -> str:
        """
        Format seconds to WebVTT timestamp format HH:MM:SS.mmm.

        :param seconds: Time in seconds.
        :return: Formatted timestamp string.
        """
        td = datetime.timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        milliseconds = int((td.total_seconds() - total_seconds) * 1000)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"

    async def run_dashboard_server(self):
        """Run the real-time monitoring dashboard using aiohttp."""
        app = web.Application()
        app.add_routes([
            web.get('/', self.handle_dashboard),
            web.get('/metrics', self.metrics_endpoint)
        ])

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', DASHBOARD_PORT)
        await site.start()
        self.logger.info(f"Monitoring dashboard started at http://localhost:{DASHBOARD_PORT}")

        # Keep the dashboard running
        while self.running:
            await asyncio.sleep(3600)

    async def handle_dashboard(self, request):
        """Handle HTTP requests to the dashboard."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Real-Time Translator Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; }}
                th {{ background-color: #f2f2f2; }}
            </style>
            <script>
                async function fetchMetrics() {{
                    const response = await fetch('/metrics');
                    const data = await response.json();
                    document.getElementById('avg_delay').innerText = data.average_processing_delay.toFixed(6);
                    document.getElementById('min_delay').innerText = data.min_processing_delay.toFixed(6);
                    document.getElementById('max_delay').innerText = data.max_processing_delay.toFixed(6);
                    document.getElementById('stddev_delay').innerText = data.stddev_processing_delay.toFixed(6);
                    document.getElementById('buffer_status').innerText = data.buffer_status;
                    document.getElementById('audio_queue_size').innerText = data.audio_queue_size;
                    document.getElementById('muxing_queue_size').innerText = data.muxing_queue_size;
                }}

                setInterval(fetchMetrics, 1000);
            </script>
        </head>
        <body>
            <h1>Real-Time Translator Dashboard</h1>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Average Processing Delay (s)</td><td id="avg_delay">0.0</td></tr>
                <tr><td>Minimum Processing Delay (s)</td><td id="min_delay">0.0</td></tr>
                <tr><td>Maximum Processing Delay (s)</td><td id="max_delay">0.0</td></tr>
                <tr><td>Std Dev Processing Delay (s)</td><td id="stddev_delay">0.0</td></tr>
                <tr><td>Audio Buffer Size</td><td id="buffer_status">0</td></tr>
                <tr><td>Translated Audio Queue Size</td><td id="audio_queue_size">0</td></tr>
                <tr><td>Muxing Queue Size</td><td id="muxing_queue_size">0</td></tr>
            </table>
        </body>
        </html>
        """
        return web.Response(text=html, content_type='text/html')

    async def metrics_endpoint(self, request):
        """Provide metrics in JSON format for the dashboard."""
        # Convert deque to list for JSON serialization
        serializable_metrics = self.metrics.copy()
        serializable_metrics["processing_delays"] = list(serializable_metrics["processing_delays"])
        return web.json_response(serializable_metrics)

    async def run(self):
        """Run the OpenAIClient."""
        # Start the dashboard
        dashboard_task = asyncio.create_task(self.run_dashboard_server())

        # Start WebSocket connection
        try:
            await self.connect()
        except Exception as e:
            self.logger.error(f"Initial connection failed: {e}")
            await self.reconnect()

        # Start message sender
        send_task = asyncio.create_task(self.send_messages())

        # Start handling responses
        handle_responses_task = asyncio.create_task(self.handle_responses())

        # Start reading input audio
        read_audio_task = asyncio.create_task(self.read_input_audio())

        # Start heartbeat
        heartbeat_task = asyncio.create_task(self.heartbeat())

        # Start video processing
        self.video_processing_task = asyncio.create_task(self.run_video_processing())

        # No need to start another muxing_task here since it's already started in __init__

        # Wait for all tasks to complete
        done, pending = await asyncio.wait(
            [
                send_task,
                handle_responses_task,
                read_audio_task,
                heartbeat_task,
                self.video_processing_task,
                dashboard_task
            ],
            return_when=asyncio.FIRST_EXCEPTION
        )

        # Cancel pending tasks
        for task in pending:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    async def heartbeat(self):
        """Send periodic heartbeat pings to keep the WebSocket connection alive."""
        while self.running:
            try:
                if self.ws and self.ws.open:
                    await self.ws.ping()
                    self.logger.debug("Sent heartbeat ping.")
                else:
                    self.logger.warning("WebSocket is closed. Attempting to reconnect...")
                    await self.reconnect()
            except Exception as e:
                self.logger.error(f"Error in heartbeat: {e}")
                await self.reconnect()
            await asyncio.sleep(30)  # Ping every 30 seconds

    async def disconnect(self):
        """Gracefully disconnect the client."""
        self.logger.info("Disconnecting the client...")
        self.running = False

        # Close WebSocket
        if self.ws:
            try:
                await self.ws.close()
                self.logger.info("WebSocket connection closed.")
            except Exception as e:
                self.logger.error(f"Error closing WebSocket: {e}")

        # Cancel playback task
        if self.playback_task:
            self.playback_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.playback_task

        # Cancel muxing task
        if self.muxing_task:
            self.muxing_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.muxing_task

        # Cancel video processing task
        if self.video_processing_task:
            self.video_processing_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.video_processing_task

        # Cancel dashboard task if necessary
        # Assuming dashboard_task is stored or accessible
        # For simplicity, it was not stored; it will be canceled in the run method.

        # Close current audio segment WAV file
        if self.current_audio_segment_wf:
            try:
                self.current_audio_segment_wf.close()
                self.logger.debug("Closed current audio segment WAV file.")
            except Exception as e:
                self.logger.error(f"Error closing current audio segment WAV file: {e}")

        # Close WAV file
        if self.output_wav:
            try:
                self.output_wav.close()
                self.logger.info("Output WAV file closed.")
            except Exception as e:
                self.logger.error(f"Error closing WAV file: {e}")

        self.logger.info("Client disconnected successfully.")

    async def shutdown(self, sig):
        """Handle shutdown signals."""
        self.logger.info(f"Received exit signal {sig.name}...")
        await self.disconnect()

    def test_video_writer(self):
        """Test function to write a sample video and verify its integrity."""
        try:
            self.logger.info("Starting test video writer.")
            test_video_path = 'output/video/test_video.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Alternative codec
            fps = 30.0
            width = 1280
            height = 720
            out = cv2.VideoWriter(test_video_path, fourcc, fps, (width, height))
            cap = cv2.VideoCapture(STREAM_URL)
            if not cap.isOpened():
                self.logger.error("Cannot open video stream for testing.")
                return
            for i in range(int(fps * 5)):  # Write 5 seconds of video
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame during test video writing.")
                    break
                out.write(frame)
            out.release()
            cap.release()
            self.logger.info(f"Test video written to {test_video_path}")
            # Verify video file
            probe = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_format', '-show_streams', test_video_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if probe.returncode == 0 and 'moov' in probe.stdout:
                self.logger.info("Test video is valid.")
            else:
                self.logger.error("Test video is invalid.")
                self.logger.error(f"FFprobe output: {probe.stderr}")
        except Exception as e:
            self.logger.error(f"Error in test_video_writer: {e}")

    async def run_video_processing(self):
        """Run video processing within the asyncio event loop."""
        await asyncio.to_thread(self.start_video_processing)

    def start_video_processing(self):
        """Start video processing with OpenCV and dynamic subtitle overlay."""
        try:
            self.logger.info("Starting video processing with OpenCV.")

            # Open video capture
            cap = cv2.VideoCapture(STREAM_URL)

            if not cap.isOpened():
                self.logger.error("Cannot open video stream.")
                self.running = False
                return

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or fps is None:
                fps = 30.0  # Default FPS if unable to get from stream
                self.logger.warning(f"Unable to get FPS from stream. Defaulting to {fps} FPS.")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

            self.logger.info(f"Video properties - FPS: {fps}, Width: {width}, Height: {height}")

            # Initialize variables for video segmentation
            segment_duration = 10  # seconds
            segment_frames = int(fps * segment_duration)
            frame_count = 0
            self.segment_start_time = time.perf_counter()

            if self.video_start_time is None:
                self.video_start_time = self.segment_start_time

            while self.running:
                # Define video writer for each segment
                segment_output_path = f'output/video/output_video_segment_{self.subtitle_index}.mp4'
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Alternative codec

                out = cv2.VideoWriter(segment_output_path, fourcc, fps, (width, height))
                if not out.isOpened():
                    self.logger.error(f"Failed to open VideoWriter for segment {self.subtitle_index}.")
                    self.running = False
                    break

                self.logger.info(f"Started recording segment {self.subtitle_index} to {segment_output_path}")

                # Initialize the corresponding audio segment file
                audio_segment_path = f'output/audio/output_audio_segment_{self.subtitle_index}.wav'
                try:
                    wf = wave.open(audio_segment_path, 'wb')
                    wf.setnchannels(AUDIO_CHANNELS)
                    wf.setsampwidth(AUDIO_SAMPLE_WIDTH)
                    wf.setframerate(AUDIO_SAMPLE_RATE)
                    self.current_audio_segment_wf = wf  # Reference for writing translated audio
                    self.logger.debug(f"Initialized audio segment file: {audio_segment_path}")
                except Exception as e:
                    self.logger.error(f"Failed to initialize audio segment file {audio_segment_path}: {e}", exc_info=True)
                    # Close VideoWriter if audio segment fails
                    out.release()
                    self.logger.info(f"Released VideoWriter for segment {self.subtitle_index} due to audio init failure.")
                    self.current_audio_segment_wf = None
                    continue

                frames_written = 0
                while frame_count < segment_frames and self.running:
                    ret, frame = cap.read()
                    if not ret:
                        self.logger.warning("Failed to read frame from video stream.")
                        break
                    out.write(frame)
                    frame_count += 1
                    frames_written += 1
                    if frames_written % int(fps) == 0:
                        self.logger.debug(f"Written {frames_written} frames to segment {self.subtitle_index}.")

                # Release the video writer
                out.release()
                self.logger.info(f"Segment {self.subtitle_index} saved to {segment_output_path}. Frames written: {frames_written}")

                # Add a short delay to ensure file system flush
                time.sleep(0.5)  # 500 milliseconds

                # Close the audio WAV file
                try:
                    wf.close()
                    self.logger.debug(f"Closed audio segment file: {audio_segment_path}")
                    self.current_audio_segment_wf = None
                except Exception as e:
                    self.logger.error(f"Failed to close audio segment file {audio_segment_path}: {e}")

                # Enqueue muxing job
                final_output_path = f'output/final/output_final_video_segment_{self.subtitle_index}.mp4'
                muxing_job = {
                    "segment_index": self.subtitle_index,
                    "video": segment_output_path,
                    "audio": audio_segment_path,
                    "subtitles": SUBTITLE_PATH,
                    "output": final_output_path
                }
                # Enqueue muxing job in a thread-safe manner
                asyncio.run_coroutine_threadsafe(self.enqueue_muxing_job(muxing_job), self.loop)

                # Reset for next segment
                self.subtitle_index += 1
                frame_count = 0
                self.segment_start_time = time.perf_counter()

        except Exception as e:
            self.logger.error(f"Error in start_video_processing: {e}", exc_info=True)
            self.running = False

    async def run_dashboard_server(self):
        """Run the real-time monitoring dashboard using aiohttp."""
        app = web.Application()
        app.add_routes([
            web.get('/', self.handle_dashboard),
            web.get('/metrics', self.metrics_endpoint)
        ])

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', DASHBOARD_PORT)
        await site.start()
        self.logger.info(f"Monitoring dashboard started at http://localhost:{DASHBOARD_PORT}")

        # Keep the dashboard running
        while self.running:
            await asyncio.sleep(3600)

    async def handle_dashboard(self, request):
        """Handle HTTP requests to the dashboard."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Real-Time Translator Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; }}
                th {{ background-color: #f2f2f2; }}
            </style>
            <script>
                async function fetchMetrics() {{
                    const response = await fetch('/metrics');
                    const data = await response.json();
                    document.getElementById('avg_delay').innerText = data.average_processing_delay.toFixed(6);
                    document.getElementById('min_delay').innerText = data.min_processing_delay.toFixed(6);
                    document.getElementById('max_delay').innerText = data.max_processing_delay.toFixed(6);
                    document.getElementById('stddev_delay').innerText = data.stddev_processing_delay.toFixed(6);
                    document.getElementById('buffer_status').innerText = data.buffer_status;
                    document.getElementById('audio_queue_size').innerText = data.audio_queue_size;
                    document.getElementById('muxing_queue_size').innerText = data.muxing_queue_size;
                }}

                setInterval(fetchMetrics, 1000);
            </script>
        </head>
        <body>
            <h1>Real-Time Translator Dashboard</h1>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Average Processing Delay (s)</td><td id="avg_delay">0.0</td></tr>
                <tr><td>Minimum Processing Delay (s)</td><td id="min_delay">0.0</td></tr>
                <tr><td>Maximum Processing Delay (s)</td><td id="max_delay">0.0</td></tr>
                <tr><td>Std Dev Processing Delay (s)</td><td id="stddev_delay">0.0</td></tr>
                <tr><td>Audio Buffer Size</td><td id="buffer_status">0</td></tr>
                <tr><td>Translated Audio Queue Size</td><td id="audio_queue_size">0</td></tr>
                <tr><td>Muxing Queue Size</td><td id="muxing_queue_size">0</td></tr>
            </table>
        </body>
        </html>
        """
        return web.Response(text=html, content_type='text/html')

    async def metrics_endpoint(self, request):
        """Provide metrics in JSON format for the dashboard."""
        # Convert deque to list for JSON serialization
        serializable_metrics = self.metrics.copy()
        serializable_metrics["processing_delays"] = list(serializable_metrics["processing_delays"])
        return web.json_response(serializable_metrics)

    async def run(self):
        """Run the OpenAIClient."""
        # Start the dashboard
        dashboard_task = asyncio.create_task(self.run_dashboard_server())

        # Start WebSocket connection
        try:
            await self.connect()
        except Exception as e:
            self.logger.error(f"Initial connection failed: {e}")
            await self.reconnect()

        # Start message sender
        send_task = asyncio.create_task(self.send_messages())

        # Start handling responses
        handle_responses_task = asyncio.create_task(self.handle_responses())

        # Start reading input audio
        read_audio_task = asyncio.create_task(self.read_input_audio())

        # Start heartbeat
        heartbeat_task = asyncio.create_task(self.heartbeat())

        # Start video processing
        self.video_processing_task = asyncio.create_task(self.run_video_processing())

        # No need to start another muxing_task here since it's already started in __init__

        # Wait for all tasks to complete
        done, pending = await asyncio.wait(
            [
                send_task,
                handle_responses_task,
                read_audio_task,
                heartbeat_task,
                self.video_processing_task,
                dashboard_task
            ],
            return_when=asyncio.FIRST_EXCEPTION
        )

        # Cancel pending tasks
        for task in pending:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    async def heartbeat(self):
        """Send periodic heartbeat pings to keep the WebSocket connection alive."""
        while self.running:
            try:
                if self.ws and self.ws.open:
                    await self.ws.ping()
                    self.logger.debug("Sent heartbeat ping.")
                else:
                    self.logger.warning("WebSocket is closed. Attempting to reconnect...")
                    await self.reconnect()
            except Exception as e:
                self.logger.error(f"Error in heartbeat: {e}")
                await self.reconnect()
            await asyncio.sleep(30)  # Ping every 30 seconds

    async def disconnect(self):
        """Gracefully disconnect the client."""
        self.logger.info("Disconnecting the client...")
        self.running = False

        # Close WebSocket
        if self.ws:
            try:
                await self.ws.close()
                self.logger.info("WebSocket connection closed.")
            except Exception as e:
                self.logger.error(f"Error closing WebSocket: {e}")

        # Cancel playback task
        if self.playback_task:
            self.playback_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.playback_task

        # Cancel muxing task
        if self.muxing_task:
            self.muxing_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.muxing_task

        # Cancel video processing task
        if self.video_processing_task:
            self.video_processing_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.video_processing_task

        # Close current audio segment WAV file
        if self.current_audio_segment_wf:
            try:
                self.current_audio_segment_wf.close()
                self.logger.debug("Closed current audio segment WAV file.")
            except Exception as e:
                self.logger.error(f"Error closing current audio segment WAV file: {e}")

        # Close WAV file
        if self.output_wav:
            try:
                self.output_wav.close()
                self.logger.info("Output WAV file closed.")
            except Exception as e:
                self.logger.error(f"Error closing WAV file: {e}")

        self.logger.info("Client disconnected successfully.")

    async def shutdown(self, sig):
        """Handle shutdown signals."""
        self.logger.info(f"Received exit signal {sig.name}...")
        await self.disconnect()

    def test_video_writer(self):
        """Test function to write a sample video and verify its integrity."""
        try:
            self.logger.info("Starting test video writer.")
            test_video_path = 'output/video/test_video.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Alternative codec
            fps = 30.0
            width = 1280
            height = 720
            out = cv2.VideoWriter(test_video_path, fourcc, fps, (width, height))
            cap = cv2.VideoCapture(STREAM_URL)
            if not cap.isOpened():
                self.logger.error("Cannot open video stream for testing.")
                return
            for i in range(int(fps * 5)):  # Write 5 seconds of video
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame during test video writing.")
                    break
                out.write(frame)
            out.release()
            cap.release()
            self.logger.info(f"Test video written to {test_video_path}")
            # Verify video file
            probe = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_format', '-show_streams', test_video_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if probe.returncode == 0 and 'moov' in probe.stdout:
                self.logger.info("Test video is valid.")
            else:
                self.logger.error("Test video is invalid.")
                self.logger.error(f"FFprobe output: {probe.stderr}")
        except Exception as e:
            self.logger.error(f"Error in test_video_writer: {e}")

    async def run_video_processing(self):
        """Run video processing within the asyncio event loop."""
        await asyncio.to_thread(self.start_video_processing)

    def start_video_processing(self):
        """Start video processing with OpenCV and dynamic subtitle overlay."""
        try:
            self.logger.info("Starting video processing with OpenCV.")

            # Open video capture
            cap = cv2.VideoCapture(STREAM_URL)

            if not cap.isOpened():
                self.logger.error("Cannot open video stream.")
                self.running = False
                return

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or fps is None:
                fps = 30.0  # Default FPS if unable to get from stream
                self.logger.warning(f"Unable to get FPS from stream. Defaulting to {fps} FPS.")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

            self.logger.info(f"Video properties - FPS: {fps}, Width: {width}, Height: {height}")

            # Initialize variables for video segmentation
            segment_duration = 10  # seconds
            segment_frames = int(fps * segment_duration)
            frame_count = 0
            self.segment_start_time = time.perf_counter()

            if self.video_start_time is None:
                self.video_start_time = self.segment_start_time

            while self.running:
                # Define video writer for each segment
                segment_output_path = f'output/video/output_video_segment_{self.subtitle_index}.mp4'
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Alternative codec

                out = cv2.VideoWriter(segment_output_path, fourcc, fps, (width, height))
                if not out.isOpened():
                    self.logger.error(f"Failed to open VideoWriter for segment {self.subtitle_index}.")
                    self.running = False
                    break

                self.logger.info(f"Started recording segment {self.subtitle_index} to {segment_output_path}")

                # Initialize the corresponding audio segment file
                audio_segment_path = f'output/audio/output_audio_segment_{self.subtitle_index}.wav'
                try:
                    wf = wave.open(audio_segment_path, 'wb')
                    wf.setnchannels(AUDIO_CHANNELS)
                    wf.setsampwidth(AUDIO_SAMPLE_WIDTH)
                    wf.setframerate(AUDIO_SAMPLE_RATE)
                    self.current_audio_segment_wf = wf  # Reference for writing translated audio
                    self.logger.debug(f"Initialized audio segment file: {audio_segment_path}")
                except Exception as e:
                    self.logger.error(f"Failed to initialize audio segment file {audio_segment_path}: {e}", exc_info=True)
                    # Close VideoWriter if audio segment fails
                    out.release()
                    self.logger.info(f"Released VideoWriter for segment {self.subtitle_index} due to audio init failure.")
                    self.current_audio_segment_wf = None
                    continue

                frames_written = 0
                while frame_count < segment_frames and self.running:
                    ret, frame = cap.read()
                    if not ret:
                        self.logger.warning("Failed to read frame from video stream.")
                        break
                    out.write(frame)
                    frame_count += 1
                    frames_written += 1
                    if frames_written % int(fps) == 0:
                        self.logger.debug(f"Written {frames_written} frames to segment {self.subtitle_index}.")

                # Release the video writer
                out.release()
                self.logger.info(f"Segment {self.subtitle_index} saved to {segment_output_path}. Frames written: {frames_written}")

                # Add a short delay to ensure file system flush
                time.sleep(0.5)  # 500 milliseconds

                # Close the audio WAV file
                try:
                    wf.close()
                    self.logger.debug(f"Closed audio segment file: {audio_segment_path}")
                    self.current_audio_segment_wf = None
                except Exception as e:
                    self.logger.error(f"Failed to close audio segment file {audio_segment_path}: {e}")

                # Enqueue muxing job
                final_output_path = f'output/final/output_final_video_segment_{self.subtitle_index}.mp4'
                muxing_job = {
                    "segment_index": self.subtitle_index,
                    "video": segment_output_path,
                    "audio": audio_segment_path,
                    "subtitles": SUBTITLE_PATH,
                    "output": final_output_path
                }
                # Enqueue muxing job in a thread-safe manner
                asyncio.run_coroutine_threadsafe(self.enqueue_muxing_job(muxing_job), self.loop)

                # Reset for next segment
                self.subtitle_index += 1
                frame_count = 0
                self.segment_start_time = time.perf_counter()

        except Exception as e:
            self.logger.error(f"Error in start_video_processing: {e}", exc_info=True)
            self.running = False
