# openai_client/client.py

import asyncio
import aiofiles
import websockets
import json
import os
import base64
import wave
import uuid
import csv
import statistics
import librosa
import numpy as np
from asyncio import Queue
from collections import deque
import datetime
import time
from contextlib import suppress
import logging
from typing import Optional, Dict, Any, Set

from config import (
    API_URL, AUDIO_SAMPLE_RATE, AUDIO_CHANNELS, AUDIO_SAMPLE_WIDTH,
    SUBTITLE_PATH, OUTPUT_WAV_PATH, MUXING_QUEUE_MAXSIZE, DASHBOARD_PORT,
    INPUT_AUDIO_PIPE, MALE_VOICE_ID, FEMALE_VOICE_ID, DEFAULT_VOICE_ID,
    SEGMENT_DURATION, MAX_BACKOFF
)
from logging_setup import setup_logging
from .audio_processing import AudioProcessor
from .video_processing import VideoProcessor
from .muxing import Muxer
from .rtmp_streamer import RTMPStreamer
from .dashboard import Dashboard
from .utils import format_timestamp_vtt

class OpenAIClient:
    def __init__(self, api_key: str):
        """
        Initialize the OpenAIClient.
        """
        self.api_key = api_key
        self.loop = asyncio.get_event_loop()
        self.session_id = str(uuid.uuid4())
        self.logger, self.muxing_logger = setup_logging(self.session_id)
        self.ws = None
        self.running = True

        # Assign configuration variables as instance attributes
        self.API_URL = API_URL
        self.AUDIO_SAMPLE_RATE = AUDIO_SAMPLE_RATE
        self.AUDIO_CHANNELS = AUDIO_CHANNELS
        self.AUDIO_SAMPLE_WIDTH = AUDIO_SAMPLE_WIDTH
        self.SUBTITLE_PATH = SUBTITLE_PATH
        self.OUTPUT_WAV_PATH = OUTPUT_WAV_PATH
        self.MUXING_QUEUE_MAXSIZE = MUXING_QUEUE_MAXSIZE
        self.DASHBOARD_PORT = DASHBOARD_PORT
        self.INPUT_AUDIO_PIPE = INPUT_AUDIO_PIPE
        self.MALE_VOICE_ID = MALE_VOICE_ID
        self.FEMALE_VOICE_ID = FEMALE_VOICE_ID
        self.DEFAULT_VOICE_ID = DEFAULT_VOICE_ID
        self.SEGMENT_DURATION = SEGMENT_DURATION
        self.MAX_BACKOFF = MAX_BACKOFF

        # Queues
        self.send_queue = Queue()
        self.sent_audio_timestamps = deque()

        # Playback Buffer
        self.audio_buffer = deque(maxlen=100)
        self.playback_event = asyncio.Event()

        # Timestamps
        self.video_start_time = None

        # Transcript
        self.current_transcript = ""

        # Subtitle Management
        self.segment_index = 1
        self.segment_index_lock = asyncio.Lock()  # Add a lock for segment_index

        # Initialize output WAV file
        self.output_wav = None
        self.init_output_wav()

        # Initialize WebVTT file
        # Removed initializing main SUBTITLE_PATH as we're using segmented subtitles
        # Initialize subtitles for the first segment
        self.logger.info(f"Segment index: {self.segment_index}")
        asyncio.create_task(self.initialize_temp_subtitles(self.segment_index))

        # Initialize metrics
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

        # Initialize components
        self.audio_processor = AudioProcessor(self, self.logger)
        self.video_processor = VideoProcessor(self, self.logger)
        self.muxer = Muxer(self, self.muxing_logger)
        self.dashboard = Dashboard(self, self.logger)

        # Delay benchmarking
        self.processing_delays = deque(maxlen=100)
        self.delay_benchmark_file = f'output/logs/delay_benchmark_{self.session_id}.csv'
        self.setup_delay_benchmarking()

        # Reconnect handling
        self.is_reconnecting = False
        self.reconnect_lock = asyncio.Lock()

        # Lock to ensure only one active response
        self.response_lock = asyncio.Lock()

        # WebSocket clients management
        self.websocket_clients: Set[websockets.WebSocketServerProtocol] = set()
        self.websocket_clients_lock = asyncio.Lock()

        # Initialize RTMPStreamer
        self.rtmp_streamer = RTMPStreamer(self, self.logger)


        # Ensure all necessary directories exist
        os.makedirs('output/audio/responses', exist_ok=True)
        os.makedirs('output/audio/output', exist_ok=True)
        os.makedirs('output/subtitles', exist_ok=True)
        os.makedirs('output/final', exist_ok=True)
        os.makedirs('output/logs', exist_ok=True)

    async def get_segment_index(self) -> int:
        """Safely get the current segment index."""
        async with self.segment_index_lock:
            return self.segment_index

    async def increment_segment_index(self):
        async with self.segment_index_lock:
            old_index = self.segment_index
            self.segment_index += 1
            self.logger.info(f"Segment index incremented from {old_index} to {self.segment_index}")
            assert self.segment_index > old_index, "Segment index did not increment correctly!"



    async def initialize_temp_subtitles(self, segment_index: int):
        """Initialize a temporary WebVTT subtitle file for the given segment."""
        temp_subtitles_path = f'output/subtitles/subtitles_segment_{segment_index}.vtt'
        try:
            os.makedirs(os.path.dirname(temp_subtitles_path), exist_ok=True)
            # Check if the file already exists to prevent overwriting
            if not os.path.exists(temp_subtitles_path):
                async with aiofiles.open(temp_subtitles_path, 'w', encoding='utf-8') as f:
                    await f.write("WEBVTT\n\n")
                self.logger.info(f"Initialized temporary WebVTT file for segment {segment_index} at {temp_subtitles_path}")
            else:
                self.logger.info(f"Temporary WebVTT file for segment {segment_index} already exists at {temp_subtitles_path}")
        except Exception as e:
            self.logger.error(f"Failed to initialize temporary WebVTT file for segment {segment_index}: {e}")


    async def enqueue_muxing_job(self, muxing_job: Dict[str, Any]):
        """
        Enqueue a muxing job to the Muxer.
        
        Args:
            muxing_job (Dict[str, Any]): The muxing job details.
        """
        if self.muxer:
            await self.muxer.enqueue_muxing_job(muxing_job)
            self.logger.debug(f"Enqueued muxing job: {muxing_job}")
        else:
            self.logger.error("Muxer instance is not initialized.")

    def init_output_wav(self):
        """Initialize the output WAV file."""
        try:
            os.makedirs(os.path.dirname(self.OUTPUT_WAV_PATH), exist_ok=True)
            self.output_wav = wave.open(self.OUTPUT_WAV_PATH, 'wb')
            self.output_wav.setnchannels(self.AUDIO_CHANNELS)
            self.output_wav.setsampwidth(self.AUDIO_SAMPLE_WIDTH)
            self.output_wav.setframerate(self.AUDIO_SAMPLE_RATE)
            self.logger.info(f"Output WAV file initialized at {self.OUTPUT_WAV_PATH}.")
        except Exception as e:
            self.logger.error(f"Failed to initialize output WAV file: {e}")
            self.output_wav = None

    def setup_delay_benchmarking(self):
        """Setup CSV for delay benchmarking."""
        try:
            os.makedirs(os.path.dirname(self.delay_benchmark_file), exist_ok=True)
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

    async def log_delay_metrics(self):
        """Log delay metrics to CSV and update monitoring metrics."""
        if not self.processing_delays:
            self.logger.debug("No processing delays to log.")
            return

        processing_delay = self.processing_delays[-1]  # Use the last processing delay
        buffer = 0.5  # Add a small buffer for seamless video/audio sync
        offset = processing_delay + buffer  # Final offset for the audio
        
        # Log the processing delay and offset
        self.logger.info(f"Processing delay: {processing_delay:.6f}s, Audio Offset: {offset:.6f}s")

        # Update metrics
        self.metrics["processing_delays"].append(processing_delay)
        self.metrics["average_processing_delay"] = statistics.mean(self.processing_delays)
        self.metrics["min_processing_delay"] = min(self.processing_delays)
        self.metrics["max_processing_delay"] = max(self.processing_delays)
        self.metrics["stddev_processing_delay"] = (
            statistics.stdev(self.processing_delays) if len(self.processing_delays) > 1 else 0.0
        )
        self.metrics["buffer_status"] = len(self.audio_buffer)
        self.metrics["audio_queue_size"] = self.audio_processor.translated_audio_queue.qsize()
        self.metrics["muxing_queue_size"] = self.muxer.muxing_queue.qsize()

        try:
            # Save metrics to the CSV for later analysis
            current_time = datetime.datetime.utcnow().isoformat()
            async with aiofiles.open(self.delay_benchmark_file, 'a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                await writer.writerow([
                    current_time,
                    f"{processing_delay:.6f}",
                    f"{offset:.6f}",  # Log the calculated offset
                    f"{self.metrics['average_processing_delay']:.6f}",
                    f"{self.metrics['min_processing_delay']:.6f}",
                    f"{self.metrics['max_processing_delay']:.6f}",
                    f"{self.metrics['stddev_processing_delay']:.6f}"
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
            self.ws = await websockets.connect(self.API_URL, extra_headers=headers)
            self.logger.info("Connected to OpenAI Realtime API.")
            await self.initialize_session()
        except Exception as e:
            self.logger.error(f"Failed to connect to Realtime API: {e}")
            raise
    
    # Without VAD
    async def initialize_session(self):
        """Initialize session with desired configurations."""
        session_update_event = {
            "type": "session.update",
            "session": {
                "instructions": (
                    "You are a real-time translator. Translate the audio you receive into German without performing Voice Activity Detection (VAD). "
                    "Ensure that the translated audio matches the input audio's duration and timing exactly to facilitate synchronization with video. "
                    "If there is silence, or no one speaking, please fill this space with silence in order to keep the same output length as input length. "
                    "If there are multiple people speaking, please try to use different voices for each of them. "
                    "Provide detailed and comprehensive translations without truncating sentences. "
                    "Do not respond conversationally."
                ),
                "modalities": ["text", "audio"],
                "voice": self.DEFAULT_VOICE_ID,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                # "turn_detection": null,
                "temperature": 1.02,
                "tools": []  # Add tool definitions here if needed
            }
        }
        await self.enqueue_message(session_update_event)
        self.logger.info("Session update event enqueued.")

    # # With VAD
    # async def initialize_session(self):
    #     """Initialize session with desired configurations."""
    #     session_update_event = {
    #         "type": "session.update",
    #         "session": {
    #             "instructions": (
    #                 "You are a real-time translator. Translate the audio you receive into German without performing Voice Activity Detection (VAD). "
    #                 "Ensure that the translated audio matches the input audio's duration and timing exactly to facilitate synchronization with video. "
    #                 "Provide detailed and comprehensive translations without truncating sentences. "
    #                 "Do not respond conversationally."
    #             ),
    #             "modalities": ["text", "audio"],
    #             "voice": self.DEFAULT_VOICE_ID,
    #             "input_audio_format": "pcm16",
    #             "output_audio_format": "pcm16",
    #             "temperature": 0.8,
    #             "turn_detection": {
    #                 "type": "server_vad",
    #                 "threshold": 0.5,
    #                 "prefix_padding_ms": 300,
    #                 "silence_duration_ms": 500
    #             },
    #             "tools": []  # Add tool definitions here if needed
    #         }
    #     }
    #     await self.enqueue_message(session_update_event)
    #     self.logger.info("Session update event enqueued.")

    async def enqueue_message(self, message: dict):
        """Enqueue a message to be sent over WebSocket."""
        await self.send_queue.put(message)
        self.logger.debug(f"Message enqueued: {message['type']}")

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
                    backoff = min(backoff * 2, self.MAX_BACKOFF)
            self.is_reconnecting = False

    async def handle_responses(self):
        """Handle incoming messages from the WebSocket."""
        while self.running:
            try:
                response = await self.ws.recv()
                event = json.loads(response)
                self.logger.debug(f"Received event: {json.dumps(event, indent=2)}")  # Log entire event for debugging
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

    # openai_client/client.py

    async def process_event(self, event: dict):
        """Process a single event from the WebSocket."""
        event_type = event.get("type")
        self.logger.debug(f"Processing event: {event_type}")

        if event_type == "input_audio_buffer.speech_started":
            self.logger.info("Speech started detected by server.")
            # No action needed as server handles VAD

        elif event_type == "response.text.delta":
            delta_text = event.get("delta", "")
            self.logger.info(f"Received text delta: '{delta_text}'")
            self.logger.debug(f"Current transcript before update: '{self.current_transcript}'")
            self.current_transcript += delta_text
            self.logger.debug(f"Updated transcript: '{self.current_transcript}'")

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
                await self.audio_processor.handle_audio_delta(audio_data)
                self.logger.debug(f"Processed audio delta of size: {len(audio_data)}")
            else:
                self.logger.debug("Received empty audio delta.")

        elif event_type == "response.audio_transcript.done":
            transcript = event.get("transcript", "").strip()
            if not transcript:
                self.logger.warning("Received empty transcript in 'response.audio_transcript.done' event.")
                return

            if self.sent_audio_timestamps:
                sent_time = self.sent_audio_timestamps.popleft()
                current_time = self.loop.time()
                processing_delay = current_time - sent_time
                self.logger.debug(f"Processing delay calculated: {processing_delay:.6f}s")

                # Assume audio_duration based on SEGMENT_DURATION since 'audio' field is missing
                audio_duration = self.SEGMENT_DURATION
                self.logger.debug(f"Assumed audio duration: {audio_duration:.3f}s")

                # Calculate start and end times relative to video_start_time
                if self.video_start_time:
                    start_time = sent_time - self.video_start_time
                    end_time = start_time + audio_duration + processing_delay
                else:
                    # If video_start_time is not set, default to current_time
                    start_time = 0.0
                    end_time = audio_duration + processing_delay

                self.logger.info(f"Received transcript: '{transcript}'")
                self.logger.info(f"Subtitle timing - Start: {start_time:.3f}s, End: {end_time:.3f}s")

                # Append processing_delay
                self.processing_delays.append(processing_delay)
                await self.log_delay_metrics()

                current_segment_index = await self.get_segment_index()
                video_segment_index = current_segment_index - 1  # Adjust for zero-based index

                start_time = (video_segment_index) * self.SEGMENT_DURATION
                end_time = (video_segment_index + 1) * self.SEGMENT_DURATION

                self.logger.info(f"Subtitle timing - Start: {start_time:.3f}s, End: {end_time:.3f}s")

                # Write subtitle to temporary WebVTT file
                await self.write_vtt_subtitle(current_segment_index, start_time, end_time, transcript)
                self.logger.debug(f"Subtitle written for segment {current_segment_index}: '{transcript}'")

                buffer = 0.5  # Add a small buffer for seamless video/audio sync
                audio_offset = processing_delay + buffer

                # Close the current audio segment
                await self.audio_processor.close_current_audio_segment(video_segment_index)

                # Enqueue muxing job with audio_offset set to zero
                muxing_job = {
                    "segment_index": video_segment_index,
                    "video": f'output/video/output_video_segment_{video_segment_index}.mp4',
                    "audio": f'output/audio/output_audio_segment_{video_segment_index}.wav',
                    "subtitles": f'output/subtitles/subtitles_segment_{video_segment_index}.vtt',
                    "output": f'output/final/output_final_segment_{video_segment_index}.mp4',
                    "audio_offset": 0.0  # Set audio offset to zero
                }
                await self.enqueue_muxing_job(muxing_job)
                self.logger.info(f"Enqueued muxing job for segment {video_segment_index}")

                # Initialize subtitle file for the next segment
                await self.initialize_temp_subtitles(current_segment_index)
                self.logger.debug(f"Initialized subtitle file for next segment: {current_segment_index}")

                # Start a new audio segment for the next video segment
                await self.audio_processor.start_new_audio_segment(current_segment_index)
                self.logger.debug(f"Started new audio segment for segment index: {current_segment_index}")

                # Reset the current transcript for the next segment
                self.current_transcript = ""
                self.logger.debug("Reset current transcript for the next segment.")

            elif event_type == "response.content_part.done":
                content_part = event.get("content_part", "")
                # Handle content part if necessary
                self.logger.info(f"Received content part: '{content_part}'")
                # Depending on content, may need to process further

            elif event_type == "response.output_item.done":
                output_item = event.get("output_item", "")
                # Handle output item if necessary
                self.logger.info(f"Received output item: '{output_item}'")
                # Depending on content, may need to process further

            elif event_type == "response.created":
                self.logger.info("Response created event received.")
                # Handle response creation if necessary

            elif event_type == "response.audio_transcript.delta":
                delta_text = event.get("delta", "")
                if delta_text:
                    self.logger.info(f"Received transcript delta: '{delta_text}'")
                    self.current_transcript += delta_text
                    self.logger.debug(f"Updated transcript with delta: '{self.current_transcript}'")
                else:
                    self.logger.debug("Received empty transcript delta.")

            elif event_type == "rate_limits.updated":
                rate_limits = event.get("rate_limits", [])
                self.logger.info(f"Rate limits updated: {rate_limits}")
                # Handle rate limit updates if necessary

            elif event_type == "response.output_item.added":
                self.logger.info("Output item added event received.")
                # Handle added output items if necessary

            elif event_type == "response.content_part.added":
                self.logger.info("Content part added event received.")
                # Handle added content parts if necessary

            elif event_type == "input_audio_buffer.committed":
                self.logger.info("Audio buffer committed.")
                # Handle audio buffer commit if necessary

            else:
                self.logger.warning(f"Unhandled event type: {event_type}")

            self.logger.debug("Event processed.")


    async def handle_function_call(self, item: dict):
        """
        Handle function call events from the server.
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
        """Create a response to trigger audio generation with dynamic voice selection."""
        async with self.response_lock:
            # Path to the latest audio segment for gender detection
            latest_audio_segment = f'output/audio/output_audio_segment_{self.segment_index - 1}.wav'

            # Detect gender
            gender = self.predict_gender(latest_audio_segment)
            if gender == 'male':
                selected_voice = self.MALE_VOICE_ID
            elif gender == 'female':
                selected_voice = self.FEMALE_VOICE_ID
            else:
                selected_voice = self.DEFAULT_VOICE_ID  # Fallback to default voice

            self.logger.info(f"Detected gender: '{gender}'. Selected voice ID: '{selected_voice}'")

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
                    # "turn_detection": null,
                    "voice": selected_voice,
                    "tools": []  # Add tool definitions here if needed
                }
            }
            self.logger.debug(f"Enqueuing response event with selected voice: {response_event}")
            await self.enqueue_message(response_event)
            self.logger.info("Created response event with dynamic voice selection.")

    def predict_gender(self, audio_path: str) -> Optional[str]:
        """
        Predict the gender of the speaker in the given audio file using a simple pitch-based heuristic.
        """
        try:
            if not os.path.exists(audio_path):
                self.logger.warning(f"Audio file for gender prediction does not exist: {audio_path}")
                return None

            y, sr = librosa.load(audio_path, sr=self.AUDIO_SAMPLE_RATE)
            # Compute fundamental frequency (pitch) using librosa
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            # Extract the highest magnitude pitch per frame
            pitches = pitches[magnitudes > np.median(magnitudes)]
            if len(pitches) == 0:
                self.logger.warning("No pitch detected for gender prediction.")
                return None
            avg_pitch = np.mean(pitches)
            self.logger.debug(f"Average pitch for gender prediction: {avg_pitch:.2f} Hz")
            # Updated heuristic: Female voices typically have higher pitch (>165 Hz)
            # Adjusted to account for realistic pitch ranges
            if 165 <= avg_pitch <= 255:
                self.logger.info(f"Predicted gender: female (avg_pitch={avg_pitch:.2f} Hz)")
                return 'female'
            elif 85 <= avg_pitch < 165:
                self.logger.info(f"Predicted gender: male (avg_pitch={avg_pitch:.2f} Hz)")
                return 'male'
            else:
                self.logger.warning(f"Unusual pitch detected: {avg_pitch:.2f} Hz. Unable to determine gender.")
                return None
        except Exception as e:
            self.logger.error(f"Error predicting gender: {e}")
            return None



    async def read_input_audio(self):
        """Read audio from a named pipe and send to Realtime API in chunks of SEGMENT_DURATION."""
        self.logger.info("Starting to read from input_audio_pipe.")
        
        # Buffer to collect audio until it matches SEGMENT_DURATION
        audio_buffer = bytearray()
        bytes_per_second = self.AUDIO_SAMPLE_RATE * self.AUDIO_CHANNELS * self.AUDIO_SAMPLE_WIDTH

        while self.running:
            try:
                async with aiofiles.open(self.INPUT_AUDIO_PIPE, 'rb') as pipe:
                    while self.running:
                        data = await pipe.read(131072)  # 128KB
                        if not data:
                            await asyncio.sleep(0.05)
                            continue

                        audio_buffer.extend(data)

                        # Check if buffer exceeds SEGMENT_DURATION
                        required_size = int(bytes_per_second * self.SEGMENT_DURATION)
                        if len(audio_buffer) >= required_size:
                            # Slice out the required segment
                            segment = audio_buffer[:required_size]
                            audio_buffer = audio_buffer[required_size:]

                            # Enqueue the segment for processing
                            await self.audio_processor.send_input_audio(segment)
                            self.logger.info(f"Sent audio segment of size: {len(segment)} bytes.")

                            # Commit the audio buffer after each segment
                            await self.commit_audio_buffer()

            except Exception as e:
                self.logger.error(f"Error reading input audio: {e}")
                await asyncio.sleep(1)  # Prevent tight loop on error



    async def write_vtt_subtitle(self, segment_index: int, start_time: float, end_time: float, text: str):
        temp_subtitles_path = f'output/subtitles/subtitles_segment_{segment_index}.vtt'
        try:
            # Check if it's the first subtitle being written
            if not os.path.exists(temp_subtitles_path):
                async with aiofiles.open(temp_subtitles_path, 'w', encoding='utf-8') as f:
                    await f.write("WEBVTT\n\n")
                self.logger.debug(f"Initialized WebVTT header for {temp_subtitles_path}")

            # Convert seconds to WebVTT timestamp format HH:MM:SS.mmm.
            start_vtt = format_timestamp_vtt(start_time)
            end_vtt = format_timestamp_vtt(end_time)

            # Prepare subtitle entry.
            subtitle_index = await self.get_next_subtitle_index(segment_index)
            subtitle = f"{subtitle_index}\n{start_vtt} --> {end_vtt}\n{text}\n\n"

            # Append to the temporary WebVTT file.
            async with aiofiles.open(temp_subtitles_path, 'a', encoding='utf-8') as f:
                await f.write(subtitle)
            self.logger.debug(f"Written subtitle {subtitle_index} to {temp_subtitles_path}: '{subtitle.strip()}'")
        except Exception as e:
            self.logger.error(f"Error writing WebVTT subtitle to {temp_subtitles_path}: {e}")


    async def get_next_subtitle_index(self, segment_index: int) -> int:
        """Get the next subtitle index for the given segment."""
        if not hasattr(self, 'subtitle_indices'):
            self.subtitle_indices = {}
        self.subtitle_indices.setdefault(segment_index, 0)
        self.subtitle_indices[segment_index] += 1
        return self.subtitle_indices[segment_index]


    async def run_dashboard_server(self):
        """Run the real-time monitoring dashboard using aiohttp."""
        # Handled by Dashboard class
        pass

    async def run(self):
        """Run the OpenAIClient."""
        # Start WebSocket connection
        try:
            await self.connect()
        except Exception as e:
            self.logger.error(f"Initial connection failed: {e}")
            await self.reconnect()

        # Initialize the first audio segment and subtitle file
        await self.initialize_temp_subtitles(self.segment_index + 1)
        await self.audio_processor.start_new_audio_segment(self.segment_index + 1)

        # Reset the current transcript for the next segment
        self.current_transcript = ""
        self.logger.debug("Reset current transcript for the next segment.")

        # Start message sender
        send_task = asyncio.create_task(self.send_messages())

        # Start handling responses
        handle_responses_task = asyncio.create_task(self.handle_responses())

        # Start reading input audio
        read_audio_task = asyncio.create_task(self.read_input_audio())

        # Start heartbeat
        heartbeat_task = asyncio.create_task(self.heartbeat())

        # Start RTMPStreamer
        self.rtmp_streamer.start()

        # Wait for all tasks to complete
        done, pending = await asyncio.wait(
            [
                send_task,
                handle_responses_task,
                read_audio_task,
                heartbeat_task,
                self.audio_processor.playback_task,
                self.muxer.muxing_task,
                self.dashboard.dashboard_task,
                self.rtmp_streamer.streaming_task
            ],
            return_when=asyncio.FIRST_EXCEPTION
        )

        # Stop RTMPStreamer
        self.rtmp_streamer.stop()

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

    async def disconnect(self, shutdown: bool = False):
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
        if self.audio_processor.playback_task:
            self.audio_processor.playback_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.audio_processor.playback_task

        # Cancel muxing task
        if self.muxer.muxing_task:
            self.muxer.muxing_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.muxer.muxing_task

        # Cancel video processing task
        if self.video_processor.video_task:
            self.video_processor.video_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.video_processor.video_task

        # Cancel dashboard task
        if self.dashboard.dashboard_task:
            self.dashboard.dashboard_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.dashboard.dashboard_task

        # Close current audio segment WAV file
        if hasattr(self, 'current_audio_segment_wf') and self.current_audio_segment_wf:
            try:
                self.current_audio_segment_wf.close()
                self.logger.debug("Closed current audio segment WAV file.")
            except Exception as e:
                self.logger.error(f"Error closing current audio segment WAV file: {e}")

        # Close output WAV file
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

    async def register_websocket(self, websocket: websockets.WebSocketServerProtocol):
        """Register a new WebSocket client."""
        async with self.websocket_clients_lock:
            self.websocket_clients.add(websocket)
            client_id = uuid.uuid4().hex
            self.logger.info(f"Registered WebSocket client {client_id}.")
            return client_id

    async def unregister_websocket(self, client_id: str):
        """Unregister a WebSocket client."""
        async with self.websocket_clients_lock:
            # Find the websocket corresponding to client_id
            # For simplicity, assuming one client; modify as needed
            for ws in self.websocket_clients:
                await ws.close()
                self.websocket_clients.remove(ws)
                self.logger.info(f"Unregistered WebSocket client {client_id}.")
                break
