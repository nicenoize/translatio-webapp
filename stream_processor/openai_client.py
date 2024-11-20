import asyncio
import aiofiles
import websockets
import json
import logging
import os
import base64
import pprint
from typing import Dict
import datetime
import srt
import errno
import subprocess
from collections import defaultdict
from contextlib import suppress

from stream_processor.audio_saver import AudioSaver

class OpenAIClient:
    def __init__(self, api_key: str, stream_url: str, output_rtmp_url: str):
        self.api_key = api_key
        self.stream_url = stream_url
        self.output_rtmp_url = output_rtmp_url
        self.ws = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.websocket_clients: Dict[int, websockets.WebSocketServerProtocol] = {}
        self.output_video_file = 'output/output_video.mp4'

        # Create directories for output if they don't exist
        os.makedirs('output/transcripts', exist_ok=True)
        os.makedirs('output/audio/input', exist_ok=True)
        os.makedirs('output/audio/output', exist_ok=True)
        os.makedirs('output/subtitles', exist_ok=True)

        # Named pipes
        self.input_audio_pipe = 'input_audio_pipe'
        self.translated_audio_pipe = 'translated_audio_pipe'
        self.create_named_pipes()

        # Transcript and subtitles
        self.transcript_file = open('output/transcripts/transcript.txt', 'w', encoding='utf-8')
        self.subtitle_file_path = 'output/subtitles/subtitles.srt'
        # Initialize subtitles with a placeholder
        self.initialize_subtitles()

        # Audio playback
        self.playback_queue = asyncio.Queue()
        self.playback_buffer = defaultdict(bytes)
        self.playback_sequence = 0
        self.playback_event = asyncio.Event()
        self.audio_sequence = 0
        self.playback_task = asyncio.create_task(self.audio_playback_handler())

        # Reconnection
        self.is_reconnecting = False
        self.reconnect_lock = asyncio.Lock()

        # Subtitles
        self.subtitle_entries = []
        self.subtitle_index = 1
        self.current_subtitle = ""
        self.subtitle_start_time = None

        # Send queue
        self.send_queue = asyncio.Queue()
        self.send_task = asyncio.create_task(self.send_messages())

        # FFmpeg process
        self.ffmpeg_process = None
        self.video_start_time = None

        # Audio Saver
        self.audio_saver = AudioSaver(output_wav_path='output/audio/output/translated_audio.wav')
        self.verify_audio_task = asyncio.create_task(self.verify_audio_writing())

    def create_named_pipes(self):
        """Create named pipes if they don't exist"""
        for pipe in [self.input_audio_pipe, self.translated_audio_pipe]:
            try:
                if not os.path.exists(pipe):
                    os.mkfifo(pipe)
                    self.logger.info(f"Created named pipe: {pipe}")
                else:
                    self.logger.info(f"Named pipe already exists: {pipe}")
            except Exception as e:
                self.logger.error(f"Error creating pipe {pipe}: {e}")
                raise

    def initialize_subtitles(self):
        """Initialize the subtitles file with a placeholder to prevent FFmpeg from blocking"""
        if not os.path.exists(self.subtitle_file_path) or os.path.getsize(self.subtitle_file_path) == 0:
            with open(self.subtitle_file_path, 'w', encoding='utf-8') as f:
                placeholder_subtitle = srt.Subtitle(
                    index=0,
                    start=datetime.timedelta(0),
                    end=datetime.timedelta(seconds=1),
                    content=" "
                )
                f.write(srt.compose([placeholder_subtitle]))
            self.logger.info("Initialized subtitles file with a placeholder.")

        # Open the subtitles file in read-plus mode
        self.subtitle_file = open(self.subtitle_file_path, 'r+', encoding='utf-8')

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
                async with aiofiles.open(self.input_audio_pipe, 'rb') as pipe:
                    while True:
                        data = await pipe.read(32768)  # Read 32KB
                        if not data:
                            await asyncio.sleep(0.1)
                            continue
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
                    if sequence == self.playback_sequence:
                        await self.process_playback_chunk(sequence, audio_data)
                        self.playback_sequence += 1
                        # Check if the next sequences are already in the buffer
                        while self.playback_sequence in self.playback_buffer:
                            buffered_data = self.playback_buffer.pop(self.playback_sequence)
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
        """Process and assemble translated audio chunks."""
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
                "temperature": 0.6
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

    async def create_response(self):
        """Send response.create message to request response generation."""
        response_create_event = {
            "type": "response.create",
            "response": {
                "modalities": ["text", "audio"],
                "instructions": "Please translate the audio you receive into German. Try to keep the original tone and emotion."
            }
        }
        await self.enqueue_message(json.dumps(response_create_event))
        self.logger.info("Enqueued response.create message")

    async def handle_responses(self):
        """Handle translation responses from OpenAI"""
        while True:
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

            try:
                event = json.loads(response)
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to decode JSON response: {e}")
                continue

            event_type = event.get("type")

            self.logger.debug(f"Received event type: {event_type}")

            if event_type == "input_audio_buffer.speech_started":
                self.logger.info("Speech started")
                self.subtitle_start_time = datetime.datetime.now()
            elif event_type == "input_audio_buffer.speech_stopped":
                self.logger.info("Speech stopped")
                if self.current_subtitle.strip():
                    self.write_subtitle(
                        self.subtitle_index,
                        self.subtitle_start_time,
                        datetime.datetime.now(),
                        self.current_subtitle.strip()
                    )
                    self.subtitle_index += 1
                    self.current_subtitle = ""
                    self.subtitle_start_time = None
                # Commit the audio buffer and create a response
                await self.commit_audio_buffer()
                await self.create_response()
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
            elif event_type == "error":
                error_info = event.get("error", {})
                self.logger.error(f"OpenAI Error: {error_info}")
            else:
                self.logger.warning(f"Unhandled event type: {event_type}")

    def write_subtitle(self, index, start_time, end_time, text):
        """Write a single subtitle entry to the SRT file using the srt library."""
        # Initialize video_start_time if not already set
        if self.video_start_time is None:
            self.video_start_time = start_time

        # Adjust start_time and end_time to match video timeline
        start_td = (start_time - self.video_start_time)
        end_td = (end_time - self.video_start_time)

        # Ensure no negative times
        if start_td.total_seconds() < 0:
            start_td = datetime.timedelta(seconds=0)
        if end_td.total_seconds() < 0:
            end_td = datetime.timedelta(seconds=0)

        subtitle = srt.Subtitle(index=index, start=start_td, end=end_td, content=text)
        self.subtitle_entries.append(subtitle)

        # Validate subtitles
        try:
            srt.validate(self.subtitle_entries)
        except ValueError as e:
            self.logger.error(f"Subtitle validation error: {e}")
            return

        # Write to the SRT file
        self.subtitle_file.seek(0)
        self.subtitle_file.truncate()
        self.subtitle_file.write(srt.compose(self.subtitle_entries))
        self.subtitle_file.flush()
        self.logger.debug(f"Written subtitle entry {index}")

    async def enqueue_audio(self, sequence: int, audio_data: bytes):
        """Enqueue translated audio data with its sequence number into playback queue and save to file."""
        # Validate sequence number
        if sequence != self.audio_sequence:
            self.logger.error(f"Audio sequence mismatch: expected {self.audio_sequence}, got {sequence}")
            # Handle out-of-order or missing sequences as needed
            # For now, let's skip this chunk
            return

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

        # Save the audio data to WAV for verification
        try:
            await self.audio_saver.save_audio_chunk(audio_data)
        except Exception as e:
            self.logger.error(f"Error saving audio to WAV: {e}")

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

            # Close subtitle file
            if self.subtitle_file:
                try:
                    self.subtitle_file.close()
                    self.logger.info("Subtitle SRT file closed")
                except Exception as e:
                    self.logger.error(f"Error closing subtitle SRT file: {e}")

            # Terminate FFmpeg process
            if hasattr(self, 'ffmpeg_process') and self.ffmpeg_process:
                self.ffmpeg_process.terminate()
                self.logger.info("Terminated FFmpeg process")

            # Terminate verification task
            if hasattr(self, 'verify_audio_task'):
                self.verify_audio_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self.verify_audio_task

            # Shutdown AudioSaver
            if hasattr(self, 'audio_saver'):
                await self.audio_saver.shutdown()

    async def start_ffmpeg_process(self):
        """Start FFmpeg process to combine streams with subtitles overlayed."""
        # Ensure the subtitles file exists and has at least one entry
        if not os.path.exists(self.subtitle_file_path) or os.path.getsize(self.subtitle_file_path) == 0:
            with open(self.subtitle_file_path, 'w', encoding='utf-8') as f:
                placeholder_subtitle = srt.Subtitle(
                    index=0,
                    start=datetime.timedelta(0),
                    end=datetime.timedelta(seconds=1),
                    content=" "
                )
                f.write(srt.compose([placeholder_subtitle]))
            self.logger.info("Initialized subtitles file with a placeholder.")

        # Build FFmpeg command with subtitles overlay
        ffmpeg_command = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-re',  # Read input at native frame rate
            '-i', self.stream_url,  # Input video stream
            '-f', 's16le',
            '-ar', '24000',
            '-ac', '1',
            '-i', self.translated_audio_pipe,  # Input translated audio from named pipe
            '-vf', f"subtitles={self.subtitle_file_path}:force_style='FontSize=24,PrimaryColour=&HFFFFFF&'",  # Overlay subtitles with styling
            '-c:v', 'libx264',  # Encode video using H.264
            '-c:a', 'aac',        # Encode audio using AAC
            '-b:a', '192k',
            '-preset', 'fast',    # Encoding speed
            '-pix_fmt', 'yuv420p',# Pixel format
            self.output_video_file  # Output to local file
        ]

        # Start FFmpeg process
        try:
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True  # To read stderr as text
            )
            self.logger.info("Started FFmpeg process to combine streams with subtitles overlayed")
            self.video_start_time = datetime.datetime.now()
        except Exception as e:
            self.logger.error(f"Failed to start FFmpeg process: {e}", exc_info=True)
            self.ffmpeg_process = None

    async def monitor_ffmpeg(self):
        """Monitor FFmpeg process for errors."""
        if not self.ffmpeg_process:
            self.logger.error("FFmpeg process is not running. Cannot monitor.")
            return

        try:
            while True:
                if self.ffmpeg_process.stderr:
                    line = await asyncio.to_thread(self.ffmpeg_process.stderr.readline)
                    if not line:
                        break
                    line = line.strip()
                    if line:
                        self.logger.error(f"FFmpeg: {line}")
                else:
                    await asyncio.sleep(1)
        except Exception as e:
            self.logger.error(f"Error monitoring FFmpeg process: {e}", exc_info=True)

    async def verify_audio_writing(self):
        """Coroutine to verify that audio data is being written to the pipe."""
        while True:
            try:
                if os.path.exists(self.translated_audio_pipe):
                    pipe_size = os.stat(self.translated_audio_pipe).st_size
                    self.logger.debug(f"Translated audio pipe size: {pipe_size} bytes")
                else:
                    self.logger.warning(f"Translated audio pipe {self.translated_audio_pipe} does not exist.")
            except Exception as e:
                self.logger.error(f"Error verifying translated_audio_pipe: {e}")
            await asyncio.sleep(10)  # Adjust the interval as needed

    async def run(self):
        """Main loop to run the OpenAIClient."""
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
                await self.start_ffmpeg_process()

                # Start monitoring FFmpeg
                if self.ffmpeg_process:
                    monitor_ffmpeg_task = asyncio.create_task(self.monitor_ffmpeg())
                else:
                    self.logger.error("FFmpeg process was not started. Skipping FFmpeg monitoring.")
                    monitor_ffmpeg_task = None

                # Wait for tasks to complete
                if monitor_ffmpeg_task:
                    await asyncio.gather(
                        handle_responses_task,
                        read_input_audio_task,
                        heartbeat_task,
                        monitor_ffmpeg_task,
                        return_exceptions=True
                    )
                else:
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
