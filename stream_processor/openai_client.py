# stream_processor/openai_client.py

import asyncio
import aiofiles
import websockets
import json
import logging
import os
import base64
import pprint
from typing import Dict
import wave
import numpy as np
import simpleaudio as sa  # For playing audio locally
import subprocess
from asyncio import PriorityQueue, Queue
from collections import defaultdict
import datetime
import srt
from contextlib import suppress

class OpenAIClient:
    def __init__(self, api_key: str, translated_audio_pipe: str):
        self.api_key = api_key
        self.ws = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.translated_audio_pipe = translated_audio_pipe
        self.websocket_clients: Dict[int, websockets.WebSocketServerProtocol] = {}
        self.rtmp_link = 'rtmp://sNVi5-egEGF.bintu-vtrans.nanocosmos.de/live'
        self.stream_url = 'https://bintu-play.nanocosmos.de/h5live/http/stream.mp4?url=rtmp://localhost/play&stream=sNVi5-kYN1t'

        # Create directories for output if they don't exist
        os.makedirs('output/transcripts', exist_ok=True)
        os.makedirs('output/audio/input', exist_ok=True)
        os.makedirs('output/audio/output', exist_ok=True)
        os.makedirs('output/subtitles', exist_ok=True)

        # Create named pipes if they don't exist
        self.create_named_pipe('input_audio_pipe')
        self.create_named_pipe('translated_audio_pipe')

        self.transcript_file = open('output/transcripts/transcript.txt', 'w', encoding='utf-8')
        self.audio_counter = 0

        # Initialize FFmpeg Process 2 (Stream with translated audio and subtitles)
        self.ffmpeg_process = self.start_ffmpeg_stream()
        self.ffmpeg_log_task = asyncio.create_task(self.read_ffmpeg_logs())

        # Initialize separate queues for playback and FFmpeg
        self.playback_queue = PriorityQueue()
        self.ffmpeg_queue = PriorityQueue()

        # Initialize playback buffer
        self.playback_buffer = defaultdict(bytes)  # Buffer to store out-of-order chunks
        self.playback_sequence = 0  # Expected sequence number for playback
        self.playback_event = asyncio.Event()  # Event to signal available audio

        # Initialize a separate sequence counter for audio chunks
        self.audio_sequence = 0  # Tracks the next expected audio sequence number

        # Create tasks for playback and FFmpeg writing
        self.playback_task = asyncio.create_task(self.audio_playback_handler())
        self.ffmpeg_writer_task = asyncio.create_task(self.ffmpeg_writer())
        self.current_sequence = 0  # Tracks the current expected sequence for FFmpeg

        # Add a lock for reconnecting to prevent race conditions
        self.is_reconnecting = False
        self.reconnect_lock = asyncio.Lock()

        # Initialize WAV file for output
        self.output_wav = wave.open('output/audio/output_audio.wav', 'wb')
        self.output_wav.setnchannels(1)
        self.output_wav.setsampwidth(2)  # 16-bit PCM
        self.output_wav.setframerate(24000)

        # Initialize subtitle variables
        self.subtitle_entries = []  # List to store subtitle entries
        self.subtitle_index = 1  # Subtitle entry index
        self.current_subtitle = ""
        self.subtitle_start_time = None
        self.subtitle_file = open('output/subtitles/subtitles.srt', 'w', encoding='utf-8')

        # Initialize the send queue
        self.send_queue = Queue()
        self.send_task = asyncio.create_task(self.send_messages())

    def create_named_pipe(self, pipe_name):
        if not os.path.exists(pipe_name):
            os.mkfifo(pipe_name)
            self.logger.info(f"Created named pipe: {pipe_name}")
        else:
            self.logger.info(f"Named pipe already exists: {pipe_name}")

    async def send_messages(self):
        """Dedicated coroutine to send messages from the send_queue."""
        while True:
            message = await self.send_queue.get()
            if message is None:
                self.logger.info("Send queue received shutdown signal.")
                break  # Allows graceful shutdown

            try:
                await self.safe_send(message)
                self.logger.debug(f"Sent message: {message}")
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
        try:
            async with aiofiles.open('input_audio_pipe', 'rb') as pipe:
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
            self.logger.error(f"Error reading from input_audio_pipe: {e}", exc_info=True)

    def start_ffmpeg_stream(self):
        """
        Start an FFmpeg process to combine translated audio with the original video stream,
        overlay subtitles, and stream to the RTMP server.
        """
        command = [
            'ffmpeg',
            '-loglevel', 'debug',  # Detailed logging
            '-f', 's16le',          # Input format for translated audio
            '-ar', '24000',         # Audio sample rate
            '-ac', '1',             # Audio channels
            '-i', self.translated_audio_pipe,  # Translated audio input
            '-i', self.stream_url,  # Original video stream
            '-vf', "subtitles=output/subtitles/subtitles.srt",  # Overlay subtitles
            '-c:v', 'libx264',      # Video codec
            '-c:a', 'aac',          # Audio codec
            '-b:a', '128k',         # Audio bitrate
            '-f', 'flv',            # Output format for RTMP
            '-flags', '-global_header',
            '-fflags', 'nobuffer',
            '-flush_packets', '0',
            '-shortest',            # Stop encoding when the shortest input ends
            self.rtmp_link          # RTMP server URL
        ]

        self.logger.info(f"Starting FFmpeg with command: {' '.join(command)}")
        return subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    async def read_ffmpeg_logs(self):
        """Read FFmpeg process stderr output"""
        while True:
            line = await asyncio.get_event_loop().run_in_executor(None, self.ffmpeg_process.stderr.readline)
            if line:
                self.logger.debug(f"FFmpeg: {line.decode().strip()}")
            else:
                self.logger.warning("FFmpeg process has terminated.")
                # Attempt to restart FFmpeg
                self.ffmpeg_process = self.start_ffmpeg_stream()
                self.ffmpeg_log_task = asyncio.create_task(self.read_ffmpeg_logs())
                break

    async def ffmpeg_writer(self):
        """Dedicated coroutine to write ordered translated audio data to FFmpeg's stdin from ffmpeg_queue"""
        buffer = {}
        while True:
            item = await self.ffmpeg_queue.get()
            if item is None:
                self.logger.info("FFmpeg writer received shutdown signal.")
                break  # Allows graceful shutdown

            sequence, audio_data = item
            if sequence == self.current_sequence:
                try:
                    if self.ffmpeg_process and self.ffmpeg_process.stdin:
                        self.ffmpeg_process.stdin.write(audio_data)
                        self.ffmpeg_process.stdin.flush()
                    self.current_sequence += 1
                    self.logger.debug(f"Written translated audio chunk to FFmpeg: {sequence}")
                except BrokenPipeError:
                    self.logger.error("FFmpeg process has terminated unexpectedly.")
                    # FFmpeg restart is handled in read_ffmpeg_logs
                # Check if the next sequences are already in the buffer
                while self.current_sequence in buffer:
                    buffered_data = buffer.pop(self.current_sequence)
                    try:
                        if self.ffmpeg_process and self.ffmpeg_process.stdin:
                            self.ffmpeg_process.stdin.write(buffered_data)
                            self.ffmpeg_process.stdin.flush()
                        self.current_sequence += 1
                        self.logger.debug(f"Written buffered translated audio chunk to FFmpeg: {self.current_sequence}")
                    except BrokenPipeError:
                        self.logger.error("FFmpeg process has terminated unexpectedly.")
                        break  # Exit the loop if FFmpeg terminates
            elif sequence > self.current_sequence:
                # Future sequence, store in buffer
                buffer[sequence] = audio_data
                self.logger.debug(f"Buffered out-of-order translated audio chunk for FFmpeg: {sequence}")
            else:
                # Duplicate or old sequence, ignore or handle accordingly
                self.logger.warning(f"Received duplicate or old translated FFmpeg chunk: {sequence}")

            self.ffmpeg_queue.task_done()

    async def audio_playback_handler(self):
        """Handle ordered playback of translated audio chunks from playback_queue"""
        while True:
            await self.playback_event.wait()
            while not self.playback_queue.empty():
                sequence, audio_data = await self.playback_queue.get()
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

    async def process_playback_chunk(self, sequence: int, audio_data: bytes):
        """Process and play a single translated audio chunk for playback and save to WAV"""
        # Write to the output WAV file
        if self.output_wav:
            self.output_wav.writeframes(audio_data)
            self.logger.debug(f"Written translated audio chunk {sequence} to WAV file")

        # Play the audio using simpleaudio
        try:
            # Convert the audio data to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            # Create a WaveObject and play
            play_obj = sa.play_buffer(audio_array, 1, 2, 24000)
            play_obj.wait_done()  # Wait until playback is finished
            self.logger.debug(f"Played translated audio chunk: {sequence}")
        except Exception as e:
            self.logger.error(f"Error playing translated audio chunk {sequence}: {e}")

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
                "object": "realtime.session",
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
            raise

    async def reconnect(self):
        """Reconnect to the WebSocket server"""
        async with self.reconnect_lock:
            if self.is_reconnecting:
                self.logger.debug("Already reconnecting. Skipping additional reconnect attempts.")
                return
            self.is_reconnecting = True
            self.logger.info("Reconnecting to OpenAI Realtime API...")
            await self.disconnect()
            try:
                await self.connect()
                self.logger.info("Reconnected to OpenAI Realtime API.")
            except Exception as e:
                self.logger.error(f"Failed to reconnect to OpenAI Realtime API: {e}", exc_info=True)
            self.is_reconnecting = False

    async def send_audio_chunk(self, base64_audio: str):
        """Send audio chunk to the OpenAI Realtime API"""
        append_event = {
            "type": "input_audio_buffer.append",
            "audio": base64_audio
        }
        await self.enqueue_message(json.dumps(append_event))
        self.logger.info(f"Enqueued audio chunk of size: {len(base64_audio)} bytes")

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
                "instructions": "Please assist the user with the translation."
            }
        }
        await self.enqueue_message(json.dumps(response_create_event))
        self.logger.info("Enqueued response.create message")

    async def handle_responses(self):
        """Handle translation responses from OpenAI"""
        try:
            while True:
                try:
                    response = await self.ws.recv()
                except (websockets.exceptions.ConnectionClosed, websockets.exceptions.ConnectionClosedError) as e:
                    self.logger.error(f"WebSocket connection closed during recv: {e}")
                    await self.reconnect()
                    continue  # Attempt to receive again after reconnecting
                except Exception as e:
                    self.logger.error(f"Exception during WebSocket recv: {e}", exc_info=True)
                    await self.reconnect()
                    continue

                event = json.loads(response)
                event_type = event.get("type")

                self.logger.debug(f"Received event: {pprint.pformat(event)}")

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
        except Exception as e:
            self.logger.error(f"Error handling OpenAI responses: {e}", exc_info=True)
        finally:
            await self.disconnect()

    def write_subtitle(self, index, start_time, end_time, text):
        """Write a single subtitle entry to the SRT file using the srt library"""
        # Convert datetime to timedelta
        start_td = datetime.timedelta(
            hours=start_time.hour,
            minutes=start_time.minute,
            seconds=start_time.second,
            microseconds=start_time.microsecond
        )
        end_td = datetime.timedelta(
            hours=end_time.hour,
            minutes=end_time.minute,
            seconds=end_time.second,
            microseconds=end_time.microsecond
        )

        subtitle = srt.Subtitle(index=index, start=start_td, end=end_td, content=text)
        self.subtitle_entries.append(subtitle)

        # Write to the SRT file
        self.subtitle_file.write(srt.compose([subtitle]))
        self.subtitle_file.flush()
        self.logger.debug(f"Written subtitle entry {index}")

    async def enqueue_audio(self, sequence: int, audio_data: bytes):
        """Enqueue translated audio data with its sequence number into both playback and FFmpeg queues"""
        await self.playback_queue.put((sequence, audio_data))
        await self.ffmpeg_queue.put((sequence, audio_data))
        self.logger.debug(f"Enqueued audio chunk: {sequence} into both queues")
        self.playback_event.set()

    async def heartbeat(self):
        """Send periodic heartbeat pings to keep the WebSocket connection alive."""
        while self.ws and self.ws.open:
            try:
                await self.ws.ping()
                self.logger.debug("Sent heartbeat ping")
            except Exception as e:
                self.logger.error(f"Heartbeat ping failed: {e}")
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

    async def disconnect(self):
        """Disconnect from OpenAI and clean up resources"""
        if self.ws:
            try:
                await self.ws.close()
                self.logger.info("Disconnected from OpenAI Realtime API")
            except Exception as e:
                self.logger.error(f"Error disconnecting from OpenAI: {e}")
            self.ws = None  # Reset the WebSocket connection

        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.stdin.close()
                self.ffmpeg_process.terminate()
                await asyncio.get_event_loop().run_in_executor(None, self.ffmpeg_process.wait)
                self.logger.info("FFmpeg Process 2 terminated")
            except Exception as e:
                self.logger.error(f"Error terminating FFmpeg Process 2: {e}")

        # Shutdown queues
        if self.playback_queue:
            await self.playback_queue.put(None)
        if self.ffmpeg_queue:
            await self.ffmpeg_queue.put(None)
        if self.playback_task:
            await self.playback_task
        if self.ffmpeg_writer_task:
            await self.ffmpeg_writer_task
        if self.send_task:
            await self.send_queue.put(None)
            await self.send_task

        # Close WAV file
        if self.output_wav:
            try:
                self.output_wav.close()
                self.logger.info("Output WAV file closed")
            except Exception as e:
                self.logger.error(f"Error closing output WAV file: {e}")

        # Close subtitle file
        if self.subtitle_file:
            try:
                self.subtitle_file.close()
                self.logger.info("Subtitle SRT file closed")
            except Exception as e:
                self.logger.error(f"Error closing subtitle SRT file: {e}")

    async def run(self):
        try:
            await self.connect()
            # Start handling responses
            handle_responses_task = asyncio.create_task(self.handle_responses())
            # Start reading and sending audio
            read_input_audio_task = asyncio.create_task(self.read_input_audio())
            # Start heartbeat
            heartbeat_task = asyncio.create_task(self.heartbeat())
            # Wait for tasks to complete
            await asyncio.gather(handle_responses_task, read_input_audio_task, heartbeat_task)
        except Exception as e:
            self.logger.error(f"Error in OpenAIClient run: {e}", exc_info=True)
        finally:
            await self.disconnect()

