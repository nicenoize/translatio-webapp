import asyncio
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
from asyncio import PriorityQueue
from collections import defaultdict

class OpenAIClient:
    def __init__(self, api_key: str, translated_audio_pipe: str):
        self.api_key = api_key
        self.ws = None
        self.logger = logging.getLogger(__name__)
        self.translated_audio_pipe = translated_audio_pipe
        self.websocket_clients: Dict[int, websockets.WebSocketServerProtocol] = {}
        self.rtmp_link = 'rtmp://sNVi5-egEGF.bintu-vtrans.nanocosmos.de/live'

        # Create directories for output if they don't exist
        os.makedirs('output/transcripts', exist_ok=True)
        os.makedirs('output/audio/input', exist_ok=True)
        os.makedirs('output/audio/output', exist_ok=True)
        self.transcript_file = open('output/transcripts/transcript.txt', 'w', encoding='utf-8')
        self.audio_counter = 0

        # Initialize FFmpeg
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

    def start_ffmpeg_stream(self):
        """Start an FFmpeg process to stream audio data to the RTMP server"""
        command = [
            'ffmpeg',
            '-loglevel', 'debug',  # Added for detailed logging
            '-f', 's16le',
            '-ar', '24000',
            '-ac', '1',
            '-i', '-',  # Read audio from stdin
            '-c:a', 'aac',
            '-b:a', '128k',
            '-f', 'flv',
            self.rtmp_link
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
                # The existing ffmpeg_writer_task will continue using the new FFmpeg process
                break

    async def ffmpeg_writer(self):
        """Dedicated coroutine to write ordered audio data to FFmpeg's stdin from ffmpeg_queue"""
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
                    self.logger.debug(f"Written audio chunk to FFmpeg: {sequence}")
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
                        self.logger.debug(f"Written buffered audio chunk to FFmpeg: {self.current_sequence}")
                    except BrokenPipeError:
                        self.logger.error("FFmpeg process has terminated unexpectedly.")
                        break  # Exit the loop if FFmpeg terminates
            elif sequence > self.current_sequence:
                # Future sequence, store in buffer
                buffer[sequence] = audio_data
                self.logger.debug(f"Buffered out-of-order chunk for FFmpeg: {sequence}")
            else:
                # Duplicate or old sequence, ignore or handle accordingly
                self.logger.warning(f"Received duplicate or old FFmpeg chunk: {sequence}")

            self.ffmpeg_queue.task_done()

    async def audio_playback_handler(self):
        """Handle ordered playback of audio chunks from playback_queue"""
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
                    self.logger.debug(f"Buffered out-of-order chunk for playback: {sequence}")
                else:
                    # Duplicate or old chunk, ignore
                    self.logger.warning(f"Ignoring duplicate or old playback chunk: {sequence}")
                self.playback_queue.task_done()
            self.playback_event.clear()

    async def process_playback_chunk(self, sequence: int, audio_data: bytes):
        """Process and play a single audio chunk for playback"""
        # Write to the output WAV file
        if self.output_wav:
            self.output_wav.writeframes(audio_data)
            self.logger.debug(f"Written audio chunk {sequence} to WAV file")
        
        # Play the audio using simpleaudio
        try:
            # Convert the audio data to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            # Create a WaveObject and play
            play_obj = sa.play_buffer(audio_array, 1, 2, 24000)
            play_obj.wait_done()  # Wait until playback is finished
            self.logger.debug(f"Played audio chunk: {sequence}")
        except Exception as e:
            self.logger.error(f"Error playing audio chunk {sequence}: {e}")
        
        # No need to enqueue back into any queue

    async def connect(self):
        """Connect to OpenAI's Realtime API and initialize session"""
        url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        self.ws = await websockets.connect(url, extra_headers=headers)
        self.logger.info("Connected to OpenAI Realtime API")

        # Configure session without invalid fields
        session_update = {
            "type": "session.update",
            "session": {
                # "object": "realtime.session",  # Ensure this field is included
                "model": "gpt-4o-realtime-preview-2024-10-01",
                "modalities": ["text", "audio"],
                "instructions": "You are a realtime translator. Please use the audio you receive and translate it into german. Try to match the tone, emotion and duration of the original audio.",
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 200
                },
                # "tools": [],
                # "tool_choice": "auto",
                "temperature": 0.8,
                "max_response_output_tokens": 500,
            }
        }
        await self.safe_send(json.dumps(session_update))
        await self.init_conversation()

    async def init_conversation(self):
        """Initialize conversation context"""
        conversation_init = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "system",
                "content": [{
                    "type": "input_text",
                    "text": (
                        "You are a realtime translator. You will receive audio from a livestream. Please use the audio and translate it into german. Try to match the tone, emotion and duration of the original audio."
                    )
                }]
            }
        }
        await self.safe_send(json.dumps(conversation_init))

    async def safe_send(self, data):
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
        if not self.ws:
            self.logger.error("WebSocket not initialized")
            return

        try:
            append_event = {
                "type": "input_audio_buffer.append",
                "audio": base64_audio
            }
            await self.safe_send(json.dumps(append_event))
            self.logger.info(f"Sent audio chunk of size: {len(base64_audio)} bytes")
        except Exception as e:
            self.logger.error(f"Error sending audio chunk: {e}", exc_info=True)

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
                elif event_type == "input_audio_buffer.speech_stopped":
                    self.logger.info("Speech stopped")
                elif event_type == "response.audio.delta":
                    self.logger.info("Received audio delta")
                    audio_data = event.get("delta", "")
                    if audio_data:
                        try:
                            decoded_audio = base64.b64decode(audio_data)
                            sequence = self.audio_sequence  # Assign current sequence number
                            await self.enqueue_audio(sequence, decoded_audio)
                            self.audio_sequence += 1  # Increment for the next chunk
                            self.logger.debug(f"Processed audio chunk: {sequence}")
                        except Exception as e:
                            self.logger.error(f"Error handling audio data: {e}")
                elif event_type == "response.audio_transcript.delta":
                    text = event.get("delta", "")
                    if text.strip():
                        self.logger.info(f"Translated text: {text}")
                        await self.broadcast_translation(text)
                elif event_type == "error":
                    error_info = event.get("error", {})
                    self.logger.error(f"OpenAI Error: {error_info}")
        except Exception as e:
            self.logger.error(f"Error handling OpenAI responses: {e}", exc_info=True)
        finally:
            await self.disconnect()

    async def enqueue_audio(self, sequence: int, audio_data: bytes):
        """Enqueue audio data with its sequence number into both playback and FFmpeg queues"""
        await self.playback_queue.put((sequence, audio_data))
        await self.ffmpeg_queue.put((sequence, audio_data))
        self.logger.debug(f"Enqueued audio chunk: {sequence} into both queues")
        self.playback_event.set()

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
        """Broadcast translation to all connected WebSocket clients"""
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
                self.logger.info("FFmpeg process terminated")
            except Exception as e:
                self.logger.error(f"Error terminating FFmpeg process: {e}")

        # Shutdown queues
        if self.playback_queue:
            await self.playback_queue.put(None)
        if self.ffmpeg_queue:
            await self.ffmpeg_queue.put(None)
        if self.playback_task:
            await self.playback_task
        if self.ffmpeg_writer_task:
            await self.ffmpeg_writer_task

        # Close WAV file
        if self.output_wav:
            try:
                self.output_wav.close()
                self.logger.info("Output WAV file closed")
            except Exception as e:
                self.logger.error(f"Error closing output WAV file: {e}")
