import asyncio
import websockets
import json
import logging
import os
import base64
import pprint
from typing import Dict
import wave
import struct
from fastapi import WebSocket
import time
import subprocess

class OpenAIClient:
    def __init__(self, api_key: str, translated_audio_pipe: str):
        self.api_key = api_key
        self.ws = None
        self.logger = logging.getLogger(__name__)
        self.translated_audio_pipe = translated_audio_pipe
        self.websocket_clients: Dict[int, WebSocket] = {}
        self.rtmp_link = 'rtmp://sNVi5-egEGF.bintu-vtrans.nanocosmos.de/live'
        self.response_in_progress = False  # Track if a response is in progress

        # Initialize audio buffers and synchronization variables
        self.outgoing_audio_buffer = bytearray()
        self.audio_buffer_lock = asyncio.Lock()
        self.min_buffer_size = 24000  # Buffer size for 0.5 seconds of audio at 24000 Hz
        self.timestamp_buffer = []  # Store timestamps of audio chunks
        self.audio_chunk_id = 0  # Unique ID for each audio chunk
        self.sent_chunks = {}  # Map chunk IDs to timestamps
        self.latency_measurements = []  # List of latency measurements
        self.max_latency_measurements = 10  # Max number of measurements to keep
        self.latency_compensation = 0.0  # Average latency compensation
        self.last_audio_chunk_sent_time = None  # Time when the last audio chunk was sent

        # Create directories for output if they don't exist
        os.makedirs('output/transcripts', exist_ok=True)
        os.makedirs('output/audio/input', exist_ok=True)
        os.makedirs('output/audio/output', exist_ok=True)
        self.transcript_file = open('output/transcripts/transcript.txt', 'w', encoding='utf-8')
        self.audio_counter = 0

        # Start FFmpeg process to stream audio to RTMP server
        self.ffmpeg_process = self.start_ffmpeg_stream()
        self.ffmpeg_log_task = asyncio.create_task(self.read_ffmpeg_logs())

    def start_ffmpeg_stream(self):
        """Start an FFmpeg process to stream audio data to the RTMP server"""
        command = [
            'ffmpeg',
            '-re',
            '-f', 's16le',
            '-ar', '24000',  # Audio sample rate
            '-ac', '1',       # Mono audio
            '-i', '-',        # Read audio from stdin
            '-c:a', 'aac',
            '-b:a', '128k',
            '-f', 'flv',
            self.rtmp_link
        ]
        return subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    async def read_ffmpeg_logs(self):
        """Read FFmpeg process stderr output"""
        while True:
            line = await asyncio.get_event_loop().run_in_executor(None, self.ffmpeg_process.stderr.readline)
            if line:
                self.logger.info(f"FFmpeg: {line.decode().strip()}")
            else:
                break

    async def connect(self):
        """Connect to OpenAI's Realtime API and initialize session"""
        url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        self.ws = await websockets.connect(url, extra_headers=headers)
        self.logger.info("Connected to OpenAI Realtime API")

        # Configure session with server-side VAD and other settings
        session_update = {
            "type": "session.update",
            "session": {
                "tools": [{
                    "type": "function",
                    "name": "translate",
                    "description": "Translate speech from English to German, maintaining natural tone and style",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The text to translate"
                            },
                            "target_language": {
                                "type": "string",
                                "enum": ["de"],
                                "default": "de"
                            }
                        },
                        "required": ["text"]
                    }
                }],
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.6,
                    "prefix_padding_ms": 500,
                    "silence_duration_ms": 400
                },
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "temperature": 0.7,
                "modalities": ["text", "audio"]
            }
        }
        await self.ws.send(json.dumps(session_update))
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
                    "text": "You are a real-time English to German translator. Try to maintain the original tone and emotion of the input."
                }]
            }
        }
        await self.ws.send(json.dumps(conversation_init))

    async def send_audio_chunk(self, base64_audio: str):
        """Append audio chunk to buffer and send when buffer size is sufficient"""
        if not self.ws:
            self.logger.error("WebSocket not initialized")
            return

        try:
            decoded_audio = base64.b64decode(base64_audio)
            timestamp = time.time()  # Capture the current time when audio is received

            async with self.audio_buffer_lock:
                self.outgoing_audio_buffer.extend(decoded_audio)
                self.timestamp_buffer.append(timestamp)

                # Send audio buffer when it reaches the minimum buffer size
                if len(self.outgoing_audio_buffer) >= self.min_buffer_size:
                    # Assign a unique ID to this audio chunk
                    self.audio_chunk_id += 1
                    chunk_id = self.audio_chunk_id

                    # Store the timestamp and sent time of this chunk
                    self.sent_chunks[chunk_id] = {
                        'timestamp': self.timestamp_buffer[0],  # Use the earliest timestamp
                        'sent_time': time.time()
                    }

                    # Prepare the event to send to OpenAI
                    append_event = {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(self.outgoing_audio_buffer).decode('utf-8')
                        # Note: The API may not accept additional fields like 'chunk_id'
                    }
                    await self.ws.send(json.dumps(append_event))

                    self.logger.info(f"Sent audio buffer of size: {len(self.outgoing_audio_buffer)} bytes, chunk_id: {chunk_id}")

                    # Update the time when the last audio chunk was sent
                    self.last_audio_chunk_sent_time = time.time()

                    # Reset buffers
                    self.outgoing_audio_buffer = bytearray()
                    self.timestamp_buffer = []

                    # Create a response if none is in progress
                    if not self.response_in_progress:
                        response_event = {
                            "type": "response.create",
                            "response": {
                                "modalities": ["text", "audio"]
                            }
                        }
                        await self.ws.send(json.dumps(response_event))
                        self.response_in_progress = True

        except Exception as e:
            self.logger.error(f"Error sending audio chunk: {e}", exc_info=True)

    async def handle_responses(self):
        """Handle translation responses from OpenAI"""
        try:
            while True:
                response = await self.ws.recv()
                event = json.loads(response)
                event_type = event.get("type")

                self.logger.debug(f"Received event: {pprint.pformat(event)}")

                if event_type == "response.audio.delta":
                    audio_data = event.get("delta", "")
                    if audio_data:
                        try:
                            decoded_audio = base64.b64decode(audio_data)
                            current_time = time.time()

                            # Measure latency and update latency compensation
                            if self.last_audio_chunk_sent_time:
                                latency = current_time - self.last_audio_chunk_sent_time
                                self.latency_measurements.append(latency)
                                if len(self.latency_measurements) > self.max_latency_measurements:
                                    self.latency_measurements.pop(0)
                                self.latency_compensation = sum(self.latency_measurements) / len(self.latency_measurements)
                                self.last_audio_chunk_sent_time = None  # Reset after measuring latency

                            # Schedule playback considering latency compensation
                            playback_time = current_time + self.latency_compensation
                            asyncio.create_task(self.play_audio_at_time(decoded_audio, playback_time))

                        except Exception as e:
                            self.logger.error(f"Error handling audio data: {e}")

                elif event_type == "response.audio_transcript.delta":
                    text = event.get("delta", "")
                    if text.strip():
                        # Handle text as needed (e.g., display or log)
                        self.logger.info(f"Translated text: {text}")

                elif event_type == "response.done":
                    # Reset the response in progress flag
                    self.response_in_progress = False

                elif event_type == "error":
                    error_info = event.get("error", {})
                    self.logger.error(f"OpenAI Error: {error_info}")
                    # Reset the response in progress flag on error
                    self.response_in_progress = False

        except websockets.exceptions.ConnectionClosed as e:
            self.logger.error(f"WebSocket connection closed: {e}")
        except Exception as e:
            self.logger.error(f"Error handling OpenAI responses: {e}")
        finally:
            await self.disconnect()

    async def play_audio_at_time(self, audio_data: bytes, playback_time: float):
        """Play the audio data at the specified playback_time"""
        delay = playback_time - time.time()
        if delay > 0:
            await asyncio.sleep(delay)
        # Write the audio data to FFmpeg stdin for streaming
        if self.ffmpeg_process and self.ffmpeg_process.stdin:
            try:
                self.ffmpeg_process.stdin.write(audio_data)
                self.ffmpeg_process.stdin.flush()
            except BrokenPipeError:
                self.logger.error("FFmpeg process has terminated unexpectedly.")

    async def register_websocket(self, websocket: WebSocket) -> int:
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
                await websocket.send_text(message)
            except Exception as e:
                self.logger.error(f"Error broadcasting to client {client_id}: {e}")
                disconnected_clients.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.unregister_websocket(client_id)

    def wrap_pcm_in_wav(self, pcm_data: bytes, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2) -> bytes:
        """Wrap PCM data in WAV format"""
        try:
            num_samples = len(pcm_data) // (sample_width * channels)
            byte_rate = sample_rate * channels * sample_width
            block_align = channels * sample_width
            data_size = len(pcm_data)
            fmt_chunk = struct.pack('<4sIHHIIHH',
                                    b'fmt ', 16, 1, channels, sample_rate,
                                    byte_rate, block_align, sample_width * 8)
            data_chunk = struct.pack('<4sI', b'data', data_size)
            wav_header = struct.pack('<4sI4s', b'RIFF', 36 + data_size, b'WAVE') + fmt_chunk + data_chunk
            return wav_header + pcm_data
        except Exception as e:
            self.logger.error(f"Error wrapping PCM data into WAV: {e}", exc_info=True)
            return b''

    async def disconnect(self):
        """Disconnect from OpenAI and clean up resources"""
        if self.ws:
            try:
                await self.ws.close()
                self.logger.info("Disconnected from OpenAI Realtime API")
            except Exception as e:
                self.logger.error(f"Error disconnecting from OpenAI: {e}")

        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.stdin.close()
                self.ffmpeg_process.terminate()
                await self.ffmpeg_process.wait()
                self.logger.info("FFmpeg process terminated")
            except Exception as e:
                self.logger.error(f"Error terminating FFmpeg process: {e}")
