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
import time

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

        # Start FFmpeg process to stream audio to RTMP server
        self.ffmpeg_process = self.start_ffmpeg_stream()
        self.ffmpeg_log_task = asyncio.create_task(self.read_ffmpeg_logs())

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

        # Configure session without invalid fields
        session_update = {
            "type": "session.update",
            "session": {
                "id": "sess_001",
                "object": "realtime.session",
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
                await self.ws.send(data)
        except (websockets.exceptions.ConnectionClosed, websockets.exceptions.ConnectionClosedError) as e:
            self.logger.error(f"WebSocket connection closed during send: {e}")
            await self.reconnect()
            await self.ws.send(data)
        except Exception as e:
            self.logger.error(f"Exception during WebSocket send: {e}", exc_info=True)
            raise

    async def reconnect(self):
        """Reconnect to the WebSocket server"""
        async with self.reconnect_lock:
            if self.is_reconnecting:
                return
            self.is_reconnecting = True
            self.logger.info("Reconnecting to OpenAI Realtime API...")
            await self.disconnect()
            await self.connect()
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
                    print("Received audio delta")
                    audio_data = event.get("delta", "")
                    if audio_data:
                        try:
                            decoded_audio = base64.b64decode(audio_data)
                            asyncio.create_task(self.play_audio(decoded_audio))
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

    async def play_audio(self, audio_data: bytes):
        """Play the audio data immediately"""
        # Write the audio data to FFmpeg stdin for streaming
        if self.ffmpeg_process and self.ffmpeg_process.stdin:
            try:
                self.ffmpeg_process.stdin.write(audio_data)
                self.ffmpeg_process.stdin.flush()
            except BrokenPipeError:
                self.logger.error("FFmpeg process has terminated unexpectedly.")

        # Write the audio data to the output WAV file
        if self.output_wav:
            self.output_wav.writeframes(audio_data)

        # Optionally, play the audio data using simpleaudio
        try:
            # Convert the audio data to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            # Play the audio data
            sa.play_buffer(audio_array, 1, 2, 24000)
        except Exception as e:
            self.logger.error(f"Error playing audio: {e}")

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
                await self.ffmpeg_process.wait()
                self.logger.info("FFmpeg process terminated")
            except Exception as e:
                self.logger.error(f"Error terminating FFmpeg process: {e}")

        if self.output_wav:
            try:
                self.output_wav.close()
                self.logger.info("Output WAV file closed")
            except Exception as e:
                self.logger.error(f"Error closing output WAV file: {e}")
