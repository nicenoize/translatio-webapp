import asyncio
import websockets
import json
import logging
import os
import base64
from contextlib import suppress
import pprint
from typing import Optional, Dict
import wave
import struct
from fastapi import WebSocket

class OpenAIClient:
    def __init__(self, api_key: str, translated_audio_pipe: str):
        self.api_key = api_key
        self.ws = None
        self.logger = logging.getLogger(__name__)
        self.translated_audio_pipe = translated_audio_pipe
        self.websocket_clients: Dict[int, WebSocket] = {}
        
        if not os.path.exists(self.translated_audio_pipe):
            raise FileNotFoundError(f"Named pipe not found: {self.translated_audio_pipe}")
            
        try:
            self.translated_audio_fd = os.open(self.translated_audio_pipe, os.O_RDWR | os.O_NONBLOCK)
            self.logger.info(f"Successfully opened translated audio pipe: {self.translated_audio_pipe}")
        except Exception as e:
            self.logger.error(f"Failed to open translated audio pipe: {e}")
            raise

        # Create output directories
        os.makedirs('output/transcripts', exist_ok=True)
        os.makedirs('output/audio', exist_ok=True)
        
        self.transcript_file = open('output/transcripts/transcript.txt', 'w', encoding='utf-8')
        self.audio_counter = 0

        # Initialize buffers for outgoing audio
        self.outgoing_audio_buffer = bytearray()
        self.audio_buffer_lock = asyncio.Lock()
        
        # Increase buffer size to 5 seconds of audio at 24kHz, 16-bit mono
        # 24000 samples/sec * 2 bytes/sample * 5 sec = 240000 bytes
        self.min_buffer_size = 240000  # 5 seconds buffer
        
        # Add translation accumulation buffer
        self.current_translation = ""
        self.translation_buffer = []

    async def connect(self):
        """Connect to OpenAI's Realtime API"""
        url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        self.ws = await websockets.connect(url, extra_headers=headers)
        self.logger.info("Connected to OpenAI Realtime API")

        # Update session configuration for better translation quality
        session_update = {
            "type": "session.update",
            "session": {
                "instructions": """You are a real-time translator. Translate all incoming English speech into natural German. 
                Maintain the speaking style and tone of the original speaker. Wait for complete thoughts or sentences before translating.
                Do not engage in conversation or add any commentary. Only translate the content.""",
                "model": "gpt-4o-realtime-preview-2024-10-01",
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1",
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.6,  # Increased threshold for better sentence detection
                    "prefix_padding_ms": 500,  # Increased padding
                    "silence_duration_ms": 400  # Increased silence duration
                },
                "temperature": 0.7,
                "modalities": ["text", "audio"],
                "max_response_output_tokens": "inf"
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
                    "text": "Translate incoming English speech to German. Maintain speaker style and wait for complete thoughts."
                }]
            }
        }
        await self.ws.send(json.dumps(conversation_init))

    async def send_audio_chunk(self, base64_audio: str):
        """Append audio chunk to buffer and send if buffer size is sufficient"""
        if not self.ws:
            self.logger.error("WebSocket not initialized")
            return

        try:
            decoded_audio = base64.b64decode(base64_audio)
            async with self.audio_buffer_lock:
                self.outgoing_audio_buffer.extend(decoded_audio)
                self.logger.debug(f"Appended {len(decoded_audio)} bytes to outgoing buffer. Total buffer size: {len(self.outgoing_audio_buffer)} bytes")

                if len(self.outgoing_audio_buffer) >= self.min_buffer_size:
                    # Wrap PCM data in WAV format
                    wav_bytes = self.wrap_pcm_in_wav(self.outgoing_audio_buffer)
                    # Send audio buffer append event
                    append_event = {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(self.outgoing_audio_buffer).decode('utf-8')
                    }
                    await self.ws.send(json.dumps(append_event))
                    self.logger.debug("Sent audio buffer append event")

                    # Commit the audio chunk for processing
                    commit_event = {
                        "type": "input_audio_buffer.commit"
                    }
                    await self.ws.send(json.dumps(commit_event))
                    self.logger.info("Committed audio chunk")

                    # Save and clear the buffer
                    self.save_audio_segment(self.outgoing_audio_buffer)
                    self.outgoing_audio_buffer = bytearray()
        except Exception as e:
            self.logger.error(f"Error sending audio chunk: {e}", exc_info=True)

    async def send_audio_commit(self):
        """Commit audio buffer and create response"""
        if not self.ws:
            return
            
        try:
            # Commit the buffer
            commit_event = {
                "type": "input_audio_buffer.commit"
            }
            await self.ws.send(json.dumps(commit_event))
            
            # Create a response
            response_event = {
                "type": "response.create",
                "response": {
                    "modalities": ["text", "audio"],
                    "instructions": "Translate to German"
                }
            }
            await self.ws.send(json.dumps(response_event))
            
            self.logger.info("Committed audio buffer and requested translation")
            
        except Exception as e:
            self.logger.error(f"Error committing audio: {e}", exc_info=True)


    async def commit_audio(self):
        """Commit audio and request translation"""
        if self.ws:
            await self.ws.send(json.dumps({
                "type": "input_audio_buffer.commit"
            }))
            # Trigger a new response after committing
            await self.ws.send(json.dumps({
                "type": "response.create",
                "response": {
                    "modalities": ["text", "audio"],
                    "instructions": "Translate to German"
                }
            }))
            self.logger.debug("Committed audio buffer for translation")

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

    async def broadcast_audio(self, audio_data: bytes):
        """Broadcast audio data to all connected WebSocket clients"""
        base64_audio = base64.b64encode(audio_data).decode('utf-8')
        message = json.dumps({
            "type": "audio",
            "audio": base64_audio
        })
        disconnected_clients = []
        
        for client_id, websocket in self.websocket_clients.items():
            try:
                await websocket.send_text(message)
            except Exception as e:
                self.logger.error(f"Error broadcasting audio to client {client_id}: {e}")
                disconnected_clients.append(client_id)
                
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.unregister_websocket(client_id)


    def save_audio_segment(self, audio_data: bytes, sample_rate: int = 24000):
        """Save audio segment as WAV file"""
        filename = f'output/audio/segment_{self.audio_counter}.wav'
        
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)
        
        self.audio_counter += 1
        self.logger.debug(f"Saved audio segment: {filename}")
        return filename

    async def handle_responses(self):
        """Handle translation responses from OpenAI"""
        try:
            wav_buffer = bytearray()
            accumulated_text = ""

            while True:
                response = await self.ws.recv()
                event = json.loads(response)
                event_type = event.get("type")
                
                self.logger.debug(f"Received event: {pprint.pformat(event)}")

                if event_type == "response.audio.delta":
                    # Handle audio data with larger chunks
                    audio_data = event.get("delta", "")
                    if audio_data:
                        try:
                            decoded_audio = base64.b64decode(audio_data)
                            wav_buffer.extend(decoded_audio)
                            self.logger.debug(f"Appended {len(decoded_audio)} bytes to received audio buffer. Total size: {len(wav_buffer)} bytes")

                            # Increased buffer size for better audio quality (10 seconds of audio)
                            if len(wav_buffer) >= 480000:  # 24kHz * 2 bytes * 10 seconds
                                wav_bytes = self.wrap_pcm_in_wav(wav_buffer)
                                await self.broadcast_audio(wav_bytes)
                                self.save_audio_segment(wav_buffer)
                                self.logger.debug("Broadcasted and saved 10-second audio segment")
                                wav_buffer = bytearray()
                                
                        except Exception as e:
                            self.logger.error(f"Error handling audio data: {e}")

                elif event_type == "response.audio_transcript.delta":
                    # Accumulate translation text
                    text = event.get("delta", "")
                    if text.strip():
                        accumulated_text += text
                        self.transcript_file.write(text)
                        self.transcript_file.flush()
                        
                        # Only broadcast complete sentences
                        if any(char in text for char in ['.', '!', '?', '\n']):
                            await self.broadcast_translation(accumulated_text)
                            self.logger.debug(f"Broadcasted accumulated translation: {accumulated_text}")
                            accumulated_text = ""

                elif event_type == "response.audio.done":
                    # Handle remaining audio and text
                    if wav_buffer:
                        wav_bytes = self.wrap_pcm_in_wav(wav_buffer)
                        await self.broadcast_audio(wav_bytes)
                        self.save_audio_segment(wav_buffer)
                        self.logger.debug("Broadcasted and saved remaining audio buffer")
                        wav_buffer = bytearray()
                    
                    if accumulated_text:
                        await self.broadcast_translation(accumulated_text)
                        self.logger.debug(f"Broadcasted remaining translation: {accumulated_text}")
                        accumulated_text = ""

                elif event_type == "error":
                    error_info = event.get("error", {})
                    self.logger.error(f"OpenAI Error: {error_info}")

        except websockets.exceptions.ConnectionClosed as e:
            self.logger.error(f"WebSocket connection closed: {e}")
        except Exception as e:
            self.logger.error(f"Error handling OpenAI responses: {e}")
        finally:
            self.cleanup()

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
        """Disconnect from OpenAI"""
        if self.ws:
            try:
                await self.ws.close()
                self.logger.info("Disconnected from OpenAI Realtime API")
            except Exception as e:
                self.logger.error(f"Error disconnecting from OpenAI: {e}")
            
        if self.translated_audio_fd is not None:
            try:
                os.close(self.translated_audio_fd)
            except OSError as e:
                self.logger.error(f"Error closing translated audio pipe: {e}")