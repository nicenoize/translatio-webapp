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
import time
import webrtcvad
import subprocess

class OpenAIClient:
    def __init__(self, api_key: str, translated_audio_pipe: str):
        self.api_key = api_key
        self.ws = None
        self.logger = logging.getLogger(__name__)
        self.translated_audio_pipe = translated_audio_pipe
        self.websocket_clients: Dict[int, WebSocket] = {}
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(1)  # 0: Aggressive, 3: Very Aggressive
        self.silence_duration = 0
        self.silence_threshold = 1.0  # seconds
        self.rtmp_link = 'rtmp://sNVi5-egEGF.bintu-vtrans.nanocosmos.de/live'
        self.ffmpeg_process = self.start_ffmpeg_stream()
        
        if not os.path.exists(self.translated_audio_pipe):
            raise FileNotFoundError(f"Named pipe not found: {self.translated_audio_pipe}")
            
        try:
            self.translated_audio_fd = os.open(self.translated_audio_pipe, os.O_RDWR | os.O_NONBLOCK)
            self.logger.info(f"Successfully opened translated audio pipe: {self.translated_audio_pipe}")
        except Exception as e:
            self.logger.error(f"Failed to open translated audio pipe: {e}")
            raise

        # Create separate directories for input and output audio
        os.makedirs('output/transcripts', exist_ok=True)
        os.makedirs('output/audio/input', exist_ok=True)
        os.makedirs('output/audio/output', exist_ok=True)
        
        self.transcript_file = open('output/transcripts/transcript.txt', 'w', encoding='utf-8')
        self.audio_counter = 0
        self.current_input_id = None  # Track current input audio ID
        self.translation_pairs = {}  # Track input-output audio pairs

        self.outgoing_audio_buffer = bytearray()
        self.audio_buffer_lock = asyncio.Lock()
        self.min_buffer_size = 240000  # 5 seconds buffer
        self.current_translation = ""
        self.translation_buffer = []


    def start_ffmpeg_stream(self):
        """Start an ffmpeg process to stream audio data to the RTMP server"""
        command = [
            'ffmpeg',
            '-re',
            '-f', 's16le',
            '-ar', '24000',
            '-ac', '1',
            '-i', '-',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-f', 'flv',
            self.rtmp_link
        ]
        return subprocess.Popen(command, stdin=subprocess.PIPE)
    
    def should_commit_audio(self):
        """Determine if audio should be committed based on silence detection"""
        # Analyze the last chunk of audio for silence
        frame_duration = 20  # ms
        num_bytes_per_frame = int(24000 * 2 * (frame_duration / 1000))  # sample_rate * sample_width * duration
        if len(self.outgoing_audio_buffer) < num_bytes_per_frame:
            return False

        frames = [self.outgoing_audio_buffer[i:i+num_bytes_per_frame] 
                  for i in range(0, len(self.outgoing_audio_buffer), num_bytes_per_frame)]
        
        is_speech = False
        for frame in frames[-5:]:  # Check the last 100ms
            if self.vad.is_speech(frame, 24000):
                is_speech = True
                break

        if not is_speech:
            self.silence_duration += frame_duration / 1000.0
        else:
            self.silence_duration = 0

        if self.silence_duration >= self.silence_threshold:
            return True
        return False

    async def connect(self):
        """Connect to OpenAI's Realtime API"""
        url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        self.ws = await websockets.connect(url, extra_headers=headers)
        self.logger.info("Connected to OpenAI Realtime API")

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
        """Append audio chunk to buffer and send if buffer size is sufficient"""
        if not self.ws:
            self.logger.error("WebSocket not initialized")
            return

        try:
            decoded_audio = base64.b64decode(base64_audio)
            async with self.audio_buffer_lock:
                self.outgoing_audio_buffer.extend(decoded_audio)
                
                if len(self.outgoing_audio_buffer) >= self.min_buffer_size:
                    # Generate unique ID for this input audio
                    self.current_input_id = f"input_{int(time.time())}_{self.audio_counter}"
                    
                    # Save input audio first
                    input_filename = self.save_audio_segment(
                        self.outgoing_audio_buffer, 
                        is_input=True, 
                        audio_id=self.current_input_id
                    )
                    
                    # Send audio buffer
                    append_event = {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(self.outgoing_audio_buffer).decode('utf-8')
                    }
                    await self.ws.send(json.dumps(append_event))
                    
                    # Commit the buffer
                    commit_event = {
                        "type": "input_audio_buffer.commit"
                    }
                    await self.ws.send(json.dumps(commit_event))

                    # Request translation
                    response_event = {
                        "type": "response.create",
                        "response": {
                            "tools": [{
                                "type": "function",
                                "name": "translate",
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
                            "modalities": ["text", "audio"]
                        }
                    }
                    await self.ws.send(json.dumps(response_event))
                    
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
            
            # Create a response with proper tool configuration
            response_event = {
                "type": "response.create",
                "response": {
                    "tools": [{
                        "type": "function",
                        "name": "translate",
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
                    "modalities": ["text", "audio"]
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


    # Differentiate between outgoing or incoming audio
    def save_audio_segment(self, audio_data: bytes, is_input: bool = True, audio_id: str = None) -> str:
        """Save audio segment as WAV file with organized naming"""
        if audio_id is None:
            audio_id = f"{'input' if is_input else 'output'}_{int(time.time())}_{self.audio_counter}"
        
        # Determine directory and filename
        subdir = 'input' if is_input else 'output'
        filename = f'output/audio/{subdir}/{audio_id}.wav'
        
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(24000)  # 24kHz
            wav_file.writeframes(audio_data)
        
        if is_input:
            self.translation_pairs[audio_id] = None  # Initialize pair tracking
        
        self.logger.debug(f"Saved {'input' if is_input else 'output'} audio: {filename}")
        self.audio_counter += 1
        return filename

    async def handle_responses(self):
        """Handle translation responses from OpenAI"""
        try:
            wav_buffer = bytearray()
            accumulated_text = ""
            current_output_id = None

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
                            wav_buffer.extend(decoded_audio)

                            if len(wav_buffer) >= 480000:  # 10 seconds buffer
                                if not current_output_id:
                                    current_output_id = f"output_{int(time.time())}_{self.audio_counter}"
                                    if self.current_input_id:
                                        self.translation_pairs[self.current_input_id] = current_output_id
                                
                                wav_bytes = self.wrap_pcm_in_wav(wav_buffer)
                                await self.broadcast_audio(wav_bytes)
                                self.save_audio_segment(wav_buffer, is_input=False, audio_id=current_output_id)
                                wav_buffer = bytearray()
                                
                        except Exception as e:
                            self.logger.error(f"Error handling audio data: {e}")

                elif event_type == "response.audio_transcript.delta":
                    text = event.get("delta", "")
                    if text.strip():
                        accumulated_text += text
                        self.transcript_file.write(f"[{self.current_input_id}] {text}\n")
                        self.transcript_file.flush()
                        
                        if any(char in text for char in ['.', '!', '?', '\n']):
                            await self.broadcast_translation(accumulated_text)
                            accumulated_text = ""

                elif event_type == "response.audio.done":
                    if wav_buffer:
                        if not current_output_id:
                            current_output_id = f"output_{int(time.time())}_{self.audio_counter}"
                            if self.current_input_id:
                                self.translation_pairs[self.current_input_id] = current_output_id
                        
                        wav_bytes = self.wrap_pcm_in_wav(wav_buffer)
                        await self.broadcast_audio(wav_bytes)
                        self.save_audio_segment(wav_buffer, is_input=False, audio_id=current_output_id)
                        wav_buffer = bytearray()
                    
                    if accumulated_text:
                        await self.broadcast_translation(accumulated_text)
                        accumulated_text = ""
                    
                    # Reset IDs for next translation
                    current_output_id = None

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