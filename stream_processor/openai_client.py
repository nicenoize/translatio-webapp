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
import errno


class OpenAIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.ws = None
        self.logger = logging.getLogger(__name__)
        self.websocket_clients: Dict[int, WebSocket] = {}
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
        os.makedirs('output/audio', exist_ok=True)
        self.transcript_file = open('output/transcripts/transcript.txt', 'w', encoding='utf-8')
        self.audio_counter = 0

        # Path to the named pipe for translated audio
        self.translated_audio_pipe_path = 'translated_audio_pipe'
        self.create_named_pipe()

        # Add a lock for reconnecting to prevent race conditions
        self.reconnect_lock = asyncio.Lock()
        self.is_reconnecting = False

    def create_named_pipe(self):
        """Create a named pipe for translated audio output"""
        if not os.path.exists(self.translated_audio_pipe_path):
            os.mkfifo(self.translated_audio_pipe_path)
            self.logger.info(f"Created named pipe at {self.translated_audio_pipe_path}")

    async def connect(self):
        """Connect to OpenAI's Realtime API and initialize session"""
        url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }

        # Add reconnection attempts
        max_retries = 5
        retry_delay = 5  # seconds
        for attempt in range(max_retries):
            try:
                self.ws = await websockets.connect(url, extra_headers=headers)
                self.logger.info("Connected to OpenAI Realtime API")
                break
            except Exception as e:
                self.logger.error(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    self.logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    self.logger.error("Max connection attempts reached. Exiting.")
                    raise

        await self.init_conversation()

    async def init_conversation(self):
        """Initialize conversation context"""
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
        await self.safe_send(json.dumps(session_update))
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

                    # Prepare the event to send to OpenAI
                    append_event = {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(self.outgoing_audio_buffer).decode('utf-8')
                    }
                    await self.safe_send(json.dumps(append_event))

                    # Send commit event
                    commit_event = {"type": "input_audio_buffer.commit"}
                    await self.safe_send(json.dumps(commit_event))

                    self.logger.info(f"Sent audio buffer of size: {len(self.outgoing_audio_buffer)} bytes, chunk_id: {chunk_id}")

                    # Reset buffers
                    self.outgoing_audio_buffer = bytearray()
                    self.timestamp_buffer = []

        except Exception as e:
            self.logger.error(f"Error sending audio chunk: {e}", exc_info=True)



    async def handle_responses(self):
        """Handle translation responses from OpenAI"""
        try:
            # Open the named pipe for writing translated audio in non-blocking mode
            try:
                pipe_fd = os.open(self.translated_audio_pipe_path, os.O_WRONLY | os.O_NONBLOCK)
                pipe = os.fdopen(pipe_fd, 'wb')
                self.logger.info("Opened named pipe for writing.")
            except OSError as e:
                if e.errno == errno.ENXIO:
                    self.logger.error("No reader is connected to the named pipe.")
                    pipe = None
                else:
                    self.logger.error(f"Error opening named pipe: {e}")
                    pipe = None

            while True:
                try:
                    response = await self.ws.recv()
                    print('Response: ', response)
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

                            # Write the audio data to the named pipe if it's open
                            if pipe:
                                try:
                                    pipe.write(decoded_audio)
                                    pipe.flush()
                                except BrokenPipeError:
                                    self.logger.error("BrokenPipeError: No reader connected to the named pipe.")
                                    pipe.close()
                                    pipe = None
                            else:
                                self.logger.warning("No pipe to write audio data.")

                        except Exception as e:
                            self.logger.error(f"Error handling audio data: {e}")

                elif event_type == "response.audio_transcript.delta":
                    text = event.get("delta", "")
                    if text.strip():
                        # Handle text as needed (e.g., display or log)
                        self.logger.info(f"Translated text: {text}")
                        # Optionally broadcast to connected WebSocket clients
                        await self.broadcast_translation(text)

                elif event_type == "response.done":
                    # Reset the response in progress flag
                    self.response_in_progress = False

                elif event_type == "error":
                    error_info = event.get("error", {})
                    self.logger.error(f"OpenAI Error: {error_info}")
                    # Reset the response in progress flag on error
                    self.response_in_progress = False

        except Exception as e:
            self.logger.error(f"Error handling OpenAI responses: {e}", exc_info=True)
        finally:
            if pipe:
                pipe.close()
            await self.disconnect()

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

    async def disconnect(self):
        """Disconnect from OpenAI and clean up resources"""
        if self.ws:
            try:
                await self.ws.close()
                self.logger.info("Disconnected from OpenAI Realtime API")
            except Exception as e:
                self.logger.error(f"Error disconnecting from OpenAI: {e}")

            self.ws = None  # Reset the WebSocket connection

        # Remove the named pipe
        if os.path.exists(self.translated_audio_pipe_path):
            os.remove(self.translated_audio_pipe_path)
            self.logger.info(f"Removed named pipe at {self.translated_audio_pipe_path}")
