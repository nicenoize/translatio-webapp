import asyncio
import websockets
import json
import logging
import os
import base64
from contextlib import suppress
import pprint

class OpenAIClient:
    def __init__(self, api_key: str, translated_audio_pipe: str):
        self.api_key = api_key
        self.ws = None
        self.logger = logging.getLogger(__name__)
        self.translated_audio_pipe = translated_audio_pipe
        
        if not os.path.exists(self.translated_audio_pipe):
            raise FileNotFoundError(f"Named pipe not found: {self.translated_audio_pipe}")
            
        try:
            self.translated_audio_fd = os.open(self.translated_audio_pipe, os.O_RDWR | os.O_NONBLOCK)
            self.logger.info(f"Successfully opened translated audio pipe: {self.translated_audio_pipe}")
        except Exception as e:
            self.logger.error(f"Failed to open translated audio pipe: {e}")
            raise

        os.makedirs('output/transcripts', exist_ok=True)
        self.transcript_file = open('output/transcripts/transcript.txt', 'w')
        
        self.audio_buffer = []
        self.min_buffer_size = 48000  # 2 seconds of audio at 24kHz

    async def connect(self):
        """Connect to OpenAI's Realtime API"""
        url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        self.ws = await websockets.connect(url, extra_headers=headers)
        self.logger.info("Connected to OpenAI Realtime API")

        # Initial session configuration
        session_update = {
            "type": "session.update",
            "session": {
                "instructions": """You are a real-time translator. Translate all incoming English speech into natural German. 
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
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 200
                },
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "You are a real-time translator. Translate all incoming English speech into natural German. Do not engage in conversation or add any commentary. Only translate the content.",
                    }
                ],
                "temperature": 0.7,
                "modalities": ["text", "audio"],
                "max_response_output_tokens": "inf"
            }
        }
        self.logger.debug(f"Sending session update: {pprint.pformat(session_update)}")
        await self.ws.send(json.dumps(session_update))

        # Create first conversation item to set context
        conversation_init = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "system",
                "content": [{
                    "type": "input_text",
                    "text": "Translate incoming English speech to German. Only provide translations, no other responses."
                }]
            }
        }
        await self.ws.send(json.dumps(conversation_init))

        # Setup response configuration
        response_create = {
            "type": "response.create",
            "response": {
                "modalities": ["text", "audio"],
                "output_audio_format": "pcm16",
                "temperature": 0.7,
                "instructions": "Translate the incoming speech from English to German. Provide only the translation."
            }
        }
        self.logger.debug(f"Sending response create: {pprint.pformat(response_create)}")
        await self.ws.send(json.dumps(response_create))

    async def send_audio_chunk(self, base64_audio: str):
        """Send audio for translation"""
        if self.ws:
            # Create a proper audio message
            audio_message = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{
                        "type": "input_audio",
                        "audio": base64_audio
                    }]
                }
            }
            await self.ws.send(json.dumps(audio_message))
            self.logger.debug("Sent audio chunk for translation")

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

    async def handle_responses(self, audio_manager, subtitle_manager):
        """Handle translation responses from OpenAI"""
        try:
            current_translation = ""
            current_audio_segment = bytearray()
            current_text_segment = []

            while True:
                response = await self.ws.recv()
                event = json.loads(response)
                event_type = event.get("type")
                
                self.logger.debug(f"Received event: {pprint.pformat(event)}")

                if event_type == "response.audio.delta":
                    # Accumulate audio data
                    audio_data = event.get("delta", "")
                    if audio_data:
                        try:
                            decoded_audio = base64.b64decode(audio_data)
                            current_audio_segment.extend(decoded_audio)
                        except Exception as e:
                            self.logger.error(f"Error decoding audio: {e}")

                elif event_type == "response.audio.done":
                    # Write accumulated audio segment
                    try:
                        os.write(self.translated_audio_fd, current_audio_segment)
                        self.logger.info(f"Wrote complete audio segment ({len(current_audio_segment)} bytes)")
                        current_audio_segment = bytearray()
                    except BlockingIOError:
                        self.logger.debug("Pipe full, skipping write")
                    except Exception as e:
                        self.logger.error(f"Error writing audio: {e}")

                elif event_type == "response.text.delta":
                    # Accumulate text
                    text = event.get("delta", "")
                    if text.strip():
                        current_text_segment.append(text)
                        self.logger.debug(f"Received text delta: {text}")

                elif event_type == "response.text.done":
                    # Process complete text segment
                    if current_text_segment:
                        complete_text = ''.join(current_text_segment)
                        current_translation += complete_text
                        self.logger.info(f"Complete translation: {complete_text}")
                        subtitle_manager.update_subtitle(complete_text)
                        self.transcript_file.write(complete_text + "\n")
                        self.transcript_file.flush()
                        current_text_segment = []

                elif event_type == "error":
                    error_info = event.get("error", {})
                    self.logger.error(f"OpenAI Error: {error_info}")

        except Exception as e:
            self.logger.error(f"Error handling OpenAI responses: {e}")
        finally:
            self.transcript_file.close()
            if self.translated_audio_fd is not None:
                try:
                    os.close(self.translated_audio_fd)
                except OSError as e:
                    self.logger.error(f"Error closing translated audio pipe: {e}")

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