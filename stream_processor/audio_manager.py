# stream_processor/audio_manager.py

import os
import base64
import errno
import logging
import asyncio
from contextlib import suppress
from stream_processor.openai_client import OpenAIClient

class AudioManager:
    def __init__(self, input_audio_pipe: str):
        self.logger = logging.getLogger(__name__)
        self.input_audio_pipe = input_audio_pipe
        self.ffmpeg_process = None
        self.buffer = b''
        
        # Ensure the pipe exists before opening
        if not os.path.exists(self.input_audio_pipe):
            raise FileNotFoundError(f"Named pipe not found: {self.input_audio_pipe}")
            
        try:
            # Open in read-write mode to prevent blocking
            self.audio_pipe_fd = os.open(self.input_audio_pipe, os.O_RDWR | os.O_NONBLOCK)
            self.logger.info(f"Successfully opened input audio pipe: {self.input_audio_pipe}")
        except Exception as e:
            self.logger.error(f"Failed to open input audio pipe: {e}")
            raise
            
        self.ffmpeg_process = None
        self.buffer = b''  # Add buffer for accumulating audio data

    def setup_audio_pipe(self):
        with suppress(FileNotFoundError):
            os.unlink(self.audio_pipe_path)
            
        try:
            os.mkfifo(self.audio_pipe_path)
            self.logger.info(f"Created named pipe at {self.audio_pipe_path}")
            
            # Open in read-write mode to prevent blocking
            self.audio_pipe_fd = os.open(self.audio_pipe_path, os.O_RDWR | os.O_NONBLOCK)
            self.logger.info("Successfully opened named pipe")
        except OSError as e:
            self.logger.error(f"Failed to setup audio pipe: {e}")
            raise

    async def process_audio(self, openai_client):
        """Process audio from the input pipe"""
        if not self.ffmpeg_process:
            self.logger.error("FFmpeg process not set in AudioManager")
            return

        try:
            chunk_size = 3200  # 100ms of audio at 16kHz, 16-bit mono
            self.logger.info("Starting audio processing loop")
            
            while True:
                try:
                    # Read from the named pipe
                    data = os.read(self.audio_pipe_fd, chunk_size)
                    if data:
                        self.buffer += data
                        self.logger.debug(f"Read {len(data)} bytes from input pipe, buffer size: {len(self.buffer)}")
                        
                        # Send when we have at least 100ms of audio
                        if len(self.buffer) >= chunk_size:
                            self.logger.info(f"Sending audio chunk of size {len(self.buffer)} to OpenAI")
                            base64_audio = base64.b64encode(self.buffer).decode('utf-8')
                            await OpenAIClient.send_audio_chunk(base64_audio)
                            await OpenAIClient.commit_audio()
                            self.buffer = b''
                    
                except BlockingIOError:
                    # No data available to read
                    await asyncio.sleep(0.01)
                except Exception as e:
                    self.logger.error(f"Error reading audio data: {e}")
                    break

                await asyncio.sleep(0.01)  # Prevent busy-waiting

        except Exception as e:
            self.logger.error(f"Error in process_audio: {e}")
        finally:
            # Send any remaining audio
            if self.buffer:
                self.logger.info(f"Sending final audio chunk of size {len(self.buffer)}")
                base64_audio = base64.b64encode(self.buffer).decode('utf-8')
                await OpenAIClient.send_audio_chunk(base64_audio)
                await OpenAIClient.commit_audio()


    def set_ffmpeg_process(self, process):
        """Set the FFmpeg process reference"""
        self.ffmpeg_process = process
        self.logger.info("FFmpeg process reference set in AudioManager")

    async def write_audio_chunk(self, base64_audio: str):
        """Write translated audio chunk to the pipe"""
        if not self.audio_pipe_fd:
            self.logger.error("Audio pipe not initialized")
            return

        try:
            audio_data = base64.b64decode(base64_audio)
            os.write(self.audio_pipe_fd, audio_data)
        except BlockingIOError:
            self.logger.debug("Pipe full, skipping write")
        except OSError as e:
            self.logger.error(f"Error writing to pipe: {e}")

    def cleanup(self):
        """Clean up resources"""
        if self.audio_pipe_fd is not None:
            try:
                os.close(self.audio_pipe_fd)
                self.audio_pipe_fd = None
            except OSError as e:
                self.logger.error(f"Error closing pipe: {e}")

        with suppress(FileNotFoundError):
            os.unlink(self.audio_pipe_path)
            self.logger.info("Cleaned up named pipe")