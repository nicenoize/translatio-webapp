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
        
        # Create debug directories
        os.makedirs('output/debug/audio/raw', exist_ok=True)
        
        # Open pipe in non-blocking mode
        try:
            self.audio_pipe_fd = os.open(input_audio_pipe, os.O_RDWR | os.O_NONBLOCK)
            self.logger.info(f"Opened input audio pipe: {input_audio_pipe}")
        except Exception as e:
            self.logger.error(f"Failed to open input audio pipe: {e}")
            raise
            
        self.chunk_counter = 0
        self.debug_file = open('output/debug/audio/audio_chunks.log', 'w')
        self.debug_file.write("timestamp,chunk_id,size,total_size\n")
        self.total_bytes = 0

        # Calculate minimum buffer size for 100ms of audio
        # At 24kHz, 16-bit (2 bytes) mono audio:
        # 24000 samples/sec * 2 bytes/sample * 0.1 sec = 4800 bytes
        self.min_buffer_size = 240000  # 5s of audio at 24kHz
        self.buffer = bytearray()

    async def process_audio(self, openai_client):
        """Process audio from FFmpeg pipe"""
        try:
            read_chunk_size = 1024  # Size to read from pipe at once
            
            while True:
                try:
                    # Read data from pipe
                    data = os.read(self.audio_pipe_fd, read_chunk_size)
                    if data:
                        self.buffer.extend(data)
                        self.logger.debug(f"Buffer size: {len(self.buffer)} bytes")

                        # Process complete chunks when we have enough data
                        while len(self.buffer) >= self.min_buffer_size:
                            # Take a chunk of exactly min_buffer_size
                            chunk = self.buffer[:self.min_buffer_size]
                            self.buffer = self.buffer[self.min_buffer_size:]

                            # Encode to base64 and send to OpenAI API
                            base64_audio = base64.b64encode(chunk).decode('utf-8')
                            await openai_client.send_audio_chunk(base64_audio)

                            self.logger.info(f"Sent audio chunk of size: {len(chunk)} bytes")
                            
                            # Increment counter and log
                            self.chunk_counter += 1
                            self.total_bytes += len(chunk)
                            self.debug_file.write(f"{self.chunk_counter},{len(chunk)},{self.total_bytes}\n")
                            self.debug_file.flush()

                    await asyncio.sleep(0.01)  # Prevent tight loop
                except BlockingIOError:
                    await asyncio.sleep(0.01)  # No data yet, retry
                except Exception as e:
                    self.logger.error(f"Error processing audio: {e}", exc_info=True)
                    break
                    
                await asyncio.sleep(0.01)
                
        except Exception as e:
            self.logger.error(f"Error in process_audio: {e}", exc_info=True)
        finally:
            self.debug_file.close()
            if len(self.buffer) >= self.min_buffer_size:
                self.logger.info(f"Processing final buffer of {len(self.buffer)} bytes")
                base64_audio = base64.b64encode(bytes(self.buffer)).decode('utf-8')
                await openai_client.send_audio_chunk(base64_audio)
                await openai_client.send_audio_commit()

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
        if hasattr(self, 'audio_pipe_fd') and self.audio_pipe_fd is not None:
            try:
                os.close(self.audio_pipe_fd)
            except OSError as e:
                self.logger.error(f"Error closing pipe: {e}")
                
        if hasattr(self, 'debug_file'):
            try:
                self.debug_file.close()
            except Exception as e:
                self.logger.error(f"Error closing debug file: {e}")