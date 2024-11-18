# stream_processor/stream_audio_processor.py

import asyncio
import logging
import signal
import os
import time
from contextlib import suppress
import base64

from .openai_client import OpenAIClient

class StreamAudioProcessor:
    def __init__(self, openai_api_key: str):
        self.logger = logging.getLogger(__name__)
        self.setup_logging()

        self.input_audio_pipe = 'input_audio_pipe'
        
        # Create named pipe for audio input
        self.create_named_pipe(self.input_audio_pipe)

        self.openai_client = OpenAIClient(openai_api_key)
        self.tasks = []
        self.running = True
        self.cleanup_lock = asyncio.Lock()

    def create_named_pipe(self, pipe_name):
        """Create named pipe if it doesn't exist"""
        try:
            if not os.path.exists(pipe_name):
                os.mkfifo(pipe_name)
                self.logger.info(f"Created named pipe: {pipe_name}")
        except FileExistsError:
            self.logger.info(f"Named pipe already exists: {pipe_name}")
        except Exception as e:
            self.logger.error(f"Error creating named pipe {pipe_name}: {e}")
            raise

    def setup_logging(self):
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("stream_processor.log")
            ]
        )

    def signal_handler(self):
        """Handle shutdown signals"""
        if self.running:
            self.logger.info("Shutdown signal received")
            self.running = False
            # Schedule cleanup in the event loop
            asyncio.create_task(self.cleanup())

    async def run(self):
        try:
            # Setup signal handlers
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, self.signal_handler)

            self.logger.info("Starting stream processor...")
            await self.openai_client.connect()

            # Start tasks
            self.tasks = [
                asyncio.create_task(self.read_audio_pipe()),
                asyncio.create_task(self.openai_client.handle_responses())
            ]

            # Wait for tasks to complete or running to become False
            while self.running:
                await asyncio.sleep(0.1)

        except Exception as e:
            self.logger.error(f"Error in StreamAudioProcessor: {e}")
        finally:
            await self.cleanup()

    async def read_audio_pipe(self):
        """Read audio data from input_audio_pipe and send to OpenAIClient"""
        try:
            with open(self.input_audio_pipe, 'rb') as pipe:
                self.logger.info("Reading audio data from input_audio_pipe...")
                while self.running:
                    data = pipe.read(4800)  # Adjust chunk size as necessary
                    if not data:
                        await asyncio.sleep(0.5)
                        continue

                    # Base64-encode the audio data
                    base64_audio = base64.b64encode(data).decode('utf-8')

                    # Send the chunk of data to the OpenAI client for processing
                    await self.openai_client.send_audio_chunk(base64_audio)

        except Exception as e:
            self.logger.error(f"Error reading audio pipe: {e}")

    async def cleanup(self):
        async with self.cleanup_lock:
            if not self.running:
                return

            self.running = False
            self.logger.info("Cleaning up resources...")

            # Cancel all tasks
            for task in self.tasks:
                if not task.done():
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task

            # Cleanup OpenAI client
            await self.openai_client.disconnect()

            # Remove named pipe
            if os.path.exists(self.input_audio_pipe):
                os.unlink(self.input_audio_pipe)
                self.logger.info(f"Removed named pipe: {self.input_audio_pipe}")

            self.logger.info("Cleanup complete")
