# stream_processor/stream_audio_processor.py

import asyncio
import logging
from .openai_client import OpenAIClient
import signal
import os
from contextlib import suppress

class StreamAudioProcessor:
    def __init__(self, openai_api_key: str, stream_url: str, output_rtmp_url: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.setup_logging()

        self.input_audio_pipe = 'input_audio_pipe'
        self.translated_audio_pipe = 'translated_audio_pipe'
        
        # Create named pipes if they don't exist
        self.create_named_pipes()

        # Initialize OpenAIClient
        self.openai_client = OpenAIClient(api_key=openai_api_key, translated_audio_pipe=self.translated_audio_pipe)
        self.stream_url = stream_url
        self.output_rtmp_url = output_rtmp_url

        self.tasks = []
        self.running = True
        self.cleanup_lock = asyncio.Lock()

    def create_named_pipes(self):
        """Create named pipes if they don't exist"""
        for pipe in [self.input_audio_pipe, self.translated_audio_pipe]:
            try:
                if os.path.exists(pipe):
                    os.unlink(pipe)
                os.mkfifo(pipe)
                self.logger.info(f"Created named pipe: {pipe}")
            except FileExistsError:
                self.logger.info(f"Pipe already exists: {pipe}")
            except Exception as e:
                self.logger.error(f"Error creating pipe {pipe}: {e}")
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
                try:
                    loop.add_signal_handler(sig, self.signal_handler)
                except NotImplementedError:
                    # Signal handlers are not implemented on some systems (e.g., Windows)
                    pass

            self.logger.info("Starting StreamAudioProcessor...")

            # Ensure that the input_audio_pipe exists
            if not os.path.exists(self.input_audio_pipe):
                self.logger.error(f"Named pipe {self.input_audio_pipe} does not exist. Please create it before running.")
                return

            # Start the OpenAIClient's main loop
            await self.openai_client.run()

        except Exception as e:
            self.logger.error(f"Error in StreamAudioProcessor: {e}", exc_info=True)
        finally:
            await self.cleanup()

    async def cleanup(self):
        async with self.cleanup_lock:
            if not self.running:
                return

            self.running = False
            self.logger.info("Cleaning up StreamAudioProcessor...")

            # Cancel all tasks
            for task in self.tasks:
                if not task.done():
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task

            # Cleanup OpenAIClient
            await self.openai_client.disconnect(shutdown=True)

            self.logger.info("Cleanup complete")
