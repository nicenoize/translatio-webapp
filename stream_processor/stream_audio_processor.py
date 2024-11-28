# stream_processor/stream_audio_processor.py

import asyncio
import logging
from .openai_client import OpenAIClient
import signal
import os
from contextlib import suppress

class StreamAudioProcessor:
    def __init__(self, openai_api_key: str, stream_url: str, output_rtmp_url: str, enable_playback: bool = True, min_subtitle_duration: float = 1.0):
        """
        Initialize the StreamAudioProcessor.

        :param openai_api_key: OpenAI API key.
        :param stream_url: URL of the input stream.
        :param output_rtmp_url: URL to stream the output via RTMP.
        :param enable_playback: Flag to enable or disable local audio playback.
        :param min_subtitle_duration: Minimum duration (in seconds) for each subtitle display.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.setup_logging()

        self.translated_audio_pipe = 'translated_audio_pipe'
        self.stream_url = stream_url
        self.output_rtmp_url = output_rtmp_url

        # Initialize OpenAIClient with the new parameters
        self.openai_client = OpenAIClient(
            api_key=openai_api_key,
            translated_audio_pipe=self.translated_audio_pipe,
            enable_playback=enable_playback,
            min_subtitle_duration=min_subtitle_duration
        )

        self.running = True
        self.cleanup_lock = asyncio.Lock()

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

            # Cleanup OpenAIClient
            await self.openai_client.disconnect(shutdown=True)

            self.logger.info("Cleanup complete")
