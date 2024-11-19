# stream_audio_processor.py

import asyncio
import logging
import os
from openai_client import OpenAIClient  # Ensure correct import

class StreamAudioProcessor:
    def __init__(self, openai_api_key: str, input_audio_pipe: str, translated_audio_pipe: str, output_rtmp_url: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        self.openai_client = OpenAIClient(api_key=openai_api_key, translated_audio_pipe=translated_audio_pipe)
        self.input_audio_pipe = input_audio_pipe
        self.translated_audio_pipe = translated_audio_pipe
        self.output_rtmp_url = output_rtmp_url

    async def run(self):
        """Run the StreamAudioProcessor."""
        try:
            self.logger.info("Starting StreamAudioProcessor...")
            # Start OpenAIClient's run method
            await self.openai_client.run(self.input_audio_pipe)
        except Exception as e:
            self.logger.error(f"Error in StreamAudioProcessor: {e}", exc_info=True)
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Cleanup resources."""
        self.logger.info("Cleaning up StreamAudioProcessor...")
        await self.openai_client.cleanup()
        # Optionally, remove named pipes if needed
        for pipe in [self.input_audio_pipe, self.translated_audio_pipe]:
            try:
                if os.path.exists(pipe):
                    os.remove(pipe)
                    self.logger.info(f"Removed named pipe: {pipe}")
            except Exception as e:
                self.logger.error(f"Error removing named pipe {pipe}: {e}")
