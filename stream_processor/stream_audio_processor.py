# stream_processor/stream_audio_processor.py

import asyncio
import logging
import signal
from .openai_client import OpenAIClient
from .audio_manager import AudioManager
from .subtitle_manager import SubtitleManager
from .ffmpeg_manager import FFmpegManager
import os
import time
from contextlib import suppress

class StreamAudioProcessor:
    def __init__(self, openai_api_key: str, stream_url: str, output_rtmp_url: str):
        self.logger = logging.getLogger(__name__)
        self.setup_logging()

        self.input_audio_pipe = 'input_audio_pipe'
        self.translated_audio_pipe = 'translated_audio_pipe'
        
        # Create named pipes first
        self.create_named_pipes()

        self.subtitle_manager = SubtitleManager()
        self.audio_manager = AudioManager(self.input_audio_pipe)
        self.openai_client = OpenAIClient(openai_api_key, self.translated_audio_pipe)
        self.ffmpeg_manager = FFmpegManager(
            stream_url,
            output_rtmp_url,
            self.input_audio_pipe,
            self.translated_audio_pipe,
            self.subtitle_manager.zmq_address
        )
        self.tasks = []
        self.running = True
        self.cleanup_lock = asyncio.Lock()

    def start_processing(self):
        # Open the input named pipe in binary mode
        with open(self.input_pipe_path, 'rb') as pipe:
            print("Starting to stream audio data from input_audio_pipe to OpenAI real-time API...")
            
            while True:
                # Read a chunk of data from the pipe
                data = pipe.read(480000)  # Adjust chunk size as necessary
                
                if not data:
                    time.sleep(0.01)  # Sleep briefly if no data is available
                    continue
                
                # Send the chunk of data to the OpenAI client for processing
                self.openai_client.send_audio_chunk(data)

                # Optional: add a small sleep interval to control data flow
                time.sleep(0.05)

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
                loop.add_signal_handler(sig, self.signal_handler)

            self.logger.info("Starting stream processor...")
            await self.openai_client.connect()

            # Start FFmpeg process
            self.ffmpeg_manager.start_ffmpeg_process()
            
            # Important: Set the FFmpeg process reference in AudioManager
            self.audio_manager.set_ffmpeg_process(self.ffmpeg_manager.ffmpeg_process)

            await asyncio.sleep(1)  # Give FFmpeg time to start

            if self.ffmpeg_manager.ffmpeg_process.poll() is not None:
                raise RuntimeError("FFmpeg failed to start")

            self.tasks = [
                asyncio.create_task(self.ffmpeg_manager.monitor_process()),
                asyncio.create_task(self.audio_manager.process_audio(self.openai_client)),
                asyncio.create_task(self.openai_client.handle_responses())
            ]

            # Wait for tasks to complete or running to become False
            while self.running:
                await asyncio.sleep(0.1)

        except Exception as e:
            self.logger.error(f"Error in StreamAudioProcessor: {e}")
        finally:
            await self.cleanup()

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

            # Cleanup components
            self.ffmpeg_manager.stop_ffmpeg_process()
            self.audio_manager.cleanup()
            self.subtitle_manager.cleanup()
            await self.openai_client.disconnect()

            self.logger.info("Cleanup complete")