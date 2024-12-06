# openai_client/rtmp_streamer.py

import asyncio
import logging
import os
import subprocess
from typing import Optional

class RTMPStreamer:
    def __init__(
        self,
        client: 'OpenAIClient',
        logger: logging.Logger,
        audio_input: str = 'output/audio/output.wav',
        rtmp_url: str = 'rtmp://bintu-vtrans.nanocosmos.de/live/sNVi5-egEGF',
        buffer_duration: int = 5,  # Buffer duration in seconds
    ):
        """
        Initialize the RTMPStreamer.

        Args:
            client (OpenAIClient): Reference to the OpenAIClient instance.
            logger (logging.Logger): Logger instance.
            audio_input (str): Path to the seamless audio file.
            rtmp_url (str): RTMP server URL to stream to.
            buffer_duration (int): Delay in seconds to buffer the stream before starting.
        """
        self.client = client
        self.logger = logger
        self.audio_input = audio_input
        self.rtmp_url = rtmp_url
        self.buffer_duration = buffer_duration
        self.ffmpeg_process: Optional[asyncio.subprocess.Process] = None
        self.running = True
        self.streaming_task = None

    def start(self):
        """Start the RTMPStreamer."""
        if self.streaming_task is None or self.streaming_task.done():
            self.streaming_task = asyncio.create_task(self.stream_audio_colored_bg())
            self.logger.info("RTMPStreamer started.")

    def stop(self):
        """Stop the RTMPStreamer."""
        self.running = False
        if self.ffmpeg_process and self.ffmpeg_process.returncode is None:
            self.ffmpeg_process.terminate()
            self.logger.info("FFmpeg streaming process terminated.")
        if self.streaming_task:
            self.streaming_task.cancel()
            self.logger.info("RTMPStreamer streaming task cancelled.")

    async def stream_audio_colored_bg(self):
        """Stream the audio with a colored video background (no subtitles yet)."""
        try:
            # Wait for the audio_input file to appear
            while not os.path.exists(self.audio_input):
                if not self.running:
                    self.logger.info("Stopped before audio input was available.")
                    return
                self.logger.info("Waiting for audio input to be available...")
                await asyncio.sleep(1)

            # Introduce buffering delay before starting
            self.logger.info(f"Buffering for {self.buffer_duration} seconds before starting the stream...")
            await asyncio.sleep(self.buffer_duration)

            # Change the background color here (e.g., blue)
            ffmpeg_cmd = [
                'ffmpeg',
                '-re',  # Read input at native frame rate
                '-i', self.audio_input,  # Input audio file
                '-f', 'lavfi',
                '-i', 'color=size=1280x720:rate=25:color=blue',  # Blue video background
                '-c:v', 'libx264',  # Video codec
                '-preset', 'veryfast',
                '-maxrate', '3000k',
                '-bufsize', '6000k',
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac',  # Audio codec
                '-b:a', '160k',
                '-f', 'flv',    # Output format
                self.rtmp_url   # RTMP server URL
            ]

            self.logger.info(f"Starting FFmpeg streaming to RTMP server: {self.rtmp_url}")
            self.logger.debug(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")

            self.ffmpeg_process = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            while self.running:
                retcode = await self.ffmpeg_process.poll()
                if retcode is not None:
                    # FFmpeg has exited
                    stdout, stderr = await self.ffmpeg_process.communicate()
                    self.logger.error(f"FFmpeg terminated with code {retcode}")
                    self.logger.error(f"FFmpeg stderr: {stderr.decode('utf-8', errors='replace')}")
                    break
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            self.logger.info("Streaming task cancelled.")
        except Exception as e:
            self.logger.error(f"Error during FFmpeg streaming: {e}")
        finally:
            # Clean up if needed
            if self.ffmpeg_process and self.ffmpeg_process.returncode is None:
                self.ffmpeg_process.terminate()
                await self.ffmpeg_process.wait()
                self.logger.info("FFmpeg streaming process terminated at cleanup.")
