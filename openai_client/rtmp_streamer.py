import asyncio
import logging
import os
import subprocess
from typing import Optional

from config import RTMP_PLAYOUT_URL

class RTMPStreamer:
    def __init__(
        self,
        logger: logging.Logger,
        segments_dir: str = 'segments',
        buffer_duration: int = 5,
        max_retries: int = 3,
        retry_delay: int = 2,
    ):
        """
        Stream video segments to an RTMP server.

        Args:
            logger (logging.Logger): Logger instance.
            segments_dir (str): Directory where video segments are stored.
            rtmp_url (str): RTMP server URL to stream to.
            buffer_duration (int): Time to wait before processing new segments.
            max_retries (int): Maximum retries for failed FFmpeg commands.
            retry_delay (int): Delay between retries for failed FFmpeg commands.
        """
        self.logger = logger
        self.segments_dir = segments_dir
        self.rtmp_url = RTMP_PLAYOUT_URL
        self.buffer_duration = buffer_duration
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.running = True
        self.processed_segments = set()

    async def stream_segments(self):
        """Stream video segments to an RTMP server."""
        try:
            self.logger.info("Starting RTMP streaming process.")

            while self.running:
                # Get list of unprocessed segments
                all_segments = sorted(
                    [os.path.join(self.segments_dir, f) for f in os.listdir(self.segments_dir)
                     if f.startswith("output_final_segment") and f not in self.processed_segments]
                )

                if not all_segments:
                    self.logger.info("No new segments found. Waiting for next segment...")
                    await asyncio.sleep(self.buffer_duration)
                    continue

                self.logger.info(f"Found {len(all_segments)} new segment(s) to process.")

                for segment in all_segments:
                    if not self.running:
                        self.logger.info("RTMP streaming process was stopped.")
                        return

                    self.logger.info(f"Streaming segment: {segment}")
                    await self.stream_segment_with_retries(segment)

                    # Mark the segment as processed
                    self.processed_segments.add(os.path.basename(segment))

                # Wait briefly before checking for new segments
                self.logger.debug(f"Buffering for {self.buffer_duration} seconds before checking for new segments...")
                await asyncio.sleep(self.buffer_duration)

        except Exception as e:
            self.logger.error(f"Unexpected error during RTMP streaming: {e}")

    async def stream_segment_with_retries(self, segment: str):
        """Stream a single segment with retry logic."""
        retries = 0
        while retries <= self.max_retries:
            process = await self._run_ffmpeg(segment)
            stdout, stderr = await process.communicate()

            # Log FFmpeg output
            if stdout:
                self.logger.debug(f"FFmpeg stdout: {stdout.decode('utf-8', errors='replace')}")
            if stderr:
                self.logger.debug(f"FFmpeg stderr: {stderr.decode('utf-8', errors='replace')}")

            if process.returncode == 0:
                self.logger.info(f"Successfully streamed segment: {segment}")
                return
            else:
                self.logger.error(f"FFmpeg failed with exit code {process.returncode} for segment {segment}.")
                self.logger.error(f"FFmpeg stderr: {stderr.decode('utf-8', errors='replace')}")
                retries += 1
                if retries <= self.max_retries:
                    self.logger.warning(f"Retrying... ({retries}/{self.max_retries})")
                    await asyncio.sleep(self.retry_delay)

        self.logger.error(f"Failed to stream segment {segment} after {self.max_retries} retries.")

    async def _run_ffmpeg(self, segment: str) -> asyncio.subprocess.Process:
        """Run FFmpeg to stream a segment."""
        ffmpeg_cmd = [
            'ffmpeg',
            '-re',  # Read input in real-time
            '-i', segment,  # Input segment file
            '-c:v', 'libx264',
            '-preset', 'veryfast',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-b:a', '160k',
            '-f', 'flv',  # RTMP streaming format
            self.rtmp_url
        ]

        self.logger.debug("FFmpeg command: " + " ".join(ffmpeg_cmd))
        return await asyncio.create_subprocess_exec(
            *ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

    def start(self):
        """Start the RTMP streaming process."""
        self.logger.info("Starting RTMP streaming task.")
        asyncio.create_task(self.stream_segments())
        self.logger.info("RTMP streaming task has been initiated.")

    def stop(self):
        """Stop the streaming process."""
        self.running = False
        self.logger.info("Stopped RTMP streaming process.")
