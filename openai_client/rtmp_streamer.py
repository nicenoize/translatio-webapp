# openai_client/rtmp_streamer.py

import asyncio
import logging
import os
import subprocess

from config import RTMP_PLAYOUT_URL

class RTMPStreamer:
    def __init__(
        self,
        client, 
        logger: logging.Logger,
        segments_dir: str = 'segments',
        buffer_duration: int = 5,
        max_retries: int = 3,
        retry_delay: int = 2,
        min_buffer_duration: int = 2,
        max_buffer_duration: int = 20,
        adjustment_interval: int = 10,
    ):
        """
        Stream video segments to an RTMP server.

        Args:
            client: Reference to the OpenAIClient instance for updating metrics.
            logger (logging.Logger): Logger instance.
            segments_dir (str): Directory where video segments are stored.
            buffer_duration (int): Initial buffer duration in seconds.
            max_retries (int): Maximum retries for failed FFmpeg commands.
            retry_delay (int): Delay between retries for failed FFmpeg commands.
            min_buffer_duration (int): Minimum allowed buffer duration.
            max_buffer_duration (int): Maximum allowed buffer duration.
            adjustment_interval (int): Time interval in seconds for buffer adjustment checks.
        """
        self.client = client
        self.logger = logger
        self.segments_dir = segments_dir
        self.rtmp_url = RTMP_PLAYOUT_URL
        self.buffer_duration = buffer_duration
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.min_buffer_duration = min_buffer_duration
        self.max_buffer_duration = max_buffer_duration
        self.adjustment_interval = adjustment_interval
        self.running = True
        self.processed_segments = set()
        self.unavailable_count = 0
        self.adjust_count = 0

        # Initialize metrics
        self.client.metrics['rtmp_total_segments_streamed'] = 0
        self.client.metrics['rtmp_ffmpeg_errors'] = 0
        self.client.metrics['rtmp_current_buffer_duration'] = self.buffer_duration

    async def stream_segments(self):
        """Stream video segments to an RTMP server."""
        try:
            self.logger.info("Starting RTMP streaming process.")
            while self.running:
                all_segments = sorted(
                    [
                        os.path.join(self.segments_dir, f)
                        for f in os.listdir(self.segments_dir)
                        if f.startswith("output_final_segment") and f not in self.processed_segments
                    ]
                )

                if not all_segments:
                    self.logger.warning("No new segments found. Increasing buffer duration.")
                    self.unavailable_count += 1
                    await self.adjust_buffer_duration()
                    await asyncio.sleep(self.buffer_duration)
                    continue

                self.logger.info(f"Found {len(all_segments)} new segment(s) to process.")
                for segment in all_segments:
                    if not self.running:
                        self.logger.info("RTMP streaming process was stopped.")
                        return

                    self.logger.info(f"Streaming segment: {segment}")
                    success = await self.stream_segment_with_retries(segment)
                    
                    if success:
                        # Increment total segments streamed
                        self.client.metrics['rtmp_total_segments_streamed'] += 1
                    else:
                        # Increment FFmpeg errors
                        self.client.metrics['rtmp_ffmpeg_errors'] += 1

                    # Mark the segment as processed
                    self.processed_segments.add(os.path.basename(segment))

                self.unavailable_count = 0  # Reset on successful processing
                await asyncio.sleep(self.buffer_duration)

        except Exception as e:
            self.logger.error(f"Unexpected error during RTMP streaming: {e}")

    async def stream_segment_with_retries(self, segment: str) -> bool:
        """Stream a single segment with retry logic.

        Args:
            segment (str): Path to the video segment file.

        Returns:
            bool: True if streaming was successful, False otherwise.
        """
        retries = 0
        while retries <= self.max_retries:
            process = await self.run_ffmpeg(segment)
            stdout, stderr = await process.communicate()

            # Log FFmpeg output
            if stdout:
                self.logger.debug(f"FFmpeg stdout: {stdout.decode('utf-8', errors='replace')}")
            if stderr:
                self.logger.debug(f"FFmpeg stderr: {stderr.decode('utf-8', errors='replace')}")

            if process.returncode == 0:
                self.logger.info(f"Successfully streamed segment: {segment}")
                return True
            else:
                self.logger.error(f"FFmpeg failed with exit code {process.returncode} for segment {segment}.")
                retries += 1
                if retries <= self.max_retries:
                    self.logger.warning(f"Retrying... ({retries}/{self.max_retries})")
                    await asyncio.sleep(self.retry_delay)

        self.logger.error(f"Failed to stream segment {segment} after {self.max_retries} retries.")
        return False

    async def run_ffmpeg(self, segment: str) -> asyncio.subprocess.Process:
        """Run FFmpeg to stream a segment.

        Args:
            segment (str): Path to the video segment file.

        Returns:
            asyncio.subprocess.Process: The FFmpeg subprocess.
        """
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

    async def adjust_buffer_duration(self):
        """Dynamically adjust the buffer duration based on streaming conditions."""
        self.adjust_count += 1
        self.client.metrics['rtmp_current_buffer_duration'] = self.buffer_duration

        if self.unavailable_count > 3:
            # Increase buffer duration if segments are frequently unavailable
            new_buffer = min(self.buffer_duration + 2, self.max_buffer_duration)
            if new_buffer != self.buffer_duration:
                self.buffer_duration = new_buffer
                self.logger.info(f"Increasing buffer duration to {self.buffer_duration}s.")
        elif self.adjust_count >= self.adjustment_interval:
            # Decrease buffer duration if segments are consistently available
            new_buffer = max(self.buffer_duration - 1, self.min_buffer_duration)
            if new_buffer != self.buffer_duration:
                self.buffer_duration = new_buffer
                self.logger.info(f"Decreasing buffer duration to {self.buffer_duration}s.")
            self.adjust_count = 0

        # Update the buffer duration in metrics
        self.client.metrics['rtmp_current_buffer_duration'] = self.buffer_duration

    def start(self):
        """Start the RTMP streaming process."""
        self.logger.info("Starting RTMP streaming task.")
        asyncio.create_task(self.stream_segments())
        self.logger.info("RTMP streaming task has been initiated.")

    def stop(self):
        """Stop the streaming process."""
        self.running = False
        self.logger.info("Stopped RTMP streaming process.")
