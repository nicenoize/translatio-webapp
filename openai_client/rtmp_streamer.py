# openai_client/rtmp_streamer.py

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
    ):
        """
        Stream video segments to an RTMP server.

        Args:
            logger (logging.Logger): Logger instance.
            segments_dir (str): Directory where video segments are stored.
            rtmp_url (str): RTMP server URL to stream to.
            buffer_duration (int): Time to wait before processing new segments.
        """
        self.logger = logger
        self.segments_dir = segments_dir
        self.rtmp_url = RTMP_PLAYOUT_URL
        self.buffer_duration = buffer_duration
        self.running = True
        self.processed_segments = set()

    async def stream_segments(self):
        """Stream video segments to an RTMP server."""
        try:
            self.logger.debug("Starting RTMP streaming process.")
            
            while self.running:
                # Get list of unprocessed segments
                all_segments = sorted(
                    [os.path.join(self.segments_dir, f) for f in os.listdir(self.segments_dir)
                     if f.startswith("output_final_segment") and f not in self.processed_segments]
                )
                if not all_segments:
                    self.logger.debug("No new segments found. Waiting for new segments...")
                    await asyncio.sleep(self.buffer_duration)
                    continue

                self.logger.debug(f"Found {len(all_segments)} new segments to process.")
                
                for segment in all_segments:
                    if not self.running:
                        self.logger.debug("RTMP streaming process was stopped.")
                        return
                    
                    self.logger.debug(f"Streaming segment: {segment}")

                    # Build FFmpeg command to stream the segment
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
                    
                    process = await asyncio.create_subprocess_exec(
                        *ffmpeg_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )

                    stdout, stderr = await process.communicate()

                    # Log FFmpeg output for debugging
                    if stdout:
                        self.logger.debug(f"FFmpeg stdout: {stdout.decode('utf-8', errors='replace')}")
                    if stderr:
                        self.logger.debug(f"FFmpeg stderr: {stderr.decode('utf-8', errors='replace')}")

                    if process.returncode != 0:
                        self.logger.error(f"FFmpeg failed with exit code {process.returncode} for segment {segment}.")
                        self.logger.error(f"FFmpeg stderr: {stderr.decode('utf-8', errors='replace')}")
                    else:
                        self.logger.debug(f"Successfully streamed segment: {segment}")
                    
                    # Mark the segment as processed
                    self.processed_segments.add(os.path.basename(segment))

                # Wait before checking for new segments
                self.logger.debug(f"Buffering for {self.buffer_duration} seconds before checking for new segments...")
                await asyncio.sleep(self.buffer_duration)

        except FileNotFoundError as fnf_error:
            self.logger.error(f"FFmpeg not found: {fnf_error}. Ensure FFmpeg is installed and in the system PATH.")
        except Exception as e:
            self.logger.error(f"Error during RTMP streaming: {e}")

    def start(self):
        """Start the RTMP streaming process."""
        self.logger.debug("Starting RTMP streaming task.")
        asyncio.create_task(self.stream_segments())
        self.logger.debug("RTMP streaming task has been initiated.")

    def stop(self):
        """Stop the streaming process."""
        self.running = False
        self.logger.debug("Stopped RTMP streaming process.")
