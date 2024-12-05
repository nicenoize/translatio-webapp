# openai_client/rtmp_streamer.py

import asyncio
import logging
import os
import subprocess
from typing import Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from collections import deque

class RTMPStreamer:
    def __init__(
        self,
        client: 'OpenAIClient',
        logger: logging.Logger,
        final_video_dir: str = 'output/final',
        rtmp_url: str = 'rtmp://bintu-vtrans.nanocosmos.de/live/sNVi5-egEGF',
        buffer_size: int = 5,  # Number of segments to buffer before starting
    ):
        """
        Initialize the RTMPStreamer.

        Args:
            client (OpenAIClient): Reference to the OpenAIClient instance.
            logger (logging.Logger): Logger instance.
            final_video_dir (str): Directory where final muxed video segments are stored.
            rtmp_url (str): RTMP server URL to stream to.
            buffer_size (int): Number of segments to buffer before starting the stream.
        """
        self.client = client
        self.logger = logger
        self.final_video_dir = final_video_dir
        self.rtmp_url = rtmp_url
        self.segment_queue = deque()
        self.buffer_size = buffer_size
        self.loop = asyncio.get_event_loop()
        self.streaming_task: Optional[asyncio.Task] = None
        self.running = True
        self.playlist_path = 'output/playlist.txt'

        # Ensure the final_video_dir exists
        os.makedirs(self.final_video_dir, exist_ok=True)

    def start(self):
        """Start the RTMPStreamer by launching the streaming task."""
        self.streaming_task = asyncio.create_task(self.stream_video())
        self.logger.info("RTMPStreamer started.")

    def stop(self):
        """Stop the RTMPStreamer."""
        self.running = False
        if self.streaming_task:
            self.streaming_task.cancel()
            self.logger.info("RTMPStreamer stopped.")

    async def stream_video(self):
        """Stream video segments to the RTMP server using FFmpeg."""
        try:
            # Wait until buffer is filled
            self.logger.info(f"Buffering {self.buffer_size} segments before starting the stream...")
            while len(self.segment_queue) < self.buffer_size and self.running:
                await asyncio.sleep(1)

            if not self.running:
                return

            # Start the FFmpeg process
            await self.start_ffmpeg_stream()

        except asyncio.CancelledError:
            self.logger.info("Streaming task cancelled.")
        except Exception as e:
            self.logger.error(f"Error during streaming: {e}")

    async def start_ffmpeg_stream(self):
        """Start the FFmpeg process to stream concatenated video segments."""
        try:
            # Generate the initial playlist file
            await self.update_playlist_file()

            # FFmpeg command to read the playlist and stream to RTMP
            ffmpeg_cmd = [
                'ffmpeg',
                '-re',  # Read input at native frame rate
                '-f', 'concat',
                '-safe', '0',
                '-i', self.playlist_path,
                '-c', 'copy',  # Copy codec (no re-encoding)
                '-f', 'flv',
                self.rtmp_url
            ]

            self.logger.info(f"Starting FFmpeg streaming to RTMP server: {self.rtmp_url}")
            self.logger.debug(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")

            # Start the FFmpeg process
            process = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Monitor for new segments and update the playlist
            while self.running:
                await asyncio.sleep(1)
                await self.update_playlist_file()
                # Optionally, check process status or handle errors

        except Exception as e:
            self.logger.error(f"Error starting FFmpeg stream: {e}")

    async def update_playlist_file(self):
        """Update the playlist file with available video segments."""
        try:
            # Scan the final_video_dir for new segments
            segment_files = sorted([
                f for f in os.listdir(self.final_video_dir)
                if f.startswith('output_final_segment_') and f.endswith('.mp4')
            ])

            # Add new segments to the queue
            for segment_file in segment_files:
                segment_path = os.path.join(self.final_video_dir, segment_file)
                if segment_path not in self.segment_queue:
                    self.segment_queue.append(segment_path)
                    self.logger.info(f"Added segment to queue: {segment_path}")

            # Generate the playlist file
            with open(self.playlist_path, 'w') as playlist_file:
                for segment_path in self.segment_queue:
                    playlist_file.write(f"file '{os.path.abspath(segment_path)}'\n")

            self.logger.debug(f"Updated playlist file with {len(self.segment_queue)} segments.")

        except Exception as e:
            self.logger.error(f"Error updating playlist file: {e}")