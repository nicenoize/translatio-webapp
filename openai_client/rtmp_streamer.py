# openai_client/rtmp_streamer.py

import asyncio
import logging
import os
import subprocess
from typing import Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class RTMPStreamer:
    def __init__(
        self,
        client: 'OpenAIClient',
        logger: logging.Logger,
        final_video_dir: str = 'output/final',
        subtitles_path: str = 'output/subtitles/subtitles.vtt',
        rtmp_url: str = 'rtmp://sNVi5-egEGF.bintu-vtrans.nanocosmos.de/live',
    ):
        """
        Initialize the RTMPStreamer.

        Args:
            client (OpenAIClient): Reference to the OpenAIClient instance.
            logger (logging.Logger): Logger instance.
            final_video_dir (str): Directory where final muxed video segments are stored.
            subtitles_path (str): Path to the WebVTT subtitles file.
            rtmp_url (str): RTMP server URL to stream to.
        """
        self.client = client
        self.logger = logger
        self.final_video_dir = final_video_dir
        self.subtitles_path = subtitles_path
        self.rtmp_url = rtmp_url
        self.observer = Observer()
        self.ffmpeg_process: Optional[subprocess.Popen] = None
        self.loop = asyncio.get_event_loop()

    def start(self):
        """Start the RTMPStreamer by setting up the file system observer."""
        event_handler = NewVideoHandler(self)
        self.observer.schedule(event_handler, self.final_video_dir, recursive=False)
        self.observer.start()
        self.logger.info(f"RTMPStreamer started, watching directory: {self.final_video_dir}")

    def stop(self):
        """Stop the RTMPStreamer and terminate FFmpeg process."""
        self.observer.stop()
        self.observer.join()
        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()
            self.logger.info("FFmpeg streaming process terminated.")

    async def stream_video(self, video_path: str):
        """
        Overlay subtitles on the video and stream to the RTMP server using FFmpeg.

        Args:
            video_path (str): Path to the muxed video file.
        """
        if not os.path.exists(video_path):
            self.logger.error(f"Video file does not exist: {video_path}")
            return

        if not os.path.exists(self.subtitles_path):
            self.logger.error(f"Subtitles file does not exist: {self.subtitles_path}")
            return

        # FFmpeg command to overlay subtitles and stream to RTMP
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output files without asking
            '-i', video_path,  # Input video file
            '-vf', f"subtitles={self.subtitles_path}",  # Overlay subtitles
            '-c:v', 'libx264',  # Video codec
            '-preset', 'veryfast',
            '-maxrate', '3000k',
            '-bufsize', '6000k',
            '-c:a', 'aac',  # Audio codec
            '-b:a', '160k',
            '-f', 'flv',  # Output format
            self.rtmp_url  # RTMP server URL
        ]

        self.logger.info(f"Starting FFmpeg streaming for: {video_path}")
        try:
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Monitor FFmpeg process
            while True:
                retcode = self.ffmpeg_process.poll()
                if retcode is not None:
                    # FFmpeg process has terminated
                    stdout, stderr = self.ffmpeg_process.communicate()
                    self.logger.error(f"FFmpeg terminated with code {retcode}")
                    self.logger.error(f"FFmpeg stderr: {stderr.decode('utf-8')}")
                    break
                await asyncio.sleep(1)
        except Exception as e:
            self.logger.error(f"Error during FFmpeg streaming: {e}")

class NewVideoHandler(FileSystemEventHandler):
    """Handler for new video files in the final_video_dir."""

    def __init__(self, streamer: RTMPStreamer):
        self.streamer = streamer

    def on_created(self, event):
        """Handle new video file creation."""
        if not event.is_directory and event.src_path.endswith(('.mp4', '.mkv', '.mov')):
            video_path = event.src_path
            self.streamer.logger.info(f"New video file detected: {video_path}")
            asyncio.run_coroutine_threadsafe(
                self.streamer.stream_video(video_path),
                self.streamer.loop
            )
