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
        subtitles_input: str = 'output/subtitles/master_subtitles.vtt',
        output_file: str = 'output/final/output_with_subs.mp4',
        buffer_duration: int = 5,
    ):
        """
        Prepare a local MP4 file with audio, colored background video, and subtitles.
        Once this works, we can adapt it for live streaming.

        Args:
            client (OpenAIClient): Reference to the OpenAIClient instance.
            logger (logging.Logger): Logger instance.
            audio_input (str): Path to the audio file.
            subtitles_input (str): Path to the subtitles file (SRT).
            output_file (str): Path to the output MP4 file to be created.
            buffer_duration (int): Time to wait before processing (simulate buffering).
        """
        self.client = client
        self.logger = logger
        self.audio_input = audio_input
        self.subtitles_input = subtitles_input
        self.output_file = output_file
        self.buffer_duration = buffer_duration
        self.running = True

    async def create_local_file(self):
        """Create a local MP4 file that combines a colored background, audio, and subtitles."""
        try:
            self.logger.info("Starting local file creation process.")

            # Wait until audio file exists
            while not os.path.exists(self.audio_input):
                self.logger.info(f"Waiting for audio file to appear at {self.audio_input}...")
                await asyncio.sleep(1)
                if not self.running:
                    self.logger.info("Local file creation process was stopped before audio file was found.")
                    return

            self.logger.info(f"Audio file found at {self.audio_input}.")

            # Wait until subtitles file exists
            while not os.path.exists(self.subtitles_input):
                self.logger.info(f"Waiting for subtitles file to appear at {self.subtitles_input}...")
                await asyncio.sleep(1)
                if not self.running:
                    self.logger.info("Local file creation process was stopped before subtitles file was found.")
                    return

            self.logger.info(f"Subtitles file found at {self.subtitles_input}.")

            # Simulate buffering
            self.logger.info(f"Buffering for {self.buffer_duration} seconds before creating local file...")
            await asyncio.sleep(self.buffer_duration)
            self.logger.info("Buffering complete.")

            # Ensure output directory exists
            output_dir = os.path.dirname(self.output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                self.logger.info(f"Ensured that output directory exists: {output_dir}")

            # Verify input files
            if not os.path.isfile(self.audio_input):
                self.logger.error(f"Audio input file does not exist: {self.audio_input}")
                return

            if not os.path.isfile(self.subtitles_input):
                self.logger.warning(f"Subtitles input file does not exist: {self.subtitles_input}. Subtitles will be skipped.")
                subtitles_filter = ""
            else:
                subtitles_filter = f"subtitles='{os.path.abspath(self.subtitles_input)}'"

            # Prepare FFmpeg filter
            if subtitles_filter:
                video_filter = subtitles_filter
            else:
                video_filter = ""

            # Build FFmpeg command
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file if it exists
                '-i', self.audio_input,  # Input audio
                '-f', 'lavfi',
                '-i', 'color=size=1280x720:rate=25:color=blue',  # Colored background video
            ]

            if video_filter:
                ffmpeg_cmd.extend([
                    '-vf', video_filter,
                ])
            else:
                self.logger.info("No subtitles filter applied; proceeding without subtitles.")

            # Add encoding options
            ffmpeg_cmd.extend([
                '-c:v', 'libx264',
                '-preset', 'veryfast',
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac',
                '-b:a', '160k',
                self.output_file
            ])

            self.logger.info("Running FFmpeg to create local MP4 file...")
            self.logger.debug("FFmpeg command: " + " ".join(ffmpeg_cmd))

            # Execute FFmpeg command
            process = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            self.logger.info("FFmpeg process started. Awaiting completion...")

            stdout, stderr = await process.communicate()

            # Log FFmpeg output for debugging
            if stdout:
                self.logger.debug(f"FFmpeg stdout: {stdout.decode('utf-8', errors='replace')}")
            if stderr:
                self.logger.debug(f"FFmpeg stderr: {stderr.decode('utf-8', errors='replace')}")

            if process.returncode != 0:
                self.logger.error(f"FFmpeg failed with exit code {process.returncode}.")
                self.logger.error(f"FFmpeg stderr: {stderr.decode('utf-8', errors='replace')}")
            else:
                self.logger.info(f"Successfully created local MP4 file: {self.output_file}")

        except FileNotFoundError as fnf_error:
            self.logger.error(f"FFmpeg not found: {fnf_error}. Ensure FFmpeg is installed and in the system PATH.")
        except Exception as e:
            self.logger.error(f"Error during local file creation: {e}")

    def start(self):
        """Start the file creation process."""
        self.logger.info("Starting local file creation task.")
        asyncio.create_task(self.create_local_file())
        self.logger.info("Local file creation task has been initiated.")

    def stop(self):
        """Stop the process if running."""
        self.running = False
        self.logger.info("Stopped local file creation process.")
