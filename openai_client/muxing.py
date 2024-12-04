# openai_client/muxing.py

import asyncio
import subprocess
import os
import logging
from config import MAX_BACKOFF
from collections import deque

class Muxer:
    def __init__(self, client, logger: logging.Logger):
        self.client = client
        self.logger = logger
        self.muxing_queue = asyncio.Queue(maxsize=self.client.MUXING_QUEUE_MAXSIZE)
        self.muxing_task = asyncio.create_task(self.mux_audio_video_subtitles())

    async def enqueue_muxing_job(self, job: dict):
        """Enqueue a muxing job to the muxing_queue."""
        try:
            await self.muxing_queue.put(job)
            self.logger.debug(f"Enqueued muxing job for segment {job.get('segment_index')}.")
        except asyncio.QueueFull:
            self.logger.error("Muxing queue is full. Failed to enqueue muxing job.")
        except Exception as e:
            self.logger.error(f"Failed to enqueue muxing job: {e}")

    async def mux_audio_video_subtitles(self):
        """Asynchronous task to mux audio, video, and subtitles using FFmpeg with enhanced reliability."""
        while self.client.running:
            try:
                # Wait for a muxing job
                job = await self.muxing_queue.get()
                if job is None:
                    self.logger.info("Muxing queue received shutdown signal.")
                    break

                video_path = job['video']
                audio_path = job['audio']
                subtitles_path = job['subtitles']
                final_output_path = job['output']
                segment_index = job['segment_index']

                self.logger.info(f"Muxing audio and video for segment {segment_index}.")

                # Check if video and audio files exist
                if not os.path.exists(video_path):
                    self.logger.error(f"Video file does not exist: {video_path}")
                    self.muxing_queue.task_done()
                    continue
                if not os.path.exists(audio_path):
                    self.logger.error(f"Audio file does not exist: {audio_path}")
                    self.muxing_queue.task_done()
                    continue

                # Check file sizes to ensure they are not empty
                min_video_size = 1000  # bytes
                min_audio_size = 100  # bytes
                try:
                    video_size = os.path.getsize(video_path)
                    audio_size = os.path.getsize(audio_path)
                    if video_size < min_video_size:
                        self.logger.error(f"Video file {video_path} is too small ({video_size} bytes). Skipping muxing.")
                        self.muxing_queue.task_done()
                        continue
                    if audio_size < min_audio_size:
                        self.logger.error(f"Audio file {audio_path} is too small ({audio_size} bytes). Skipping muxing.")
                        self.muxing_queue.task_done()
                        continue
                except Exception as e:
                    self.logger.error(f"Error getting file sizes: {e}")
                    self.muxing_queue.task_done()
                    continue

                # Implement a retry mechanism with a maximum number of attempts
                max_attempts = 5
                attempt = 1
                while attempt <= max_attempts:
                    # Ensure video file is properly closed by checking with ffprobe
                    is_valid = await self.verify_file(video_path, 'video')
                    if is_valid:
                        break
                    else:
                        self.logger.warning(f"Video file {video_path} is invalid. Retrying in 2 seconds... (Attempt {attempt}/{max_attempts})")
                        await asyncio.sleep(2)
                        attempt += 1

                if not is_valid:
                    self.logger.error(f"Video file {video_path} remains invalid after {max_attempts} attempts.")
                    self.muxing_queue.task_done()
                    continue

                # Verify audio file
                is_audio_valid = await self.verify_file(audio_path, 'audio')
                if not is_audio_valid:
                    self.logger.error(f"Audio file {audio_path} is invalid.")
                    self.muxing_queue.task_done()
                    continue

                # Introduce a small delay to ensure file system has completed writing
                await asyncio.sleep(0.5)  # 500 milliseconds

                # FFmpeg command to mux audio, video, and subtitles
                ffmpeg_command = [
                    'ffmpeg',
                    '-y',  # Overwrite output files without asking
                    '-i', video_path,
                    '-i', audio_path,
                    '-vf', f"subtitles={subtitles_path}:force_style='FontSize=24,PrimaryColour=&HFFFFFF&'",
                    '-c:v', 'libx264',  # Re-encode video to ensure compatibility
                    '-c:a', 'aac',       # Encode audio to AAC for better compatibility
                    '-strict', 'experimental',
                    final_output_path
                ]

                self.logger.debug(f"Running FFmpeg command: {' '.join(ffmpeg_command)}")
                process = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                if process.returncode == 0:
                    self.logger.info(f"Successfully muxed video to {final_output_path}")
                else:
                    self.logger.error(f"FFmpeg muxing failed for segment {segment_index}.")
                    self.logger.error(f"FFmpeg stderr: {process.stderr}")
                    self.logger.error(f"FFmpeg stdout: {process.stdout}")

                self.muxing_queue.task_done()

            except Exception as e:
                self.logger.error(f"Error in mux_audio_video_subtitles: {e}")

    async def verify_file(self, file_path: str, file_type: str) -> bool:
        """Verify if the file is valid using ffprobe."""
        try:
            probe = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_format', '-show_streams', file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if probe.returncode == 0:
                if file_type == 'video' and 'video' in probe.stdout:
                    self.logger.debug(f"Video file {file_path} is valid.")
                    return True
                elif file_type == 'audio' and 'audio' in probe.stdout:
                    self.logger.debug(f"Audio file {file_path} is valid.")
                    return True
            self.logger.error(f"{file_type.capitalize()} file {file_path} is invalid. FFprobe output: {probe.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"Error verifying {file_type} file {file_path}: {e}")
            return False
