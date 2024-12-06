# openai_client/muxing.py

import asyncio
import subprocess
import os
import logging
from typing import Dict

class Muxer:
    def __init__(self, client, logger: logging.Logger):
        self.client = client
        self.logger = logger
        self.muxing_queue = asyncio.Queue(maxsize=self.client.MUXING_QUEUE_MAXSIZE)
        self.muxing_task = asyncio.create_task(self.mux_audio_video_subtitles())
        self.processed_segments = set()

    async def enqueue_muxing_job(self, job: Dict):
        segment_index = job["segment_index"]
        try:
            if segment_index in self.processed_segments:
                self.logger.warning(f"Segment {segment_index} already processed. Skipping.")
                return
            self.processed_segments.add(segment_index)
            await self.muxing_queue.put(job)
            self.logger.info(f"Enqueued muxing job for segment {segment_index}")
        except asyncio.QueueFull:
            self.logger.error("Muxing queue is full. Failed to enqueue muxing job.")
        except Exception as e:
            self.logger.error(f"Failed to enqueue muxing job: {e}")

    async def mux_audio_video_subtitles(self):
        while self.client.running:
            try:
                job = await self.muxing_queue.get()
                if job is None:
                    self.logger.info("Muxing queue received shutdown signal.")
                    break

                video_path = job['video']
                audio_path = job['audio']
                subtitles_path = job['subtitles']
                final_output_path = job['output']
                segment_index = job['segment_index']
                audio_offset = job.get('audio_offset', 0.0)

                if not self.validate_files(video_path, audio_path, subtitles_path, segment_index):
                    self.muxing_queue.task_done()
                    continue

                # Since audio is now adjusted beforehand, we just mux directly:
                self.run_ffmpeg_command(video_path, audio_path, subtitles_path, final_output_path, segment_index, audio_offset)

                # Clean up subtitles if present
                if os.path.exists(subtitles_path):
                    os.remove(subtitles_path)

                self.muxing_queue.task_done()

            except Exception as e:
                self.logger.error(f"Error in mux_audio_video_subtitles: {e}")

    def validate_files(self, video_path: str, audio_path: str, subtitles_path: str, segment_index: int) -> bool:
        try:
            if not os.path.exists(video_path):
                self.logger.error(f"Missing video file for segment {segment_index}.")
                return False
            if not os.path.exists(audio_path):
                self.logger.error(f"Missing audio file for segment {segment_index}.")
                return False
            # Subtitles can be optional
            return True
        except Exception as e:
            self.logger.error(f"Error validating files for segment {segment_index}: {e}")
            return False

    def run_ffmpeg_command(self, video_path: str, audio_path: str, subtitles_path: str, output_path: str, segment_index: int, audio_offset: float = 0.0):
        offset_log_file = "output/logs/audio_offsets.log"
        os.makedirs(os.path.dirname(offset_log_file), exist_ok=True)

        try:
            command = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-itsoffset', str(audio_offset), '-i', audio_path
            ]

            if os.path.exists(subtitles_path):
                command.extend([
                    '-vf', f"subtitles={subtitles_path}:force_style='FontName=Arial,FontSize=24,PrimaryColour=&HFFFFFF&'",
                    '-c:v', 'libx264'
                ])
            else:
                command.extend([
                    '-c:v', 'copy'
                ])

            command.extend([
                '-c:a', 'aac',
                '-strict', 'experimental',
                '-map', '0:v', '-map', '1:a',
                output_path
            ])

            self.logger.debug(f"Running FFmpeg command: {' '.join(command)}")

            process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            with open(offset_log_file, 'a') as log_file:
                log_entry = (
                    f"Segment: {segment_index}, "
                    f"Video: {video_path}, "
                    f"Audio: {audio_path}, "
                    f"Offset: {audio_offset:.2f}, "
                    f"Output: {output_path}, "
                    f"Return Code: {process.returncode}\n"
                )
                log_file.write(log_entry)

            if process.returncode == 0:
                self.logger.info(f"Successfully muxed segment {segment_index}. Output: {output_path}")
            else:
                self.logger.error(f"FFmpeg failed for segment {segment_index}. Error: {process.stderr}")
        except Exception as e:
            self.logger.error(f"Error running FFmpeg for segment {segment_index}: {e}")
