import asyncio
import subprocess
import os
import logging
from config import MAX_BACKOFF, SEGMENT_DURATION
from typing import Dict

class Muxer:
    def __init__(self, client, logger: logging.Logger):
        self.client = client
        self.logger = logger
        self.muxing_queue = asyncio.Queue(maxsize=self.client.MUXING_QUEUE_MAXSIZE)
        self.muxing_task = asyncio.create_task(self.mux_audio_video_subtitles())
        # Track processed segments in Muxer
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
        """Asynchronous task to mux audio, video, and subtitles."""
        while self.client.running:
            try:
                # Wait for a muxing job
                job = await self.muxing_queue.get()
                if job is None:
                    self.logger.info("Muxing queue received shutdown signal.")
                    break

                # Inside mux_audio_video_subtitles
                video_path = job['video']
                audio_path = job['audio']
                subtitles_path = job['subtitles']
                final_output_path = job['output']
                segment_index = job['segment_index']
                audio_offset = job.get('audio_offset', 0.0)

                # Validate file existence
                if not self.validate_files(video_path, audio_path, subtitles_path, segment_index):
                    self.muxing_queue.task_done()
                    continue

                # Adjust audio duration to match video
                adjusted_audio_path = f"{os.path.splitext(audio_path)[0]}_adjusted.wav"
                self.adjust_audio_duration(audio_path, video_path, adjusted_audio_path)

                # FFmpeg muxing
                self.run_ffmpeg_command(video_path, adjusted_audio_path, subtitles_path, final_output_path, segment_index, audio_offset)

                # Clean up temporary files
                if os.path.exists(subtitles_path):
                    os.remove(subtitles_path)
                if os.path.exists(adjusted_audio_path):
                    os.remove(adjusted_audio_path)

                self.muxing_queue.task_done()

            except Exception as e:
                self.logger.error(f"Error in mux_audio_video_subtitles: {e}")

    def validate_files(self, video_path: str, audio_path: str, subtitles_path: str, segment_index: int) -> bool:
        """Validate video, audio, and subtitles files for muxing."""
        try:
            if not os.path.exists(video_path):
                self.logger.error(f"Missing video file for segment {segment_index}.")
                return False
            if not os.path.exists(audio_path):
                self.logger.error(f"Missing audio file for segment {segment_index}.")
                return False
            if not os.path.exists(subtitles_path):
                self.logger.warning(f"Subtitles file {subtitles_path} does not exist for segment {segment_index}. Subtitles will be skipped.")
                # Subtitles are optional
            return True
        except Exception as e:
            self.logger.error(f"Error validating files for segment {segment_index}: {e}")
            return False

    def adjust_audio_duration(self, audio_path: str, video_path: str, output_audio_path: str):
        """Adjust the audio duration to match the video duration."""
        try:
            video_duration = self.get_duration_from_ffprobe(video_path)
            audio_duration = self.get_duration_from_ffprobe(audio_path)
            self.logger.debug(f"Video duration: {video_duration}, Audio duration: {audio_duration}")

            if audio_duration == 0 or video_duration == 0:
                self.logger.error("Invalid duration. Skipping audio adjustment.")
                return

            duration_ratio = video_duration / audio_duration

            # Create FFmpeg command to adjust audio
            if abs(video_duration - audio_duration) < 0.1:
                # Durations are close enough; no adjustment needed
                self.logger.info("Audio and video durations are close enough. No adjustment needed.")
                os.rename(audio_path, output_audio_path)
            elif audio_duration < video_duration:
                # Audio is shorter; pad with silence
                pad_duration = video_duration - audio_duration
                command = [
                    'ffmpeg', '-y',
                    '-i', audio_path,
                    '-af', f'apad=pad_dur={pad_duration}',
                    '-c:a', 'pcm_s16le',
                    output_audio_path
                ]
                self.logger.info(f"Padding audio with {pad_duration:.2f} seconds of silence.")
            else:
                # Audio is longer; time-shrink the audio
                # FFmpeg's atempo supports tempo adjustments between 0.5 and 2.0
                atempo_value = min(max(duration_ratio, 0.5), 2.0)
                command = [
                    'ffmpeg', '-y',
                    '-i', audio_path,
                    '-filter:a', f'atempo={atempo_value}',
                    '-c:a', 'pcm_s16le',
                    output_audio_path
                ]
                self.logger.info(f"Adjusting audio tempo by a factor of {atempo_value:.3f} to match video duration.")

            # Run the FFmpeg command
            process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if process.returncode != 0:
                self.logger.error(f"FFmpeg audio adjustment failed: {process.stderr}")
                # If adjustment fails, copy the original audio
                os.rename(audio_path, output_audio_path)
            else:
                self.logger.info("Audio duration adjusted successfully.")

        except Exception as e:
            self.logger.error(f"Error adjusting audio duration: {e}")
            # If there's an error, use the original audio
            os.rename(audio_path, output_audio_path)

    def get_duration_from_ffprobe(self, file_path: str) -> float:
        """Retrieve file duration using ffprobe."""
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries',
                 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            duration_str = result.stdout.strip()
            if not duration_str or duration_str == 'N/A':
                self.logger.error(f"ffprobe could not retrieve duration for file {file_path}. Output: {duration_str}")
                return 0.0
            return float(duration_str)
        except Exception as e:
            self.logger.error(f"Error retrieving duration from ffprobe for file {file_path}: {e}")
            return 0.0

    def run_ffmpeg_command(self, video_path: str, audio_path: str, subtitles_path: str, output_path: str, segment_index: int, audio_offset: float = 0.0):
        """Run FFmpeg command to mux video, audio, and optionally subtitles."""
        # Define offset log file
        offset_log_file = "output/logs/audio_offsets.log"
        os.makedirs(os.path.dirname(offset_log_file), exist_ok=True)

        try:
            # Prepare FFmpeg command
            command = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-itsoffset', str(audio_offset), '-i', audio_path,
            ]

            # Initialize codec options
            if os.path.exists(subtitles_path):
                # If subtitles are present, apply filter and set video codec to libx264
                command.extend([
                    '-vf', f"subtitles={subtitles_path}:force_style='FontName=Arial,FontSize=24,PrimaryColour=&HFFFFFF&'",
                    '-c:v', 'libx264',  # Specify a codec that allows re-encoding
                ])
            else:
                # If no subtitles, copy the video stream
                command.extend([
                    '-c:v', 'copy',
                ])

            # Always set audio codec to AAC
            command.extend([
                '-c:a', 'aac',
                '-strict', 'experimental',
                '-map', '0:v', '-map', '1:a',
                output_path
            ])

            # Log the FFmpeg command
            self.logger.debug(f"Running FFmpeg command: {' '.join(command)}")

            # Execute FFmpeg command
            process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # Log offset-specific details
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

            # Check FFmpeg result
            if process.returncode == 0:
                self.logger.info(f"Successfully muxed segment {segment_index}. Output: {output_path}")
            else:
                self.logger.error(f"FFmpeg failed for segment {segment_index}. Error: {process.stderr}")
        except Exception as e:
            self.logger.error(f"Error running FFmpeg for segment {segment_index}: {e}")


    def vtt_timestamp_to_seconds(self, timestamp: str) -> float:
        """Convert VTT timestamp to seconds."""
        h, m, s = map(float, timestamp.replace(',', '.').split(':'))
        return h * 3600 + m * 60 + s
