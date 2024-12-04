# openai_client/muxing.py

import asyncio
import subprocess
import os
import logging
from config import MAX_BACKOFF, SEGMENT_DURATION
from typing import Dict
import webvtt

class Muxer:
    def __init__(self, client, logger: logging.Logger):
        self.client = client
        self.logger = logger
        self.muxing_queue = asyncio.Queue(maxsize=self.client.MUXING_QUEUE_MAXSIZE)
        self.muxing_task = asyncio.create_task(self.mux_audio_video_subtitles())
        # Track processed segments in Muxer
        self.processed_segments = set()
        self.expected_segments = set(range(1, 5))  # work with 5 segments for now (later without max)


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

                original_video_path = job['video']
                audio_path = job['audio']
                final_output_path = job['output']
                segment_index = job['segment_index']
                audio_offset = job.get('audio_offset', 0.0)

                # Use the previous segment's video if it exists
                previous_segment_index = segment_index - 1
                fallback_video_path = f'output/video/output_video_segment_{previous_segment_index}.mp4'
                video_path = fallback_video_path if os.path.exists(fallback_video_path) else original_video_path

                self.logger.info(f"Using video for segment {segment_index}: {video_path}")

                # Validate file existence
                if not self.validate_files(video_path, audio_path, segment_index):
                    self.muxing_queue.task_done()
                    continue

                # Validate duration consistency
                if not self.check_duration_match(video_path, audio_path):
                    self.logger.error(f"Duration mismatch for segment {segment_index}. Skipping muxing.")
                    self.muxing_queue.task_done()
                    continue

                # Extract subtitles
                temp_subtitles_path = f'output/subtitles/subtitles_segment_{segment_index}.vtt'
                self.extract_subtitles_for_segment(segment_index, temp_subtitles_path)

                # FFmpeg muxing
                self.run_ffmpeg_command(video_path, audio_path, temp_subtitles_path, final_output_path, segment_index, audio_offset)

                # Clean up temporary files
                if os.path.exists(temp_subtitles_path):
                    os.remove(temp_subtitles_path)

                self.muxing_queue.task_done()

            except Exception as e:
                self.logger.error(f"Error in mux_audio_video_subtitles: {e}")


    def validate_files(self, video_path: str, audio_path: str, segment_index: int) -> bool:
        """Validate video and audio files for muxing."""
        try:
            if not os.path.exists(video_path) or not os.path.exists(audio_path):
                self.logger.error(f"Missing video or audio file for segment {segment_index}.")
                return False

            video_size = os.path.getsize(video_path)
            audio_size = os.path.getsize(audio_path)
            if video_size < 1000 or audio_size < 100:
                self.logger.error(f"Invalid file size for segment {segment_index}. Video: {video_size}, Audio: {audio_size}.")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Error validating files for segment {segment_index}: {e}")
            return False


    def check_duration_match(self, video_path: str, audio_path: str) -> bool:
        """Check if video and audio durations match closely."""
        try:
            video_duration = self.get_duration_from_ffprobe(video_path)
            audio_duration = self.get_duration_from_ffprobe(audio_path)
            return abs(video_duration - audio_duration) < 0.1  # Allow small tolerance
        except Exception as e:
            self.logger.error(f"Error checking duration match: {e}")
            return False

    def get_duration_from_ffprobe(self, file_path: str) -> float:
        """Retrieve file duration using ffprobe."""
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return float(result.stdout.strip())
        except Exception as e:
            self.logger.error(f"Error retrieving duration from ffprobe for file {file_path}: {e}")
            return 0.0

    def extract_subtitles_for_segment(self, segment_index: int, output_path: str):
        """Extract subtitles for the given segment."""
        try:
            segment_start_time = (segment_index - 1) * SEGMENT_DURATION
            segment_end_time = segment_start_time + SEGMENT_DURATION

            subs = webvtt.read(self.client.SUBTITLE_PATH)
            segment_subs = [
                caption for caption in subs if segment_start_time <= self.vtt_timestamp_to_seconds(caption.start) < segment_end_time
            ]

            new_vtt = webvtt.WebVTT()
            if segment_subs:
                for caption in segment_subs:
                    new_vtt.captions.append(caption)
                self.logger.info(f"Extracted subtitles for segment {segment_index}.")
            else:
                self.logger.warning(f"No subtitles found for segment {segment_index}. Generating empty subtitle file.")

            new_vtt.save(output_path)

        except Exception as e:
            self.logger.error(f"Error extracting subtitles for segment {segment_index}: {e}")


    def run_ffmpeg_command(self, video_path: str, audio_path: str, subtitles_path: str, output_path: str, segment_index: int, audio_offset: float = 0.0):
        """Run FFmpeg command to mux video, audio, and optionally subtitles."""
        # Define offset log file
        offset_log_file = "output/logs/audio_offsets.log"
        os.makedirs(os.path.dirname(offset_log_file), exist_ok=True)
            
        try:
            # Prepare FFmpeg command
            subtitles_path = f"output/subtitles/subtitles_segment_{segment_index}.vtt"
            command = [
                'ffmpeg', '-y',
                '-itsoffset', str(video_delay), '-i', video_path,  # Apply delay to video
                '-itsoffset', str(audio_offset), '-i', audio_path,  # Apply offset to audio
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-strict', 'experimental'
            ]

            # Add subtitles if they exist
            if os.path.exists(subtitles_path):
                command.extend(['-vf', f"subtitles={subtitles_path}:force_style='FontName=Arial,FontSize=24,PrimaryColour=&HFFFFFF&'"])

            # Map video and offset audio streams
            command.extend(['-map', '0:v', '-map', '1:a'])
            command.append(output_path)

            # Log the FFmpeg command
            self.logger.debug(f"Running FFmpeg command: {' '.join(command)}")

            # Execute FFmpeg command
            process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # Log offset-specific details
            with open("output/logs/audio_offsets.log", 'a') as log_file:
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
