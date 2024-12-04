# openai_client/muxing.py

import asyncio
import subprocess
import os
import logging
from config import MAX_BACKOFF, SEGMENT_DURATION
from typing import Dict, Any
import webvtt  # Ensure webvtt-py is installed: pip install webvtt-py

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

                # Extract relevant subtitles for this segment
                subtitles_path = self.client.SUBTITLE_PATH  # Path to the main subtitles.vtt
                temp_subtitles_path = f'output/subtitles/subtitles_segment_{segment_index}.vtt'
                self.extract_subtitles_for_segment(segment_index, temp_subtitles_path)

                # Check if any subtitles were extracted
                if os.path.exists(temp_subtitles_path):
                    self.logger.info(f"Temporary subtitles file created: {temp_subtitles_path}")
                else:
                    self.logger.warning(f"No subtitles to overlay for segment {segment_index}. Proceeding without subtitles.")

                # FFmpeg command to mux audio, video, and subtitles
                # If subtitles are available, include them; otherwise, skip the subtitle filter
                ffmpeg_command = [
                    'ffmpeg',
                    '-y',  # Overwrite output files without asking
                    '-i', video_path,  # Input video file
                    '-i', audio_path,  # Input audio file
                ]

                if os.path.exists(temp_subtitles_path):
                    # Add subtitle overlay
                    ffmpeg_command += [
                        '-vf', f"subtitles={temp_subtitles_path}:force_style='FontName=Arial,FontSize=24,PrimaryColour=&HFFFFFF&'",
                    ]
                else:
                    # No subtitles to overlay
                    ffmpeg_command += [
                        '-vf', 'null',  # No video filter
                    ]

                # Continue building the FFmpeg command
                ffmpeg_command += [
                    '-c:v', 'libx264',  # Re-encode video to ensure compatibility
                    '-c:a', 'aac',       # Encode audio to AAC for better compatibility
                    '-strict', 'experimental',
                    final_output_path  # Output file
                ]

                self.logger.debug(f"Running FFmpeg command: {' '.join(ffmpeg_command)}")
                process = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                if process.returncode == 0:
                    self.logger.info(f"Successfully muxed video to {final_output_path}")
                    # Verify that the final video contains audio
                    if os.path.exists(final_output_path):
                        self.logger.info(f"Final muxed video file exists: {final_output_path}")
                        # Optionally, perform a quick check on the video file
                        # e.g., check duration matches expected
                    else:
                        self.logger.error(f"Final muxed video file was not found: {final_output_path}")
                else:
                    self.logger.error(f"FFmpeg muxing failed for segment {segment_index}.")
                    self.logger.error(f"FFmpeg stderr: {process.stderr}")
                    self.logger.error(f"FFmpeg stdout: {process.stdout}")

                # Clean up temporary subtitles file
                if os.path.exists(temp_subtitles_path):
                    os.remove(temp_subtitles_path)
                    self.logger.debug(f"Removed temporary subtitles file: {temp_subtitles_path}")

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

    def extract_subtitles_for_segment(self, segment_index: int, output_path: str):
        """
        Extract subtitles for the given segment index and save to output_path.

        Args:
            segment_index (int): The index of the current video segment.
            output_path (str): Path to save the extracted subtitles.
        """
        try:
            segment_start_time = (segment_index - 1) * SEGMENT_DURATION
            segment_end_time = segment_start_time + SEGMENT_DURATION

            self.logger.debug(f"Extracting subtitles for segment {segment_index}: Start={segment_start_time}s, End={segment_end_time}s")

            # Parse the main subtitles.vtt
            subs = webvtt.read(self.client.SUBTITLE_PATH)
            segment_subs = []

            for caption in subs:
                # Convert VTT timestamps to seconds
                start_sec = self.vtt_timestamp_to_seconds(caption.start)
                end_sec = self.vtt_timestamp_to_seconds(caption.end)

                # Check if the subtitle falls within the current segment
                if segment_start_time <= start_sec < segment_end_time:
                    # Adjust the timestamps relative to the segment start
                    adjusted_start = start_sec - segment_start_time
                    adjusted_end = end_sec - segment_start_time

                    # Ensure adjusted times are non-negative
                    adjusted_start = max(adjusted_start, 0)
                    adjusted_end = max(adjusted_end, 0)

                    # Create a new caption with adjusted timings
                    adjusted_caption = webvtt.Caption(
                        start=self.seconds_to_vtt_timestamp(adjusted_start),
                        end=self.seconds_to_vtt_timestamp(adjusted_end),
                        text=caption.text
                    )
                    segment_subs.append(adjusted_caption)

            if segment_subs:
                # Ensure the output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Write the adjusted subtitles to the output_path
                new_vtt = webvtt.WebVTT()
                for caption in segment_subs:
                    new_vtt.captions.append(caption)
                new_vtt.save(output_path)
                self.logger.debug(f"Extracted {len(segment_subs)} subtitles for segment {segment_index} and saved to {output_path}.")
            else:
                # No subtitles for this segment; do not create the file
                self.logger.warning(f"No subtitles found for segment {segment_index} (Start: {segment_start_time}s, End: {segment_end_time}s).")

        except Exception as e:
            self.logger.error(f"Error extracting subtitles for segment {segment_index}: {e}")


    def vtt_timestamp_to_seconds(self, timestamp: str) -> float:
        """Convert VTT timestamp to seconds."""
        try:
            parts = timestamp.split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds, millis = map(float, parts[2].split('.'))
            total_seconds = hours * 3600 + minutes * 60 + seconds + millis / 1000.0
            return total_seconds
        except Exception as e:
            self.logger.error(f"Error converting VTT timestamp to seconds: {timestamp} - {e}")
            return 0.0

    def seconds_to_vtt_timestamp(self, seconds: float) -> str:
        """Convert seconds to VTT timestamp."""
        try:
            millis = int((seconds - int(seconds)) * 1000)
            sec = int(seconds) % 60
            min = (int(seconds) // 60) % 60
            hour = int(seconds) // 3600
            return f"{hour:02}:{min:02}:{sec:02}.{millis:03}"
        except Exception as e:
            self.logger.error(f"Error converting seconds to VTT timestamp: {seconds} - {e}")
            return "00:00:00.000"
