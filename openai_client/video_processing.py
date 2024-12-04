# openai_client/video_processing.py

import asyncio
import cv2
import time
import os
import wave
import logging
from config import STREAM_URL, SEGMENT_DURATION
from collections import deque

class VideoProcessor:
    def __init__(self, client, logger: logging.Logger):
        self.client = client
        self.logger = logger
        self.segment_duration = SEGMENT_DURATION
        self.video_task = asyncio.create_task(self.run_video_processing())

    async def run_video_processing(self):
        """Run video processing within the asyncio event loop."""
        await asyncio.to_thread(self.start_video_processing)

    def start_video_processing(self):
        """Start video processing with OpenCV and dynamic subtitle overlay."""
        try:
            self.logger.info("Starting video processing with OpenCV.")

            # Open video capture
            cap = cv2.VideoCapture(STREAM_URL)

            if not cap.isOpened():
                self.logger.error("Cannot open video stream.")
                self.client.running = False
                return

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or fps is None:
                fps = 30.0  # Default FPS if unable to get from stream
                self.logger.warning(f"Unable to get FPS from stream. Defaulting to {fps} FPS.")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

            self.logger.info(f"Video properties - FPS: {fps}, Width: {width}, Height: {height}")

            # Initialize variables for video segmentation
            segment_frames = int(fps * self.segment_duration)
            frame_count = 0
            self.segment_start_time = time.perf_counter()

            if self.client.video_start_time is None:
                self.client.video_start_time = self.segment_start_time

            while self.client.running:
                # Retrieve current segment_index safely
                loop = self.client.loop
                try:
                    current_segment_index_future = asyncio.run_coroutine_threadsafe(
                        self.client.get_segment_index(),
                        loop
                    )
                    current_segment_index = current_segment_index_future.result()
                except Exception as e:
                    self.logger.error(f"Error retrieving segment_index: {e}")
                    self.client.running = False
                    break

                # Define video writer for each segment
                segment_output_path = f'output/video/output_video_segment_{current_segment_index}.mp4'
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Alternative codec

                out = cv2.VideoWriter(segment_output_path, fourcc, fps, (width, height))
                if not out.isOpened():
                    self.logger.error(f"Failed to open VideoWriter for segment {current_segment_index}.")
                    self.client.running = False
                    break

                self.logger.info(f"Started recording segment {current_segment_index} to {segment_output_path}")

                # Initialize the corresponding audio segment file
                audio_segment_path = f'output/audio/output_audio_segment_{current_segment_index}.wav'
                try:
                    os.makedirs(os.path.dirname(audio_segment_path), exist_ok=True)
                    wf = wave.open(audio_segment_path, 'wb')
                    wf.setnchannels(self.client.AUDIO_CHANNELS)
                    wf.setsampwidth(self.client.AUDIO_SAMPLE_WIDTH)
                    wf.setframerate(self.client.AUDIO_SAMPLE_RATE)
                    self.client.current_audio_segment_wf = wf  # Reference for writing translated audio
                    self.logger.debug(f"Initialized audio segment file: {audio_segment_path}")
                except Exception as e:
                    self.logger.error(f"Failed to initialize audio segment file {audio_segment_path}: {e}", exc_info=True)
                    # Close VideoWriter if audio segment fails
                    out.release()
                    self.logger.info(f"Released VideoWriter for segment {current_segment_index} due to audio init failure.")
                    self.client.current_audio_segment_wf = None
                    continue

                frames_written = 0
                while frame_count < segment_frames and self.client.running:
                    ret, frame = cap.read()
                    if not ret:
                        self.logger.warning("Failed to read frame from video stream.")
                        break
                    out.write(frame)
                    frame_count += 1
                    frames_written += 1
                    if frames_written % int(fps) == 0:
                        self.logger.debug(f"Written {frames_written} frames to segment {current_segment_index}.")

                # Release the video writer
                out.release()
                self.logger.info(f"Segment {current_segment_index} saved to {segment_output_path}. Frames written: {frames_written}")

                # Introduce a small delay to ensure file system has completed writing
                time.sleep(2)  # 2 seconds

                # Log file sizes
                try:
                    video_size = os.path.getsize(segment_output_path)
                    audio_size = os.path.getsize(audio_segment_path)
                    self.logger.info(f"Video segment size: {video_size} bytes")
                    self.logger.info(f"Audio segment size: {audio_size} bytes")
                except Exception as e:
                    self.logger.error(f"Error getting file sizes: {e}")

                # Close the audio WAV file
                try:
                    wf.close()
                    self.logger.debug(f"Closed audio segment file: {audio_segment_path}")
                    self.client.current_audio_segment_wf = None
                except Exception as e:
                    self.logger.error(f"Failed to close audio segment file {audio_segment_path}: {e}")

                # Enqueue muxing job
                final_output_path = f'output/final/output_final_segment_{current_segment_index}.mp4'
                muxing_job = {
                    "segment_index": current_segment_index,
                    "video": segment_output_path,
                    "audio": audio_segment_path,
                    "subtitles": f'output/subtitles/subtitles_segment_{current_segment_index}.vtt',
                    "output": final_output_path
                }
                # Enqueue muxing job in a thread-safe manner
                asyncio.run_coroutine_threadsafe(
                    self.client.muxer.enqueue_muxing_job(muxing_job),
                    self.client.loop
                )

                # Reset for next segment
                frame_count = 0
                self.segment_start_time = time.perf_counter()

        except Exception as e:
            self.logger.error(f"Error in start_video_processing: {e}", exc_info=True)
            self.client.running = False
