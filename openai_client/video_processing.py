# openai_client/video_processing.py

import asyncio
import cv2
import time
import os
import logging
from config import STREAM_URL, SEGMENT_DURATION

class VideoProcessor:
    def __init__(self, client, logger: logging.Logger):
        self.client = client
        self.logger = logger
        self.segment_duration = SEGMENT_DURATION
        self.video_task = asyncio.create_task(self.run_video_processing())

    async def run_video_processing(self):
        await asyncio.to_thread(self.start_video_processing)

    def start_video_processing(self):
        try:
            self.logger.info("Starting video processing with OpenCV.")
            cap = cv2.VideoCapture(STREAM_URL)

            if not cap.isOpened():
                self.logger.error("Cannot open video stream.")
                self.client.running = False
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or fps is None or fps != fps:
                fps = 30.0
                self.logger.warning(f"Unable to get FPS from stream. Defaulting to {fps} FPS.")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

            self.logger.info(f"Video properties - FPS: {fps}, Width: {width}, Height: {height}")

            while self.client.running:
                frame_count = 0
                frames_written = 0
                segment_start_time = time.perf_counter()

                if self.client.video_start_time is None:
                    self.client.video_start_time = segment_start_time

                try:
                    current_segment_index_future = asyncio.run_coroutine_threadsafe(
                        self.client.get_segment_index(),
                        self.client.loop
                    )
                    current_segment_index = current_segment_index_future.result()
                except Exception as e:
                    self.logger.error(f"Error retrieving segment_index: {e}")
                    self.client.running = False
                    break

                segment_output_path = f'output/video/output_video_segment_{current_segment_index}.mp4'
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')

                out = cv2.VideoWriter(segment_output_path, fourcc, fps, (width, height))
                if not out.isOpened():
                    self.logger.error(f"Failed to open VideoWriter for segment {current_segment_index}.")
                    self.client.running = False
                    break

                self.logger.info(f"Started recording segment {current_segment_index} to {segment_output_path}")

                segment_frames = round(fps * self.segment_duration)

                while frame_count < segment_frames and self.client.running:
                    ret, frame = cap.read()
                    if not ret:
                        self.logger.warning("Failed to read frame from video stream.")
                        break
                    out.write(frame)
                    frame_count += 1
                    frames_written += 1

                out.release()
                self.logger.info(f"Segment {current_segment_index} saved to {segment_output_path}. Frames: {frames_written}")

                try:
                    video_size = os.path.getsize(segment_output_path)
                    self.logger.info(f"Video segment size: {video_size} bytes")
                except Exception as e:
                    self.logger.error(f"Error getting file size: {e}")

                try:
                    asyncio.run_coroutine_threadsafe(
                        self.client.increment_segment_index(),
                        self.client.loop
                    ).result()
                    self.logger.info("Segment index incremented.")
                except Exception as e:
                    self.logger.error(f"Error incrementing segment index: {e}")
                    self.client.running = False
                    break

        except Exception as e:
            self.logger.error(f"Error in start_video_processing: {e}", exc_info=True)
            self.client.running = False
