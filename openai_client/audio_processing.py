import asyncio
import base64
import wave
import uuid
import os
from asyncio import Queue
import simpleaudio as sa
import numpy as np
import logging
from typing import Optional
from .utils import format_timestamp_vtt
from config import AUDIO_CHANNELS, AUDIO_SAMPLE_RATE, AUDIO_SAMPLE_WIDTH, SEGMENT_DURATION
from collections import deque
import aiofiles  # Ensure aiofiles is imported

class AudioProcessor:
    def __init__(self, client, logger: logging.Logger, initial_buffer_size: int = 5):
        self.client = client
        self.logger = logger
        self.translated_audio_queue = Queue()
        self.audio_buffer = deque(maxlen=100)
        self.segment_lock = asyncio.Lock()  # To synchronize access to segments
        self.output_audio_path = os.environ.get('OUTPUT_AUDIO_PIPE')
        self.initial_buffer_size = initial_buffer_size
        self.playback_started = False
        self.playback_task = asyncio.create_task(self.audio_playback_handler())

    async def write_to_output_audio_pipe(self, audio_data: bytes):
        """Write audio data to the output audio named pipe for streaming."""
        try:
            async with aiofiles.open(self.output_audio_path, 'ab') as f:
                await f.write(audio_data)
            self.logger.debug("Written audio data to output audio pipe.")
        except Exception as e:
            self.logger.error(f"Error writing to output audio pipe: {e}")

    async def handle_audio_delta(self, audio_data: str):
        """Handle incoming audio delta from the server."""
        try:
            decoded_audio = base64.b64decode(audio_data)
            if not decoded_audio:
                self.logger.warning("Decoded audio data is empty.")
                return

            # Save raw audio for verification
            response_audio_filename = f"{uuid.uuid4()}.wav"
            response_audio_path = os.path.join('output/audio/responses', response_audio_filename)
            os.makedirs(os.path.dirname(response_audio_path), exist_ok=True)
            with wave.open(response_audio_path, 'wb') as wf_response:
                wf_response.setnchannels(AUDIO_CHANNELS)
                wf_response.setsampwidth(AUDIO_SAMPLE_WIDTH)
                wf_response.setframerate(AUDIO_SAMPLE_RATE)
                wf_response.writeframes(decoded_audio)
            self.logger.info(f"Saved raw audio response to {response_audio_path}")

            # Enqueue audio for playback
            await self.translated_audio_queue.put(decoded_audio)
            self.logger.debug("Enqueued translated audio for playback.")

        except Exception as e:
            self.logger.error(f"Error handling audio delta: {e}")

    async def send_input_audio(self, audio_data: bytes):
        """Send input audio to the Realtime API and record the send time."""
        try:
            pcm_base64 = base64.b64encode(audio_data).decode('utf-8')
            audio_event = {
                "type": "input_audio_buffer.append",
                "audio": pcm_base64
            }
            await self.client.enqueue_message(audio_event)
            sent_time = asyncio.get_event_loop().time()
            self.client.sent_audio_timestamps.append(sent_time)
            self.logger.debug(f"Sent input audio buffer append event at {sent_time}.")
        except Exception as e:
            self.logger.error(f"Failed to send input audio: {e}")

    async def audio_playback_handler(self):
        """Handle playback of translated audio with initial buffering."""
        # Wait for initial buffer to accumulate
        self.logger.info(f"Waiting until we have at least {self.initial_buffer_size} segments buffered before starting playback.")
        while self.client.running:
            if self.translated_audio_queue.qsize() >= self.initial_buffer_size:
                self.logger.info(f"Initial buffer reached ({self.translated_audio_queue.qsize()} segments), starting playback.")
                self.playback_started = True
                break
            else:
                await asyncio.sleep(0.1)

        # Now continuously play back and write to output pipe
        while self.client.running:
            try:
                audio_data = await self.translated_audio_queue.get()
                await self.play_audio(audio_data)
                self.translated_audio_queue.task_done()

                # Write audio data to the seamless output audio pipe
                await self.write_to_output_audio_pipe(audio_data)
            except Exception as e:
                self.logger.error(f"Error in audio_playback_handler: {e}")

    async def play_audio(self, audio_data: bytes):
        """Play audio locally and write it to current segment and output WAV file."""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            # Local playback (blocking, off the main thread)
            play_obj = sa.play_buffer(audio_array, AUDIO_CHANNELS, AUDIO_SAMPLE_WIDTH, AUDIO_SAMPLE_RATE)
            await asyncio.to_thread(play_obj.wait_done)

            # Write to current audio segment file
            async with self.segment_lock:
                if hasattr(self.client, 'current_audio_segment_wf') and self.client.current_audio_segment_wf:
                    self.client.current_audio_segment_wf.writeframes(audio_data)
                    self.logger.debug("Written audio data to current audio segment WAV file.")
                else:
                    self.logger.warning("No current audio segment WAV file to write to.")

            # Write to output WAV
            if self.client.output_wav:
                self.client.output_wav.writeframes(audio_data)
                self.logger.debug("Written translated audio chunk to output WAV file.")
            else:
                self.logger.warning("Output WAV file is not initialized.")

        except Exception as e:
            self.logger.error(f"Error playing or buffering audio: {e}")

    async def start_new_audio_segment(self, segment_index: int = None):
        """Initialize a new audio segment WAV file for the given segment index."""
        async with self.segment_lock:
            try:
                if segment_index is None:
                    segment_index = await self.client.get_segment_index()
                audio_segment_path = f'output/audio/output_audio_segment_{segment_index}.wav'
                os.makedirs(os.path.dirname(audio_segment_path), exist_ok=True)
                wf = wave.open(audio_segment_path, 'wb')
                wf.setnchannels(AUDIO_CHANNELS)
                wf.setsampwidth(AUDIO_SAMPLE_WIDTH)
                wf.setframerate(AUDIO_SAMPLE_RATE)
                self.client.current_audio_segment_wf = wf
                self.logger.info(f"Initialized new audio segment file: {audio_segment_path}")
            except Exception as e:
                self.logger.error(f"Failed to initialize audio segment file for segment {segment_index}: {e}")
                self.client.current_audio_segment_wf = None

    async def close_current_audio_segment(self, segment_index: int = None):
        """Close the current audio segment WAV file."""
        async with self.segment_lock:
            try:
                if segment_index is None:
                    segment_index = await self.client.get_segment_index() - 1  # Adjust for the previous segment
                if hasattr(self.client, 'current_audio_segment_wf') and self.client.current_audio_segment_wf:
                    self.client.current_audio_segment_wf.close()
                    self.logger.info(f"Closed audio segment file for segment {segment_index}.")
                    self.client.current_audio_segment_wf = None
            except Exception as e:
                self.logger.error(f"Failed to close audio segment file for segment {segment_index}: {e}")
