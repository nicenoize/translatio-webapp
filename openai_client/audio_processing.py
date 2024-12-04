# openai_client/audio_processing.py

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
from config import AUDIO_CHANNELS, AUDIO_SAMPLE_RATE, AUDIO_SAMPLE_WIDTH
from collections import deque

class AudioProcessor:
    def __init__(self, client, logger: logging.Logger):
        self.client = client
        self.logger = logger
        self.translated_audio_queue = Queue()
        self.playback_task = asyncio.create_task(self.audio_playback_handler())
        self.audio_buffer = deque(maxlen=100)

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
            with wave.open(response_audio_path, 'wb') as wf_response:
                wf_response.setnchannels(AUDIO_CHANNELS)
                wf_response.setsampwidth(AUDIO_SAMPLE_WIDTH)
                wf_response.setframerate(AUDIO_SAMPLE_RATE)
                wf_response.writeframes(decoded_audio)
            self.logger.info(f"Saved raw audio response to {response_audio_path}")

            # Enqueue audio for playback only once
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
        """Handle playback of translated audio."""
        while self.client.running:
            try:
                audio_data = await self.translated_audio_queue.get()
                await self.play_audio(audio_data)
                self.translated_audio_queue.task_done()
            except Exception as e:
                self.logger.error(f"Error in audio_playback_handler: {e}")

    async def play_audio(self, audio_data: bytes):
        """Play audio using simpleaudio and write to both output WAV and current audio segment WAV."""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            play_obj = sa.play_buffer(audio_array, AUDIO_CHANNELS, AUDIO_SAMPLE_WIDTH, AUDIO_SAMPLE_RATE)
            await asyncio.to_thread(play_obj.wait_done)
            self.logger.debug("Played translated audio chunk.")

            # Write to output WAV
            if self.client.output_wav:
                self.client.output_wav.writeframes(audio_data)
                self.logger.debug("Written translated audio chunk to output WAV file.")

            # Write to current audio segment WAV
            if self.client.current_audio_segment_wf:
                self.client.current_audio_segment_wf.writeframes(audio_data)
                self.logger.debug("Written translated audio chunk to current audio segment WAV file.")

        except Exception as e:
            self.logger.error(f"Error playing audio: {e}")
