import asyncio
import wave
import logging

class AudioSaver:
    def __init__(self, output_wav_path: str, sample_width: int = 2, frame_rate: int = 24000, channels: int = 1):
        self.output_wav_path = output_wav_path
        self.sample_width = sample_width
        self.frame_rate = frame_rate
        self.channels = channels
        self.queue = asyncio.Queue()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.writer_task = asyncio.create_task(self.writer())
        self.wave_file = None

    async def writer(self):
        """Async task to write PCM data to WAV file."""
        try:
            self.wave_file = wave.open(self.output_wav_path, 'wb')
            self.wave_file.setnchannels(self.channels)
            self.wave_file.setsampwidth(self.sample_width)
            self.wave_file.setframerate(self.frame_rate)
            self.logger.info(f"Opened WAV file for writing: {self.output_wav_path}")

            while True:
                audio_data = await self.queue.get()
                if audio_data is None:
                    self.logger.info("AudioSaver received shutdown signal.")
                    break
                self.wave_file.writeframes(audio_data)
                self.logger.debug(f"Wrote {len(audio_data)} bytes to WAV file.")
                self.queue.task_done()
        except Exception as e:
            self.logger.error(f"Error in AudioSaver.writer: {e}", exc_info=True)
        finally:
            if self.wave_file:
                self.wave_file.close()
                self.logger.info(f"Closed WAV file: {self.output_wav_path}")

    async def save_audio_chunk(self, audio_data: bytes):
        """Enqueue audio data to be saved."""
        await self.queue.put(audio_data)
        self.logger.debug(f"Enqueued audio chunk of size {len(audio_data)} bytes to AudioSaver.")

    async def shutdown(self):
        """Shutdown the AudioSaver."""
        await self.queue.put(None)
        await self.writer_task
