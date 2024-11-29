import wave
from typing import Dict, Optional
import asyncio
import os

class AudioManager:
    def __init__(self):
        self.writers: Dict[int, wave.Wave_write] = {}
        self.lock = asyncio.Lock()

    async def initialize_segment(self, segment_index: int):
        async with self.lock:
            if segment_index in self.writers:
                return

            audio_path = f'output/audio/output_audio_segment_{segment_index}.wav'
            try:
                wf = wave.open(audio_path, 'wb')
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                self.writers[segment_index] = wf
            except Exception as e:
                raise RuntimeError(f"Failed to initialize audio segment {segment_index}: {e}")

    async def write_audio(self, segment_index: int, audio_data: bytes):
        async with self.lock:
            if segment_index not in self.writers:
                await self.initialize_segment(segment_index)
            self.writers[segment_index].writeframes(audio_data)

    async def close_segment(self, segment_index: int) -> Optional[str]:
        async with self.lock:
            if segment_index in self.writers:
                try:
                    self.writers[segment_index].close()
                    audio_path = f'output/audio/output_audio_segment_{segment_index}.wav'
                    if os.path.exists(audio_path):
                        return audio_path
                finally:
                    self.writers.pop(segment_index, None)
        return None

    async def cleanup(self):
        async with self.lock:
            for writer in self.writers.values():
                try:
                    writer.close()
                except Exception:
                    pass
            self.writers.clear()