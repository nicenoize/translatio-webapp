from dataclasses import dataclass
from typing import List, Optional
import datetime
import asyncio
import aiofiles
import srt

@dataclass
class SubtitleEntry:
    index: int
    start_time: float
    end_time: float
    text: str
    segment_index: int

class SubtitleManager:
    def __init__(self):
        self.entries: List[SubtitleEntry] = []
        self.current_index = 1
        self.lock = asyncio.Lock()

    async def add_subtitle(self, text: str, start_time: float, end_time: float, segment_index: int):
        async with self.lock:
            entry = SubtitleEntry(
                index=self.current_index,
                start_time=start_time,
                end_time=end_time,
                text=text,
                segment_index=segment_index
            )
            self.entries.append(entry)
            self.current_index += 1
            await self._write_to_srt(entry)

    async def _write_to_srt(self, entry: SubtitleEntry):
        srt_path = f'output/subtitles/subtitles_segment_{entry.segment_index}.srt'
        subtitle = srt.Subtitle(
            index=entry.index,
            start=datetime.timedelta(seconds=entry.start_time),
            end=datetime.timedelta(seconds=entry.end_time),
            content=entry.text
        )
        
        async with aiofiles.open(srt_path, 'a', encoding='utf-8') as f:
            await f.write(f"{srt.compose([subtitle])}\n")

    def get_subtitles_for_segment(self, segment_index: int) -> List[SubtitleEntry]:
        return [entry for entry in self.entries if entry.segment_index == segment_index]

    async def clear_segment(self, segment_index: int):
        async with self.lock:
            self.entries = [entry for entry in self.entries if entry.segment_index != segment_index]
            # Optionally clear the SRT file
            srt_path = f'output/subtitles/subtitles_segment_{segment_index}.srt'
            try:
                async with aiofiles.open(srt_path, 'w', encoding='utf-8') as f:
                    await f.write('')
            except Exception:
                pass  # Ignore if file doesn't exist