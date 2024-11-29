from dataclasses import dataclass, field
from typing import Optional, Dict, List
import time

@dataclass
class EventData:
    timestamp: float
    audio_data: bytes
    translated_text: Optional[str] = None
    translated_audio: Optional[bytes] = None
    processing_complete: bool = False
    duration: Optional[float] = None

class EventTracker:
    def __init__(self):
        self.event_map: Dict[str, EventData] = {}
        self.pending_events: List[str] = []
        self.last_processed_timestamp: Optional[float] = None

    def register_input_event(self, event_id: str, timestamp: float, audio_data: bytes):
        self.event_map[event_id] = EventData(timestamp=timestamp, audio_data=audio_data)
        self.pending_events.append(event_id)

    def update_event(self, event_id: str, **updates):
        if event_id in self.event_map:
            for key, value in updates.items():
                setattr(self.event_map[event_id], key, value)
            
            if updates.get('processing_complete', False):
                if event_id in self.pending_events:
                    self.pending_events.remove(event_id)
                self.last_processed_timestamp = self.event_map[event_id].timestamp

    def is_event_complete(self, event_id: str) -> bool:
        if event_id not in self.event_map:
            return False
        event = self.event_map[event_id]
        return bool(event.translated_text and event.translated_audio and event.processing_complete)

    def get_pending_events(self) -> List[str]:
        return sorted(
            self.pending_events,
            key=lambda x: self.event_map[x].timestamp if x in self.event_map else float('inf')
        )

    def get_event_data(self, event_id: str) -> Optional[EventData]:
        return self.event_map.get(event_id)

    def clear_old_events(self, threshold: float = 300):  # 5 minutes
        current_time = time.time()
        old_events = [
            event_id for event_id, data in self.event_map.items()
            if current_time - data.timestamp > threshold
        ]
        for event_id in old_events:
            self.event_map.pop(event_id)
            if event_id in self.pending_events:
                self.pending_events.remove(event_id)