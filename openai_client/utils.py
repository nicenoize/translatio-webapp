# openai_client/utils.py

import datetime

def format_timestamp_vtt(seconds: float) -> str:
    """
    Format seconds to WebVTT timestamp format HH:MM:SS.mmm.
    """
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    milliseconds = int((td.total_seconds() - total_seconds) * 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"
