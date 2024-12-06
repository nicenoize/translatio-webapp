# openai_client/utils.py

import datetime

def format_timestamp_srt(seconds: float) -> str:
    """
    Format seconds to SRT timestamp format HH:MM:SS,mmm.
    
    Args:
        seconds (float): Time in seconds.
    
    Returns:
        str: Formatted timestamp string in SRT format.
    """
    try:
        td = datetime.timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        milliseconds = int(round((td.total_seconds() - total_seconds) * 1000))
        
        # Handle cases where rounding milliseconds could push it to 1000
        if milliseconds == 1000:
            total_seconds += 1
            milliseconds = 0
        
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        
        return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"
    
    except Exception as e:
        raise ValueError(f"Invalid input for timestamp formatting: {e}")

