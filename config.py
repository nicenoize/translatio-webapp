# config.py

import os

# WebSocket API URL
API_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"

# Audio Configuration
AUDIO_SAMPLE_RATE = 24000  # 24kHz
AUDIO_CHANNELS = 1         # Mono
AUDIO_SAMPLE_WIDTH = 2     # 16-bit PCM

# Paths
SUBTITLE_PATH = 'output/subtitles/subtitles.srt'
OUTPUT_WAV_PATH = 'output/audio/output_audio.wav'
STREAM_URL = 'https://bintu-play.nanocosmos.de/h5live/http/stream.mp4?url=rtmp://localhost/play&stream=sNVi5-kYN1t'
RTMP_PLAYOUT_URL = 'rtmp://bintu-vtrans.nanocosmos.de/live/sNVi5-egEGF'
INPUT_AUDIO_PIPE = os.path.abspath('input_audio_pipe')  # Absolute path for the named pipe

# Muxing
MUXING_QUEUE_MAXSIZE = 100
DASHBOARD_PORT = 8080

# Voice Identifiers
MALE_VOICE_ID = "shimmer"
FEMALE_VOICE_ID = "coral"
DEFAULT_VOICE_ID = "ash"

# Other Configurations
SEGMENT_DURATION = 5  # seconds
MAX_BACKOFF = 60       # seconds for reconnection attempts
