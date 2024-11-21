#!/bin/bash

# Enable strict mode
set -euo pipefail

# Check if port 8000 is in use and kill the process
if lsof -i :8000 -t >/dev/null; then
    echo "Port 8000 is in use. Attempting to free it..."
    lsof -i :8000 -t | xargs kill -9
    echo "Port 8000 has been freed."
fi

# Define absolute paths based on your project directory
PROJECT_DIR="/Users/seebo/Documents/Uni/Masterarbeit/repo/translatio-webapp"

# Define named pipes with absolute paths
INPUT_AUDIO_PIPE="$PROJECT_DIR/input_audio_pipe"
TRANSLATED_AUDIO_PIPE="$PROJECT_DIR/translated_audio_pipe"
VIDEO_PIPE="$PROJECT_DIR/video_pipe"

# Define log files with absolute paths
FFMPEG_AUDIO_LOG="$PROJECT_DIR/output/logs/ffmpeg_audio.log"
FFMPEG_VIDEO_LOG="$PROJECT_DIR/output/logs/ffmpeg_video.log"
FFMPEG_MIXER_LOG="$PROJECT_DIR/output/logs/ffmpeg_mixer.log"
PYTHON_LOG="$PROJECT_DIR/output/logs/python.log"

# Define output video path
OUTPUT_VIDEO="$PROJECT_DIR/output/video/output_video.mp4"

# Define subtitles path
SUBTITLES_PATH="$PROJECT_DIR/output/subtitles/subtitles.srt"

# Create output directories if they don't exist
mkdir -p "$PROJECT_DIR/output/logs"
mkdir -p "$PROJECT_DIR/output/video"
mkdir -p "$PROJECT_DIR/output/subtitles"

# Function to create named pipe if it doesn't exist
create_pipe() {
    local pipe_name="$1"
    if [[ ! -p "$pipe_name" ]]; then
        mkfifo "$pipe_name"
        echo "Created named pipe: $pipe_name"
    else
        echo "Named pipe already exists: $pipe_name"
    fi
}

# Create all required pipes
create_pipe "$INPUT_AUDIO_PIPE"
create_pipe "$TRANSLATED_AUDIO_PIPE"
create_pipe "$VIDEO_PIPE"

# Ensure pipes have correct permissions
chmod 666 "$INPUT_AUDIO_PIPE" "$TRANSLATED_AUDIO_PIPE" "$VIDEO_PIPE"

# Start FFmpeg Audio Pipe
ffmpeg -y -re -fflags nobuffer -flags low_delay \
    -i "https://bintu-play.nanocosmos.de/h5live/http/stream.mp4?url=rtmp://localhost/play&stream=sNVi5-kYN1t" \
    -vn -acodec pcm_s16le -ac 1 -ar 24000 -f s16le \
    "$INPUT_AUDIO_PIPE" > "$FFMPEG_AUDIO_LOG" 2>&1 &

FFMPEG_AUDIO_PID=$!
echo "Started FFmpeg Audio Pipe with PID: $FFMPEG_AUDIO_PID"

# Start FFmpeg Video Pipe
ffmpeg -y -re -fflags nobuffer -flags low_delay \
    -i "https://bintu-play.nanocosmos.de/h5live/http/stream.mp4?url=rtmp://localhost/play&stream=sNVi5-kYN1t" \
    -an -c:v copy -f mpegts \
    "$VIDEO_PIPE" > "$FFMPEG_VIDEO_LOG" 2>&1 &

FFMPEG_VIDEO_PID=$!
echo "Started FFmpeg Video Pipe with PID: $FFMPEG_VIDEO_PID"

# Start the Python Application
# Assuming your Python script is named `main.py` and is in the project directory
echo "Starting Python application..."
python3 "$PROJECT_DIR/main.py" > "$PYTHON_LOG" 2>&1 &
PYTHON_PID=$!
echo "Started Python application with PID: $PYTHON_PID"

# Allow some time for the Python application to open the translated_audio_pipe
sleep 2

# Check if subtitles file exists; wait until it does
echo "Waiting for subtitles file to be available..."
while [[ ! -f "$SUBTITLES_PATH" ]]; do
    echo "Subtitles file not found. Waiting..."
    sleep 1
done
echo "Subtitles file detected: $SUBTITLES_PATH"

# Start FFmpeg Mixer
ffmpeg -y \
    -f s16le -ar 24000 -ac 1 -i "$INPUT_AUDIO_PIPE" \
    -f s16le -ar 24000 -ac 1 -i "$TRANSLATED_AUDIO_PIPE" \
    -i "$VIDEO_PIPE" \
    -filter_complex "\
        [0:a]volume=0.3[a1]; \
        [1:a]volume=1.0[a2]; \
        [a1][a2]amix=inputs=2:duration=first[aout]; \
        [2:v]subtitles='$SUBTITLES_PATH'[vout] \
    " \
    -map "[vout]" \
    -map "[aout]" \
    -c:v libx264 \
    -c:a aac \
    -b:a 192k \
    -ar 24000 \
    -shortest \
    "$OUTPUT_VIDEO" > "$FFMPEG_MIXER_LOG" 2>&1 &

FFMPEG_MIXER_PID=$!
echo "Started FFmpeg Mixer with PID: $FFMPEG_MIXER_PID"

# Function to clean up background processes and named pipes
cleanup() {
    echo "Cleaning up..."
    # Kill all background processes
    kill "$FFMPEG_AUDIO_PID" "$FFMPEG_VIDEO_PID" "$FFMPEG_MIXER_PID" "$PYTHON_PID" 2>/dev/null || true
    # Optionally, wait for them to terminate
    wait "$FFMPEG_AUDIO_PID" "$FFMPEG_VIDEO_PID" "$FFMPEG_MIXER_PID" "$PYTHON_PID" 2>/dev/null || true
    # Remove named pipes
    rm -f "$INPUT_AUDIO_PIPE" "$TRANSLATED_AUDIO_PIPE" "$VIDEO_PIPE"
    echo "Cleanup complete."
}

# Trap EXIT and common termination signals to ensure cleanup
trap cleanup EXIT
trap "exit" INT TERM

# Function to monitor all processes
monitor_processes() {
    while true; do
        sleep 5
        if ! ps -p "$FFMPEG_AUDIO_PID" > /dev/null; then
            echo "FFmpeg Audio Pipe (PID: $FFMPEG_AUDIO_PID) has stopped."
            exit 1
        fi
        if ! ps -p "$FFMPEG_VIDEO_PID" > /dev/null; then
            echo "FFmpeg Video Pipe (PID: $FFMPEG_VIDEO_PID) has stopped."
            exit 1
        fi
        if ! ps -p "$FFMPEG_MIXER_PID" > /dev/null; then
            echo "FFmpeg Mixer (PID: $FFMPEG_MIXER_PID) has stopped."
            exit 1
        fi
        if ! ps -p "$PYTHON_PID" > /dev/null; then
            echo "Python application (PID: $PYTHON_PID) has stopped."
            exit 1
        fi
    done
}

# Start monitoring in the background
monitor_processes &
MONITOR_PID=$!
echo "Started process monitor with PID: $MONITOR_PID"

# Wait for all background processes
wait
