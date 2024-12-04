#!/bin/bash

# Enable strict mode
set -euo pipefail

# Check if port 8000 is in use and kill the process
if lsof -i :8000 -t >/dev/null; then
    echo "Port 8000 is in use. Attempting to free it..."
    lsof -i :8000 -t | xargs kill -9
    lsof -i :8080 -t | xargs kill -9
    echo "Port 8000 has been freed."
fi

# Define absolute paths based on your project directory
PROJECT_DIR="/Users/seebo/Documents/Uni/Masterarbeit/repo/translatio-webapp"

# Define named pipes with absolute paths
INPUT_AUDIO_PIPE="$PROJECT_DIR/input_audio_pipe"

# Define log files with absolute paths
APP_LOG="$PROJECT_DIR/output/logs/app.log"
STREAM_AUDIO_PROCESSOR_LOG="$PROJECT_DIR/output/logs/stream_audio_processor.log"
MAIN_LOG="$PROJECT_DIR/output/logs/main.log"

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

# Create required pipes
create_pipe "$INPUT_AUDIO_PIPE"

# Ensure pipes have correct permissions
chmod 666 "$INPUT_AUDIO_PIPE" 

# Clear existing log files to prevent them from becoming too large
truncate -s 0 "$APP_LOG" "$STREAM_AUDIO_PROCESSOR_LOG" "$MAIN_LOG"
echo "Cleared existing log files."

# Start FFmpeg Audio Pipe
ffmpeg -y -re -fflags nobuffer -flags low_delay \
    -i "https://bintu-play.nanocosmos.de/h5live/http/stream.mp4?url=rtmp://localhost/play&stream=sNVi5-kYN1t" \
    -vn -acodec pcm_s16le -ac 1 -ar 24000 -f s16le \
    "$INPUT_AUDIO_PIPE" > "$STREAM_AUDIO_PROCESSOR_LOG" 2>&1 &

FFMPEG_AUDIO_PID=$!
echo "Started FFmpeg Audio Pipe with PID: $FFMPEG_AUDIO_PID"

# Start the Python Application
echo "Starting Python application..."
python3 "$PROJECT_DIR/main.py" > "$APP_LOG" 2>&1 &
PYTHON_PID=$!
echo "Started Python application with PID: $PYTHON_PID"

# Function to clean up background processes and named pipes
cleanup() {
    echo "Cleaning up..."
    # Kill all background processes
    kill "$FFMPEG_AUDIO_PID" "$PYTHON_PID" 2>/dev/null || true
    # Optionally, wait for them to terminate
    wait "$FFMPEG_AUDIO_PID" "$PYTHON_PID" 2>/dev/null || true
    # Remove named pipes
    rm -f "$INPUT_AUDIO_PIPE"

    lsof -i :8000 -t | xargs kill -9
    lsof -i :8080 -t | xargs kill -9
    
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
