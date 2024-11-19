#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to create named pipes if they don't exist
create_named_pipes() {
    PIPE1="input_audio_pipe"
    PIPE2="translated_audio_pipe"

    if [[ ! -p $PIPE1 ]]; then
        mkfifo $PIPE1
        echo "Created named pipe: $PIPE1"
    else
        echo "Named pipe already exists: $PIPE1"
    fi

    if [[ ! -p $PIPE2 ]]; then
        mkfifo $PIPE2
        echo "Created named pipe: $PIPE2"
    else
        echo "Named pipe already exists: $PIPE2"
    fi
}

# Function to start FFmpeg Process 1 (Audio Extraction)
start_ffmpeg_process1() {
    ffmpeg -i "https://bintu-play.nanocosmos.de/h5live/http/stream.mp4?url=rtmp://localhost/play&stream=sNVi5-kYN1t" \
           -vn \
           -acodec pcm_s16le \
           -ac 1 \
           -ar 24000 \
           -f s16le \
           input_audio_pipe &
    FFMPEG1_PID=$!
    echo "FFmpeg Process 1 PID: $FFMPEG1_PID"
}

# Function to start FFmpeg Process 2 (Streaming with Subtitles)
start_ffmpeg_process2() {
    ffmpeg -loglevel debug \
           -f s16le \
           -ar 24000 \
           -ac 1 \
           -i translated_audio_pipe \
           -i "https://bintu-play.nanocosmos.de/h5live/http/stream.mp4?url=rtmp://localhost/play&stream=sNVi5-kYN1t" \
           -vf subtitles=output/subtitles/subtitles.srt \
           -c:v libx264 \
           -c:a aac \
           -b:a 128k \
           -f flv \
           -flags -global_header \
           -fflags nobuffer \
           -flush_packets 0 \
           -shortest \
           rtmp://sNVi5-egEGF.bintu-vtrans.nanocosmos.de/live \
           &
    FFMPEG2_PID=$!
    echo "FFmpeg Process 2 PID: $FFMPEG2_PID"
}

# Function to start the Python application
start_python_app() {
    # Use 'python3' to ensure the correct interpreter is used
    python3 main.py &
    PYTHON_PID=$!
    echo "Python application PID: $PYTHON_PID"
}

# Function to handle termination and cleanup
cleanup() {
    echo "Terminating all processes..."
    # Kill all background processes using their PIDs
    kill $FFMPEG1_PID $FFMPEG2_PID $PYTHON_PID 2>/dev/null || true
    # Wait for processes to terminate
    wait $FFMPEG1_PID $FFMPEG2_PID $PYTHON_PID 2>/dev/null || true
    # Remove named pipes
    rm -f input_audio_pipe translated_audio_pipe
    echo "Named pipes removed."
    echo "Cleanup complete. Exiting."
    exit 0
}

# Trap SIGINT and SIGTERM to execute cleanup
trap cleanup SIGINT SIGTERM

# Set the script to run in its own process group
# This allows us to send signals to all child processes
# The 'setsid' command starts the script in a new session
# Alternatively, use 'set -m' to enable job control
# Here, we'll capture the current process group ID
GROUP_ID=$(ps -o pgid= $$ | tr -d ' ')

# Create named pipes
create_named_pipes

# Start the Python application first to ensure it reads from 'input_audio_pipe' before FFmpeg writes to it
start_python_app

# Start FFmpeg Process 1 (Audio Extraction)
start_ffmpeg_process1

# Start FFmpeg Process 2 (Streaming with Subtitles)
start_ffmpeg_process2

# Wait for all background processes
wait
