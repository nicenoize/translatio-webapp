# stream_processor/ffmpeg_manager.py

import subprocess
import logging
import asyncio
import os
import signal
import time

class FFmpegManager:
    def __init__(self, input_stream_url: str, output_rtmp_url: str, input_audio_pipe: str, translated_audio_pipe: str, zmq_address: str):
        self.input_stream_url = input_stream_url
        self.output_rtmp_url = output_rtmp_url
        self.input_audio_pipe = input_audio_pipe
        self.translated_audio_pipe = translated_audio_pipe
        self.zmq_address = zmq_address.replace('tcp://', '')
        self.ffmpeg_process = None
        self.logger = logging.getLogger(__name__)
        self.running = True

        # Create named pipes if they don't exist
        for pipe in [self.input_audio_pipe, self.translated_audio_pipe]:
            try:
                os.mkfifo(pipe)
            except FileExistsError:
                pass

    def start_ffmpeg_process(self):
        """Start the FFmpeg process with input stream"""
        command = [
            'ffmpeg',
            '-loglevel', 'debug',
            '-i', self.input_stream_url,  # Input 0: video + original audio

            # Output the input audio to a named pipe for the AudioManager
            '-map', '0:a',
            '-f', 's16le',
            '-ac', '1',
            '-ar', '16000',
            self.input_audio_pipe,  # Write audio to input_audio_pipe

            # Read the translated audio from a named pipe
            '-f', 's16le',
            '-ar', '16000',
            '-ac', '1',
            '-i', self.translated_audio_pipe,   # Input 1: translated audio

            # Map video from Input 0
            '-map', '0:v',

            # Map translated audio from Input 1
            '-map', '1:a',

            # Apply subtitles via ZMQ
            '-vf', f"drawtext=text='':fontcolor=white:fontsize=24:x=(w-text_w)/2:y=h-50:box=1:boxcolor=black@0.5:reload=1:zeromq=bind_address=tcp\\://{self.zmq_address}",

            # Output settings
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-f', 'flv',
            self.output_rtmp_url
        ]

        self.logger.info(f"Starting FFmpeg with command: {' '.join(command)}")

        try:
            self.ffmpeg_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                bufsize=10**8
            )

            # Wait a moment and check if process is still running
            time.sleep(2)
            if self.ffmpeg_process.poll() is not None:
                stderr = self.ffmpeg_process.stderr.read().decode()
                self.logger.error(f"FFmpeg failed to start. Error: {stderr}")
                raise RuntimeError("FFmpeg failed to start")

            self.logger.info("FFmpeg process started successfully")

        except Exception as e:
            self.logger.error(f"Failed to start FFmpeg: {e}")
            raise

    async def monitor_process(self):
        """Monitor FFmpeg process and log its output"""
        self.logger.info("Starting FFmpeg process monitoring")
        try:
            while self.running and self.ffmpeg_process and self.ffmpeg_process.poll() is None:
                try:
                    line = await asyncio.get_event_loop().run_in_executor(
                        None, self.ffmpeg_process.stderr.readline
                    )
                    if line:
                        message = line.decode().strip()
                        if "error" in message.lower():
                            self.logger.error(f"FFmpeg: {message}")
                        else:
                            self.logger.debug(f"FFmpeg: {message}")
                except Exception as e:
                    self.logger.error(f"Error reading FFmpeg output: {e}")
                    break

            if self.ffmpeg_process.poll() is not None:
                self.logger.error(f"FFmpeg process exited with code: {self.ffmpeg_process.poll()}")
                stderr_output = self.ffmpeg_process.stderr.read().decode()
                self.logger.error(f"FFmpeg stderr: {stderr_output}")

        except Exception as e:
            self.logger.error(f"Error monitoring FFmpeg process: {e}")
        finally:
            self.logger.info("FFmpeg process monitoring ended")

    def stop_ffmpeg_process(self):
        """Stop the FFmpeg process"""
        self.running = False
        if self.ffmpeg_process:
            try:
                # Try graceful shutdown first
                self.ffmpeg_process.terminate()
                try:
                    self.ffmpeg_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if necessary
                    self.ffmpeg_process.kill()
                    self.ffmpeg_process.wait()
            except Exception as e:
                self.logger.error(f"Error stopping FFmpeg process: {e}")
            finally:
                self.logger.info("FFmpeg process stopped")

        # Clean up named pipes
        for pipe in [self.input_audio_pipe, self.translated_audio_pipe]:
            try:
                if os.path.exists(pipe):
                    os.unlink(pipe)
            except Exception as e:
                self.logger.error(f"Error cleaning up named pipe {pipe}: {e}")
