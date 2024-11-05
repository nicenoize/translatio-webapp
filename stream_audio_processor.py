import asyncio
import websockets
import json
import base64
import subprocess
import os
from datetime import datetime
import logging
import wave
import pathlib
from typing import Optional, Dict
from dotenv import load_dotenv

load_dotenv()

class StreamAudioProcessor:
    def __init__(self, openai_api_key: str, stream_url: str, output_dir: str = "output"):
        if not openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        self.openai_api_key = openai_api_key
        self.stream_url = stream_url
        self.ws = None
        self.ffmpeg_process = None
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different types of output
        self.audio_dir = self.output_dir / "audio"
        self.subtitles_dir = self.output_dir / "subtitles"
        self.audio_dir.mkdir(exist_ok=True)
        self.subtitles_dir.mkdir(exist_ok=True)
        
        # Initialize storage for current session
        self.current_translation_audio = bytearray()
        self.current_subtitles = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"stream_processor_{datetime.now():%Y%m%d}.log")
            ]
        )
        self.logger = logging.getLogger(__name__)

    async def connect_to_openai(self) -> bool:
        """Establish WebSocket connection with OpenAI's Realtime API"""
        MAX_RETRIES = 3
        retry_count = 0
        
        while retry_count < MAX_RETRIES:
            try:
                url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
                headers = {
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "OpenAI-Beta": "realtime=v1"
                }
                
                self.ws = await websockets.connect(url, extra_headers=headers)
                self.logger.info("Connected to OpenAI Realtime API")
                
                # Initialize the session with audio configuration
                # Corrected message format as per API requirements
                await self.ws.send(json.dumps({
                    "type": "session.update",
                    "input_audio_transcription": {
                        "enabled": True,
                        "model": "whisper-1"
                    }
                }))
                
                # Create initial response with translation settings
                await self.ws.send(json.dumps({
                    "type": "response.create",
                    "modalities": ["text", "audio"],
                    "instructions": "Translate the input to English and provide both audio and text output."
                }))
                
                return True
                
            except Exception as e:
                retry_count += 1
                self.logger.error(f"Connection attempt {retry_count} failed: {str(e)}")
                if retry_count < MAX_RETRIES:
                    await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                else:
                    self.logger.error("Failed to connect after maximum retries")
                    return False

    def start_ffmpeg_stream(self) -> bool:
        """Start FFmpeg process to extract audio from stream"""
        try:
            command = [
                'ffmpeg',
                '-i', self.stream_url,
                '-vn',  # Disable video
                '-ar', '24000',  # Sample rate 24kHz
                '-ac', '1',  # Mono channel
                '-f', 's16le',  # 16-bit PCM
                '-',  # Output to pipe
                '-loglevel', 'error'  # Show only error messages
            ]
            
            self.ffmpeg_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,  # Capture stderr
                bufsize=10**8  # Increase buffer size
            )
            
            self.logger.info("FFmpeg stream started")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start FFmpeg: {str(e)}")
            return False

    def save_audio(self, audio_data: bytes, filename: str):
        """Save audio data as WAV file"""
        try:
            filepath = self.audio_dir / filename
            with wave.open(str(filepath), 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(24000)  # 24kHz
                wav_file.writeframes(audio_data)
            self.logger.info(f"Saved audio file: {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save audio file: {str(e)}")

    def save_subtitles(self, subtitles: list, filename: str):
        """Save subtitles in SRT format"""
        try:
            filepath = self.subtitles_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                for i, subtitle in enumerate(subtitles, 1):
                    f.write(f"{i}\n")
                    f.write(f"{subtitle['timestamp']}\n")
                    f.write(f"{subtitle['text']}\n\n")
            self.logger.info(f"Saved subtitles file: {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save subtitles file: {str(e)}")

    async def process_audio_chunks(self):
        """Process audio chunks and send to OpenAI"""
        CHUNK_SIZE = 4096
        buffer = bytearray()
        
        try:
            while True:
                # Read chunk from FFmpeg output
                audio_chunk = self.ffmpeg_process.stdout.read(CHUNK_SIZE)
                if not audio_chunk:
                    # Check if FFmpeg has exited
                    return_code = self.ffmpeg_process.poll()
                    if return_code is not None:
                        stderr_output = self.ffmpeg_process.stderr.read().decode()
                        self.logger.error(f"FFmpeg process exited with code {return_code}, stderr: {stderr_output}")
                        break
                    else:
                        # No data yet, continue
                        await asyncio.sleep(0.1)
                        continue

                self.logger.info(f"Received audio chunk of size {len(audio_chunk)}")

                # Optional: Write raw audio data to a temp file for debugging
                # with open('temp_audio.raw', 'ab') as temp_audio_file:
                #     temp_audio_file.write(audio_chunk)

                buffer.extend(audio_chunk)
                
                # Process buffer when it reaches sufficient size
                if len(buffer) >= CHUNK_SIZE:
                    # Encode audio chunk to base64
                    base64_audio = base64.b64encode(bytes(buffer)).decode('utf-8')
                    
                    # Send audio chunk to OpenAI
                    await self.ws.send(json.dumps({
                        "type": "input.audio",
                        "audio": base64_audio
                    }))
                    
                    buffer.clear()
                    await asyncio.sleep(0.1)  # Prevent overwhelming the API
                
        except Exception as e:
            self.logger.error(f"Error processing audio chunk: {str(e)}")
        finally:
            if buffer:
                # Process any remaining audio in buffer
                base64_audio = base64.b64encode(bytes(buffer)).decode('utf-8')
                await self.ws.send(json.dumps({
                    "type": "input.audio",
                    "audio": base64_audio
                }))

    async def handle_openai_responses(self):
        """Handle responses from OpenAI"""
        try:
            while True:
                response = await self.ws.recv()
                event = json.loads(response)
                
                # Handle different event types
                event_type = event.get("type")
                
                if event_type == "error":
                    error_info = event.get("error", {})
                    self.logger.error(f"OpenAI Error: {error_info}")
                    if error_info.get("code") == "invalid_api_key":
                        raise ValueError("Invalid API key provided")
                
                elif event_type == "transcript":
                    timestamp = datetime.now().strftime("%H:%M:%S,000")
                    self.current_subtitles.append({
                        "timestamp": timestamp,
                        "text": event.get("text", "")
                    })
                    self.logger.info(f"Transcription: {event.get('text', '')}")
                
                elif event_type == "response.audio.delta":
                    # Accumulate translated audio
                    audio_data = base64.b64decode(event.get("audio", ""))
                    self.current_translation_audio.extend(audio_data)
                
                elif event_type == "response.audio.end":
                    # Save accumulated audio
                    filename = f"translation_{self.session_id}_{len(self.current_translation_audio)}.wav"
                    self.save_audio(bytes(self.current_translation_audio), filename)
                    self.current_translation_audio = bytearray()
                
                elif event_type == "response.text.delta":
                    # Add to current subtitles
                    text = event.get("text", "")
                    if text.strip():
                        timestamp = datetime.now().strftime("%H:%M:%S,000")
                        self.current_subtitles.append({
                            "timestamp": timestamp,
                            "text": text
                        })
                        self.logger.info(f"Response text: {text}")
                
                elif event_type == "response.text.end":
                    # Save accumulated subtitles
                    filename = f"subtitles_{self.session_id}.srt"
                    self.save_subtitles(self.current_subtitles, filename)
                    self.current_subtitles = []
        
        except ValueError as ve:
            self.logger.error(f"Validation error: {str(ve)}")
            raise
        except Exception as e:
            self.logger.error(f"Error handling OpenAI responses: {str(e)}")

    async def run(self):
        """Main run method"""
        try:
            # Connect to OpenAI
            if not await self.connect_to_openai():
                return
            
            # Start FFmpeg stream
            if not self.start_ffmpeg_stream():
                return
            
            # Create tasks for processing audio and handling responses
            tasks = [
                asyncio.create_task(self.process_audio_chunks()),
                asyncio.create_task(self.handle_openai_responses())
            ]
            
            # Wait for both tasks to complete
            await asyncio.gather(*tasks)
            
        except Exception as e:
            self.logger.error(f"Error in main run method: {str(e)}")
        
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Cleanup method"""
        try:
            if self.ffmpeg_process:
                self.ffmpeg_process.terminate()
                await asyncio.sleep(1)
                self.ffmpeg_process.kill()  # Force kill if still running
                self.ffmpeg_process.wait()
            
            if self.ws:
                await self.ws.close()
            
            # Save any remaining data
            if self.current_translation_audio:
                filename = f"translation_{self.session_id}_final.wav"
                self.save_audio(bytes(self.current_translation_audio), filename)
            
            if self.current_subtitles:
                filename = f"subtitles_{self.session_id}_final.srt"
                self.save_subtitles(self.current_subtitles, filename)
                
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

# Usage example
async def main():
    # Get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    stream_url = "https://bintu-play.nanocosmos.de/h5live/http/stream.mp4?url=rtmp://localhost/play&stream=sNVi5-kYN1t"
    
    processor = StreamAudioProcessor(
        openai_api_key=api_key,
        stream_url=stream_url,
        output_dir="stream_output"
    )
    
    try:
        await processor.run()
    except KeyboardInterrupt:
        await processor.cleanup()
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        await processor.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
