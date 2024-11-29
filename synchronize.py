import asyncio
import aiofiles
import os
import wave
import subprocess
import logging
from logging.handlers import RotatingFileHandler
import srt
import datetime
from typing import List, Optional

class Synchronizer:
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Synchronizer initialized.")

        # Directories
        self.video_dir = 'output/video'
        self.audio_dir = 'output/audio'
        self.subtitle_dir = 'output/subtitles'
        self.final_dir = 'output/final'

        # Ensure directories exist
        self.create_directories()

    def setup_logging(self):
        """Setup logging with RotatingFileHandler."""
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        # Create rotating file handler for the synchronizer log
        os.makedirs('output/logs', exist_ok=True)
        sync_handler = RotatingFileHandler("output/logs/synchronizer.log", maxBytes=5*1024*1024, backupCount=5)
        sync_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        sync_handler.setFormatter(sync_formatter)
        logger.addHandler(sync_handler)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    def create_directories(self):
        """Create necessary directories for outputs."""
        directories = [
            self.video_dir,
            self.audio_dir,
            self.subtitle_dir,
            self.final_dir
        ]
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
                self.logger.debug(f"Ensured directory exists: {directory}")
            except Exception as e:
                self.logger.error(f"Failed to create directory {directory}: {e}", exc_info=True)

    def get_segment_indices(self) -> List[int]:
        """Retrieve sorted list of segment indices based on available video files."""
        video_files = [f for f in os.listdir(self.video_dir) if f.startswith('output_video_segment_') and f.endswith('.mp4')]
        indices = []
        for file in video_files:
            try:
                index = int(file.replace('output_video_segment_', '').replace('.mp4', ''))
                indices.append(index)
            except ValueError:
                self.logger.warning(f"Unexpected video file format: {file}")
        indices.sort()
        return indices

    async def synchronize_segments(self):
        """Synchronize video, audio, and subtitle segments."""
        segment_indices = self.get_segment_indices()
        if not segment_indices:
            self.logger.warning("No video segments found for synchronization.")
            return

        for index in segment_indices:
            self.logger.info(f"Synchronizing segment {index}.")

            video_path = os.path.join(self.video_dir, f'output_video_segment_{index}.mp4')
            audio_path = os.path.join(self.audio_dir, f'output_audio_segment_{index}.wav')
            subtitle_path = os.path.join(self.subtitle_dir, f'subtitles_segment_{index}.srt')
            final_output_path = os.path.join(self.final_dir, f'output_final_video_segment_{index}.mp4')

            # Check if all required files exist
            missing_files = []
            for path in [video_path, audio_path, subtitle_path]:
                if not os.path.exists(path):
                    missing_files.append(path)
            if missing_files:
                self.logger.error(f"Missing files for segment {index}: {missing_files}. Skipping this segment.")
                continue

            # Verify audio file integrity
            if not self.is_valid_wav(audio_path):
                self.logger.error(f"Audio file {audio_path} is invalid or empty. Skipping segment {index}.")
                continue

            # Adjust subtitle timings
            adjusted_subtitle_path = await self.adjust_subtitle_timings(subtitle_path, video_path, index)
            if not adjusted_subtitle_path:
                self.logger.error(f"Failed to adjust subtitles for segment {index}. Skipping.")
                continue

            # Overlay subtitles onto video
            temp_video_with_subs = os.path.join(self.video_dir, f'output_video_segment_{index}_with_subs.mp4')
            success = self.overlay_subtitles(video_path, adjusted_subtitle_path, temp_video_with_subs)
            if not success:
                self.logger.error(f"Failed to overlay subtitles for segment {index}. Skipping.")
                continue

            # Mux video with translated audio
            success = self.mux_video_audio(temp_video_with_subs, audio_path, final_output_path, index)
            if not success:
                self.logger.error(f"Failed to mux video and audio for segment {index}. Skipping.")
                continue

            # Clean up temporary files
            self.cleanup_files([temp_video_with_subs])

            self.logger.info(f"Segment {index} successfully synchronized and muxed.")

    def is_valid_wav(self, wav_path: str) -> bool:
        """Check if the WAV file is valid and contains audio data."""
        try:
            with wave.open(wav_path, 'rb') as wf:
                frames = wf.getnframes()
                if frames == 0:
                    self.logger.warning(f"WAV file {wav_path} contains no frames.")
                    return False
                return True
        except wave.Error as e:
            self.logger.error(f"Error reading WAV file {wav_path}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error while validating WAV file {wav_path}: {e}", exc_info=True)
            return False

    async def adjust_subtitle_timings(self, subtitle_path: str, video_path: str, segment_index: int) -> Optional[str]:
        """
        Adjust subtitle timings to match the video's timeline.
        Returns the path to the adjusted subtitle file.
        """
        try:
            # Get video duration
            video_duration = self.get_video_duration(video_path)
            if video_duration is None:
                self.logger.error(f"Cannot determine duration for video {video_path}.")
                return None

            # Read subtitle file
            async with aiofiles.open(subtitle_path, 'r', encoding='utf-8') as f:
                subtitle_content = await f.read()
            subtitles = list(srt.parse(subtitle_content))

            # Adjust timings (assuming subtitles are relative to the start of the segment)
            # If subtitles have absolute timings, additional logic is needed
            # Here, we assume they are relative

            # Calculate start time based on segment index and duration
            segment_duration = self.get_video_segment_duration(video_path)
            if segment_duration is None:
                self.logger.error(f"Cannot determine segment duration for video {video_path}.")
                return None

            # Calculate the absolute start time
            absolute_start_time = (segment_index - 1) * self.get_video_segment_duration(video_path)

            # Adjust subtitle timings
            for subtitle in subtitles:
                subtitle.start += datetime.timedelta(seconds=absolute_start_time)
                subtitle.end += datetime.timedelta(seconds=absolute_start_time)

            # Write adjusted subtitles to a new file
            adjusted_subtitle_path = os.path.join(self.subtitle_dir, f'subtitles_segment_{segment_index}_adjusted.srt')
            async with aiofiles.open(adjusted_subtitle_path, 'w', encoding='utf-8') as f:
                await f.write(srt.compose(subtitles))

            self.logger.info(f"Adjusted subtitles saved to {adjusted_subtitle_path}.")
            return adjusted_subtitle_path

        except Exception as e:
            self.logger.error(f"Error adjusting subtitle timings for segment {segment_index}: {e}", exc_info=True)
            return None

    def get_video_duration(self, video_path: str) -> Optional[float]:
        """Retrieve the duration of the video in seconds."""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries',
                'format=duration',
                '-of',
                'default=noprint_wrappers=1:nokey=1',
                video_path
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                self.logger.error(f"FFprobe error for {video_path}: {result.stderr}")
                return None
            duration = float(result.stdout.strip())
            self.logger.debug(f"Video {video_path} duration: {duration} seconds.")
            return duration
        except Exception as e:
            self.logger.error(f"Error getting video duration for {video_path}: {e}", exc_info=True)
            return None

    def get_video_segment_duration(self, video_path: str) -> Optional[float]:
        """
        Assuming all video segments have the same duration.
        Alternatively, calculate based on frame rate and number of frames.
        """
        return self.get_video_duration(video_path)

    def overlay_subtitles(self, video_path: str, subtitles_path: str, output_path: str) -> bool:
        """Overlay subtitles onto the video using FFmpeg."""
        try:
            ffmpeg_command = [
                'ffmpeg',
                '-y',
                '-i', video_path,
                '-vf', f"subtitles={subtitles_path}:force_style='FontSize=24,PrimaryColour=&HFFFFFF&'",
                '-c:v', 'libx264',  # Re-encode video to ensure compatibility
                '-c:a', 'copy',      # Copy the audio stream without re-encoding
                output_path
            ]
            self.logger.info(f"Overlaying subtitles onto video {video_path}...")
            process = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if process.returncode == 0:
                self.logger.info(f"Successfully overlaid subtitles onto {output_path}")
                return True
            else:
                self.logger.error(f"FFmpeg subtitle overlay failed for {video_path}.")
                self.logger.error(f"FFmpeg stderr: {process.stderr}")
                return False
        except Exception as e:
            self.logger.error(f"Error during FFmpeg subtitle overlay: {e}", exc_info=True)
            return False

    def mux_video_audio(self, video_path: str, audio_path: str, final_output_path: str, segment_index: int) -> bool:
        """Mux video with translated audio using FFmpeg."""
        try:
            # Check if audio needs padding to match video duration
            video_duration = self.get_video_duration(video_path)
            audio_duration = self.get_audio_duration(audio_path)

            if video_duration is None or audio_duration is None:
                self.logger.error(f"Cannot determine durations for muxing segment {segment_index}.")
                return False

            required_duration = video_duration
            pad_duration = required_duration - audio_duration

            if pad_duration > 0:
                self.logger.debug(f"Padding audio with {pad_duration:.3f} seconds of silence for segment {segment_index}.")
                padded_audio_path = os.path.join(self.audio_dir, f'output_audio_segment_{segment_index}_padded.wav')
                self.generate_silence_audio(padded_audio_path, pad_duration, audio_path)
                audio_input_path = padded_audio_path
            else:
                audio_input_path = audio_path

            # FFmpeg command to mux video with audio
            ffmpeg_mux_command = [
                'ffmpeg',
                '-y',
                '-i', video_path,
                '-i', audio_input_path,
                '-c:v', 'copy',      # Copy video stream without re-encoding
                '-c:a', 'aac',        # Encode audio to AAC
                '-strict', 'experimental',
                final_output_path
            ]
            self.logger.info(f"Muxing video {video_path} with audio {audio_input_path} into {final_output_path}...")
            process = subprocess.run(ffmpeg_mux_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if process.returncode == 0:
                self.logger.info(f"Successfully muxed into {final_output_path}")
                return True
            else:
                self.logger.error(f"FFmpeg muxing failed for segment {segment_index}.")
                self.logger.error(f"FFmpeg stderr: {process.stderr}")
                return False
        except Exception as e:
            self.logger.error(f"Error during FFmpeg muxing for segment {segment_index}: {e}", exc_info=True)
            return False

    def get_audio_duration(self, audio_path: str) -> Optional[float]:
        """Retrieve the duration of the audio in seconds."""
        try:
            with wave.open(audio_path, 'rb') as wf:
                frames = wf.getnframes()
                framerate = wf.getframerate()
                duration = frames / float(framerate)
                self.logger.debug(f"Audio {audio_path} duration: {duration} seconds.")
                return duration
        except wave.Error as e:
            self.logger.error(f"Wave error for {audio_path}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error getting audio duration for {audio_path}: {e}", exc_info=True)
            return None

    def generate_silence_audio(self, silence_path: str, duration: float, original_audio_path: str):
        """Generate a silent audio file of specified duration."""
        try:
            # Get original audio's framerate
            with wave.open(original_audio_path, 'rb') as wf:
                framerate = wf.getframerate()

            ffmpeg_command = [
                'ffmpeg',
                '-y',
                '-f', 'lavfi',
                '-i', f'anullsrc=channel_layout=mono:sample_rate={framerate}',
                '-t', f'{duration:.3f}',
                '-c:a', 'pcm_s16le',
                silence_path
            ]
            self.logger.debug(f"Generating {duration} seconds of silence at {silence_path}...")
            process = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if process.returncode == 0:
                self.logger.debug(f"Successfully generated silence audio at {silence_path}.")
            else:
                self.logger.error(f"FFmpeg failed to generate silence audio for duration {duration}.")
                self.logger.error(f"FFmpeg stderr: {process.stderr}")
        except Exception as e:
            self.logger.error(f"Error generating silence audio at {silence_path}: {e}", exc_info=True)

    def cleanup_files(self, file_paths: List[str]):
        """Remove temporary files."""
        for path in file_paths:
            try:
                os.remove(path)
                self.logger.debug(f"Removed temporary file: {path}")
            except Exception as e:
                self.logger.warning(f"Failed to remove temporary file {path}: {e}")

async def main():
    synchronizer = Synchronizer()
    await synchronizer.synchronize_segments()

if __name__ == "__main__":
    asyncio.run(main())
