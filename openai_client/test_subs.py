import asyncio
from openai_client.client import OpenAIClient

async def test_subtitle_extraction():
    client = OpenAIClient(api_key='dummy_api_key')
    client.segment_index = 1  # Initialize segment index

    muxing_job = {
        'segment_index': client.segment_index,
        'video': 'output/video/output_video_segment_1.mp4',
        'audio': 'output/audio/output_audio_segment_1.wav',
        'subtitles': client.SUBTITLE_PATH,
        'output': 'output/final/output_final_segment_1.mp4'
    }

    await client.muxer.enqueue_muxing_job(muxing_job)
    # Allow some time for the muxing process to handle the job
    await asyncio.sleep(2)

    # Check if the temporary subtitle file was created
    temp_subtitles_path = 'output/subtitles/subtitles_segment_1.vtt'
    if os.path.exists(temp_subtitles_path):
        print(f"Temporary subtitle file created at {temp_subtitles_path}")
    else:
        print("Failed to create temporary subtitle file for segment 1.")

asyncio.run(test_subtitle_extraction())
