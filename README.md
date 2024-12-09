
# Low Latency Translation

This application aims to provide a low-latency translation service, leveraging OpenAI's Realtime API.

# Setup & Execution

1. Install Requirements:

```pip install -r requirements.txt```

2. Use run_all.sh to fully run the application:
```bash run_all.sh```

3. Use run_pipes to just initialize the pipes for debugging
```bash run_pipes.sh```

# Application
After executing the script, you can access two sites:

- **[localhost:8000](http://localhost:8000)**: Displays two streams — the original and the translated one.
- **[localhost:8080](http://localhost:8080)**: Displays the monitoring dashboard.

---

The application generates **segments** (currently set to 5 seconds, configurable in `config.py`) and processes them one by one. 

- Processed segments are stored in the `output/final/` directory.
- These files contain the **muxed results**, including:
  - Translated audio.
  - Generated subtitles.
  - The video part.


# File Structure

```
├── openai_client/                  # Core modules for audio, video, and streaming logic
│   ├── audio_processing.py         # Handles audio input, segmentation, and processing
│   ├── client.py                   # Main client that orchestrates the entire pipeline
│   ├── dashboard.py                # Dashboard for monitoring the pipeline
│   ├── muxing.py                   # Combines audio, video, and subtitles
│   ├── rtmp_streamer.py            # Streams output to an RTMP endpoint
│   ├── utils.py                    # Utility functions for processing
│   ├── video_processing.py         # Handles video segmentation and adjustments
│
├── output/                         # Directory for processed outputs
│   ├── audio/                      # Audio segment outputs
│   ├── final/                      # Final synchronized MP4 files
│   ├── logs/                       # Log files
│   ├── subtitles/                  # Generated subtitle (SRT) files
│   ├── transcripts/                # Text transcripts
│   ├── video/                      # Video segment outputs
│
├── static/                         # Static files for the dashboard
│   ├── dashboard_style.css         # CSS for the monitoring latency
│
├── templates/                      # HTML templates
│   ├── index.html                  # Simple page to show streams 
│
├── .env                            # Environment variables (API Keu)
├── config.py                       # Configuration file for global settings
├── main.py                         # Main entry point for the entire pipeline
├── processing.svg                  # Visualization of the pipeline process
├── README.md                       # Documentation
├── requirements.txt                # Dependencies for the project
├── run_all.sh                      # Script to start all pipes and processes
├── run_pipes.sh                    # Script to initialize pipes

```

