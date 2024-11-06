# main.py

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

from stream_processor.stream_audio_processor import StreamAudioProcessor

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Stream URLs
STREAM_URLS = {
    "original": "https://demo.nanocosmos.de/nanoplayer/embed/1.3.3/nanoplayer.html?group.id=32ed6d97-b58f-40c0-ac4c-349e6cdb3777&options.adaption.rule=deviationOfMean2&startIndex=0&playback.latencyControlMode=balancedadaptive",
    "translated": "https://demo.nanocosmos.de/nanoplayer/embed/1.3.3/nanoplayer.html?group.id=52e4d770-7c2d-4615-b1c7-d51bc34350c4&options.adaption.rule=deviationOfMean2&startIndex=0&playback.latencyControlMode=balancedadaptive"
}

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "stream_urls": STREAM_URLS}
    )

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import asyncio
from dotenv import load_dotenv
import signal

load_dotenv()

from stream_processor.stream_audio_processor import StreamAudioProcessor

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global reference to the processor
processor = None
should_exit = False

# Stream URLs
STREAM_URLS = {
    "original": "https://demo.nanocosmos.de/nanoplayer/embed/1.3.3/nanoplayer.html?group.id=32ed6d97-b58f-40c0-ac4c-349e6cdb3777&options.adaption.rule=deviationOfMean2&startIndex=0&playback.latencyControlMode=balancedadaptive",
    "translated": "https://demo.nanocosmos.de/nanoplayer/embed/1.3.3/nanoplayer.html?group.id=52e4d770-7c2d-4615-b1c7-d51bc34350c4&options.adaption.rule=deviationOfMean2&startIndex=0&playback.latencyControlMode=balancedadaptive"
}

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "stream_urls": STREAM_URLS}
    )

async def shutdown_event():
    global processor, should_exit
    should_exit = True
    if processor:
        await processor.cleanup()

def signal_handler():
    asyncio.create_task(shutdown_event())

@app.on_event("startup")
async def startup_event():
    global processor
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
        
    stream_url = "https://bintu-play.nanocosmos.de/h5live/http/stream.mp4?url=rtmp://localhost/play&stream=sNVi5-kYN1t"
    output_rtmp_url = "rtmp://sNVi5-egEGF.bintu-vtrans.nanocosmos.de/live"

    processor = StreamAudioProcessor(
        openai_api_key=api_key,
        stream_url=stream_url,
        output_rtmp_url=output_rtmp_url
    )
    
    # Set up signal handlers
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)
    
    # Start the processor in the background
    asyncio.create_task(processor.run())

@app.on_event("shutdown")
async def shutdown():
    await shutdown_event()

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000,
        log_level="debug",
        reload=False  # Disable reload to prevent multiple processor instances
    )
