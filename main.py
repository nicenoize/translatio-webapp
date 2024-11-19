# main.py

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import asyncio
from dotenv import load_dotenv
import signal
import logging

load_dotenv()

from stream_processor.stream_audio_processor import StreamAudioProcessor  # Ensure this module imports OpenAIClient correctly

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

# Global reference to the processor
processor = None
should_exit = False

# Initialize a logger for main.py
main_logger = logging.getLogger("main")
main_logger.setLevel(logging.DEBUG)
main_logger.addHandler(logging.StreamHandler())
main_logger.addHandler(logging.FileHandler("main.log"))

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

    # Initialize StreamAudioProcessor with the updated OpenAIClient
    processor = StreamAudioProcessor(
        openai_api_key=api_key,
        stream_url=stream_url,
        output_rtmp_url=output_rtmp_url
    )
    
    # Set up signal handlers
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Signal handlers are not implemented on some systems (e.g., Windows)
            pass
    
    # Start the processor's run method
    asyncio.create_task(processor.run())

@app.on_event("shutdown")
async def shutdown():
    await shutdown_event()

@app.websocket("/ws/translations")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Register the WebSocket client
    try:
        client_id = await processor.openai_client.register_websocket(websocket)
    except Exception as e:
        main_logger.error(f"Failed to register WebSocket client: {e}")
        await websocket.close()
        return
    
    try:
        while True:
            # Keep the connection alive by awaiting any messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        main_logger.info(f"WebSocket client {client_id} disconnected.")
    except Exception as e:
        main_logger.error(f"WebSocket error: {e}")
    finally:
        # Unregister the WebSocket client when the connection is closed
        await processor.openai_client.unregister_websocket(client_id)

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        reload=False  # Disable reload to prevent multiple processor instances
    )
