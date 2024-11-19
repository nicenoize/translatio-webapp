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
from contextlib import asynccontextmanager

from stream_processor.openai_client import OpenAIClient 

load_dotenv()

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Initialize a logger for main.py
main_logger = logging.getLogger("main")
main_logger.setLevel(logging.DEBUG)
main_logger.addHandler(logging.StreamHandler())
main_logger.addHandler(logging.FileHandler("application.log"))

# Global reference to the client
client = None
should_exit = False

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
        
    translated_audio_pipe = 'translated_audio_pipe'
    input_audio_pipe = 'input_audio_pipe'
    output_rtmp_url = "rtmp://sNVi5-egEGF.bintu-vtrans.nanocosmos.de/live"

    # Initialize OpenAIClient
    client = OpenAIClient(
        api_key=api_key,
        translated_audio_pipe=translated_audio_pipe
    )
    
    # Start the client
    asyncio.create_task(client.run_client())

    # Yield control to the application
    try:
        yield
    finally:
        # Disconnect the client on shutdown
        await client.disconnect()

    # Optional: Remove named pipes on shutdown
    try:
        os.remove('input_audio_pipe')
        os.remove('translated_audio_pipe')
        main_logger.info("Named pipes removed.")
    except Exception as e:
        main_logger.error(f"Error removing named pipes: {e}")

app = FastAPI(lifespan=lifespan)

@app.websocket("/ws/translations")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Register the WebSocket client
    try:
        client_id = await client.register_websocket(websocket)
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
        await client.unregister_websocket(client_id)

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=5000,
        log_level="info",
        reload=False  # Disable reload to prevent multiple client instances
    )
