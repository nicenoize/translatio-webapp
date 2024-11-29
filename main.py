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
from logging.handlers import RotatingFileHandler


load_dotenv()

from stream_processor.openai_client import OpenAIClient  # Ensure correct import path

app = FastAPI()


# Stream URLs
STREAM_URLS = {
    "original": "https://demo.nanocosmos.de/nanoplayer/embed/1.3.3/nanoplayer.html?group.id=32ed6d97-b58f-40c0-ac4c-349e6cdb3777&options.adaption.rule=deviationOfMean2&startIndex=0&playback.latencyControlMode=balancedadaptive",
    "translated": "https://demo.nanocosmos.de/nanoplayer/embed/1.3.3/nanoplayer.html?group.id=52e4d770-7c2d-4615-b1c7-d51bc34350c4&options.adaption.rule=deviationOfMean2&startIndex=0&playback.latencyControlMode=balancedadaptive"
}

async def main():
    # Get API key from environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Get event loop
    loop = asyncio.get_event_loop()
    
    # Initialize client
    client = OpenAIClient(api_key, loop)
    
    try:
        # Run the client
        await client.run()
    except KeyboardInterrupt:
        # Handle graceful shutdown
        await client.shutdown(signal.SIGINT)
    finally:
        # Ensure cleanup
        await client.disconnect(shutdown=True)

if __name__ == "__main__":
    # Run the application
    asyncio.run(main())
