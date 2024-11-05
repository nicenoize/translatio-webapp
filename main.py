from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)