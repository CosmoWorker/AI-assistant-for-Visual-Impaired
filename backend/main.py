# backend/main.py
import sys
import os

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
import time
import base64
from PIL import Image
from io import BytesIO
import cv2
from BLIP_CAM import CaptionGenerator, load_models, get_gpu_usage, setup_logging

app = FastAPI()
logger = setup_logging()

BLIP_CAPTION_GENERATOR = None

def load_blip_model():
    global BLIP_CAPTION_GENERATOR
    logger.info("Loading BLIP model...")
    # Assuming your model file is within the BLIP_CAM directory
    blip_processor, blip_model, device = load_models()
    if None in (blip_processor, blip_model):
        logger.error("Failed to load the BLIP model. Exiting.")
        sys.exit(1)
    logger.info(f"Using {device.upper()} for inference.")
    BLIP_CAPTION_GENERATOR = CaptionGenerator(blip_processor, blip_model, device)
    return BLIP_CAPTION_GENERATOR

def predict_caption(image_data: str):
    if BLIP_CAPTION_GENERATOR is None:
        logger.warning("BLIP Caption Generator not initialized!")
        return "BLIP not ready."
    try:
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        import cv2
        image_cv = cv2.cvtColor(pil_image_to_cv2(image), cv2.COLOR_RGB2BGR)
        BLIP_CAPTION_GENERATOR.update_frame(image_cv)
        caption = BLIP_CAPTION_GENERATOR.get_caption()
        return caption
    except Exception as e:
        logger.error(f"Error during caption generation: {e}")
        return "Error generating caption."

def pil_image_to_cv2(pil_image):
    import numpy as np
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected via WebSocket (FastAPI)")
    await websocket.send_json({"status": "Backend (FastAPI) connected successfully!"})
    if BLIP_CAPTION_GENERATOR is None:
        load_blip_model()
    try:
        while True:
            data = await websocket.receive_json()
            if "frame" in data:
                image_data = data["frame"]
                caption = predict_caption(image_data)
                gpu_info = get_gpu_usage()
                await websocket.send_json({"caption": {"text": caption, "gpu_info": gpu_info}})
            else:
                print("Received unknown data:", data)
    except WebSocketDisconnect:
        print("Client disconnected from WebSocket (FastAPI)")
    except Exception as e:
        print(f"WebSocket error: {e}")
        if BLIP_CAPTION_GENERATOR:
            BLIP_CAPTION_GENERATOR.stop()

@app.on_event("shutdown")
def shutdown_event():
    if BLIP_CAPTION_GENERATOR:
        logger.info("Stopping BLIP Caption Generator...")
        BLIP_CAPTION_GENERATOR.stop()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)