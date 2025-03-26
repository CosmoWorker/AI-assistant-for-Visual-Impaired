from fastapi import FastAPI
from fastrtc import Stream
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

app=FastAPI()

async def handler(frame):
    print("Recieved Frame")
    

stream = Stream(handler=handler)

stream.mount(app)

@app.get('/')
def read_root():
    return {"message" : "FastRTC backend is running"}