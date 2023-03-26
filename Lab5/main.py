import uvicorn
import requests
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np
import cv2
import base64
import json
from typing import List
import asyncio


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")




class ImageRequest(BaseModel):
    image: str
    name: str
    surname: str
    numbers: List[int]


# encode image as base64 string
def encode_image(image):
    _, encoded_image = cv2.imencode(".jpg", image)
    return "data:image/jpeg;base64," + base64.b64encode(encoded_image).decode()

# decode base64 string to image
def decode_image(image_string):
    encoded_data = image_string.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def apply_canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

@app.post("/process-image")
async def process_image(image_request: ImageRequest):
    image = decode_image(image_request.image)
    edges = apply_canny(image)
    processed_image = encode_image(edges)

    return {"name": image_request.name,
            "surname": image_request.surname,
            "numbers": image_request.numbers,
            "processed_image": processed_image}

@app.get("/")
async def index(request: Request):
    image_file = 'bas.png'
    # Load the image
    image        = cv2.imread(image_file)
    image_string = encode_image(image)
    image = decode_image(image_string)
    edges = apply_canny(image)
    processed_image = encode_image(edges)
    return templates.TemplateResponse('index.html', {'request': request, 'processed':processed_image, 'original': image_string})

