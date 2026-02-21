import io
import os
import cv2
import numpy as np
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN
from pydantic import BaseModel
import base64
import requests
from fastapi import BackgroundTasks
from segmentation_engine import OnnxSegmenter, Detection, NailImage, NailOrientation

# --- Configuration ---
ONNX_MODEL_PATH = "onnx/nails_seg_s_yolov8_v1.onnx"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
IMG_SIZE = 640
API_KEY = os.environ.get("API_KEY", "019ae483-85c3-703d-8a20-d8c34ef0503c")
API_KEY_NAME = "x-api-key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Store the model and device globally
MODEL_STATE = {}

# --- Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup, clean up on shutdown."""
    try:
        segmenter = OnnxSegmenter(ONNX_MODEL_PATH, CONF_THRESHOLD, IOU_THRESHOLD, IMG_SIZE)
        segmenter.load_model()
        MODEL_STATE['segmenter'] = segmenter
        print(f"Segmentation Engine loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise RuntimeError(f"Failed to load ML model: {e}")
    yield
    MODEL_STATE.clear()

# --- FastAPI App Initialization ---
app = FastAPI(title="Nail Segmentation API (Polygon)", lifespan=lifespan)

# --- Response Model (Pydantic for automatic documentation) ---

# Detection and NailImage models are imported from segmentation_engine

class SegmentationResponse(BaseModel):
    status: str
    detections: list[Detection]
    image_width: int
    image_height: int

class NailExtractionResponse(BaseModel):
    status: str
    nails: list[NailImage]
    image_width: int
    image_height: int

class ExtractionRequest(BaseModel):
    imageUrl: str
    designId: str
    callbackUrl: str

# --- Core Processing Logic ---

def get_segmenter():
    """Dependency injection function to provide the segmenter."""
    if 'segmenter' not in MODEL_STATE:
        raise HTTPException(status_code=500, detail="Model not initialized.")
    return MODEL_STATE['segmenter']

async def verify_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    else:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Could not validate credentials"
        )

# --- Health Endpoint ---

@app.get("/health")
async def health():
    """Unauthenticated health check for Cloud Run warm-up pings."""
    return {
        "status": "healthy",
        "model_loaded": "segmenter" in MODEL_STATE
    }

# --- API Endpoints ---

@app.post(
    "/segment",
    response_model=SegmentationResponse,
    summary="Run Nail Segmentation on Uploaded Image"
)
async def segment_image(
    file: UploadFile = File(...),
    segmenter = Depends(get_segmenter),
    api_key: str = Depends(verify_api_key)
):
    """
    Accepts an image file and returns bounding boxes and polygon coordinates
    (as a list of [x, y] points) for all detected nails.
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a JPEG, PNG, or WebP image."
        )

    try:
        contents = await file.read()
        file_bytes = np.frombuffer(contents, dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")

        # Run segmentation
        detections = segmenter.segment(img)

        image_height, image_width = img.shape[:2]

        if not detections:
            return SegmentationResponse(
                status="no_detections",
                detections=[],
                image_width=image_width,
                image_height=image_height
            )

        return SegmentationResponse(
            status="success",
            detections=detections,
            image_width=image_width,
            image_height=image_height
        )

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

def process_and_callback(request: ExtractionRequest, segmenter):
    """
    Background task to download image, extract nails, and call the callback URL.
    """
    try:
        print(f"Starting background extraction for designId: {request.designId}")

        # 1. Download Image
        response = requests.get(request.imageUrl)
        if response.status_code != 200:
            print(f"Failed to download image from {request.imageUrl}")
            return

        file_bytes = np.frombuffer(response.content, dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            print(f"Could not decode image for designId: {request.designId}")
            return

        # 2. Extract Nails
        extracted_nails = segmenter.extract_nails(img)

        if not extracted_nails:
            print(f"No nails detected for designId: {request.designId}")

        # 3. Prepare Payload
        nails_data = [nail.dict() for nail in extracted_nails]

        payload = {
            "designId": request.designId,
            "nails": nails_data,
            "status": "success" if extracted_nails else "no_detections"
        }

        # 4. Send Callback
        print(f"Sending callback to {request.callbackUrl}")
        cb_response = requests.post(request.callbackUrl, json=payload, headers={"x-api-key": API_KEY})

        if cb_response.status_code == 200:
            print(f"Callback successful for designId: {request.designId}")
        else:
            print(f"Callback failed with status {cb_response.status_code}: {cb_response.text}")

    except Exception as e:
        print(f"Error in background task: {e}")


@app.post(
    "/extract_nails",
    summary="Trigger Async Nail Extraction"
)
async def extract_nails_endpoint(
    request: ExtractionRequest,
    background_tasks: BackgroundTasks,
    segmenter = Depends(get_segmenter),
    api_key: str = Depends(verify_api_key)
):
    """
    Accepts a JSON payload with imageUrl, designId, and callbackUrl.
    Starts the extraction process in the background and returns immediately.
    """
    background_tasks.add_task(process_and_callback, request, segmenter)
    return {"status": "accepted", "message": "Extraction started in background"}

@app.post(
    "/extract_nails_sync",
    response_model=NailExtractionResponse,
    summary="Extract Nail Images with Transparent Background (Sync)"
)
async def extract_nails_sync_endpoint(
    file: UploadFile = File(...),
    segmenter = Depends(get_segmenter),
    api_key: str = Depends(verify_api_key)
):
    """
    Accepts an image file and returns individual nail images as base64 encoded PNGs
    with transparent backgrounds.
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a JPEG, PNG, or WebP image."
        )

    try:
        contents = await file.read()
        file_bytes = np.frombuffer(contents, dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")

        # Extract nails
        extracted_nails = segmenter.extract_nails(img)

        image_height, image_width = img.shape[:2]

        if not extracted_nails:
            return NailExtractionResponse(
                status="no_detections",
                nails=[],
                image_width=image_width,
                image_height=image_height
            )

        return NailExtractionResponse(
            status="success",
            nails=extracted_nails,
            image_width=image_width,
            image_height=image_height
        )

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# --- Server Run Command ---
# To run the server on your VM:
# uvicorn app_fastapi:app --host 0.0.0.0 --port 8000 --workers 4
