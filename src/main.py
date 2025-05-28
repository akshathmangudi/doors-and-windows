"""
Main FastAPI application for the blueprint object detection service.
Exposes an endpoint to detect doors and windows using YOLOv8 models.
"""

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from src.detection import run_detection

from .config import AVAILABLE_MODELS, DEFAULT_MODEL

app = FastAPI(title="Blueprint Object Detector", version="1.0")


@app.post("/detect")
async def detect(
    image: UploadFile = File(...),
    model_name: str = Query(DEFAULT_MODEL, description="Choose model variant"),
) -> JSONResponse:
    """
    Endpoint to detect doors/windows in uploaded blueprint image.

    Args:
        image (UploadFile): Uploaded blueprint image.
        model_name (str): YOLO model name (e.g., 'yolov8n', 'yolov8s').

    Returns:
        JSONResponse: Formatted detection output.
    """
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400, detail=f"Invalid model name. Available: {AVAILABLE_MODELS}"
        )

    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Upload an image."
        )

    image_bytes = await image.read()

    try:
        result = run_detection(
            image_bytes, filename=image.filename, model_name=model_name
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Detection failed: {str(e)}"
        ) from e
