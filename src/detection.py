# app/detection.py

from ultralytics import YOLO
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
from typing import List, Dict, Any
from .config import MODEL_PATHS, CONFIDENCE_THRESHOLD

# Global model cache
MODEL_CACHE = {}

def load_model(model_name: str) -> YOLO:
    """
    Load and cache a YOLO model.

    Args:
        model_name (str): Key name of the model.

    Returns:
        YOLO: Loaded YOLO model instance.
    """
    if model_name not in MODEL_CACHE:
        model_path = MODEL_PATHS.get(model_name)
        if model_path is None:
            raise ValueError(f"Model path for {model_name} not found.")
        MODEL_CACHE[model_name] = YOLO(model_path)

    return MODEL_CACHE[model_name]

def read_image_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Converts uploaded image bytes to a NumPy array.

    Args:
        image_bytes (bytes): Image file bytes.

    Returns:
        np.ndarray: OpenCV-compatible image array (BGR).
    """
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def run_detection(image_bytes: bytes, filename: str, model_name: str) -> Dict[str, Any]:
    """
    Perform detection using a YOLO model.

    Args:
        image_bytes (bytes): Raw image bytes.
        filename (str): Name of the uploaded file.
        model_name (str): Key name for the model to use.

    Returns:
        dict: Structured detection output.
    """
    model = load_model(model_name)
    image = read_image_bytes(image_bytes)
    results = model(image, conf=CONFIDENCE_THRESHOLD)

    detections = []
    for result in results:
        for box in result.boxes:
            bbox = box.xyxy[0].tolist()
            label = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            detections.append({
                "label": label,
                "confidence": round(confidence, 3),
                "bbox": [round(coord, 2) for coord in bbox]
            })

    return {
        "filename": filename,
        "detections": detections
    }
