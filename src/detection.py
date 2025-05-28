"""
The script that contains the driver code behind app.py. This file contains methods for:
- loading the model
- reading the image during inference (from streamlit)
- running the detection itself.
"""

from io import BytesIO
from typing import Any, Dict

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from .config import CONFIDENCE_THRESHOLD, MODEL_PATHS

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


def run_detection(
    image_bytes: bytes, filename: str, model_name: str, conf_threshold: float = None
) -> Dict[str, Any]:
    """
    Perform detection using a YOLO model.

    Args:
        image_bytes (bytes): Image file bytes.
        filename (str): Name of the uploaded file.
        model_name (str): Name of the model to use.
        conf_threshold (float, optional): Confidence threshold. Uses config default if None.

    Returns:
        Dict[str, Any]: Detection results with filename and detections list.
    """
    model = load_model(model_name)
    image = read_image_bytes(image_bytes)

    # Use provided threshold or fall back to config
    threshold = conf_threshold if conf_threshold is not None else CONFIDENCE_THRESHOLD
    results = model.predict(image, conf=threshold, verbose=True)

    detections = []
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            num_boxes = len(result.boxes)
            for i in range(num_boxes):
                cls_id = int(result.boxes.cls[i].item())
                conf = float(result.boxes.conf[i].item())
                bbox = result.boxes.xyxy[i].tolist()

                detections.append(
                    {
                        "label": model.names[cls_id],
                        "confidence": round(conf, 3),
                        "bbox": [round(coord, 2) for coord in bbox],
                    }
                )

    return {
        "filename": filename,
        "detections": detections,
        "total_detections": len(detections),
        "confidence_threshold": threshold,
        "image_shape": image.shape,
    }
