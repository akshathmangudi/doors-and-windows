import os
import cv2
import torch
import numpy as np
import yaml

# Load model config once
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_PATHS = config["model_paths"]
MODEL_CACHE = {}

from ultralytics import YOLO

def get_model(model_name: str):
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unsupported model: {model_name}. Choose from {list(MODEL_PATHS.keys())}")

    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    model_path = MODEL_PATHS[model_name]
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model weight file not found: {model_path}")

    model = YOLO(model_path)  # ✅ Proper YOLOv8 loading
    MODEL_CACHE[model_name] = model
    return model


def run_detection(image_bytes: bytes, filename: str, model_name: str) -> dict:
    """
    Run object detection on uploaded image using selected model.
    """
    # Decode image bytes to OpenCV format
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Failed to decode image. Check if it's a valid image file.")

    model = get_model(model_name)
    results = model(img)[0]  # first (and only) result from list

    detections = []
    if results.boxes is not None:
        for box in results.boxes:
            xyxy = box.xyxy.cpu().numpy().flatten().tolist()
            conf = float(box.conf.cpu().item())
            cls = int(box.cls.cpu().item())
            detections.append({
                "bbox": [round(x, 2) for x in xyxy],
                "confidence": round(conf, 2),
                "class_id": cls,
                "label": model.names[cls]
            })

    return {
        "filename": filename,
        "model_used": model_name,
        "num_detections": len(detections),
        "detections": detections
    }
