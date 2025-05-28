from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from detection import run_detection
import yaml

app = FastAPI()

# Load config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_PATHS = config["model_paths"]
DEFAULT_MODEL = config.get("default_model", "yolov8n")
AVAILABLE_MODELS = list(MODEL_PATHS.keys())


@app.post("/detect")
async def detect(
    image: UploadFile = File(...),
    model_name: str = Query(DEFAULT_MODEL, description="Model name to use")
):
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model name. Choose from: {AVAILABLE_MODELS}"
        )

    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image.")

    image_bytes = await image.read()

    try:
        result = run_detection(image_bytes, filename=image.filename, model_name=model_name)
        formatted = {
            "detections": [
                {
                    "label": det["label"],
                    "confidence": det["confidence"],
                    "bbox": det["bbox"]
                }
                for det in result["detections"]
            ]
        }
        return JSONResponse(content=formatted)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
