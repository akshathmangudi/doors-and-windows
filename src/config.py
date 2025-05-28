import os
import yaml

from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Load YAML config
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# YOLO model paths
MODEL_PATHS = config.get("model_paths", {})
DEFAULT_MODEL = next(iter(MODEL_PATHS), "yolov8n")  # Use first model if none explicitly set
AVAILABLE_MODELS = list(MODEL_PATHS.keys())

CONFIDENCE_THRESHOLD = config.get("params", {}).get("confidence_threshold")
DATASET_CONFIG = config.get("palcode-ai-1", {})
ROBOFLOW_API = os.getenv("ROBOFLOW_API")
WORKSPACE_NAME = os.getenv("WORKSPACE_NAME")
PROJECT_NAME = os.getenv("PROJECT_NAME")
DEBUG = True