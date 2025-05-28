"""
Configuration loader for YOLO models, dataset paths, and environment variables.

- Loads model paths and parameters from config.yaml
- Loads environment variables from .env
- Exposes config globals like DEFAULT_MODEL, CONFIDENCE_THRESHOLD, etc.
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

# Load YAML configuration
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Model configurations
MODEL_PATHS = config.get("model_paths", {})
DEFAULT_MODEL = next(iter(MODEL_PATHS), "yolov8n")  # Fallback to 'yolov8n'
AVAILABLE_MODELS = list(MODEL_PATHS.keys())

CONFIDENCE_THRESHOLD = config.get("params", {}).get("confidence_threshold")
DATASET_CONFIG = config.get("palcode-ai-1", {})

ROBOFLOW_API = os.getenv("ROBOFLOW_API", "")
WORKSPACE_NAME = os.getenv("WORKSPACE_NAME", "")
PROJECT_NAME = os.getenv("PROJECT_NAME", "")
