# doors-and-windows

### More details into the report is present under `report.pdf`

A lightweight web application that detects **doors** and **windows** in architectural floor plans. 

This application has been built using Streamlit for the front-end and supports three YOLO variants for the prediction, namely: 
- YOLOv8n
- YOLOv8s
- YOLOV8m

## Features: 
- Upload blueprint images in the form of either '.jpg', '.jpeg', or '.png'.
- Select from different YOLOv8 variants. 
- Visual detection output along with bounding boxes. 
- Output includes label, confidence scores, and bounding box coordinates. 

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/akshathmangudi/doors-and-windows.git
cd doors-and-windows/
```

### Create and activate a virtual environment

#### For virtualvenv
```bash
python -m venv <env_name> 
source <env_name>/bin/activate # On Windows use 'venv\Scripts\activate
```

#### For conda
```bash
conda create -n <env_name> python=3.10
conda activate <env_name>
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Streamlit App
```bash
streamlit run app.py
```
This app will launch in your browser at http://localhost:7860

### To run the API Locally (FastAPI Example)
If you're running the backend as an API: 
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```
And view the API in `http://0.0.0.0:8000/docs`

## Linting and Formatting
To ensure the code has been maintained properly and professionally, we use **pylint**, **black** and **isort** for linting, formatting and import sorting respectively. 

## Example API Usage

### To test the public API: 
Go to the link that's present in the "About" section or click <a href="https://huggingface.co/spaces/akshathmangudi/d-and-w">here</a>.

### Endpoint
```bash
POST /detect
Content-Type: multipart/form-data
```

### Request Fields
| Field        | Type   | Description                    |
| ------------ | ------ | ------------------------------ |
| `image`      | file   | Floorplan image (JPG/PNG)      |
| `model_name` | string | Model variant (e.g. `yolov8n`) |


### Example Response
```json
{
  "filename": "image.jpg",
  "detections": [
    {
      "label": "door",
      "confidence": 0.848,
      "bbox": [852.94, 467.47, 975.9, 613.08]
    },
    {
      "label": "door",
      "confidence": 0.842,
      "bbox": [378.89, 268.19, 504.86, 414.77]
    },
  ],
  "total_detections": 2,
  "confidence_threshold": 0.5,
  "image_shape": [1021, 1143, 3]
}
```

## Project Structure
```bash
├── app.py
├── config.yaml
├── contract.yaml
├── Dockerfile
├── __init__.py
├── LICENSE
├── model_comparison.txt
├── models
│   ├── onnx
│   │   ├── yolov8m.onnx
│   │   ├── yolov8n.onnx
│   │   └── yolov8s.onnx
│   └── pt
│       ├── yolov8m.pt
│       ├── yolov8n.pt
│       └── yolov8s.pt
├── README.md
├── requirements.txt
└── src
    ├── config.py
    ├── detection.py
    ├── __init__.py
    ├── main.py
    └── utils.py
```

## License
See the LICENSE file or more info. 
