# streamlit_app.py

import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import io
from src.config import AVAILABLE_MODELS, DEFAULT_MODEL
from src.detection import load_model, read_image_bytes, run_detection

st.set_page_config(page_title="Blueprint Object Detection", layout="centered")

st.title("Door & Window Detection from Blueprints")
st.markdown("Upload a floorplan and select a model to detect doors and windows.")

# Sidebar - Model selection
model_choice = st.sidebar.selectbox("Choose YOLOv8 Model", AVAILABLE_MODELS, index=AVAILABLE_MODELS.index(DEFAULT_MODEL))

# Upload image
uploaded_file = st.file_uploader("Upload a blueprint image", type=["jpg", "jpeg", "png"])

import os
os.environ["TORCH_HOME"] = "/tmp/torch_home"  # Optional, to reduce torch path issues

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


if uploaded_file:
    # Show the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Run Detection"):
        # Inference - Use the run_detection function instead
        image_bytes = uploaded_file.getvalue()
        results = run_detection(image_bytes, uploaded_file.name, model_choice)
        detections = results["detections"]

        # Draw bounding boxes
        draw = ImageDraw.Draw(image)
        for det in detections:
            label = det["label"]
            conf = det["confidence"]
            bbox = det["bbox"]
            # bbox is in format [x1, y1, x2, y2]
            draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline="red", width=2)
            draw.text((bbox[0], bbox[1] - 10), f"{label} ({conf:.2f})", fill="red")

        st.image(image, caption="Detections", use_container_width=True)

        # Show detection results
        st.markdown("### Detection Results")
        if detections:
            st.success(f"Found {len(detections)} detections!")
            st.dataframe(detections)
        else:
            st.warning("No detections found. This could be due to:")
            st.write("- Low confidence threshold")
            st.write("- Model not trained on this type of image")
            st.write("- Image quality or resolution issues")
            st.write("- Objects too small or unclear in the image")
            
            # Add some debugging info
            st.markdown("### Debug Information")
            st.write(f"Image size: {image.size}")
            st.write(f"Model: {model_choice}")
            st.write("Try adjusting the confidence threshold in your config file.")