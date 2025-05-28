"""
The main streamlit web app code to detect doors and windows
in a floor plan, with the freedom to choose the yolo-v8 variant.
"""

import os
import warnings

import streamlit as st
from PIL import Image, ImageDraw

from src.config import AVAILABLE_MODELS, DEFAULT_MODEL
from src.detection import run_detection

# Reduce PyTorch path issues (especially on platforms like Spaces)
os.environ["TORCH_HOME"] = "/tmp/torch_home"

# Suppress known warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Streamlit App Config
st.set_page_config(page_title="Blueprint Object Detection", layout="centered")
st.title("Door & Window Detection from Blueprints")
st.markdown("Upload a floorplan and select a model to detect doors and windows.")


def main():
    """
    The main function that runs the streamlit app.
    """
    model_choice = st.sidebar.selectbox(
        "Choose YOLOv8 Model",
        AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(DEFAULT_MODEL),
    )

    # File upload
    uploaded_file = st.file_uploader(
        "Upload a blueprint image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Run Detection"):
            image_bytes = uploaded_file.getvalue()
            results = run_detection(image_bytes, uploaded_file.name, model_choice)
            detections = results.get("detections", [])

            if detections:
                # Draw bounding boxes
                image_with_boxes = image.copy()
                draw = ImageDraw.Draw(image_with_boxes)

                for det in detections:
                    label = det["label"]
                    conf = det["confidence"]
                    bbox = det["bbox"]
                    if len(bbox) != 4:
                        continue  # skip malformed detections

                    draw.rectangle(
                        [(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline="red", width=2
                    )
                    draw.text(
                        (bbox[0], bbox[1] - 10), f"{label} ({conf:.2f})", fill="red"
                    )

                st.image(
                    image_with_boxes, caption="Detections", use_container_width=True
                )
                st.success(f"✅ Found {len(detections)} objects")
                st.dataframe(detections)

            else:
                st.warning("⚠️ No detections found.")
                st.markdown("### Possible Reasons")
                st.write("- Confidence threshold too high")
                st.write("- Unclear or low-resolution image")
                st.write("- Model not trained for this style of blueprint")

            # Debug info
            st.markdown("### Debug Information")
            st.write(f"Image size: {image.size}")
            st.write(f"Selected Model: `{model_choice}`")
            st.write("Try adjusting `confidence_threshold` in your `config.yaml`.")


if __name__ == "__main__":
    main()
