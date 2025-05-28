"""
The script that contains the necessary util functions for the whole pipeline.

This file contains methods for:
- creating the dataset.
- logging model performances onto a text file.
- exporting .pt and onnx-type weights to a specified directory.
"""

import os
import random
import shutil
from pathlib import Path
from typing import Dict

from ultralytics import YOLO

from .config import DATASET_CONFIG, MODEL_PATHS

random.seed(42)


def create_dataset(val_split: float = 0.2) -> None:
    """
    Splits images and annotations into training and validation datasets.

    Args:
        val_split (float): Fraction of data to reserve for validation. Default is 0.2.
    """
    input_dir = Path(DATASET_CONFIG.get("input_dir", "images/"))
    annos_dir = Path(DATASET_CONFIG.get("annos_dir", "annotations/"))
    output_dir = Path(DATASET_CONFIG.get("output_dir", "dataset/"))

    for split in ["train", "val"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    random.shuffle(image_files)

    split_index = int(len(image_files) * (1 - val_split))
    train_images = image_files[:split_index]
    val_images = image_files[split_index:]

    def process(images, split):
        for img_path in images:
            label_path = annos_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                print(f"[WARNING] Missing label for: {img_path.name}")
                continue
            shutil.copy2(img_path, output_dir / split / "images" / img_path.name)
            shutil.copy2(label_path, output_dir / split / "labels" / label_path.name)

    process(train_images, "train")
    process(val_images, "val")

    print(f"[DONE] Dataset created at: {output_dir}")
    print(f" - Training images: {len(train_images)}")
    print(f" - Validation images: {len(val_images)}")


def log_model_comparisons(
    gflops_map: Dict[str, float], log_file: str = "model_comparison_log.txt"
) -> None:
    """
    Logs performance metrics for each YOLO model variant to a file.

    Args:
        gflops_map (Dict[str, float]): Mapping from model names to their GFLOPs.
        log_file (str): Path to the output log file. Default is 'model_comparison_log.txt'.
    """
    data_yaml = DATASET_CONFIG.get("data_yaml", "./dataset/data.yaml")

    with open(log_file, "w", encoding="utf-8") as f:
        for name, path in MODEL_PATHS.items():
            print(f"Validating model: {name}, path: {path}")
            model = YOLO(path)
            metrics = model.val(data=data_yaml)

            f.write(f"\n{'='*40}\n{name} Validation Results\n{'='*40}\n")
            f.write(f"Parameters: {sum(p.numel() for p in model.model.parameters())}\n")
            f.write(f"GFLOPs: {gflops_map.get(name, 'N/A')}\n")
            f.write(f"Speed (inference): {metrics.speed['inference']:.2f} ms/image\n\n")

            f.write("Class     | Precision | Recall\n")
            for i, cls in enumerate(model.model.names):
                prec = metrics.box.p[i].item()
                rec = metrics.box.r[i].item()
                f.write(f"{cls:<10} | {prec:.3f}     | {rec:.3f}\n")

            f.write("\nOverall:\n")
            f.write(f"mAP@50: {metrics.box.map50:.3f}\n")
            f.write(f"mAP@50-95: {metrics.box.map:.3f}\n")


def export_models_to_weights_dir(output_dir: str = "weights") -> None:
    """
    Exports YOLO model weights in both .pt and ONNX formats to the specified directory.

    Args:
        output_dir (str): Directory to save the exported model weights. Default is 'weights'.
    """
    os.makedirs(output_dir, exist_ok=True)

    for model_name, model_path in MODEL_PATHS.items():
        print(f"\nExporting {model_name} from {model_path}...")

        model = YOLO(model_path)

        # Copy .pt file
        pt_dest = os.path.join(output_dir, f"{model_name.lower()}.pt")
        shutil.copy(model_path, pt_dest)
        print(f"Saved: {pt_dest}")

        # Export to ONNX format
        onnx_path = model.export(format="onnx")
        exported_onnx_path = os.path.join(output_dir, f"{model_name.lower()}.onnx")
        shutil.move(onnx_path, exported_onnx_path)
        print(f"Saved: {exported_onnx_path}")
