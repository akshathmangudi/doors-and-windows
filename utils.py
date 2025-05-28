import os 
from pathlib import Path
import shutil
import random
from typing import Union, Dict
from ultralytics import YOLO
import onnx

random.seed(42)

def create_dataset(input_dir: Union[str, Path], annos_dir: Union[str, Path], output_dir: Union[str, Path], val_split=0.2):
    input_dir = Path(input_dir)
    annos_dir = Path(annos_dir)
    output_dir = Path(output_dir)

    # Define train/val image + label directories
    for split in ["train", "val"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Gather all image files
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

            # Copy image
            shutil.copy2(img_path, output_dir / split / "images" / img_path.name)
            # Copy label
            shutil.copy2(label_path, output_dir / split / "labels" / label_path.name)

    process(train_images, "train")
    process(val_images, "val")

    print(f"[DONE] Dataset created at: {output_dir}")
    print(f" - Training images: {len(train_images)}")
    print(f" - Validation images: {len(val_images)}")


def log_model_comparisons(models: Union[dict, Dict[str, str]], gflops_map: Union[dict, Dict[str, float]], log_file: str = "model_comparison_log.txt"):
    with open(log_file, "w") as f:        
        for name, path in models.items():
            print(f"Loading model: {name}, from path: {path}")  # Add this
            model = YOLO(path)
            metrics = model.val(data="./palcode-ai-1/data.yaml")

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

def export_models_to_weights_dir(model_paths: dict, output_dir: str = "weights"):
    os.makedirs(output_dir, exist_ok=True)

    for model_name, model_path in model_paths.items():
        print(f"\nProcessing {model_name} from {model_path}...")

        # Load the YOLO model
        model = YOLO(model_path)

        # Copy .pt file with renamed filename
        pt_dest = os.path.join(output_dir, f"{model_name.lower()}.pt")
        shutil.copy(model_path, pt_dest)
        print(f"Saved: {pt_dest}")

        # Export to ONNX format
        onnx_path = model.export(format="onnx")
        exported_onnx_path = os.path.join(output_dir, f"{model_name.lower()}.onnx")
        shutil.move(onnx_path, exported_onnx_path)
        print(f"Saved: {exported_onnx_path}")
    

# if __name__ == "__main__":
#     input_dir = "./images/"
#     annos_dir = "./annotations"
#     output_dir = "./dataset"
    
#     # Usage
#     gflops_map = {
#         "YOLOv8n": 8.1,
#         "YOLOv8s": 28.4,
#         "YOLOv8m": 78.7
#     }

#     models = {
#         "YOLOv8n": "runs/detect/train2/weights/best.pt",
#         "YOLOv8s": "runs/detect/train3/weights/best.pt",
#         "YOLOv8m": "runs/detect/train4/weights/best.pt"
#     }
#       
#   create_dataset(input_dir, annos_dir, output_dir, val_split=0.2)
#   log_model_comparisons(models, gflops_map)
#   export_models_to_weights_dir(models)