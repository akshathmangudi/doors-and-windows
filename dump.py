from utils import create_dataset, log_model_comparisons

def init_roboflow(): 
    from roboflow import Roboflow
    rf = Roboflow(api_key="Y4lXRdUJquUR4BvzLqFQ")
    project = rf.workspace("testingmodels").project("palcode-ai")
    dataset = project.version(1).download("yolov5")

if __name__ == "__main__": 
    input_dir = "./images/"
    annos_dir = "./annotations"
    output_dir = "./dataset"

    # create_dataset(input_dir, annos_dir, output_dir, val_split=0.2)
    # init_roboflow()

    gflops_map = {
    "YOLOv8n": 8.1,
    "YOLOv8s": 28.4,
    "YOLOv8m": 78.7
    }

    models = {
        "YOLOv8n": "weights/yolov8n.pt",
        "YOLOv8s": "weights/yolov8s.pt",
        "YOLOv8m": "weights/yolov8m.pt"
    }

    log_model_comparisons(models, gflops_map)