from ultralytics import YOLO

def load_model(path):
    return YOLO(path) 