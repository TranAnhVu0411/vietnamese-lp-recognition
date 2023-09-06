from ultralytics import YOLO
import cv2
import torch
from pathlib import Path
import os
import argparse

parser = argparse.ArgumentParser(description='Align License Plate')
parser.add_argument('-img_path', '--img_path', type=str, help='lp img path for alignment', required=True)
parser.add_argument('-model_path', '--model_path', type=str, help='model path for lp detection', default = 'model/best_lp_detect.pt')
parser.add_argument('-save_path', '--save_path', type=str, help='saved path for lp aligment', required=True)
args = parser.parse_args()

def load_model(path):
    return YOLO(path) 

def lp_detect(img_path=None, model=None, save_path=None):
    if save_path == None:
        return 'Please set save path'
    if img_path == None:
        return 'Please set image path'
    img = cv2.imread(img_path)
    img_copy = img.copy()
    name = Path(img_path).stem
    if model == None:
        return 'Please set model'
    results = model(img_path)

    for result in results:
        for idx, coord in enumerate(result.boxes.xyxy):
            coord = coord.type(torch.int32).numpy()
            print((coord[0], coord[1]))
            lp = img[coord[1]:coord[3], coord[0]:coord[2]]
            cv2.rectangle(img_copy, (coord[0], coord[1]), (coord[2], coord[3]), (0, 0, 255), 2)
            cv2.imwrite(os.path.join(save_path, '{}-[{},{},{},{}].jpg'.format(name, coord[0], coord[1], coord[2], coord[3])), lp)

    cv2.imwrite(os.path.join(save_path, '{}-lp_detection_result.jpg'.format(name)), img_copy)

if __name__ == "__main__":
    img_path = args.img_path
    save_path = args.save_path
    model = load_model(args.model_path)
    lp_detect(img_path, model, save_path)