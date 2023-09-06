import cv2
import torch
from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser(description='Align License Plate')
parser.add_argument('-img_path', '--img_path', type=str, help='lp img path for alignment', required=True)
parser.add_argument('-model_path', '--model_path', type=str, help='model path for lp recognition', default='model/best_corner_detect.pt')
args = parser.parse_args()

def group_lines(points):
    # Calculate the average y-coordinate
    average_y = sum(y for _, _, y in points) / len(points)

    # Group points into two lines based on y-coordinate
    line1 = []
    line2 = []

    for class_idx, x, y in points:
        if y <= average_y:
            line1.append((class_idx, x, y))
        else:
            line2.append((class_idx, x, y))

    return line1, line2

characters = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'S', 'T', 'U', 'V',  'X', 'Y', 'Z',  '0' ]

# Model lp recognition 
def load_model(path):
    return YOLO(path)

def lp_recognition(img_path=None, model=None):
    if img_path == None:
        return 'Please set image path'
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    if model == None:
        return 'Please set model'
    results = model(img_path)
    lp_content = ''
    for result in results:
        new_data = []
        for data in result.boxes.boxes:
            data = data.type(torch.int32)
            # Data chỉ cần quan tâm đến tâm + class idx
            new_data.append([data[-1], data[0], data[1]])
        # Xử lý biển vuông
        if abs(w/h - 330/165) < abs(w/h - 520/110):
            upper_data, lower_data = group_lines(new_data)

            sorted_upper_data = sorted(upper_data, key=lambda point: (point[1]))
            sorted_lower_data = sorted(lower_data, key=lambda point: (point[1]))
            for i in sorted_upper_data:
                lp_content += characters[i[0]]
            lp_content += ' '   
            for i in sorted_lower_data:
                lp_content += characters[i[0]]
        # Xử lý biển chữ nhật dài
        else:
            sorted_label_data = sorted(new_data, key=lambda point: (point[1]))
            
            for i in sorted_label_data:
                lp_content += characters[i[0]]
    return lp_content

if __name__ == "__main__":
    img_path = args.image_path
    model = load_model(args.model_path)
    print(lp_recognition(img_path, model))
