'''
The following link gives additional guidance:
https://roboflow.com/how-to-create-a-confusion-matrix/yolov8
'''

import supervision as sv
from ultralytics import YOLO
import numpy as np
import os

class ConfusionMatrix:

    # Class constructor
    def __init__(self, 
                image_path = os.path.abspath("../ml/Circuit-Component-Detection-2/valid/images"), 
                labels_path = os.path.abspath("../ml/Circuit-Component-Detection-2/valid/labels"), 
                data_yaml_path = os.path.abspath("../ml/Circuit-Component-Detection-2/data.yaml"),
                model_path = os.path.abspath("../ml/runs/detect/train2/weights/best.pt")):
        
        self.dataset = sv.DetectionDataset.from_yolo(image_path, labels_path, data_yaml_path) # Load dataset
        self.model = YOLO(model_path) # Load model
        self.confusion_matrix = sv.ConfusionMatrix.benchmark(dataset = self.dataset, callback = self.callback)

    def callback(self, image: np.ndarray) -> sv.Detections:
        result = self.model(image)[0]
        return sv.Detections.from_ultralytics(result)

    def plot_normalized_confusion_matrix(self):
        self.confusion_matrix.plot(save_path = os.path.abspath("../ml/images/normalized_confusion_matrix1.png"), normalize= True)

    def plot_confusion_matrix(self):
        self.confusion_matrix.plot(save_path = os.path.abspath("../ml/images/confusion_matrix1.png"))

    
if __name__ == "__main__":
    confusion_matrix = ConfusionMatrix()
    confusion_matrix.plot_confusion_matrix()
    confusion_matrix.plot_normalized_confusion_matrix()
    