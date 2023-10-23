import torch
from roboflow import Roboflow
from ultralytics import YOLO
import numpy as np
import os


class ComponentDetection:

    # Initialize or load model 
    def __init__(self, 
                initialize_random_model = False, 
                initialize_pretrained_model = False, 
                local_model = True,
                local_dataset = True):
        
        # Set up device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using device: ", self.device)
        
        # If initialize to random weights
        if initialize_random_model:
            self.model = self.initialize_model()
            
        # If initialize to pretrained weights
        elif initialize_pretrained_model:
            self.model = self.initialize_pretrained_model()
            
        # If loading a local model
        elif local_model:
            self.model = self.load_model()
        
        # If loading a deployed model
        else: 
            self.model = self.load_deployed_model()
        
        # If dataset is local
        if local_dataset: 
            self.dataset = self.load_local_dataset()  
        
        # If dataset is to be loaded from Roboflow
        else: 
            self.dataset = self.load_deployed_dataset()
         
    # Train model
    def train_model(self, 
                    dataset = "../ml/Circuit-Component-Detection-2/data.yaml", 
                    epochs = 50):
        self.model.train(data = dataset, epochs = epochs, plots=True)  # train the model

    # Validate model
    def validate_model(self):
        self.model.val(data="../ml/Circuit-Component-Detection-2/data.yaml", conf=0.2, iou=0.5, batch = 5, split='val')
        
    # Load a standard uninitialized model
    def initialize_model(self):
        model = YOLO("yolov8n.yaml")  # build a new model from scratch - this is the smallest YOLOv8 model
        return model
    
    # Load a standard pre-trained model
    def initialize_pretrained_model(self):
        model = YOLO("yolov8n.pt")  # build a new model from scratch - this is the smallest YOLOv8 model
        return model
     
    # Load a local model
    def load_model(self, path = "../ml/runs/detect/train2/weights/best.pt"):
        model = YOLO(path)
        return model
    
    # Load a local dataset
    def load_local_dataset(self, 
                           dataset_path = "../ml/Circuit-Component-Detection-2/data.yaml"):
        dataset = dataset_path
        return dataset
         
    # Load dataset deployed on Roboflow
    def load_deployed_dataset(self,
                            api_key = "PSTIimpCaCht5fF8ZxGR",
                            workspace = "jonathanapps",
                            project = "circuit-component-detection",
                            version = 2,
                            model = "yolov8"):
        rf = Roboflow(api_key = api_key)
        project = rf.workspace(workspace).project(project)
        dataset = project.version(version).download(model)
        return dataset

    # Load model deployed on Roboflow
    def load_deployed_model(self,
                            api_key = "PSTIimpCaCht5fF8ZxGR",
                            workspace = "jonathanapps",
                            project = "circuit-component-detection",
                            version = 2,
                            model = "yolov8"):
        rf = Roboflow(api_key = api_key)
        project = rf.workspace(workspace).project(project)
        model = project.version(version).model
        return model
    
    # Predicts objects in a given image
    def predict(self, images, save = False, conf = 0.5):
        results = self.model(images, save = save, conf = conf)
        return results
    
    # Get bboxes information for a given set of images
    def get_images_bboxes(self, results):
        xyxys = []
        confidences = []
        class_ids = []
        for result in results: # Extract component detections for each image in directory
            boxes = result.boxes.cpu().numpy()
            xyxys.append(boxes.xyxy)
            confidences.append(boxes.conf)
            class_ids.append(boxes.cls)
        return xyxys, confidences, class_ids
    
    # Get bboxes information for a given image
    def get_image_bboxes(self, results):
        boxes = results[0].boxes.cpu().numpy()
        xyxys = boxes.xyxy
        confidences =  boxes.conf
        class_ids = boxes.cls
        return xyxys, confidences, class_ids
    
    # Compute and render confusion matrix
    def render_confusion_matrix(self, detections, labels, conf_threshold=0.3, iou_threshold=0.5):
        return 0
    
if __name__ == "__main__":
    image_path = "../ml/Circuit-Component-Detection-2/valid/images/"
    component_detector = ComponentDetection()
    results = component_detector.predict(image_path, save = True, conf = 0.5)
    xyxys, confidences, class_ids = component_detector.get_image_bboxes(results)
    results = component_detector.validate_model()