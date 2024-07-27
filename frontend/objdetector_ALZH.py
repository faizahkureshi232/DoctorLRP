import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

class AL_ObjectDetector:
    def __init__(self):
        self.model_path="/home/hackathon/frontend/best_alzh.pt" #yolo model path for alzhimers
        self.model = YOLO(self.model_path)
        self.class_list = self._load_class_labels('/home/hackathon/frontend/coco_ALZH.txt')
        self.output_dir = "/home/hackathon/frontend/output_OBJ"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _load_class_labels(self, class_list_path):
        with open(class_list_path, "r") as f:
            class_list = f.read().split("\n")
        return class_list
    
    def detect_objects(self, image):
        # Convert PIL image to OpenCV format
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Perform detection
        results = self.model.predict(image)
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Extract bounding boxes
        scores = results[0].boxes.conf.cpu().numpy()  # Extract confidence scores
        classes = results[0].boxes.cls.cpu().numpy()  # Extract class labels
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            label = self.class_list[int(classes[i])]
            confidence = scores[i]
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(image, f'{label} {confidence:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
        
        # Convert back to RGB for display
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def process_image(self, image):
        # Load the image
        
        
        # Detect objects
        output_image = self.detect_objects(image)
        
        # Display the output image
        plt.figure(figsize=(10, 10))
        plt.imshow(output_image)
        plt.axis('off')
        #plt.show()
        
        # Save the output image
        output_path = os.path.join(self.output_dir, "detected_image.jpg")
        Image.fromarray(output_image).save(output_path)
        plt.savefig(output_path)
        plt.close()
        return output_path
