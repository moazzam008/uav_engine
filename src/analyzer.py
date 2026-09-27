import cv2
import numpy as np
import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger("UAVEngine")

class AssetAnalyzer:
    def __init__(self, mode="hsv", hsv_lower=None, hsv_upper=None, min_area=100, conf_threshold=0.25):
        self.mode = mode
        self.conf_threshold = conf_threshold
        self.report_data = []
        
        if self.mode == "hsv":
            self.hsv_lower = np.array(hsv_lower, dtype=np.uint8) if hsv_lower else np.array([35, 100, 100])
            self.hsv_upper = np.array(hsv_upper, dtype=np.uint8) if hsv_upper else np.array([85, 255, 255])
            self.min_area = min_area
            logger.info(f"Initialized Spectral (HSV) Analyzer [Range: {self.hsv_lower} - {self.hsv_upper}]")
        elif self.mode == "ai":
            from ultralytics import YOLO
            import os
            
            logger.info("Initializing Aerial Neural Engine...")
            try:
                from huggingface_hub import hf_hub_download
                logger.info("Connecting to Hugging Face to download VisDrone model...")
                weights_path = hf_hub_download(
                    repo_id="dronefreak/visdrone-yolov8n", 
                    filename="best.pt",
                    local_dir="."
                )
                logger.info("Successfully loaded aerial VisDrone weights!")
                self.is_visdrone = True
            except Exception as e:
                logger.warning(f"Failed to fetch VisDrone model: {e}")
                logger.warning("Falling back to standard YOLOv8n (Street Level)...")
                weights_path = 'yolov8n.pt'
                self.is_visdrone = False
                
            self.model = YOLO(weights_path)
            logger.info("AI Neural Engine Initialized.")

    def analyze(self, image):
        if image is None: return 0, None, None
        self.report_data = []
        
        if self.mode == "hsv":
            return self._analyze_hsv(image)
        elif self.mode == "ai":
            return self._analyze_ai(image)
            
    def _analyze_hsv(self, image):
        logger.info("Executing Morphological Transformations...")
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        count = 0
        output_image = image.copy()
        for cnt in contours:
            if cv2.contourArea(cnt) > self.min_area:
                count += 1
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(output_image, (cx, cy), 5, (0, 255, 0), -1)
                    self.report_data.append({"Type": "Spectral_Target", "X": cx, "Y": cy, "Area_px": cv2.contourArea(cnt)})
                    
        df = pd.DataFrame(self.report_data) if self.report_data else pd.DataFrame(columns=["Type", "X", "Y", "Area_px"])
        logger.info(f"Spectral analysis complete. Isolated {count} assets.")
        return count, output_image, df

    def _analyze_ai(self, image):
        logger.info(f"Executing Deep Learning Inference (Conf: {self.conf_threshold})...")
        results = self.model(image, conf=self.conf_threshold)
        count = 0
        output_image = image.copy()
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if self.is_visdrone:
                    # VisDrone classes: 3=car, 4=van, 5=truck, 8=bus
                    valid_classes = [3, 4, 5, 8]
                else:
                    # COCO classes: 2=car, 3=motorcycle, 5=bus, 7=truck
                    valid_classes = [2, 3, 5, 7]
                    
                if class_id in valid_classes: 
                    count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    
                    cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 0), 4)
                    label = f"Aerial_Vehicle {conf:.2f}" if self.is_visdrone else f"Vehicle {conf:.2f}"
                    cv2.putText(output_image, label, (x1, max(10, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    
                    self.report_data.append({"Type": "Vehicle", "X": cx, "Y": cy, "Confidence": round(conf, 4)})
                    
        df = pd.DataFrame(self.report_data) if self.report_data else pd.DataFrame(columns=["Type", "X", "Y", "Confidence"])
        logger.info(f"Inference complete. Identified {count} spatial targets.")
        return count, output_image, df
