from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import os

# Load the pre-trained YOLOv8 model
model = YOLO('best (6).pt')  # Replace with your trained YOLOv8 model path

# Helper function to load an image
def load_image(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')
    return np.array(img)

# Function to detect damages
def detect_damages(image):
    results = model(image)
    damage_info = []

    for detection in results[0].boxes:
        class_id = int(detection.cls.cpu().numpy())
        confidence = float(detection.conf.cpu().numpy())
        bbox = detection.xyxy.cpu().numpy()

        damage_info.append({
            "class_id": class_id,
            "confidence": confidence,
            "bbox": bbox.flatten()
        })

    return results, damage_info

# Function to estimate repair costs
damage_cost_map = {
    0: ('crack_and_hole', 500, 2500),           # AED
    1: ('medium_deformation', 400, 600),       # AED
    2: ('severe_deformation', 1000, 3000),     # AED
    3: ('severe_scratch', 700, 2000),          # AED (per panel)
    4: ('slight_deformation', 200, 400),       # AED
    5: ('slight_scratch', 100, 250),           # AED (per panel)
    6: ('windshield_damage', 200, 3000)        # AED (repair vs. replacement)
}

def estimate_cost(damage_info):
    total_cost = 0
    cost_breakdown = []

    for damage in damage_info:
        class_id = damage['class_id']
        confidence = damage['confidence']
        min_cost, max_cost = damage_cost_map[class_id][1:3]
        estimated_cost = min_cost + (max_cost - min_cost) * confidence
        total_cost += estimated_cost

        cost_breakdown.append({
            "type": damage_cost_map[class_id][0],
            "estimated_cost": estimated_cost
        })

    return total_cost, cost_breakdown

# Main pipeline function to process an image and return JSON and annotated image path
def car_damage_pipeline(image_path):
    image = load_image(image_path)
    image_copy = image.copy()

    # Detect damages
    results, damage_info = detect_damages(image)

    # Draw bounding boxes on the image copy
    for damage in damage_info:
        x_min, y_min, x_max, y_max = map(int, damage['bbox'])
        confidence = damage['confidence']
        class_id = damage['class_id']
        label = f"{damage_cost_map[class_id][0]} ({confidence:.2f})"
        
        # Draw the bounding box and add white background with blue text
        cv2.rectangle(image_copy, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.rectangle(image_copy, (x_min, y_min - 20), (x_max, y_min), (255, 255, 255), -1)
        cv2.putText(image_copy, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Save annotated image
    annotated_image_path = "static/annotated_image.jpg"  # Save it in a static folder
    Image.fromarray(image_copy).save(annotated_image_path)

    # Estimate cost
    total_cost, cost_breakdown = estimate_cost(damage_info)

    # Return results along with the annotated image path
    return {
        "total_cost": total_cost,
        "cost_breakdown": cost_breakdown,
        "annotated_image_path": annotated_image_path
    }