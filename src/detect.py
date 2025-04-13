
# src/detect.py

import cv2
from ultralytics import YOLO
import numpy as np
import os

# --- Configuration Constants ---
# This is the *default* threshold, can be overridden by function calls
MIN_IOU_THRESHOLD_FOR_LABELING = 0.35 # Minimum IoU between two vehicles

# --- Helper Functions ---

def calculate_iou(box1, box2):
    """Calculates Intersection over Union (IoU) between two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    x_left = max(x1_1, x1_2); y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2); y_bottom = min(y2_1, y2_2)
    if x_right < x_left or y_bottom < y_top: return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area
    if union_area == 0: return 0.0
    iou = intersection_area / union_area
    return iou

# --- Find Highest Interaction Pair ---

def find_highest_interaction(detections, min_iou_threshold):
    """
    Finds the pair of vehicles with the highest IoU above a threshold.
    Args:
        detections: List of tuples [(class_name, confidence, box), ...].
        min_iou_threshold: The minimum IoU to consider for labeling.
    Returns: Tuple (max_iou_score, involved_boxes, involved_indices)
    """
    vehicles = []; vehicle_indices = []
    relevant_classes = ['car', 'truck', 'bus', 'motorcycle']
    for i, (class_name, conf, box) in enumerate(detections):
        if class_name in relevant_classes:
            vehicles.append({'box': box})
            vehicle_indices.append(i)
    if len(vehicles) < 2: return 0.0, [], []
    max_iou_found = 0.0; best_pair_indices = []; original_indices = []
    for i in range(len(vehicles)):
        for j in range(i + 1, len(vehicles)):
            iou = calculate_iou(vehicles[i]['box'], vehicles[j]['box'])
            if iou > max_iou_found:
                max_iou_found = iou
                best_pair_indices = [i, j]
    if max_iou_found >= min_iou_threshold and best_pair_indices:
        involved_boxes = [list(vehicles[idx]['box']) for idx in best_pair_indices]
        original_indices = [vehicle_indices[idx] for idx in best_pair_indices]
        return max_iou_found, involved_boxes, original_indices
    else:
        return 0.0, [], []

# --- Model Loading ---

def load_model(model_path):
    """Loads the YOLOv8 model, handling potential relative paths."""
    try:
        # Check absolute path first
        if not os.path.isabs(model_path):
             # If not absolute, assume relative to project root (one level up from src)
             script_dir = os.path.dirname(__file__) # Gets the directory of detect.py (src)
             model_path = os.path.join(script_dir, '..', model_path) # Go up one level
             model_path = os.path.abspath(model_path) # Get absolute path

        if not os.path.exists(model_path):
             print(f"Error: Model file not found at resolved path: {model_path}")
             return None

        model = YOLO(model_path)
        print(f"Model loaded successfully from {model_path}.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# --- Main Detection and Labeling Function ---

def detect_and_label_interaction(model, frame, min_iou_threshold=MIN_IOU_THRESHOLD_FOR_LABELING):
    """
    Detects objects, finds interaction, labels severity, highlights vehicles.
    Args:
        model: The loaded YOLO model.
        frame: The input image/frame (NumPy array).
        min_iou_threshold: Override for the minimum IoU threshold.
    Returns: Tuple (results, annotated_frame, detections_list, severity_score)
    """
    if model is None: return None, frame, [], 0.0
    detections_list = []; annotated_frame = frame.copy()
    results = model(frame)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = round(float(box.conf[0]), 2); cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            detections_list.append((class_name, conf, (x1, y1, x2, y2)))

    severity_score, involved_boxes, involved_indices = find_highest_interaction(detections_list, min_iou_threshold)

    default_color = (0, 255, 0); involved_color = (0, 0, 255)
    for i, (class_name, conf, box) in enumerate(detections_list):
        x1, y1, x2, y2 = box
        color = involved_color if i in involved_indices else default_color
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        # Optional: Add class name label to non-involved boxes
        # if color == default_color:
        #    cv2.putText(annotated_frame, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    if severity_score > 0 and involved_boxes:
        label_text = f"Severity {severity_score * 100:.1f}%"
        label_x, label_y = involved_boxes[0][0], involved_boxes[0][1] - 10
        (tw, th), bl = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        bg_y1=max(0,label_y-th-bl); bg_y2=max(0,label_y+bl); bg_x1=max(0,label_x); bg_x2=max(0,label_x+tw)
        cv2.rectangle(annotated_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0,0,0), -1)
        text_y = max(th, label_y - bl//2)
        cv2.putText(annotated_frame, label_text, (label_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, involved_color, 2)

    return results, annotated_frame, detections_list, severity_score

# --- Direct Execution Block (Optional: For quick testing within detect.py) ---
if __name__ == '__main__':
    print("Running detect.py directly. This is for basic testing.")
    print("Use image_test.py, video_test.py, or webcam_test.py for proper CLI usage.")

    # Configuration for direct test
    test_model_path = '../model/yolov8s.pt' # Relative path from src
    # *** CHANGE THIS TO YOUR TEST IMAGE FOR DIRECT RUN ***
    test_image_path_direct = 'C:/Users/rithi/OneDrive/Desktop/job/vista/image2.jpg'
    test_iou_threshold = 0.35 # Use the default or set a specific one here

    model = load_model(test_model_path)
    if model:
        frame = cv2.imread(test_image_path_direct)
        if frame is not None:
            _, annotated_frame, _, score = detect_and_label_interaction(model, frame, test_iou_threshold)
            print(f"Direct test score: {score}")
            cv2.imshow("Direct Detect.py Test", annotated_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"Direct test error: Could not read image {test_image_path_direct}")