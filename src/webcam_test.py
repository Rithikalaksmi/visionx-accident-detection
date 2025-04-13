# src/webcam_test.py

import cv2
import os
# import argparse # No longer needed
# Import necessary functions from detect.py
from detect import load_model, detect_and_label_interaction, MIN_IOU_THRESHOLD_FOR_LABELING

def process_webcam(camera_index, model_path, iou_threshold):
    """Processes webcam feed using detect.py logic."""
    # This function remains largely the same as before

    print(f"--- Webcam Test Script ---")
    model = load_model(model_path) # Uses the load_model which handles path resolution
    if model is None:
        print("Error: Model loading failed. Exiting script.")
        return

    print(f"Attempting to open webcam (index: {camera_index})...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open webcam {camera_index}. Check if it's connected or try a different index.")
        return

    print("Webcam opened successfully. Press 'q' in the window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam. Exiting.")
            break # Exit loop if frame capture fails

        # Perform detection and labeling
        results, annotated_frame, detections, score = detect_and_label_interaction(
            model, frame, min_iou_threshold=iou_threshold
        )

        # Show the frame
        cv2.imshow(f"Interaction Test - Webcam {camera_index} (Press 'q' to quit)", annotated_frame)

        # Check for 'q' key press to exit - waitKey must be > 0 to process events
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n'q' pressed, exiting.")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam feed closed.")


# --- Main execution block ---
if __name__ == "__main__":

    # --- !!! EDIT THESE VALUES DIRECTLY !!! ---

    # 1. Camera Index: 0 is usually the default built-in webcam.
    # Try 1, 2, etc., if you have multiple cameras.
    camera_index_to_use = 0

    # 2. Model Path: Path relative to the PROJECT ROOT directory
    model_path_relative = "model/yolov8s.pt"
    # Example for different model:
    # model_path_relative = "model/yolov8n.pt"

    # 3. IoU Threshold: Set the desired IoU threshold value
    iou_threshold_to_use = 0.35
    # Example: Use a different threshold
    # iou_threshold_to_use = 0.45

    # --- END OF EDITABLE VALUES ---


    # Call the processing function with the values defined above
    print(f"Starting webcam test with hardcoded settings...")
    print(f"Camera Index: {camera_index_to_use}")
    print(f"Model: {model_path_relative}")
    print(f"IoU Threshold: {iou_threshold_to_use}")


    process_webcam(
        camera_index=camera_index_to_use,
        model_path=model_path_relative,
        iou_threshold=iou_threshold_to_use
    )