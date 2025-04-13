# src/image_test.py

import cv2
import os
# import argparse  # No longer needed
import glob
# Import necessary functions from detect.py
from detect import load_model, detect_and_label_interaction, MIN_IOU_THRESHOLD_FOR_LABELING

def process_images(input_path, model_path, output_dir, show_output, iou_threshold):
    """Processes a single image or a directory of images using detect.py logic."""
    # This function remains largely the same as before
    print(f"--- Image Test Script ---")
    model = load_model(model_path) # Uses the load_model which handles path resolution
    if model is None:
        print("Error: Model loading failed. Exiting script.")
        return

    if os.path.isdir(input_path):
        print(f"Processing images in directory: {input_path}")
        image_files = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff'):
            image_files.extend(glob.glob(os.path.join(input_path, ext)))
        if not image_files:
            print(f"No image files found directly in {input_path}.")
            return
        print(f"Found {len(image_files)} images.")
    elif os.path.isfile(input_path):
        print(f"Processing single image: {input_path}")
        image_files = [input_path]
    else:
        print(f"Error: Input path '{input_path}' is not a valid file or directory.")
        return

    output_full_path = None
    if output_dir:
        # Resolve output path relative to project root (assuming script is in src)
        script_dir = os.path.dirname(__file__)
        output_full_path = os.path.abspath(os.path.join(script_dir, '..', output_dir))
        os.makedirs(output_full_path, exist_ok=True)
        print(f"Attempting to save annotated images to: {output_full_path}")

    for img_file in image_files:
        print(f"\nProcessing: {os.path.basename(img_file)}")
        frame = cv2.imread(img_file)
        if frame is None:
            print(f"Warning: Could not read image {img_file}. Skipping.")
            continue

        results, annotated_frame, detections, score = detect_and_label_interaction(
            model, frame, min_iou_threshold=iou_threshold
        )

        if score > 0:
            print(f"-> Interaction Detected! Severity (IoU based): {score * 100:.1f}%")
        else:
            print(f"-> No significant interaction detected (Threshold: {iou_threshold}).")

        if output_full_path:
            try:
                output_filename = os.path.join(output_full_path, f"annotated_{os.path.basename(img_file)}")
                cv2.imwrite(output_filename, annotated_frame)
                print(f"Saved annotated image to {output_filename}")
            except Exception as e:
                 print(f"Error saving image {output_filename}: {e}")

        if show_output:
            cv2.imshow(f"Interaction Test - {os.path.basename(img_file)}", annotated_frame)
            print("Press any key to continue to the next image (or 'q' to stop showing)...")
            key = cv2.waitKey(0)
            if key == ord('q'):
                 print("'q' pressed, stopping image display.")
                 show_output = False # Stop showing for subsequent images
                 cv2.destroyAllWindows()

    if show_output:
        cv2.destroyAllWindows()
    print("\n--- Image processing finished ---")


# --- Main execution block ---
if __name__ == "__main__":

    # --- !!! EDIT THESE VALUES DIRECTLY !!! ---

    # 1. Input Path: Put the FULL path to your image or directory here
    input_path_to_test = "C:/Users/rithi/OneDrive/Desktop/job/vista/image2.jpg"
    # Example for directory:
    # input_path_to_test = "C:/Users/rithi/OneDrive/Desktop/job/vista/test_images_folder"

    # 2. Model Path: Path relative to the PROJECT ROOT directory
    model_path_relative = "model/yolov8s.pt"
    # Example for different model:
    # model_path_relative = "model/yolov8n.pt"

    # 3. Output Directory: Path relative to the PROJECT ROOT directory
    # Set to None to disable saving. Directory will be created if it doesn't exist.
    output_dir_relative = "output_images"
    # Example: Disable saving
    # output_dir_relative = None

    # 4. Show Output Window? True or False
    show_image_window = True

    # 5. IoU Threshold: Set the desired IoU threshold value
    iou_threshold_to_use = 0.35
    # Example: Use a different threshold
    # iou_threshold_to_use = 0.45

    # --- END OF EDITABLE VALUES ---


    # Call the processing function with the values defined above
    print(f"Starting image test with hardcoded settings...")
    print(f"Input: {input_path_to_test}")
    print(f"Model: {model_path_relative}")
    print(f"Output Dir: {output_dir_relative}")
    print(f"Show Window: {show_image_window}")
    print(f"IoU Threshold: {iou_threshold_to_use}")

    process_images(
        input_path=input_path_to_test,
        model_path=model_path_relative,
        output_dir=output_dir_relative,
        show_output=show_image_window,
        iou_threshold=iou_threshold_to_use
    )