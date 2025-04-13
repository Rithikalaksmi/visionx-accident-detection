# VisionX: Accident Detection Project 

This project, developed for the VISTA VisionX Hackathon, aims to detect potential traffic accidents using computer vision techniques. It utilizes two distinct approaches:

1.  **YOLOv8 Object Detection + IoU Heuristic:** For analyzing single images, directories of images (via CLI), and live webcam feeds. This method detects vehicles and flags potential interactions based on the overlap (Intersection over Union - IoU) of their bounding boxes.
2.  **Convolutional Neural Network (CNN) Classification:** For analyzing video files. This method uses a dedicated CNN model (located in `src/video_test/`) trained presumably to classify frames as containing an accident or not.

The project provides both a web-based user interface (built with Streamlit) and command-line scripts for testing different input types.

## Features

*   **Image Analysis (Single/Directory):** Uses YOLOv8 + IoU heuristic via UI (single) or CLI script (single/directory).
*   **Video Analysis:** Uses a dedicated CNN classifier via UI or CLI script.
*   **Webcam Analysis:** Uses YOLOv8 + IoU heuristic via UI or CLI script.
*   **Web User Interface:** Interactive UI built with Streamlit for easy file uploads and live feed analysis.
*   **Command-Line Scripts:** Separate scripts (`image_test.py`, `video_test.py`, `webcam_test.py` located in `src/`) for testing directly from the terminal (using hardcoded configurations within the scripts).
*   **Security Validations:** Includes basic file size and content-type (magic number) validation for uploads in the Streamlit UI.

## Project Structure

accident-detection-project/
├── model/
│ └── yolov8s.pt # Pre-trained YOLOv8 model (or other variant)
├── src/
│ ├── detect.py # Core logic for YOLOv8 detection + IoU heuristic
│ ├── image_test.py # CLI script for image testing (hardcoded paths)
│ ├── video_test.py # CLI script for video testing (CNN - hardcoded paths)
│ ├── webcam_test.py # CLI script for webcam testing (hardcoded paths)
│ ├── init.py
│ └── video_test/ # Subdirectory for CNN video logic
│ ├── detection.py # Contains AccidentDetectionModel class for CNN
│ ├── model.json # CNN model architecture
│ ├── model_weights.weights.h5 # CNN model weights
│ ├── video_test.py # Potentially the original standalone script (called by UI)
│ └── init.py # Makes video_test a package
├── ui/
│ └── app.py # Main Streamlit application script
├── cloud/
│ └── deployment_docs.md # Deployment instructions (AWS/Streamlit Cloud)
├── output_images/ # Default output directory for image_test.py
├── output_videos/ # Default output directory for video_test.py
├── README.md # This file
├── requirements.txt # Python dependencies
└── SECURITY.md # Security implementation notes


## Prerequisites

*   **Python:** Version 3.9 or higher recommended.
*   **pip:** Python package installer.
*   **venv:** (Recommended) For creating virtual environments.
*   **Git:** For cloning the repository.
*   **System Libraries:**
    *   **For `python-magic` (File Validation):**
        *   **Linux (Debian/Ubuntu):** `sudo apt-get update && sudo apt-get install -y libmagic1`
        *   **macOS:** `brew install libmagic`
        *   **Windows:** No separate system library needed if using `python-magic-bin`.
    *   **For OpenCV:** May require system libraries depending on your OS (e.g., `libgl1` on Linux). Usually handled by `opencv-python-headless` or `opencv-python`.

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd accident-detection-project
    ```

2.  **Create and Activate Virtual Environment (Recommended):**
    ```bash
    # Create venv
    python -m venv venv

    # Activate venv
    # Windows (Command Prompt/PowerShell):
    venv\Scripts\activate
    # Linux/macOS (Bash/Zsh):
    source venv/bin/activate
    ```

3.  **Install Python Dependencies:**
    *   **(Windows Specific):** If you encounter issues installing `python-magic` later, you might need `python-magic-bin`. First, try installing requirements as is. If `magic` import fails in `ui/app.py`, uninstall `python-magic` and install `python-magic-bin`:
        ```bash
        # pip uninstall python-magic # If needed
        # pip install python-magic-bin # If needed
        ```
    *   Install all requirements:
        ```bash
        pip install -r requirements.txt
        ```

4.  **Place Model Files:**
    *   **YOLOv8 Model:** Download a pre-trained YOLOv8 model (e.g., `yolov8s.pt`) from the [Ultralytics GitHub](https://github.com/ultralytics/ultralytics) releases. Place the `.pt` file inside the `model/` directory.
    *   **CNN Model:** Ensure your trained CNN model files (`model.json` and `model_weights.weights.h5`) are placed inside the `src/video_test/` directory.

## Usage

You can interact with the project using the Streamlit Web UI or the command-line test scripts. **Ensure your virtual environment is activated** for all methods.

### 1. Running the Streamlit Web UI

This is the primary way to use the application interactively.

1.  **Navigate to Project Root:** Open your terminal and make sure you are in the main `accident-detection-project` directory (the one containing `ui/`, `src/`, etc.).
2.  **Activate Environment:** (If not already active) `venv\Scripts\activate` or `source venv/bin/activate`.
3.  **Run Streamlit:**
    ```bash
    streamlit run ui/app.py
    ```
4.  **Access UI:** Open your web browser and go to the local URL provided by Streamlit (usually `http://localhost:8501`).
5.  **Use Tabs:**
    *   **Image Tab:** Upload a single image. Analysis uses YOLOv8+IoU. Adjust the IoU threshold using the slider.
    *   **Video Tab:** Upload a single video. Analysis uses the CNN classifier from `src/video_test/`. Press the checkbox to start the analysis (which calls the `startapplication` function, likely opening a separate window). Note: Real-time results from this external process might not show directly in Streamlit.
    *   **Webcam Tab:** Start your default webcam feed. Analysis uses YOLOv8+IoU. Adjust the IoU threshold using the slider.

### 2. Running the Command-Line Test Scripts

These scripts are useful for batch processing or testing without the UI. **Note:** These specific versions (`image_test.py`, `video_test.py`, `webcam_test.py` provided previously) require you to **edit the configuration paths and settings directly within each Python script file** instead of using command-line arguments.

1.  **Navigate to `src` Directory:** Open your terminal and change into the `src` directory:
    ```bash
    cd src
    ```
2.  **Activate Environment:** (If not already active) `..\venv\Scripts\activate` or `source ../venv/bin/activate`.
3.  **Edit the Script:** Open the desired script (`image_test.py`, `video_test.py`, or `webcam_test.py`) in a text editor. Find the section marked `--- !!! EDIT THESE VALUES DIRECTLY !!! ---` within the `if __name__ == "__main__":` block. Modify the variables like `input_path_to_test`, `input_video_path`, `model_path_relative`, `output_dir_relative`, `show_image_window`, `iou_threshold_to_use`, etc., according to your needs. **Remember to use correct path separators for your OS (forward slashes `/` usually work on all systems).**
4.  **Run the Script:**
    ```bash
    # Example for image testing
    python image_test.py

    # Example for video testing (CNN)
    python video_test.py

    # Example for webcam testing
    python webcam_test.py
    ```
5.  **Output:** The script will print information to the console. If configured, it might save annotated files to the output directory (relative to the project root, e.g., `output_images/`) and/or display results in an OpenCV window.

## Models & Methods Explained

*   **Image & Webcam (YOLOv8 + IoU Heuristic):**
    *   Uses the pre-trained YOLOv8 model (`model/yolov8s.pt`) to detect objects like cars, trucks, etc.
    *   Calculates the Intersection over Union (IoU) between pairs of detected vehicle bounding boxes.
    *   If the maximum IoU between any pair exceeds a configurable threshold (`iou_threshold`), it flags the interaction and labels it with a "Severity X%" based on the IoU value.
    *   **Limitation:** This is a **heuristic**. High overlap doesn't always mean a crash, and crashes can occur with low overlap. It doesn't analyze visual damage.

*   **Video (CNN Classification):**
    *   Uses a dedicated CNN model (`src/video_test/model.json`, `src/video_test/model_weights.weights.h5`).
    *   This model processes individual video frames (after resizing and normalization).
    *   It classifies each frame, likely outputting a probability indicating the likelihood of an accident being depicted in that frame.
    *   **Limitation:** Relies heavily on the quality and representativeness of the data it was trained on. Analyzes frames independently, lacking motion context. Preprocessing in `ui/app.py` must exactly match training.

## Security

Basic security measures are implemented in the Streamlit UI (`ui/app.py`):
*   File upload size limits.
*   File content type validation using `python-magic`.
Refer to `SECURITY.md` for more details.

## Limitations and Future Work

*   The IoU heuristic is basic and prone to false positives/negatives.
*   CNN performance depends heavily on training data and preprocessing matching.
*   Single-frame analysis misses crucial motion information.
*   Performance on low-quality or challenging (e.g., night, weather) footage might be poor.
*   The external call for video processing limits UI integration.

**Potential Improvements:**

*   Implement object tracking (e.g., DeepSORT, BoT-SORT) with YOLOv8 to analyze motion patterns.
*   Fine-tune YOLOv8 or the CNN on accident-specific datasets.
*   Train models that explicitly detect vehicle damage.
*   Explore 3D CNNs or Recurrent Neural Networks (RNNs/LSTMs) for video analysis to incorporate temporal information.
*   Integrate video processing directly within Streamlit (if feasible) instead of calling an external script, for better UI feedback.
