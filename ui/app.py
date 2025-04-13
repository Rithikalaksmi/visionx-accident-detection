# ui/app.py

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sys
import os
import tempfile
try:
    import magic # For file validation
    magic_available = True
except ImportError:
    magic_available = False
    print("WARN: python-magic library not found. File content type validation will be skipped.")

# --- Page Configuration (Set First) ---
st.set_page_config(
    page_title="Accident Detector ",
    page_icon="💥", # Favicon emoji
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
project_root = os.path.abspath(os.path.join(current_dir, '..')) # Get project root
if src_dir not in sys.path:
    sys.path.append(src_dir)
    print(f"Added '{src_dir}' to sys.path")

# --- Import Core Logic ---
# Import YOLO + IoU logic
yolo_logic_available = False
try:
    from detect import load_model as load_yolo_model, detect_and_label_interaction, MIN_IOU_THRESHOLD_FOR_LABELING
    yolo_logic_available = True
    print("YOLO+IoU logic imported successfully.")
except ImportError as e:
    print(f"WARN: Could not import YOLO logic: {e}.")

# Import CNN Class and startapplication function
cnn_logic_available = False
startapplication_available = False
try:
    from video_test.detection import AccidentDetectionModel # type: ignore # Import the class
    cnn_logic_available = True
    print("CNN AccidentDetectionModel class imported successfully.")
    from video_test.video_test import startapplication # Import startapplication
    startapplication_available = True
    print("startapplication function imported successfully.")
except ImportError as e:
    print(f"WARN: Could not import from src.video_test: {e}.")
except ModuleNotFoundError:
     print("WARN: Could not find module src.video_test.")


# --- Configuration ---
DEFAULT_YOLO_MODEL_PATH_REL = "model/yolov8s.pt" # Relative to project root
# Paths for CNN Model Files - RELATIVE TO PROJECT ROOT
CNN_JSON_PATH_REL = "model/model.json"
CNN_WEIGHTS_PATH_REL = "model/model_weights.weights.h5"
# --- Preprocessing constants (Verify these!) ---
APP_CNN_TARGET_WIDTH = 250
APP_CNN_TARGET_HEIGHT = 250
APP_ACCIDENT_CLASS_INDEX = 0 # Verify Index (0 or 1?)

# --- Security Validation Settings ---
MAX_IMAGE_SIZE_MB = 15
MAX_VIDEO_SIZE_MB = 150
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024
MAX_VIDEO_SIZE_BYTES = MAX_VIDEO_SIZE_MB * 1024 * 1024
ALLOWED_IMAGE_MIME_TYPES = ['image/jpeg', 'image/png', 'image/bmp']
ALLOWED_VIDEO_MIME_TYPES = ['video/mp4', 'video/x-msvideo', 'video/quicktime', 'video/x-matroska', 'video/avi']
ALLOWED_IMAGE_EXTENSIONS = ["png", "jpg", "jpeg", "bmp"]
ALLOWED_VIDEO_EXTENSIONS = ["mp4", "avi", "mov", "mkv"]

# --- Main App Title & Intro ---
st.title("💥Accident Detection system💥")
st.divider()
st.markdown(
    """
    Welcome! This application analyzes visual media for potential vehicle accidents using different AI techniques.
    Select a tab below to begin.
    """
)
with st.expander("ℹ️ How It Works"):
    st.markdown(
        """
        *   **Image & Webcam Tabs:** Utilize **YOLOv8** + **IoU Heuristic** to flag potential interactions based on bounding box overlap. *Severity % is based on overlap, not damage.*
        *   **Video Tab:** Employs a dedicated **CNN** to classify frames. The analysis is launched via `startapplication`, likely opening a separate window.
        """
    )
st.divider()

# --- Security Helper Function ---
def validate_uploaded_file(uploaded_file, allowed_mime_types, max_size_bytes, file_type_label="File"):
    """Validates uploaded file size and content type."""
    if uploaded_file is None: return False, f"No {file_type_label} uploaded."
    if uploaded_file.size > max_size_bytes: return False, f"{file_type_label} is too large (Max: {max_size_bytes // (1024*1024)} MB)."
    if not magic_available:
         print("Skipping magic number validation (library unavailable).")
         return True, None
    try:
        file_content_start = uploaded_file.read(2048); uploaded_file.seek(0)
        actual_mime_type = magic.from_buffer(file_content_start, mime=True)
        if actual_mime_type not in allowed_mime_types: return False, f"Invalid {file_type_label} type ({actual_mime_type}). Allowed: {', '.join(allowed_mime_types)}."
    except Exception as e: return False, f"Could not verify {file_type_label} type. Error: {e}"
    return True, None

# --- Model Loading (Cached) ---
@st.cache_resource
def load_models_cached():
    models = {'yolo': None, 'cnn': None}
    print("Attempting to load models (cached)...")
    abs_project_root = project_root

    # Load YOLO
    if yolo_logic_available:
        try:
            abs_yolo_path = os.path.join(abs_project_root, DEFAULT_YOLO_MODEL_PATH_REL)
            models['yolo'] = load_yolo_model(abs_yolo_path)
        except Exception as e: print(f"Error loading YOLO model: {e}")

    # Load CNN
    if cnn_logic_available:
        try:
            # Construct absolute paths to CNN files in model/ directory
            abs_cnn_json_path = os.path.join(abs_project_root, CNN_JSON_PATH_REL)
            abs_cnn_weights_path = os.path.join(abs_project_root, CNN_WEIGHTS_PATH_REL)
            print(f"Attempting to init CNN with JSON: {abs_cnn_json_path}")
            print(f"Attempting to init CNN with Weights: {abs_cnn_weights_path}")
            # Instantiate the class by passing the absolute paths
            models['cnn'] = AccidentDetectionModel(abs_cnn_json_path, abs_cnn_weights_path)
        except Exception as e: print(f"Error initializing CNN model object: {e}")

    return models

models = load_models_cached()
yolo_model = models['yolo']
cnn_model_object = models['cnn'] # This is the instantiated AccidentDetectionModel object

# --- Sidebar ---
with st.sidebar:
    # st.image("path/to/your/logo.png", width=100) # Optional logo
    st.header("⚙️ Configuration")
    st.divider()
    st.subheader("Models Status")
    if yolo_model: st.success("✅ YOLO Model Loaded")
    else: st.warning("⚠️ YOLO Model Unavailable")
    if cnn_model_object: st.success("✅ CNN Model Initialized")
    else: st.warning("⚠️ CNN Model Unavailable")

    st.divider()
    st.subheader("Analysis Parameters")
    iou_threshold_value = MIN_IOU_THRESHOLD_FOR_LABELING
    if yolo_logic_available:
        iou_threshold_value = st.slider("IoU Threshold (Image/Webcam)", 0.05, 0.95, MIN_IOU_THRESHOLD_FOR_LABELING, 0.05)

    cnn_display_threshold_value = 0.5 # Default needed even if logic unavailable
    if cnn_logic_available:
        cnn_display_threshold_value = st.slider("CNN Display Threshold (Video)", 0.0, 1.0, 0.5, 0.05)
        st.caption(f"(Video assumes 'Accident' is Class Index: **{APP_ACCIDENT_CLASS_INDEX}**)")

    st.divider()
    st.info("Adjust parameters and select a tab above.")
    st.markdown("---")
    st.caption("VisionX Hackathon Project")


# --- UI Tabs ---
tab1, tab2, tab3 = st.tabs(["🖼️ Image Analysis", "🎬 Video Analysis", "📷 Live Webcam"])

# --- Image Analysis Tab ---
with tab1:
    st.subheader("🖼️ Analyze an Image (YOLO + IoU Heuristic)")
    st.markdown("Upload a single image to detect vehicles and flag potential interactions based on bounding box overlap.")
    st.divider()

    if not yolo_logic_available or not yolo_model:
        st.error("❌ YOLO Model/Logic not available. Cannot perform Image Analysis.")
    else:
        uploaded_image = st.file_uploader("Select Image File", type=ALLOWED_IMAGE_EXTENSIONS, key="img_upload",
                                          label_visibility="collapsed", help=f"Max size: {MAX_IMAGE_SIZE_MB} MB.")

        if uploaded_image is not None:
            is_valid, error_msg = validate_uploaded_file(uploaded_image, ALLOWED_IMAGE_MIME_TYPES, MAX_IMAGE_SIZE_BYTES, "Image")

            if not is_valid:
                st.error(f"❌ Validation Failed: {error_msg}")
            else:
                st.success("✅ Image validated successfully.")
                col1, col2 = st.columns(2)
                try:
                    image = Image.open(uploaded_image).convert('RGB')
                    frame = np.array(image)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    with col1:
                        st.image(frame, caption='Original Uploaded Image', use_column_width=True)

                    analyze_img_button = st.button("✨ Analyze Image Interaction", key="img_analyze", use_container_width=True)
                    placeholder_col2 = col2.empty() # Placeholder for results image

                    if analyze_img_button:
                        with st.spinner(f"🧠 Analyzing (IoU Threshold: {iou_threshold_value:.2f})..."):
                            _, annotated_frame_bgr, detections, score = detect_and_label_interaction(
                                yolo_model, frame_bgr, min_iou_threshold=iou_threshold_value
                            )
                            annotated_frame_rgb = cv2.cvtColor(annotated_frame_bgr, cv2.COLOR_BGR2RGB)

                        placeholder_col2.image(annotated_frame_rgb, caption='Analysis Result', use_column_width=True)

                        st.divider()
                        result_cols = st.columns(2)
                        with result_cols[0]:
                            if score > 0:
                                st.metric(label="Detected Interaction Severity (IoU)", value=f"{score * 100:.1f}%")
                                st.warning("🚨 Potential vehicle interaction detected based on overlap.", icon="⚠️")
                            else:
                                st.metric(label="Detected Interaction Severity (IoU)", value="None")
                                st.success(f"✅ No significant interaction detected (Threshold: {iou_threshold_value:.2f}).", icon="👍")
                        with result_cols[1]:
                             st.metric(label="Total Objects Detected", value=len(detections) if detections else 0)

                        with st.expander("View Raw Detection Data"):
                             if detections: st.dataframe(detections, column_config={"2": "Bounding Box (x1,y1,x2,y2)"}, use_container_width=True)
                             else: st.write("No objects detected.")

                except Exception as e:
                    st.error(f"😞 Error processing image: {e}")

# --- Video Analysis Tab ---
with tab2:
    st.subheader("🎬 Analyze a Video File (CNN Classification)")
    st.markdown("Upload a video file. Analysis uses a dedicated CNN model and is run via `startapplication`, likely opening in a separate window.")
    st.divider()

    # Check if *both* the CNN object and startapplication are available
    if not cnn_logic_available or not cnn_model_object or not startapplication_available:
        st.error("❌ CNN Model/Logic or the required `startapplication` function is not available.")
    else:
        uploaded_video = st.file_uploader("Select Video File", type=ALLOWED_VIDEO_EXTENSIONS, key="vid_upload_cnn", label_visibility="collapsed", help=f"Max size: {MAX_VIDEO_SIZE_MB} MB.")

        if uploaded_video is not None:
            is_valid, error_msg = validate_uploaded_file(uploaded_video, ALLOWED_VIDEO_MIME_TYPES, MAX_VIDEO_SIZE_BYTES, "Video")

            if not is_valid:
                 st.error(f"❌ Validation Failed: {error_msg}")
            else:
                st.success("✅ Video validated successfully.")
                tfile = None; video_path = None
                try:
                    # Save uploaded video to temp file
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_video.name.split(".")[-1]}')
                    uploaded_video.seek(0) # Reset pointer after validation read
                    tfile.write(uploaded_video.read()); video_path = tfile.name; tfile.close()
                    st.info(f"Video ready to be processed externally.")
                    st.info("Analysis via `startapplication` runs separately. Results may not display in this Streamlit window.")

                    if st.button("🚀 Start External Video Analysis", key="video_process_cnn", use_container_width=True):
                        if video_path and os.path.exists(video_path):
                            try:
                                st.info("⏳ Attempting to launch external analysis...")
                                startapplication(video_path) # Call the external function
                                st.success("✅ Analysis process initiated (check external window/console).")
                            except Exception as e:
                                st.error(f"😞 Error running startapplication: {e}")
                        else:
                             st.error("❌ Temporary video file path error.")
                except Exception as e:
                    st.error(f"😞 Error preparing video: {e}")
                finally: # Cleanup temp file
                     if video_path and os.path.exists(video_path):
                         try: os.remove(video_path); print(f"Cleaned up temp video file: {video_path}")
                         except Exception as del_e: st.warning(f"Could not delete temp file {video_path}: {del_e}")


# --- Webcam Feed Tab ---
with tab3:
    st.subheader("📷 Live Webcam Analysis (YOLO + IoU Heuristic)")
    st.markdown("Activate your webcam to perform real-time vehicle interaction detection using the YOLO+IoU method.")
    st.divider()

    if not yolo_logic_available or not yolo_model:
        st.error("❌ YOLO Model/Logic not available for Webcam Analysis.")
    else:
        col_webcam_1, col_webcam_2 = st.columns([1, 3]) # Control column smaller

        with col_webcam_1:
             st.write("**Controls**")
             run_webcam = st.toggle("Activate Feed", key="webcam_run_yolo", value=False)
             status_placeholder = st.empty()
             metrics_placeholder = st.empty()
             status_placeholder.info("⚪ Webcam Inactive") # Initial status

        with col_webcam_2:
             stframe_webcam = st.empty() # Placeholder for the video feed image

        camera_index = 0
        cap_webcam = None

        if run_webcam:
            try:
                cap_webcam = cv2.VideoCapture(camera_index)
                if not cap_webcam.isOpened():
                    status_placeholder.error(f"❌ Error opening webcam {camera_index}.")
                else:
                    status_placeholder.success("🟢 Webcam Active")
                    while run_webcam: # Loop controlled by toggle state
                        ret, frame = cap_webcam.read()
                        if not ret:
                            status_placeholder.error("❌ Capture Failed.")
                            break

                        # Use threshold from sidebar
                        _, annotated_frame, _, score = detect_and_label_interaction(yolo_model, frame, min_iou_threshold=iou_threshold_value)

                        # Display frame
                        stframe_webcam.image(annotated_frame, channels="BGR")

                        # Display metrics in the control column
                        with metrics_placeholder.container():
                             if score > 0: st.metric(label="Interaction Score (IoU)", value=f"{score * 100:.1f}%")
                             else: st.metric(label="Interaction Score (IoU)", value="None")

                        # Check state inside loop
                        run_webcam = st.session_state.get('webcam_run_yolo', False)
                        if not run_webcam:
                             status_placeholder.info("⚪ Feed stopped by user.")
                             break # Exit loop if toggle switched off

            except Exception as e:
                st.error(f"😞 Error during webcam processing: {e}")
            finally:
                 if cap_webcam is not None and cap_webcam.isOpened():
                     cap_webcam.release()
                     print("Webcam released.")
                 # Update status if stopped by user or error
                 if not st.session_state.get('webcam_run_yolo', False): # If toggle is off now
                     status_placeholder.info("⚪ Webcam Inactive.")
                     metrics_placeholder.empty() # Clear metrics

        # Ensure placeholders are cleared if toggle is off initially or turned off
        if not run_webcam:
             stframe_webcam.empty()
             metrics_placeholder.empty()