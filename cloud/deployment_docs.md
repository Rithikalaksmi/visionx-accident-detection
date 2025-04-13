# Cloud Deployment Instructions

Deploying this application involves considerations for both the Streamlit UI and the different model types (YOLO and CNN) and their dependencies. Streamlit Cloud is often the simplest for the UI itself.

## Important Considerations Before Deployment

*   **CNN Model Location:** The CNN model (`model.json`, `model_weights.weights.h5`) is currently expected within the `src/video_test/` directory by the `AccidentDetectionModel` class. Ensure this structure is maintained in your repository and deployment environment.
*   **`python-magic` Dependencies:** The `python-magic` library used for file validation requires the underlying `libmagic` C library.
    *   **Streamlit Cloud:** May or may not have `libmagic` pre-installed. If deployment fails related to `magic`, this might be the cause. Consider making validation optional or finding alternatives if it's problematic.
    *   **EC2/Docker:** You *must* install `libmagic` (e.g., `sudo apt-get install libmagic1` on Debian/Ubuntu) in the instance or container *before* installing `python-magic`.
*   **TensorFlow:** The CNN requires TensorFlow. Ensure `tensorflow` or `tensorflow-cpu` is in `requirements.txt`. TensorFlow can consume significant resources.

## Option 1: Streamlit Cloud (Recommended for UI)

**Prerequisites:**
*   Project pushed to GitHub with the correct structure.
*   `requirements.txt` is accurate (including `tensorflow`, `python-magic`, `ultralytics`, etc.).
*   YOLO model (`.pt`) in the `model/` directory (keep size under ~25MB or use download logic).
*   CNN model (`.json`, `.h5`) in the `src/video_test/` directory.

**Steps:**
1.  **Sign up/Log in:** Go to [share.streamlit.io](https://share.streamlit.io/) and connect GitHub.
2.  **Deploy an app:** Click "New app".
3.  **Connect Repository:** Select the repo, branch, and set **Main file path** to `ui/app.py`.
4.  **Deploy:** Click "Deploy!". Monitor the build logs carefully.
    *   **Potential Issues:** Watch for errors related to installing `tensorflow` (can be large/slow) or `python-magic` (missing `libmagic`). If `python-magic` fails, you might need to remove the validation feature or deploy using Docker/EC2 where you control the system libraries. Errors related to the CNN model might occur if the paths within `src/video_test/detection.py` are not resolving correctly in the Streamlit Cloud environment.

**Security:** Provides automatic HTTPS.

## Option 2: AWS EC2 Instance

Gives full control over the environment, necessary if system libraries like `libmagic` are essential and unavailable on Streamlit Cloud.

**Prerequisites:**
*   AWS Account, EC2 knowledge.

**Steps:**
1.  **Launch EC2 Instance:**
    *   Choose AMI (e.g., Ubuntu Server).
    *   Select instance type (consider `t3.large` or `c5.large` or larger due to TensorFlow's requirements).
    *   Configure Security Group: Allow inbound TCP on port `8501` (Streamlit) and `22` (SSH).
2.  **Connect via SSH & Setup:**
    ```bash
    # Update & Install Basics
    sudo apt update && sudo apt upgrade -y
    sudo apt install python3-pip python3-venv git -y

    # Install System Dependencies FOR BOTH MODELS/VALIDATION
    sudo apt install libgl1-mesa-glx libglib2.0-0 -y # For OpenCV
    sudo apt install libmagic1 -y # <<< FOR python-magic VALIDATION

    # Clone Repository
    git clone <your-github-repo-url>
    cd <your-repo-directory-name> # e.g., accident-detection-project

    # Setup Python Environment
    python3 -m venv venv
    source venv/bin/activate

    # Install Python Requirements
    pip install -r requirements.txt

    # Verify Model Files are Present (Git should have pulled them)
    ls model/
    ls src/video_test/
    ```
3.  **Run the Application:**
    ```bash
    # Run Streamlit allowing external connections
    streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0
    ```
    *   Use `nohup`, `screen`, or `tmux` for background execution.
4.  **Access:** `http://<your-ec2-public-ip>:8501`.
5.  **HTTPS Setup (Optional but Recommended):** Requires a domain name, reverse proxy (Nginx/Caddy) + Let's Encrypt, or an AWS ALB.

## Option 3: Docker

Best for consistency and managing complex dependencies like `libmagic`.

1.  **Create `Dockerfile`:** In the project root.
    ```dockerfile
    # Choose appropriate base image (e.g., Python + TF or build TF yourself)
    FROM python:3.10-slim # Or a version compatible with your TF

    WORKDIR /app

    # Install System Dependencies (OpenCV, libmagic)
    RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libmagic1 \
        && rm -rf /var/lib/apt/lists/*

    # Copy requirements first for layer caching
    COPY requirements.txt .
    # Install Python dependencies (Consider using --no-cache-dir)
    RUN pip install --no-cache-dir -r requirements.txt

    # Copy the entire project context
    # Ensure your .dockerignore excludes venv, __pycache__, etc.
    COPY . .

    # Expose the Streamlit port
    EXPOSE 8501

    # Command to run the Streamlit application
    # Use healthcheck if deploying to orchestrators
    HEALTHCHECK CMD streamlit hello
    ENTRYPOINT ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
    ```
2.  **Create `.dockerignore`:** (Crucial to avoid copying virtual envs etc.)
    ```
    venv/
    __pycache__/
    *.pyc
    *.pyo
    *.pyd
    .git/
    .gitignore
    .dockerignore
    output_images/
    output_videos/
    *.log
    # Add other files/dirs to ignore
    ```
3.  **Build:** `docker build -t accident-detector-app .`
4.  **Run:** `docker run -p 8501:8501 accident-detector-app`
5.  **Deploy Container:** Use AWS ECS, Fargate, Google Cloud Run, Kubernetes, etc. These platforms often integrate well with load balancers for HTTPS.