# Security Considerations & Features

This document outlines security measures implemented or considered for the Accident Interaction Detection project, following the VisionX Hackathon guidelines. The application uses two main methods: YOLOv8+IoU (Images/Webcam) and a separate CNN classifier (Video).

## Implemented Features

1.  **Input Validation for File Uploads (Enhanced):**
    *   **Extension Filtering:** The Streamlit `st.file_uploader` component in `ui/app.py` uses the `type` parameter for basic filtering based on common extensions (`ALLOWED_IMAGE_EXTENSIONS`, `ALLOWED_VIDEO_EXTENSIONS`).
    *   **File Size Limits:** Uploads are checked against maximum size limits (`MAX_IMAGE_SIZE_BYTES`, `MAX_VIDEO_SIZE_BYTES`) defined in `ui/app.py`. This helps mitigate basic resource exhaustion attacks from excessively large uploads.
    *   **Content Type Verification (Magic Numbers):** The actual content type of uploaded files is verified using the `python-magic` library by inspecting the initial bytes of the file (`magic.from_buffer`). This checks against a list of allowed MIME types (`ALLOWED_IMAGE_MIME_TYPES`, `ALLOWED_VIDEO_MIME_TYPES`) and provides stronger validation than relying solely on file extensions, helping prevent the upload of mislabeled malicious files (e.g., an executable renamed to `.jpg`). Files failing validation are rejected with an error message before further processing. *Note: `python-magic` may require the `libmagic` system library on Linux/macOS or the `python-magic-bin` package on Windows.*
    *   **Mitigation:** These combined checks significantly reduce the risk of users uploading excessively large files or non-media files disguised with media extensions.

2.  **HTTPS (Via Deployment Platform):**
    *   The core application code does not implement HTTPS directly.
    *   **Implementation:** HTTPS relies on the chosen deployment platform. If deployed using **Streamlit Cloud**, HTTPS is automatically configured and enforced. If deployed on **AWS EC2** behind a properly configured Load Balancer (ALB) or reverse proxy (Nginx/Caddy with certificates), HTTPS will be enabled.
    *   **Mitigation:** Encrypts data in transit between the user's browser and the server, protecting against eavesdropping and modification.

3.  **Temporary File Handling:**
    *   Uploaded videos intended for the CNN analysis are saved to temporary files using Python's `tempfile.NamedTemporaryFile`.
    *   Robust cleanup using a `try...finally` block ensures `os.remove()` is called on the temporary file path even if errors occur during processing, minimizing leftover files on the server.

## Optional Security Measures (Considered / Not Implemented)

*   **Input Sanitization for External Process:**
    *   *Status:* Basic path validation exists (tempfile generation).
    *   *Consideration:* The `startapplication` function in `src/video_test/video_test.py` receives a file path. While `tempfile` generates safe paths, if this function were more complex (e.g., taking other user inputs), more rigorous sanitization would be needed within `startapplication` itself to prevent command injection or path traversal if it interacts with the shell or filesystem based on user input.
*   **Basic Authentication / Login:**
    *   *Status:* Not implemented.
    *   *Consideration:* For private deployments or sensitive data, user authentication is recommended.
*   **API Key Management:**
    *   *Status:* Not applicable currently.
    *   *Consideration:* Essential if integrating external cloud services. Use secure methods (env vars, secrets managers).

## Other Security Considerations

*   **Resource Exhaustion (Processing):**
    *   *Risk:* While upload size is limited, intensive processing by either the YOLO/IoU logic or the external CNN `startapplication` script could still consume significant CPU/memory.
    *   *Possible Mitigations (Not Implemented):* Limiting video processing duration, implementing timeouts within processing functions or for the external process call.
*   **External Script Security (`startapplication`):**
    *   *Risk:* The security of the video analysis tab depends significantly on the code within the `startapplication` function and any libraries it uses. Vulnerabilities or insecure practices (like unsafe file handling, lack of error checking) within that script could impact the overall application.
    *   *Mitigation:* Code review and secure coding practices should be applied to the `src/video_test/video_test.py` and `src/video_test/detection.py` files.
*   **Model Security:**
    *   *Risk:* Loading compromised model files (`.pt`, `.json`, `.h5`) could lead to security issues.
    *   *Mitigation:* Ensure models are obtained from trusted sources (official releases, your own verified training).
*   **Dependency Security:**
    *   *Risk:* Vulnerabilities in libraries (`streamlit`, `opencv`, `numpy`, `tensorflow`, `ultralytics`, `python-magic`, etc.).
    *   *Mitigation:* Regularly update dependencies (`pip install -U ...`) and consider using security scanning tools (`safety check`, Dependabot).