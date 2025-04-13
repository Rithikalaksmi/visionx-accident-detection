import cv2
from .detection import AccidentDetectionModel
import numpy as np
import os
import platform

# Initialize the model
model = AccidentDetectionModel("src/video_test/model.json", 
                               'src/video_test/model_weights.weights.h5')
font = cv2.FONT_HERSHEY_SIMPLEX

# This version accepts the video path as argument
def startapplication(video_path):
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = video.read()
        if not ret:
            print("End of video or failed to read frame.")
            break

        try:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            roi = cv2.resize(gray_frame, (250, 250))
            pred, prob = model.predict_accident(roi[np.newaxis, :, :])

            if pred == "Accident":
                prob = round(prob[0][0] * 100, 2)
                if prob > 90:
                    if platform.system() == 'Windows':
                        import winsound
                        winsound.Beep(1000, 500)
                    else:
                        os.system("say beep")

                cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
                cv2.putText(frame, f"{pred} {prob}%", (20, 30), font, 1, (255, 255, 0), 2)

            cv2.imshow('Video', frame)

            if cv2.waitKey(33) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"Error during prediction: {e}")
            continue

    video.release()
    cv2.destroyAllWindows()
