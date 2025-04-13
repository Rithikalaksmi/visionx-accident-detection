from tensorflow.keras.models import model_from_json  # type: ignore # Use tensorflow.keras
import numpy as np

class AccidentDetectionModel(object):

    class_nums = ['Accident', "No Accident"]

    def __init__(self, model_json_file, model_weights_file):
        try:
            # Load model architecture
            with open(model_json_file, "r") as json_file:
                loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

            # Load weights
            self.loaded_model.load_weights(model_weights_file)

            # Compile the model
            self.loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        except Exception as e:
            print(f"❌ Failed to initialize AccidentDetectionModel: {e}")
            raise e

    def predict_accident(self, img):
        try:
            preds = self.loaded_model.predict(img)
            label = AccidentDetectionModel.class_nums[np.argmax(preds)]
            return label, preds
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None, None
