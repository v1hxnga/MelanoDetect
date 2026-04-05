import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VALIDATOR_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "lesion_input_validator.keras")

VALIDATOR_IMG_SIZE = (224, 224)
VALIDATOR_THRESHOLD = 0.80

validator_model = load_model(VALIDATOR_MODEL_PATH, compile=False)

def preprocess_validator_image(img_path, target_size=VALIDATOR_IMG_SIZE):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def validate_lesion_image(img_path):
    img_array = preprocess_validator_image(img_path)
    prediction = validator_model.predict(img_array, verbose=0)[0][0]

    is_valid = prediction >= VALIDATOR_THRESHOLD
    return is_valid, float(prediction)