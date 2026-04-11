import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "efficientnet_stage1_best.keras"
)

THRESHOLD = 0.45
IMG_SIZE = (260, 260)

# Load final improved EfficientNetB0 model for inference
model = load_model(MODEL_PATH, compile=False)


def preprocess_image(img_path, target_size=IMG_SIZE):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_image(img_path):
    img_array = preprocess_image(img_path)
    prediction = float(model.predict(img_array, verbose=0)[0][0])

    label = "malignant" if prediction > THRESHOLD else "benign"
    confidence = prediction if prediction > THRESHOLD else 1 - prediction

    return label, confidence, img_array, model