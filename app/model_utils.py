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


def is_safe_path(path, base):
    """
    Prevent path traversal by ensuring the resolved path stays
    inside the expected base directory.
    """
    return os.path.realpath(path).startswith(os.path.realpath(base))


def is_valid_image(path):
    """
    Basic image validation by checking file existence and allowed extension.
    """
    return os.path.exists(path) and path.lower().endswith((".jpg", ".jpeg", ".png"))


# Security: check model file exists before loading
if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
    raise FileNotFoundError(f"Model file missing or corrupted: {MODEL_PATH}")


# Load final improved EfficientNetB0 model for inference
model = load_model(MODEL_PATH, compile=False)


def preprocess_image(img_path, target_size=IMG_SIZE):
    """
    Validate the image path before loading and preprocessing.
    """
    if not is_safe_path(img_path, BASE_DIR):
        raise ValueError("Unsafe file path detected.")

    if not is_valid_image(img_path):
        raise ValueError("Invalid image file.")

    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_image(img_path):
    """
    Run the full prediction pipeline and return:
    label, confidence, preprocessed image array, model
    """
    img_array = preprocess_image(img_path)
    prediction = float(model.predict(img_array, verbose=0)[0][0])

    label = "malignant" if prediction > THRESHOLD else "benign"
    confidence = prediction if prediction > THRESHOLD else 1 - prediction

    return label, confidence, img_array, model