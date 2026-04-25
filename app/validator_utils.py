import os
import numpy as np
from PIL import Image
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


def heuristic_reject_non_lesion(img_path):
    """
    Extra safety filter before model classification.
    Rejects screenshots, diagrams, AI graphics, charts, Grad-CAM collages,
    and non-skin images.
    """
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize((256, 256))
        arr = np.asarray(img).astype(np.float32)

        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]

        brightness = arr.mean()

        # Reject very dark / infographic style images
        if brightness < 60:
            return True

        # Reject blue-heavy digital/AI graphics
        blue_ratio = ((b > r + 25) & (b > g + 20)).mean()
        if blue_ratio > 0.22:
            return True

        # Skin-like pixel ratio
        skin_mask = (
            (r > 80) &
            (g > 40) &
            (b > 25) &
            (r >= g * 0.90) &
            (g >= b * 0.70) &
            ((r - b) > 12)
        )

        skin_ratio = skin_mask.mean()

        # Real lesion/dermoscopic images should contain enough skin-like area
        if skin_ratio < 0.30:
            return True

        # Reject highly saturated heatmap/chart images
        max_c = arr.max(axis=2)
        min_c = arr.min(axis=2)
        saturation = (max_c - min_c) / (max_c + 1)

        high_saturation_ratio = (saturation > 0.58).mean()
        if high_saturation_ratio > 0.30:
            return True

        # Reject screenshots/collages with many sharp edges/text/table lines
        gray = (0.299 * r + 0.587 * g + 0.114 * b)

        gx = np.abs(np.diff(gray, axis=1))
        gy = np.abs(np.diff(gray, axis=0))

        edge_density = ((gx > 35).mean() + (gy > 35).mean()) / 2

        if edge_density > 0.18:
            return True

        # Valid lesion image usually has some darker localized region
        dark_region_ratio = ((r < 135) & (g < 110) & (b < 105)).mean()

        if dark_region_ratio < 0.012:
            return True

        return False

    except Exception:
        return True


def validate_lesion_image(img_path):
    """
    Final validation:
    1. Heuristic safety rejection
    2. Trained lesion validator model
    """

    # Step 1: reject obvious invalid images before ML validator
    if heuristic_reject_non_lesion(img_path):
        return False, 0.0

    # Step 2: trained validator model
    img_array = preprocess_validator_image(img_path)
    prediction = validator_model.predict(img_array, verbose=0)[0][0]

    is_valid = prediction >= VALIDATOR_THRESHOLD
    return is_valid, float(prediction)