import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VALIDATOR_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "lesion_input_validator.keras")

VALIDATOR_IMG_SIZE = (224, 224)
VALIDATOR_THRESHOLD = 0.55

validator_model = load_model(VALIDATOR_MODEL_PATH, compile=False)


def preprocess_validator_image(img_path, target_size=VALIDATOR_IMG_SIZE):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def reject_report_or_collage(img_path):
    img = Image.open(img_path).convert("RGB")
    arr = np.asarray(img.resize((512, 512))).astype(np.float32)

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    white_ratio = ((r > 225) & (g > 225) & (b > 225)).mean()

    gray = (0.299 * r + 0.587 * g + 0.114 * b)
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))

    edge_density = ((gx > 55).mean() + (gy > 55).mean()) / 2
    vertical_line_score = (gx > 70).mean(axis=0).max()
    horizontal_line_score = (gy > 70).mean(axis=1).max()

    if white_ratio > 0.18 and edge_density > 0.035:
        return True

    if vertical_line_score > 0.18 or horizontal_line_score > 0.18:
        return True

    return False


def reject_portrait_or_general_photo(img_path):
    """
    Rejects general photographs of people, clothing, rooms, objects.
    Real dermoscopic/close-up lesion images usually fill most of the frame
    with skin-like pixels and do not contain large colourful clothing/background areas.
    """
    img = Image.open(img_path).convert("RGB")
    img = img.resize((256, 256))
    arr = np.asarray(img).astype(np.float32)

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Skin-like pixels
    skin_mask = (
        (r > 80) &
        (g > 35) &
        (b > 25) &
        (r >= g * 0.80) &
        (g >= b * 0.55) &
        ((r - b) > 8)
    )

    skin_ratio = skin_mask.mean()

    # Non-skin area
    non_skin_ratio = 1 - skin_ratio

    # Colourful clothing/background detector
    max_c = arr.max(axis=2)
    min_c = arr.min(axis=2)
    saturation = (max_c - min_c) / (max_c + 1)

    colourful_non_skin_ratio = ((~skin_mask) & (saturation > 0.28)).mean()

    # Background/object variety detector
    gray = (0.299 * r + 0.587 * g + 0.114 * b)
    global_std = gray.std()

    # Face/portrait/general photo often has low skin fill + high non-skin colourful regions
    if skin_ratio < 0.55 and colourful_non_skin_ratio > 0.18:
        return True

    # If most of the image is non-skin, reject
    if non_skin_ratio > 0.55:
        return True

    # Clothing/room/object photos usually have high scene variation
    if skin_ratio < 0.65 and global_std > 45:
        return True

    return False


def heuristic_reject_non_lesion(img_path):
    try:
        if reject_report_or_collage(img_path):
            return True

        if reject_portrait_or_general_photo(img_path):
            return True

        img = Image.open(img_path).convert("RGB")
        img = img.resize((256, 256))
        arr = np.asarray(img).astype(np.float32)

        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]

        brightness = arr.mean()

        if brightness < 45:
            return True

        # Reject blue digital / infographic images
        blue_ratio = ((b > r + 35) & (b > g + 30)).mean()
        if blue_ratio > 0.35:
            return True

        max_c = arr.max(axis=2)
        min_c = arr.min(axis=2)
        saturation = (max_c - min_c) / (max_c + 1)

        high_saturation_ratio = (saturation > 0.70).mean()
        if high_saturation_ratio > 0.45:
            return True

        gray = (0.299 * r + 0.587 * g + 0.114 * b)
        gx = np.abs(np.diff(gray, axis=1))
        gy = np.abs(np.diff(gray, axis=0))
        edge_density = ((gx > 45).mean() + (gy > 45).mean()) / 2

        if edge_density > 0.28:
            return True

        # Final relaxed skin check for valid pale lesion images
        skin_mask = (
            (r > 80) &
            (g > 35) &
            (b > 25) &
            (r >= g * 0.80) &
            (g >= b * 0.55)
        )

        skin_ratio = skin_mask.mean()

        if skin_ratio < 0.15:
            return True

        return False

    except Exception:
        return True


def validate_lesion_image(img_path):
    if heuristic_reject_non_lesion(img_path):
        return False, 0.0

    img_array = preprocess_validator_image(img_path)
    prediction = validator_model.predict(img_array, verbose=0)[0][0]

    is_valid = prediction >= VALIDATOR_THRESHOLD
    return is_valid, float(prediction)