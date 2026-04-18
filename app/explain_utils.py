import os
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _is_valid_image_path(path: str) -> bool:
    """
    Check file exists and has an allowed image extension.
    """
    return (
        os.path.exists(path)
        and path.lower().endswith((".jpg", ".jpeg", ".png"))
    )


def _load_rgb(image_path: str) -> np.ndarray:
    """
    Safely load an RGB image after validating the path.
    """
    if not _is_valid_image_path(image_path):
        raise ValueError("Invalid image path or format")

    try:
        img = Image.open(image_path).convert("RGB")
        return np.array(img)
    except Exception:
        raise ValueError("Corrupted or unreadable image")


def _resize_heatmap_to_image(heatmap: np.ndarray, width: int, height: int) -> np.ndarray:
    heatmap_img = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255))
    heatmap_img = heatmap_img.resize((width, height))
    return np.array(heatmap_img).astype("float32") / 255.0


def _safe_mean(arr: np.ndarray, default: float = 0.0) -> float:
    """
    Safe mean for empty arrays.
    """
    if arr.size == 0:
        return default
    return float(np.mean(arr))


def _safe_std(arr: np.ndarray, default: float = 0.0) -> float:
    """
    Safe std for empty arrays.
    """
    if arr.size == 0:
        return default
    return float(np.std(arr))


def _compute_metrics(image_path: str, heatmap: np.ndarray) -> dict:
    rgb = _load_rgb(image_path)
    h, w = rgb.shape[:2]

    heatmap_resized = _resize_heatmap_to_image(heatmap, w, h)
    gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype("float32")

    # Attention mask from Grad-CAM
    attention_mask = heatmap_resized > 0.45
    attention_ratio = float(np.mean(attention_mask))

    if not np.any(attention_mask):
        attention_mask = heatmap_resized > 0.25

    ys, xs = np.where(attention_mask)

    if xs.size == 0 or ys.size == 0:
        x_min, x_max, y_min, y_max = 0, w - 1, 0, h - 1
    else:
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())

    roi_gray = gray[y_min:y_max + 1, x_min:x_max + 1]
    roi_rgb = rgb[y_min:y_max + 1, x_min:x_max + 1]
    roi_mask = attention_mask[y_min:y_max + 1, x_min:x_max + 1]

    focused_gray = roi_gray[roi_mask] if roi_gray.size else np.array([])
    focused_rgb = roi_rgb[roi_mask] if roi_rgb.size else np.array([])

    darkness = 1.0 - (_safe_mean(focused_gray, default=180.0) / 255.0)
    color_variation = _safe_std(focused_rgb, default=20.0) / 255.0

    # Simple gradient strength
    gy, gx = np.gradient(gray)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    focused_grad = grad_mag[attention_mask] if attention_mask.any() else np.array([])
    edge_strength = _safe_mean(focused_grad, default=10.0) / 255.0

    # Simple asymmetry estimate inside bounding box
    asymmetry_score = 0.0
    if roi_gray.size > 0:
        roi_norm = roi_gray / 255.0

        # left-right asymmetry
        mid_x = roi_norm.shape[1] // 2
        left = roi_norm[:, :mid_x]
        right = roi_norm[:, roi_norm.shape[1] - mid_x:]

        if left.size > 0 and right.size > 0:
            right_flipped = np.fliplr(right)
            min_h = min(left.shape[0], right_flipped.shape[0])
            min_w = min(left.shape[1], right_flipped.shape[1])
            lr_diff = np.mean(np.abs(left[:min_h, :min_w] - right_flipped[:min_h, :min_w]))
        else:
            lr_diff = 0.0

        # top-bottom asymmetry
        mid_y = roi_norm.shape[0] // 2
        top = roi_norm[:mid_y, :]
        bottom = roi_norm[roi_norm.shape[0] - mid_y:, :]

        if top.size > 0 and bottom.size > 0:
            bottom_flipped = np.flipud(bottom)
            min_h = min(top.shape[0], bottom_flipped.shape[0])
            min_w = min(top.shape[1], bottom_flipped.shape[1])
            tb_diff = np.mean(np.abs(top[:min_h, :min_w] - bottom_flipped[:min_h, :min_w]))
        else:
            tb_diff = 0.0

        asymmetry_score = float((lr_diff + tb_diff) / 2.0)

    return {
        "attention_ratio": attention_ratio,
        "darkness": darkness,
        "color_variation": color_variation,
        "edge_strength": edge_strength,
        "asymmetry_score": asymmetry_score,
    }


def _attention_phrase(attention_ratio: float) -> str:
    if attention_ratio >= 0.28:
        return "a broader and more scattered attention pattern"
    if attention_ratio >= 0.16:
        return "a moderately distributed attention pattern"
    return "a relatively compact and localized attention pattern"


def _darkness_phrase(darkness: float) -> str:
    if darkness >= 0.55:
        return "darker high-response regions"
    if darkness >= 0.38:
        return "moderately dark regions"
    return "lighter and less intense regions"


def _variation_phrase(color_variation: float, edge_strength: float) -> str:
    if color_variation >= 0.20 or edge_strength >= 0.18:
        return "strong variation in tone and structure"
    if color_variation >= 0.12 or edge_strength >= 0.11:
        return "moderate variation in color and texture"
    return "more uniform tone and structure"


def _asymmetry_phrase(asymmetry_score: float) -> str:
    if asymmetry_score >= 0.18:
        return "with a more asymmetric visual pattern"
    if asymmetry_score >= 0.10:
        return "with mild asymmetry in the attended region"
    return "with a relatively balanced visual pattern"


def generate_case_explanation(
    image_path: str,
    heatmap: np.ndarray,
    label: str,
    confidence_pct: float,
    risk_badge: str,
    benign_pct: float,
    malignant_pct: float,
) -> str:
    metrics = _compute_metrics(image_path, heatmap)

    attention_phrase = _attention_phrase(metrics["attention_ratio"])
    darkness_phrase = _darkness_phrase(metrics["darkness"])
    variation_phrase = _variation_phrase(metrics["color_variation"], metrics["edge_strength"])
    asymmetry_phrase = _asymmetry_phrase(metrics["asymmetry_score"])

    if label.lower() == "malignant":
        explanation = (
            f"The AI model predicts this lesion as malignant with {confidence_pct}% confidence and classifies it as {risk_badge}. "
            f"In this image, the Grad-CAM explanation shows {attention_phrase}, focusing mainly on {darkness_phrase}. "
            f"The model also detected {variation_phrase} {asymmetry_phrase}. "
            f"These image regions contributed more strongly to the high-risk prediction than the surrounding areas. "
            f"This explanation reflects how the model responded to the visual patterns in this specific image and should be used for screening support only."
        )
    else:
        explanation = (
            f"The AI model predicts this lesion as benign with {confidence_pct}% confidence and classifies it as {risk_badge}. "
            f"For this image, the Grad-CAM explanation shows {attention_phrase}, focusing on {darkness_phrase}. "
            f"Compared with higher-risk cases, the model observed {variation_phrase} {asymmetry_phrase}. "
            f"Overall, the attended regions appeared less suspicious to the model, which supported the lower-risk prediction. "
            f"This explanation reflects how the model responded to the visual patterns in this specific image and should be used for screening support only."
        )

    return explanation