import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.utils import load_img, img_to_array, array_to_img


def find_last_conv_layer_name(model):
    """
    Automatically find the last usable 4D feature layer for Grad-CAM.
    This avoids hardcoded names like 'efficientnetb0'.
    """
    for layer in reversed(model.layers):
        try:
            output_shape = layer.output.shape
            if len(output_shape) == 4:
                return layer.name
        except Exception:
            continue
    raise ValueError("No suitable 4D convolutional layer found in the model.")


def build_gradcam_model(model, last_conv_layer_name=None):
    """
    Build a Grad-CAM model that returns:
    - the output of the chosen last convolution-like layer
    - the final model prediction
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer_name(model)

    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    return grad_model, last_conv_layer_name


def make_gradcam_heatmap(img_array, model, grad_model=None, last_conv_layer_name=None):
    """
    Generate Grad-CAM heatmap for the predicted output.
    """
    if grad_model is None:
        grad_model, last_conv_layer_name = build_gradcam_model(model, last_conv_layer_name)

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        class_channel = predictions[:, 0]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy(), last_conv_layer_name


def save_gradcam_overlay(img_path, heatmap, output_path, alpha=0.4):
    """
    Save Grad-CAM overlay image.
    """
    img = load_img(img_path)
    img = img_to_array(img)

    heatmap = np.uint8(255 * heatmap)
    jet = plt.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_img = array_to_img(superimposed_img)
    result_img.save(output_path)


def generate_gradcam(img_path, img_array, model, output_path):
    """
    Full Grad-CAM pipeline:
    1. automatically find the last conv-like layer
    2. generate heatmap
    3. save overlay
    4. return the selected layer name
    """
    grad_model, last_conv_layer_name = build_gradcam_model(model)
    heatmap, last_conv_layer_name = make_gradcam_heatmap(
        img_array=img_array,
        model=model,
        grad_model=grad_model,
        last_conv_layer_name=last_conv_layer_name
    )
    save_gradcam_overlay(img_path, heatmap, output_path)
    return last_conv_layer_name