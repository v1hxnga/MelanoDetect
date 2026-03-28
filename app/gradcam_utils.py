import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array, array_to_img

BASE_MODEL_NAME = "efficientnetb0"


def find_last_conv_layer_name(base_model):
    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the base model.")


def build_gradcam_model(model, base_model_name=BASE_MODEL_NAME):
    base_model = model.get_layer(base_model_name)
    last_conv_layer_name = find_last_conv_layer_name(base_model)
    last_conv_layer = base_model.get_layer(last_conv_layer_name)

    # Extract both the last conv output and the base model output
    feature_extractor = tf.keras.Model(
        inputs=base_model.input,
        outputs=[last_conv_layer.output, base_model.output]
    )

    return feature_extractor, base_model, last_conv_layer_name


def make_gradcam_heatmap(img_array, model, feature_extractor, base_model):
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, base_output = feature_extractor(img_tensor, training=False)

        x = base_output

        # Pass through layers after EfficientNetB0
        passed_base = False
        for layer in model.layers:
            if layer.name == base_model.name:
                passed_base = True
                continue

            if passed_base:
                try:
                    x = layer(x, training=False)
                except TypeError:
                    x = layer(x)

        predictions = x
        class_channel = predictions[:, 0]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def save_gradcam_overlay(img_path, heatmap, output_path, alpha=0.4):
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

    result_img = array_to_img(superimposed_img)
    result_img.save(output_path)