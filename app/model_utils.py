import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

MODEL_PATH = "../models/skin_cancer_cnn_baseline.keras"

model = load_model(MODEL_PATH)

def preprocess_image(img_path, target_size=(224, 224)):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array, verbose=0)[0][0]

    label = "malignant" if prediction > 0.5 else "benign"
    confidence = float(prediction if prediction > 0.5 else 1 - prediction)

    return label, confidence, img_array, model