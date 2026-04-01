import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from model_utils import predict_image
from gradcam_utils import build_gradcam_model, make_gradcam_heatmap, save_gradcam_overlay

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
GRADCAM_FOLDER = os.path.join(BASE_DIR, "static", "gradcam")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["GRADCAM_FOLDER"] = GRADCAM_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "No selected file"

    if not (file and allowed_file(file.filename)):
        return "Invalid file format. Please upload JPG, JPEG, or PNG."

    filename = secure_filename(file.filename)
    upload_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(upload_path)

    label, confidence, img_array, model = predict_image(upload_path)

    feature_extractor, base_model, last_conv_layer_name = build_gradcam_model(model)
    heatmap = make_gradcam_heatmap(img_array, model, feature_extractor, base_model)

    gradcam_filename = f"gradcam_{filename}"
    gradcam_path = os.path.join(app.config["GRADCAM_FOLDER"], gradcam_filename)
    save_gradcam_overlay(upload_path, heatmap, gradcam_path)

    explanation = (
        "The model identified image patterns associated with malignant skin lesions."
        if label == "malignant"
        else "The model identified image patterns more consistent with a benign skin lesion."
    )

    return render_template(
        "result.html",
        label=label,
        confidence=round(confidence * 100, 2),
        uploaded_image=f"static/uploads/{filename}",
        gradcam_image=f"static/gradcam/{gradcam_filename}",
        explanation=explanation,
        last_conv_layer=last_conv_layer_name
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)