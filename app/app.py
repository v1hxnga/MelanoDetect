import os
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename

from model_utils import predict_image
from gradcam_utils import build_gradcam_model, make_gradcam_heatmap, save_gradcam_overlay
from db_utils import init_db, create_doctor, authenticate_user
from validator_utils import validate_lesion_image

app = Flask(__name__)
app.secret_key = "melanodetect-secret-key-change-this"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
GRADCAM_FOLDER = os.path.join(BASE_DIR, "static", "gradcam")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["GRADCAM_FOLDER"] = GRADCAM_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# initialize database on startup
init_db()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def logged_in():
    return "user_role" in session


@app.route("/")
def home():
    if logged_in():
        return redirect(url_for("upload_page"))
    return redirect(url_for("login_page"))


@app.route("/login", methods=["GET", "POST"])
def login_page():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()

        user = authenticate_user(email, password)

        if not user:
            flash("Invalid email or password.", "error")
            return redirect(url_for("login_page"))

        session["user_id"] = user["id"]
        session["user_name"] = user["full_name"]
        session["user_email"] = user["email"]
        session["user_role"] = user["role"]

        return redirect(url_for("upload_page"))

    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup_page():
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()

        if not full_name or not email or not password or not confirm_password:
            flash("Please fill in all fields.", "error")
            return redirect(url_for("signup_page"))

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return redirect(url_for("signup_page"))

        success, message = create_doctor(full_name, email, password)

        if not success:
            flash(message, "error")
            return redirect(url_for("signup_page"))

        flash("Doctor account created successfully. Please log in.", "success")
        return redirect(url_for("login_page"))

    return render_template("signup.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))


@app.route("/upload")
def upload_page():
    if not logged_in():
        return redirect(url_for("login_page"))
    return render_template("upload.html")


@app.route("/predict", methods=["POST"])
def predict():
    if not logged_in():
        return redirect(url_for("login_page"))

    if "file" not in request.files:
        flash("No file uploaded.", "error")
        return redirect(url_for("upload_page"))

    file = request.files["file"]

    if file.filename == "":
        flash("No file selected.", "error")
        return redirect(url_for("upload_page"))

    if not (file and allowed_file(file.filename)):
        flash("Invalid file format. Please upload JPG, JPEG, or PNG.", "error")
        return redirect(url_for("upload_page"))

    filename = secure_filename(file.filename)
    upload_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(upload_path)

    # STEP 1: Validate whether the uploaded image is really a lesion image
    is_valid, valid_score = validate_lesion_image(upload_path)

    if not is_valid:
        # optional: remove invalid upload file
        if os.path.exists(upload_path):
            os.remove(upload_path)

        flash(
            f"Unsupported image. Please upload a clear skin-lesion or dermoscopic image. Validator score: {valid_score:.2f}",
            "error"
        )
        return redirect(url_for("upload_page"))

    # STEP 2: Only valid lesion images go to the main classifier
    label, confidence, img_array, model = predict_image(upload_path)

    feature_extractor, base_model, last_conv_layer_name = build_gradcam_model(model)
    heatmap = make_gradcam_heatmap(img_array, model, feature_extractor, base_model)

    gradcam_filename = f"gradcam_{filename}"
    gradcam_path = os.path.join(app.config["GRADCAM_FOLDER"], gradcam_filename)
    save_gradcam_overlay(upload_path, heatmap, gradcam_path)

    confidence_pct = round(confidence * 100, 2)

    if label == "malignant":
        malignant_pct = confidence_pct
        benign_pct = round(100 - confidence_pct, 2)
        risk_badge = "High Risk"
        explanation = (
            f"The AI model predicts this lesion as malignant with {confidence_pct}% confidence. "
            "This result is for screening support only and should be reviewed by a medical professional."
        )
        next_steps = [
            "Consult a dermatologist as soon as possible.",
            "Do not self-diagnose using the AI output alone.",
            "Keep a copy of this result for reference."
        ]
    else:
        benign_pct = confidence_pct
        malignant_pct = round(100 - confidence_pct, 2)
        risk_badge = "Lower Risk"
        explanation = (
            f"The AI model predicts this lesion as benign with {confidence_pct}% confidence. "
            "However, only a qualified medical professional can confirm the final diagnosis."
        )
        next_steps = [
            "Monitor the lesion if any visible changes occur.",
            "Seek professional advice if symptoms persist.",
            "Use this result only as screening support."
        ]

    session["last_result"] = {
        "label": label,
        "confidence": confidence_pct,
        "benign_pct": benign_pct,
        "malignant_pct": malignant_pct,
        "risk_badge": risk_badge,
        "uploaded_image": f"uploads/{filename}",
        "gradcam_image": f"gradcam/{gradcam_filename}",
        "explanation": explanation,
        "next_steps": next_steps,
        "last_conv_layer": last_conv_layer_name,
        "validator_score": round(valid_score, 2)
    }

    return redirect(url_for("results_page"))


@app.route("/results")
def results_page():
    if not logged_in():
        return redirect(url_for("login_page"))

    result = session.get("last_result")
    if not result:
        return redirect(url_for("upload_page"))

    return render_template("result.html", result=result)


@app.route("/gradcam")
def gradcam_page():
    if not logged_in():
        return redirect(url_for("login_page"))

    result = session.get("last_result")
    if not result:
        return redirect(url_for("upload_page"))

    return render_template("gradcam.html", result=result)


@app.route("/admin")
def admin_page():
    if not logged_in():
        return redirect(url_for("login_page"))

    if session.get("user_role") != "Admin":
        return redirect(url_for("upload_page"))

    metrics = {
        "accuracy": "85.7%",
        "precision_malignant": "69.1%",
        "recall_malignant": "48.1%",
        "f1_malignant": "56.7%",
        "threshold": "0.50",
        "model_name": "EfficientNetB0 Big260 Fine-Tuned",
        "train_samples": "7010",
        "validation_samples": "1502",
        "test_samples": "1503",
        "input_size": "260 x 260",
        "optimizer": "Adam",
        "epochs": "Stage 1 + Stage 2",
        "loss_function": "Binary Focal Loss"
    }

    return render_template("admin.html", metrics=metrics)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)