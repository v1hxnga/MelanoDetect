import os
from datetime import datetime
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

init_db()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def logged_in():
    return "user_role" in session


@app.route("/")
def home():
    if logged_in():
        return redirect(url_for("dashboard_page"))
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

        if "history" not in session:
            session["history"] = []

        return redirect(url_for("dashboard_page"))

    return render_template("login.html", active_page="login")


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

        user = authenticate_user(email, password)
        session["user_id"] = user["id"]
        session["user_name"] = user["full_name"]
        session["user_email"] = user["email"]
        session["user_role"] = user["role"]
        session["history"] = []

        flash("Account created successfully. Welcome to MelanoDetect.", "success")
        return redirect(url_for("dashboard_page"))

    return render_template("signup.html", active_page="signup")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))


@app.route("/dashboard")
def dashboard_page():
    if not logged_in():
        return redirect(url_for("login_page"))

    history = session.get("history", [])
    latest_result = session.get("last_result")

    stats = {
        "total_uploads": len(history),
        "latest_prediction": latest_result["label"].capitalize() if latest_result else "No result yet",
        "latest_confidence": f'{latest_result["confidence"]}%' if latest_result else "--",
        "role": session.get("user_role", "Doctor")
    }

    return render_template(
        "dashboard.html",
        active_page="dashboard",
        stats=stats,
        latest_result=latest_result,
        history=history[:3]
    )


@app.route("/profile")
def profile_page():
    if not logged_in():
        return redirect(url_for("login_page"))

    return render_template("profile.html", active_page="profile")


@app.route("/upload")
def upload_page():
    if not logged_in():
        return redirect(url_for("login_page"))
    return render_template("upload.html", active_page="upload")


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

    is_valid, valid_score = validate_lesion_image(upload_path)

    if not is_valid:
        if os.path.exists(upload_path):
            os.remove(upload_path)

        flash(
            f"Unsupported image. Please upload a clear skin-lesion or dermoscopic image. Validator score: {valid_score:.2f}",
            "error"
        )
        return redirect(url_for("upload_page"))

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

    result = {
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
        "validator_score": round(valid_score, 2),
        "filename": filename,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
    }

    session["last_result"] = result

    history = session.get("history", [])
    history.insert(0, {
        "filename": filename,
        "label": label.capitalize(),
        "confidence": f"{confidence_pct}%",
        "timestamp": result["timestamp"]
    })
    session["history"] = history[:10]

    return redirect(url_for("results_page"))


@app.route("/results")
def results_page():
    if not logged_in():
        return redirect(url_for("login_page"))

    result = session.get("last_result")
    if not result:
        flash("No analysis result available yet. Upload an image first.", "error")
        return redirect(url_for("upload_page"))

    return render_template("result.html", result=result, active_page="results")


@app.route("/gradcam")
def gradcam_page():
    if not logged_in():
        return redirect(url_for("login_page"))

    result = session.get("last_result")
    if not result:
        flash("No Grad-CAM result available yet. Upload an image first.", "error")
        return redirect(url_for("upload_page"))

    return render_template("gradcam.html", result=result, active_page="results")


@app.route("/history")
def history_page():
    if not logged_in():
        return redirect(url_for("login_page"))

    history = session.get("history", [])
    return render_template("history.html", history=history, active_page="history")


@app.route("/admin")
def admin_page():
    if not logged_in():
        return redirect(url_for("login_page"))

    if session.get("user_role") != "Admin":
        return redirect(url_for("dashboard_page"))

    return redirect(url_for("dashboard_page"))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)