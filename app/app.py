import os
import re
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename

from model_utils import predict_image
from gradcam_utils import build_gradcam_model, make_gradcam_heatmap, save_gradcam_overlay
from db_utils import (
    init_db,
    create_doctor,
    authenticate_user,
    save_analysis_result,
    get_user_history,
    delete_history_item
)
from validator_utils import validate_lesion_image
from explain_utils import generate_case_explanation

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


def is_valid_email(email):
    pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
    return re.match(pattern, email) is not None


def is_strong_password(password):
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."

    if not re.search(r"[A-Z]", password):
        return False, "Password must include at least one uppercase letter."

    if not re.search(r"[a-z]", password):
        return False, "Password must include at least one lowercase letter."

    if not re.search(r"[0-9]", password):
        return False, "Password must include at least one number."

    if not re.search(r"[!@#$%^&*(),.?\":{}|<>_\-+=/\\[\];'`~]", password):
        return False, "Password must include at least one special character."

    return True, "Strong password."


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
        login_role = request.form.get("login_role", "").strip()

        if not email or not password or not login_role:
            flash("Please fill in all login fields.", "error")
            return redirect(url_for("login_page"))

        if login_role not in ["Doctor", "Admin"]:
            flash("Please select a valid login role.", "error")
            return redirect(url_for("login_page"))

        user = authenticate_user(email, password)

        if not user:
            flash("Invalid email or password.", "error")
            return redirect(url_for("login_page"))

        if user["role"] != login_role:
            if login_role == "Admin":
                flash("This account is not registered as an Admin account.", "error")
            else:
                flash("This account is not registered as a Doctor account.", "error")
            return redirect(url_for("login_page"))

        session["user_id"] = user["id"]
        session["user_name"] = user["full_name"]
        session["user_email"] = user["email"]
        session["user_role"] = user["role"]
        session["user_gender"] = user.get("gender", "male")

        flash(f"Welcome back, {user['full_name']}.", "success")
        return redirect(url_for("dashboard_page"))

    return render_template("login.html", active_page="login")


@app.route("/signup", methods=["GET", "POST"])
def signup_page():
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()
        gender = request.form.get("gender", "").strip().lower()

        if not full_name or not email or not password or not confirm_password or not gender:
            flash("Please fill in all fields.", "error")
            return redirect(url_for("signup_page"))

        if len(full_name) < 3:
            flash("Full name must be at least 3 characters long.", "error")
            return redirect(url_for("signup_page"))

        if not full_name.replace(" ", "").isalpha():
            flash("Full name should contain only letters and spaces.", "error")
            return redirect(url_for("signup_page"))

        if not is_valid_email(email):
            flash("Please enter a valid email address.", "error")
            return redirect(url_for("signup_page"))

        if gender not in ["male", "female"]:
            flash("Please select a valid gender.", "error")
            return redirect(url_for("signup_page"))

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return redirect(url_for("signup_page"))

        strong, message = is_strong_password(password)
        if not strong:
            flash(message, "error")
            return redirect(url_for("signup_page"))

        success, message = create_doctor(full_name, email, password, gender)

        if not success:
            flash(message, "error")
            return redirect(url_for("signup_page"))

        user = authenticate_user(email, password)

        if not user:
            flash("Account created, but automatic login failed. Please log in manually.", "error")
            return redirect(url_for("login_page"))

        session["user_id"] = user["id"]
        session["user_name"] = user["full_name"]
        session["user_email"] = user["email"]
        session["user_role"] = user["role"]
        session["user_gender"] = user.get("gender", "male")

        flash(f"Welcome, {user['full_name']}. Your account has been created successfully.", "success")
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

    history_rows = get_user_history(session["user_id"])
    history = [dict(item) for item in history_rows]

    latest_result = session.get("last_result")

    if not latest_result and history:
        latest = history[0]
        latest_result = {
            "label": latest["label"],
            "confidence": latest["confidence"],
            "benign_pct": latest["benign_pct"],
            "malignant_pct": latest["malignant_pct"],
            "risk_badge": latest["risk_badge"],
            "uploaded_image": latest["uploaded_image"],
            "gradcam_image": latest["gradcam_image"],
            "explanation": latest["explanation"],
            "next_steps": [],
            "last_conv_layer": latest["last_conv_layer"],
            "validator_score": latest["validator_score"],
            "timestamp": latest["timestamp"]
        }

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

    gender = session.get("user_gender", "male").lower()

    if gender == "female":
        doctor_image = "images/doc_female.png"
    else:
        doctor_image = "images/doc_male.png"

    return render_template(
        "profile.html",
        active_page="profile",
        doctor_image=doctor_image
    )


@app.route("/upload")
def upload_page():
    if not logged_in():
        return redirect(url_for("login_page"))
    return render_template("upload.html", active_page="upload")


@app.route("/predict", methods=["POST"])
def predict():
    if not logged_in():
        return redirect(url_for("login_page"))

    try:
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
        timestamp_for_file = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp_for_file}_{filename}"

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

        grad_model, last_conv_layer_name = build_gradcam_model(model)

        heatmap, last_conv_layer_name = make_gradcam_heatmap(
            img_array=img_array,
            model=model,
            grad_model=grad_model,
            last_conv_layer_name=last_conv_layer_name
        )

        gradcam_filename = f"gradcam_{filename}"
        gradcam_path = os.path.join(app.config["GRADCAM_FOLDER"], gradcam_filename)
        save_gradcam_overlay(upload_path, heatmap, gradcam_path)

        confidence_pct = round(confidence * 100, 1)

        if label == "malignant":
            malignant_pct = confidence_pct
            benign_pct = round(100 - confidence_pct, 1)
            risk_badge = "High Risk"
            next_steps = [
                "Consult a dermatologist as soon as possible.",
                "Do not self-diagnose using the AI output alone.",
                "Keep a copy of this result for reference."
            ]
        else:
            benign_pct = confidence_pct
            malignant_pct = round(100 - confidence_pct, 1)
            risk_badge = "Lower Risk"
            next_steps = [
                "Monitor the lesion if any visible changes occur.",
                "Seek professional advice if symptoms persist.",
                "Use this result only as screening support."
            ]

        explanation = generate_case_explanation(
            image_path=upload_path,
            heatmap=heatmap,
            label=label,
            confidence_pct=confidence_pct,
            risk_badge=risk_badge,
            benign_pct=benign_pct,
            malignant_pct=malignant_pct,
        )

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
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        session["last_result"] = result
        save_analysis_result(session["user_id"], result)

        flash("Image analyzed successfully.", "success")
        return redirect(url_for("results_page"))

    except Exception as e:
        app.logger.exception("Prediction failed")
        flash(f"Prediction failed: {str(e)}", "error")
        return redirect(url_for("upload_page"))


@app.route("/results")
def results_page():
    if not logged_in():
        return redirect(url_for("login_page"))

    result = session.get("last_result")
    if not result:
        history_rows = get_user_history(session["user_id"])
        history = [dict(item) for item in history_rows]

        if history:
            latest = history[0]
            result = {
                "label": latest["label"],
                "confidence": latest["confidence"],
                "benign_pct": latest["benign_pct"],
                "malignant_pct": latest["malignant_pct"],
                "risk_badge": latest["risk_badge"],
                "uploaded_image": latest["uploaded_image"],
                "gradcam_image": latest["gradcam_image"],
                "explanation": latest["explanation"],
                "next_steps": [],
                "last_conv_layer": latest["last_conv_layer"],
                "validator_score": latest["validator_score"],
                "timestamp": latest["timestamp"]
            }
        else:
            flash("No analysis result available yet. Upload an image first.", "error")
            return redirect(url_for("upload_page"))

    return render_template("result.html", result=result, active_page="results")


@app.route("/gradcam")
def gradcam_page():
    if not logged_in():
        return redirect(url_for("login_page"))

    result = session.get("last_result")
    if not result:
        history_rows = get_user_history(session["user_id"])
        history = [dict(item) for item in history_rows]

        if history:
            latest = history[0]
            result = {
                "label": latest["label"],
                "confidence": latest["confidence"],
                "benign_pct": latest["benign_pct"],
                "malignant_pct": latest["malignant_pct"],
                "risk_badge": latest["risk_badge"],
                "uploaded_image": latest["uploaded_image"],
                "gradcam_image": latest["gradcam_image"],
                "explanation": latest["explanation"],
                "next_steps": [],
                "last_conv_layer": latest["last_conv_layer"],
                "validator_score": latest["validator_score"],
                "timestamp": latest["timestamp"]
            }
        else:
            flash("No Grad-CAM result available yet. Upload an image first.", "error")
            return redirect(url_for("upload_page"))

    return render_template("gradcam.html", result=result, active_page="results")


@app.route("/history")
def history_page():
    if not logged_in():
        return redirect(url_for("login_page"))

    history_rows = get_user_history(session["user_id"])
    history = [dict(item) for item in history_rows]

    return render_template("history.html", history=history, active_page="history")


@app.route("/history/delete/<int:history_id>", methods=["POST"])
def delete_history_page(history_id):
    if not logged_in():
        return redirect(url_for("login_page"))

    deleted = delete_history_item(session["user_id"], history_id)

    if deleted:
        flash("History record deleted successfully.", "success")
    else:
        flash("Unable to delete that record.", "error")

    return redirect(url_for("history_page"))


@app.route("/admin")
def admin_page():
    if not logged_in():
        return redirect(url_for("login_page"))

    if session.get("user_role") != "Admin":
        return redirect(url_for("dashboard_page"))

    return redirect(url_for("dashboard_page"))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)