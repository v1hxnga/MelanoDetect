import os
import re
import uuid
from datetime import datetime, timedelta
from functools import wraps

from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from PIL import Image

from flask_wtf.csrf import CSRFProtect

from model_utils import predict_image
from gradcam_utils import build_gradcam_model, make_gradcam_heatmap, save_gradcam_overlay
from db_utils import (
    init_db,
    create_doctor,
    authenticate_user,
    save_analysis_result,
    get_user_history,
    delete_history_item,
    get_all_users,
    delete_user,
    reset_user_password,
    get_all_analysis,
    delete_analysis_admin,
    get_admin_stats
)
from validator_utils import validate_lesion_image
from explain_utils import generate_case_explanation

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "melanodetect-secret-key-change-this")

# ========================
# SECURITY CONFIG
# ========================
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=30),
    MAX_CONTENT_LENGTH=50 * 1024 * 1024  # 50MB upload limit
)

if os.environ.get("FLASK_ENV") == "production":
    app.config["SESSION_COOKIE_SECURE"] = True

csrf = CSRFProtect(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
GRADCAM_FOLDER = os.path.join(BASE_DIR, "static", "gradcam")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["GRADCAM_FOLDER"] = GRADCAM_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

init_db()

# ========================
# HELPERS
# ========================

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def logged_in():
    return "user_role" in session


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not logged_in():
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return wrapper


def is_valid_image(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False


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


def build_result_from_history_item(item):
    return {
        "label": item["label"],
        "confidence": item["confidence"],
        "benign_pct": item["benign_pct"],
        "malignant_pct": item["malignant_pct"],
        "risk_badge": item["risk_badge"],
        "uploaded_image": item["uploaded_image"],
        "gradcam_image": item["gradcam_image"],
        "explanation": item["explanation"],
        "next_steps": [],
        "last_conv_layer": item["last_conv_layer"],
        "validator_score": item["validator_score"],
        "timestamp": item["timestamp"]
    }

@app.errorhandler(413)
def request_entity_too_large(error):
    flash("The uploaded image is too large. Please upload an image smaller than 50MB.", "error")
    return redirect(url_for("upload_page"))

# ========================
# ROUTES
# ========================

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

        session.clear()
        session.permanent = True

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

        session.clear()
        session.permanent = True

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
@login_required
def dashboard_page():
    history_rows = get_user_history(session["user_id"])
    history = [dict(item) for item in history_rows]

    latest_result = session.get("last_result")

    if not latest_result and history:
        latest_result = build_result_from_history_item(history[0])

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
@login_required
def profile_page():
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
@login_required
def upload_page():
    return render_template("upload.html", active_page="upload")


@app.route("/predict", methods=["POST"])
@login_required
def predict():
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

        filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"

        upload_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(upload_path)

        if not is_valid_image(upload_path):
            if os.path.exists(upload_path):
                os.remove(upload_path)
            flash("Invalid image file. Please upload a real JPG or PNG image.", "error")
            return redirect(url_for("upload_page"))

        is_valid, valid_score = validate_lesion_image(upload_path)

        if not is_valid:
            if os.path.exists(upload_path):
                os.remove(upload_path)

            flash(
                "Unsupported image. Please upload a clear skin-lesion or dermoscopic image.",
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
@login_required
def results_page():
    result = session.get("last_result")

    if not result:
        history_rows = get_user_history(session["user_id"])
        history = [dict(item) for item in history_rows]

        if history:
            result = build_result_from_history_item(history[0])
        else:
            flash("No analysis result available yet. Upload an image first.", "error")
            return redirect(url_for("upload_page"))

    return render_template("result.html", result=result, active_page="results")


@app.route("/gradcam")
@login_required
def gradcam_page():
    result = session.get("last_result")

    if not result:
        history_rows = get_user_history(session["user_id"])
        history = [dict(item) for item in history_rows]

        if history:
            result = build_result_from_history_item(history[0])
        else:
            flash("No Grad-CAM result available yet. Upload an image first.", "error")
            return redirect(url_for("upload_page"))

    return render_template("gradcam.html", result=result, active_page="results")


@app.route("/history")
@login_required
def history_page():
    history_rows = get_user_history(session["user_id"])
    history = [dict(item) for item in history_rows]

    return render_template("history.html", history=history, active_page="history")


@app.route("/history/delete/<int:history_id>", methods=["POST"])
@login_required
def delete_history_page(history_id):
    deleted = delete_history_item(session["user_id"], history_id)

    if deleted:
        session.pop("last_result", None)
        flash("History record deleted successfully.", "success")
    else:
        flash("Unable to delete that record.", "error")

    return redirect(url_for("history_page"))

# ========================
# ADMIN PANEL
# ========================

@app.route("/admin")
@login_required
def admin_page():
    if session.get("user_role") != "Admin":
        return redirect(url_for("dashboard_page"))

    users = get_all_users()
    analyses = get_all_analysis()
    stats = get_admin_stats()

    return render_template(
        "admin.html",
        users=users,
        analyses=analyses,
        stats=stats,
        active_page="admin"
    )


@app.route("/admin/delete_user/<int:user_id>", methods=["POST"])
@login_required
def admin_delete_user(user_id):
    if session.get("user_role") != "Admin":
        return redirect(url_for("dashboard_page"))

    delete_user(user_id)
    flash("User deleted successfully.", "success")
    return redirect(url_for("admin_page"))


@app.route("/admin/reset_password/<int:user_id>", methods=["POST"])
@login_required
def admin_reset_password(user_id):
    if session.get("user_role") != "Admin":
        return redirect(url_for("dashboard_page"))

    new_password = "Temp@123"
    reset_user_password(user_id, new_password)
    flash(f"Password reset successfully. Temporary password: {new_password}", "success")
    return redirect(url_for("admin_page"))


@app.route("/admin/delete_analysis/<int:history_id>", methods=["POST"])
@login_required
def admin_delete_analysis(history_id):
    if session.get("user_role") != "Admin":
        return redirect(url_for("dashboard_page"))

    delete_analysis_admin(history_id)
    flash("Analysis deleted successfully.", "success")
    return redirect(url_for("admin_page"))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)