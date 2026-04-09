import os
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "melanodetect.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'Doctor',
            gender TEXT NOT NULL DEFAULT 'male'
        )
    """)

    cursor.execute("PRAGMA table_info(users)")
    columns = [column["name"] for column in cursor.fetchall()]

    if "gender" not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN gender TEXT NOT NULL DEFAULT 'male'")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            uploaded_image TEXT NOT NULL,
            gradcam_image TEXT,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            benign_pct REAL NOT NULL,
            malignant_pct REAL NOT NULL,
            risk_badge TEXT NOT NULL,
            explanation TEXT NOT NULL,
            validator_score REAL,
            last_conv_layer TEXT,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    admin_email = "admin@melanodetect.com"
    admin_password = "Admin@123"

    cursor.execute("SELECT * FROM users WHERE email = ?", (admin_email,))
    admin = cursor.fetchone()

    if not admin:
        cursor.execute("""
            INSERT INTO users (full_name, email, password_hash, role, gender)
            VALUES (?, ?, ?, ?, ?)
        """, (
            "System Admin",
            admin_email,
            generate_password_hash(admin_password),
            "Admin",
            "male"
        ))
    else:
        cursor.execute("""
            UPDATE users
            SET gender = COALESCE(gender, 'male')
            WHERE email = ?
        """, (admin_email,))

    conn.commit()
    conn.close()


def create_doctor(full_name, email, password, gender):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    existing_user = cursor.fetchone()

    if existing_user:
        conn.close()
        return False, "Email already exists."

    cursor.execute("""
        INSERT INTO users (full_name, email, password_hash, role, gender)
        VALUES (?, ?, ?, ?, ?)
    """, (
        full_name,
        email,
        generate_password_hash(password),
        "Doctor",
        gender
    ))

    conn.commit()
    conn.close()
    return True, "Doctor account created successfully."


def authenticate_user(email, password):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()

    if user and check_password_hash(user["password_hash"], password):
        return dict(user)

    return None


def save_analysis_result(user_id, result):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO analysis_history (
            user_id,
            uploaded_image,
            gradcam_image,
            label,
            confidence,
            benign_pct,
            malignant_pct,
            risk_badge,
            explanation,
            validator_score,
            last_conv_layer,
            timestamp
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        result["uploaded_image"],
        result["gradcam_image"],
        result["label"],
        result["confidence"],
        result["benign_pct"],
        result["malignant_pct"],
        result["risk_badge"],
        result["explanation"],
        result["validator_score"],
        result["last_conv_layer"],
        result["timestamp"]
    ))

    conn.commit()
    conn.close()


def get_user_history(user_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM analysis_history
        WHERE user_id = ?
        ORDER BY id DESC
    """, (user_id,))

    history = cursor.fetchall()
    conn.close()
    return history


def get_user_by_id(user_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()

    if user:
        return dict(user)

    return None