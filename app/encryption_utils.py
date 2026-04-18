# encryption_utils.py

import os
from cryptography.fernet import Fernet

# Load key from environment variable (BEST PRACTICE)
SECRET_KEY = os.environ.get("DB_SECRET_KEY")

# If not set (dev mode only), generate one
if not SECRET_KEY:
    SECRET_KEY = Fernet.generate_key()
    print("WARNING: Using temporary encryption key (NOT for production)")

cipher = Fernet(SECRET_KEY)


def encrypt_data(data: str) -> str:
    """Encrypt plain text into secure format"""
    if data is None:
        return None
    return cipher.encrypt(data.encode()).decode()


def decrypt_data(data: str) -> str:
    """Decrypt encrypted text back to plain text"""
    if data is None:
        return None
    return cipher.decrypt(data.encode()).decode()
