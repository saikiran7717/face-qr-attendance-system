# =============================================================================
# config.py — Central configuration for the Face + QR Attendance System
# =============================================================================

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
FACES_DIR  = os.path.join(BASE_DIR, "faces")
QR_DIR     = os.path.join(BASE_DIR, "qr_codes")
MODELS_DIR = os.path.join(BASE_DIR, "models")  # reserved for future model files

# ── MySQL Database ─────────────────────────────────────────────────────────────
# Set DB_PASSWORD environment variable, or edit the default below.
DB_CONFIG = {
    "host":     os.environ.get("DB_HOST",     "localhost"),
    "port":     int(os.environ.get("DB_PORT", "3306")),
    "user":     os.environ.get("DB_USER",     "root"),
    "password": os.environ.get("DB_PASSWORD", "YOUR_MYSQL_PASSWORD"),  # ← change this
    "database": os.environ.get("DB_NAME",     "attendance_db"),
}

# ── Face Recognition ──────────────────────────────────────────────────────────
# L2 Euclidean distance threshold for Facenet512 embeddings.
# Cosine distance range: 0 (identical) → 2 (opposite).
# Same person: ~0.10–0.40  |  Different person: ~0.60–1.00
# Scale-invariant — works with raw (un-normalised) DeepFace vectors.
FACE_MATCH_THRESHOLD: float = 0.40

# Number of face samples captured per student during registration
REGISTRATION_SAMPLES: int = 5

# ── Liveness Detection (Eye Aspect Ratio) ────────────────────────────────────
# EAR below this value is considered a blink
EAR_THRESHOLD: float = 0.25

# Consecutive frames below EAR_THRESHOLD to count as one blink
EAR_CONSEC_FRAMES: int = 2

# Number of blinks required before accepting authentication
REQUIRED_BLINKS: int = 1

# ── Webcam ────────────────────────────────────────────────────────────────────
CAMERA_INDEX: int = 0          # 0 = default webcam
FRAME_WIDTH:  int = 640
FRAME_HEIGHT: int = 480

# ── QR Code ───────────────────────────────────────────────────────────────────
QR_BOX_SIZE:   int = 10
QR_BORDER:     int = 4

# ── Attendance ─────────────────────────────────────────────────────────────────
# Seconds to wait for QR scan after face is matched
QR_SCAN_TIMEOUT: int = 15
