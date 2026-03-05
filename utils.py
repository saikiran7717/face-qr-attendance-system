# =============================================================================
# utils.py — Shared utilities: face embedding, QR generation & decoding
#
# Uses DeepFace (Facenet512 backend) for face embeddings.
# No compiler required — installs via pip on all platforms.
# =============================================================================

from __future__ import annotations

import logging
import os
from typing import Optional

import cv2
import numpy as np
import qrcode
from deepface import DeepFace
from pyzbar import pyzbar

from config import QR_BORDER, QR_BOX_SIZE, QR_DIR

# Suppress DeepFace/TensorFlow verbosity
logging.getLogger("deepface").setLevel(logging.ERROR)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# DeepFace model to use for embeddings.
# "Facenet512" → 512-d vector, excellent accuracy, downloads ~90 MB on first run.
# Alternatives: "VGG-Face", "ArcFace", "Facenet"
_MODEL_NAME    = "Facenet512"
_DETECTOR_BACKEND = "opencv"   # fast; alternatives: "retinaface", "mtcnn"


# ── Face embedding ─────────────────────────────────────────────────────────────

def extract_embedding(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect the largest face in *image_bgr* and return its embedding vector.

    Tries opencv detector first; falls back to skip_detection if no face found.

    Returns
    -------
    np.ndarray (512,) from Facenet512, or None if no face detected.
    """
    # Try with face detection first
    for enforce in (True, False):
        try:
            result = DeepFace.represent(
                img_path=image_bgr,
                model_name=_MODEL_NAME,
                detector_backend=_DETECTOR_BACKEND,
                enforce_detection=enforce,
                align=True,
            )
            if result:
                return np.array(result[0]["embedding"], dtype=np.float32)
        except Exception:
            continue
    return None


def average_embedding(embeddings: list[np.ndarray]) -> np.ndarray:
    """
    Return the element-wise mean of multiple face embeddings.
    Averaging over several samples improves robustness.
    """
    if not embeddings:
        raise ValueError("[Utils] Cannot average an empty list of embeddings.")
    return np.mean(np.stack(embeddings, axis=0), axis=0)


def compare_embedding(
    stored: np.ndarray,
    candidate: np.ndarray,
    threshold: float,
) -> tuple[bool, float]:
    """
    Compare two face embeddings using cosine distance.

    Cosine distance is scale-invariant — works correctly with raw
    (un-normalised) Facenet512 vectors whose magnitudes vary per frame.

    Same person  : cosine distance ~ 0.10 - 0.40
    Diff person  : cosine distance ~ 0.60 - 1.00

    Returns
    -------
    (is_match, distance) : (bool, float)
    """
    a = stored.flatten().astype(np.float64)
    b = candidate.flatten().astype(np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return (False, 1.0)
    cosine_sim = float(np.dot(a, b) / denom)
    distance   = 1.0 - cosine_sim          # cosine distance in [0, 2]
    return (distance <= threshold, distance)


def find_best_match(
    candidate: np.ndarray,
    students: list[dict],
    threshold: float,
) -> Optional[dict]:
    """
    Search *students* for the closest face embedding to *candidate*.
    Prints distances to console to help with threshold tuning.

    Returns
    -------
    Best-matching student dict (with added 'distance' key) or None.
    """
    best_match    = None
    best_distance = float("inf")

    for student in students:
        _, dist = compare_embedding(student["embedding"], candidate, threshold)
        print(f"[FaceMatch] Student '{student['name']}' → distance={dist:.4f}  (threshold={threshold})")
        if dist < best_distance:
            best_distance = dist
            best_match    = {**student, "distance": dist}

    if best_match and best_distance <= threshold:
        return best_match
    return None


# ── QR code operations ─────────────────────────────────────────────────────────

def generate_qr_code(student_id: int, save_dir: str = QR_DIR) -> str:
    """
    Generate a QR code PNG that encodes only the student_id integer.

    Parameters
    ----------
    student_id : int
    save_dir   : str   Directory where the PNG is saved.

    Returns
    -------
    str   Absolute path to the saved QR PNG.
    """
    os.makedirs(save_dir, exist_ok=True)
    data    = str(student_id)           # QR payload = student ID only
    qr      = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=QR_BOX_SIZE,
        border=QR_BORDER,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img     = qr.make_image(fill_color="black", back_color="white")
    path    = os.path.join(save_dir, f"student_{student_id}.png")
    img.save(path)
    print(f"[Utils] QR code saved → {path}")
    return os.path.abspath(path)


def decode_qr_from_frame(frame: np.ndarray) -> Optional[int]:
    """
    Scan a webcam frame for a QR code and decode the student_id.

    Returns
    -------
    int (student_id) or None if no valid QR detected.
    """
    decoded = pyzbar.decode(frame)
    for obj in decoded:
        if obj.type == "QRCODE":
            raw = obj.data.decode("utf-8", errors="ignore").strip()
            if raw.isdigit():
                return int(raw)
    return None


def draw_qr_box(frame: np.ndarray) -> np.ndarray:
    """Draw decoded QR bounding polygons on *frame* for visual feedback."""
    decoded = pyzbar.decode(frame)
    for obj in decoded:
        pts = np.array([p for p in obj.polygon], np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        raw = obj.data.decode("utf-8", errors="ignore")
        cv2.putText(frame, f"QR: {raw}", (pts[0][0][0], pts[0][0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


# ── Frame capture helpers ─────────────────────────────────────────────────────

def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    """Open camera and configure resolution, raising on failure."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"[Utils] Cannot open camera at index {index}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def read_frame(cap: cv2.VideoCapture) -> Optional[np.ndarray]:
    """Read one frame, returns None on failure."""
    ret, frame = cap.read()
    return frame if ret else None
