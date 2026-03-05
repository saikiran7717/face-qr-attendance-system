# =============================================================================
# liveness.py — Blink-based liveness detection using OpenCV Haar Cascades
#
# Uses OpenCV's built-in face + eye cascades (no extra packages needed).
# Blink = face detected but eyes NOT detected for EAR_CONSEC_FRAMES frames,
#         followed by eyes being detected again.
# =============================================================================

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple

from config import EAR_CONSEC_FRAMES, EAR_THRESHOLD, REQUIRED_BLINKS

# ── Load OpenCV built-in Haar cascades ────────────────────────────────────────
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)


# ── Liveness session state ────────────────────────────────────────────────────

@dataclass
class LivenessState:
    """Mutable state carried across frames for a single authentication session."""
    blink_counter:   int  = 0    # consecutive frames where face found but eyes closed
    total_blinks:    int  = 0    # confirmed blink events
    liveness_passed: bool = False
    _eyes_were_open: bool = False  # tracks eye state transition


# ── Public API ────────────────────────────────────────────────────────────────

def get_ear_from_frame(frame: np.ndarray) -> Tuple[float, bool]:
    """
    Detect face and eyes in *frame* using Haar cascades.

    Returns a pseudo-EAR:
      - 0.30  (eyes open  → above EAR_THRESHOLD → no blink)
      - 0.10  (eyes closed → below EAR_THRESHOLD → blink candidate)
      - 0.00  (no face found)

    Returns
    -------
    (pseudo_ear, face_found) : (float, bool)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )

    if len(faces) == 0:
        return 0.0, False

    # Use largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    roi_gray = gray[y: y + h, x: x + w]

    eyes = _eye_cascade.detectMultiScale(
        roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
    )

    if len(eyes) >= 2:
        return 0.30, True   # eyes open  → EAR above threshold
    else:
        return 0.10, True   # eyes closed/blinking → EAR below threshold


def update_liveness_state(
    state: LivenessState,
    ear: float,
    face_found: bool,
) -> LivenessState:
    """
    Update liveness state from the latest pseudo-EAR reading.

    Blink logic:
      eyes open  (ear >= EAR_THRESHOLD) → set _eyes_were_open = True
      eyes closed (ear < EAR_THRESHOLD) → increment blink_counter
      eyes reopen after being closed    → count as one blink event
    """
    if not face_found:
        return state

    if ear >= EAR_THRESHOLD:
        # Eyes are open
        if state.blink_counter >= EAR_CONSEC_FRAMES and state._eyes_were_open:
            # Transition: was open → closed → now open again = 1 blink
            state.total_blinks += 1
        state.blink_counter  = 0
        state._eyes_were_open = True
    else:
        # Eyes appear closed
        if state._eyes_were_open:
            state.blink_counter += 1

    if state.total_blinks >= REQUIRED_BLINKS:
        state.liveness_passed = True

    return state


def draw_liveness_overlay(
    frame: np.ndarray,
    state: LivenessState,
    ear: float,
) -> np.ndarray:
    """Annotate *frame* with eye state, blink count, and liveness status."""
    color      = (0, 200, 0) if state.liveness_passed else (0, 100, 255)
    eye_status = "Eyes: OPEN" if ear >= EAR_THRESHOLD else "Eyes: CLOSED"

    cv2.putText(frame, eye_status,
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Blinks: {state.total_blinks}/{REQUIRED_BLINKS}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    label = "Liveness: PASS" if state.liveness_passed else "Blink to verify..."
    cv2.putText(frame, label, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame


def reset_liveness() -> LivenessState:
    """Return a fresh LivenessState for a new authentication attempt."""
    return LivenessState()
