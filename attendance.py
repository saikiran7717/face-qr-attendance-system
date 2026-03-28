# =============================================================================
# attendance.py — Live Face + QR attendance with liveness detection
#
# Usage:
#   python attendance.py
#
# Flow:
#   Phase 1 — Liveness  : blink to prove liveness.
#   Phase 2 — Face match: DeepFace runs in a background thread; UI stays live.
#                         Moves to QR scan IMMEDIATELY when face matches.
#   Phase 3 — QR scan   : show QR code to camera.
#   Phase 4 — Done      : show result, auto-reset.
# =============================================================================

from __future__ import annotations

import os
import threading
import time
from enum import Enum, auto
from typing import Optional

# Suppress TensorFlow startup noise
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import cv2
import numpy as np

from config import (
    CAMERA_INDEX,
    FACE_MATCH_THRESHOLD,
    FACE_MATCH_MARGIN,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    QR_SCAN_TIMEOUT,
)
from db import get_all_students, has_attended_today, mark_attendance
from liveness import (
    LivenessState,
    draw_liveness_overlay,
    get_ear_from_frame,
    reset_liveness,
    update_liveness_state,
)
from utils import (
    decode_qr_from_frame,
    draw_qr_box,
    extract_embedding,
    find_best_match,
    open_camera,
    read_frame,
)


# ── Phase enum ────────────────────────────────────────────────────────────────

class Phase(Enum):
    LIVENESS   = auto()
    FACE_MATCH = auto()
    QR_SCAN    = auto()
    DONE       = auto()


# ── Session state ─────────────────────────────────────────────────────────────

class AttendanceSession:
    """All mutable state for one attendance attempt."""

    def __init__(self) -> None:
        self.phase:           Phase          = Phase.LIVENESS
        self.liveness:        LivenessState  = reset_liveness()
        self.matched_student: Optional[dict] = None
        self.qr_scan_start:   float          = 0.0
        self.result_message:  str            = ""
        self.result_color:    tuple          = (200, 200, 200)

        # Background DeepFace thread state
        self._scan_running:    bool            = False
        self._scan_result:     Optional[dict]  = None   # written by bg thread on match
        self._scan_status:     str             = "Initialising…"
        self._spinner_idx:     int             = 0
        self._next_scan_time:  float           = 0.0   # earliest time to launch next scan

    def transition(self, new_phase: Phase) -> None:
        self.phase = new_phase
        if new_phase == Phase.QR_SCAN:
            self.qr_scan_start = time.time()

    def is_qr_expired(self) -> bool:
        return (time.time() - self.qr_scan_start) > QR_SCAN_TIMEOUT


# ── Background worker ─────────────────────────────────────────────────────────

def _recognition_worker(
    frame: np.ndarray,
    students: list[dict],
    threshold: float,
    session: AttendanceSession,
) -> None:
    """
    Daemon thread: extract embedding from *frame*, compare to all students.
    Writes match dict into session._scan_result on success.
    Always clears session._scan_running when finished.
    """
    try:
        embedding = extract_embedding(frame)
        if embedding is None:
            session._scan_status = "No face detected — look straight at camera"
            return
        match = find_best_match(embedding, students, threshold, FACE_MATCH_MARGIN)
        if match:
            session._scan_result = match
            session._scan_status = f"Matched: {match['name']}!"
        else:
            session._scan_status = "Face not recognised — hold still & retry"
    except Exception as exc:
        session._scan_status = f"Error: {exc}"
    finally:
        session._scan_running = False


# ── Phase processors ──────────────────────────────────────────────────────────

def _process_liveness(frame: np.ndarray, session: AttendanceSession) -> np.ndarray:
    """Phase 1: blink detection. On pass → FACE_MATCH."""
    ear, face_found = get_ear_from_frame(frame)
    session.liveness = update_liveness_state(session.liveness, ear, face_found)
    frame = draw_liveness_overlay(frame, session.liveness, ear)

    if session.liveness.liveness_passed:
        print("[Attendance] Liveness PASSED → starting face recognition.")
        session.transition(Phase.FACE_MATCH)

    return frame


_SPINNER = ["|", "/", "-", "\\"]


def _process_face_match(
    frame: np.ndarray,
    session: AttendanceSession,
    students: list[dict],
) -> np.ndarray:
    """
    Phase 2: background DeepFace recognition.
    - Launches a new thread whenever none is running.
    - Transitions to QR_SCAN IMMEDIATELY when a match is found.
    - Never auto-resets; press 'r' to restart manually.
    """
    # ── Check if background thread returned a result ──────────────────────────
    if session._scan_result is not None:
        match = session._scan_result
        session._scan_result  = None
        session._scan_running = False
        session.matched_student = match
        print(
            f"[Attendance] Matched: {match['name']} "
            f"(ID={match['id']}, dist={match['distance']:.4f}) → show QR code."
        )
        session.transition(Phase.QR_SCAN)
        return frame

    # ── Launch a new background thread if none is running & cooldown elapsed ──
    now = time.time()
    if not session._scan_running and now >= session._next_scan_time:
        session._scan_running   = True
        session._next_scan_time = now + 1.5   # wait 1.5 s before next attempt
        threading.Thread(
            target=_recognition_worker,
            args=(frame.copy(), students, FACE_MATCH_THRESHOLD, session),
            daemon=True,
        ).start()

    # ── Spinner overlay ───────────────────────────────────────────────────────
    session._spinner_idx = (session._spinner_idx + 1) % len(_SPINNER)
    spin = _SPINNER[session._spinner_idx]
    color = (0, 200, 255) if session._scan_running else (0, 80, 200)
    cv2.putText(frame, f"{spin}  {session._scan_status}",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv2.putText(frame, "Keep face visible — recognition running in background",
                (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (170, 170, 170), 1)
    return frame


def _process_qr_scan(frame: np.ndarray, session: AttendanceSession) -> np.ndarray:
    """Phase 3: QR decode + attendance marking."""
    student = session.matched_student
    assert student is not None

    frame = draw_qr_box(frame)

    remaining = max(0.0, QR_SCAN_TIMEOUT - (time.time() - session.qr_scan_start))
    cv2.putText(frame,
                f"Matched: {student['name']}  |  Show QR — {remaining:.0f}s",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

    if session.is_qr_expired():
        print("[Attendance] QR scan timed out. Restarting.")
        _reset_session(session)
        return frame

    qr_id = decode_qr_from_frame(frame)
    if qr_id is None:
        return frame

    face_id = student["id"]

    if qr_id != face_id:
        print(f"[Attendance] MISMATCH — face={face_id}  QR={qr_id}. REJECTED.")
        session.result_message = "REJECTED: QR does not match face!"
        session.result_color   = (0, 0, 220)
        session.transition(Phase.DONE)
        return frame

    # IDs match
    if has_attended_today(face_id):
        msg = f"{student['name']} already marked PRESENT today."
        session.result_color = (0, 160, 255)
    else:
        ok = mark_attendance(student_id=face_id, status="PRESENT")
        if ok:
            msg = f"ATTENDANCE MARKED — {student['name']}. Welcome!"
            session.result_color = (0, 200, 50)
        else:
            msg = "DB error — could not mark attendance."
            session.result_color = (0, 0, 200)

    print(f"[Attendance] {msg}")
    session.result_message = msg
    session.transition(Phase.DONE)
    return frame


def _reset_session(session: AttendanceSession) -> None:
    """Reset all state back to Phase 1 (liveness)."""
    session.phase              = Phase.LIVENESS
    session.liveness           = reset_liveness()
    session.matched_student    = None
    session.result_message     = ""
    session.result_color       = (200, 200, 200)
    session._scan_running      = False
    session._scan_result       = None
    session._scan_status       = "Initialising…"
    session._spinner_idx       = 0
    session._next_scan_time    = 0.0


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_attendance() -> None:
    """Open webcam and run the 3-phase attendance loop."""
    print("[Attendance] Loading registered students …")
    students = get_all_students()
    if not students:
        print("[Attendance] No students registered. Run register.py first.")
        return
    print(f"[Attendance] Loaded {len(students)} student(s).")

    # Pre-warm DeepFace / Facenet512 model so Phase 2 is instant
    print("[Attendance] Warming up face recognition model (may take ~30s first time) …")
    try:
        from utils import extract_embedding as _warmup
        _warmup(np.zeros((100, 100, 3), dtype=np.uint8))   # dummy — just loads weights
        print("[Attendance] Model ready. Opening camera …")
    except Exception as _e:
        print(f"[Attendance] Pre-warm note: {_e}")
        print("[Attendance] Continuing anyway …")

    cap     = open_camera(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)
    session = AttendanceSession()
    done_at: float = 0.0

    try:
        while True:
            frame = read_frame(cap)
            if frame is None:
                continue

            # Phase routing
            if session.phase == Phase.LIVENESS:
                cv2.putText(frame, "Phase 1: Blink to verify liveness",
                            (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 50), 1)
                frame = _process_liveness(frame, session)

            elif session.phase == Phase.FACE_MATCH:
                cv2.putText(frame, "Phase 2: Face Recognition — keep still",
                            (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 50), 1)
                frame = _process_face_match(frame, session, students)

            elif session.phase == Phase.QR_SCAN:
                cv2.putText(frame, "Phase 3: Show your QR code",
                            (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 50), 1)
                frame = _process_qr_scan(frame, session)
                if session.phase == Phase.DONE:
                    done_at = time.time()

            elif session.phase == Phase.DONE:
                cv2.putText(frame, session.result_message,
                            (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            session.result_color, 2)
                if time.time() - done_at >= 4:
                    _reset_session(session)

            # HUD
            cv2.putText(frame, "r = reset  |  q = quit",
                        (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
            cv2.imshow("Face + QR Attendance System", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[Attendance] Exiting.")
                break
            elif key == ord("r"):
                print("[Attendance] Session reset.")
                _reset_session(session)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_attendance()
