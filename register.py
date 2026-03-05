# =============================================================================
# register.py — Student registration module
#
# Usage:
#   python register.py
#
# Workflow:
#   1. Prompt for student name & email
#   2. Open webcam, capture N face samples
#   3. Generate average face embedding
#   4. Store record in MySQL
#   5. Generate and save QR code
# =============================================================================

from __future__ import annotations

import os
import sys
import time

import cv2

from config import (
    CAMERA_INDEX,
    FACES_DIR,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    REGISTRATION_SAMPLES,
)
from db import register_student, student_exists_by_email
from utils import (
    average_embedding,
    extract_embedding,
    generate_qr_code,
    open_camera,
    read_frame,
)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _save_face_sample(frame: "np.ndarray", student_name: str, idx: int) -> None:
    """Persist a face sample image to disk for auditing purposes."""
    safe_name = student_name.replace(" ", "_").lower()
    folder    = os.path.join(FACES_DIR, safe_name)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"sample_{idx}.jpg")
    cv2.imwrite(path, frame)


def _collect_face_samples(
    student_name: str,
    n_samples: int = REGISTRATION_SAMPLES,
) -> list["np.ndarray"]:
    """
    Open webcam and collect *n_samples* frames that contain a detectable face.

    Returns
    -------
    list of np.ndarray (BGR frames), one per successful sample.
    """
    cap = open_camera(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)
    samples: list = []
    collected = 0

    print(f"\n[Register] Webcam open. Look at the camera.")
    print(f"[Register] Collecting {n_samples} face samples — press 'c' to capture, 'q' to quit.\n")

    try:
        while collected < n_samples:
            frame = read_frame(cap)
            if frame is None:
                continue

            # Live preview
            cv2.putText(
                frame,
                f"Sample {collected}/{n_samples}  —  press 'c' to capture",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 200, 255),
                2,
            )
            cv2.imshow("Registration — Press 'c' to capture, 'q' to quit", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[Register] Registration cancelled by user.")
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)

            if key == ord("c"):
                embedding = extract_embedding(frame)
                if embedding is None:
                    print("[Register] No face detected in frame. Try again.")
                    continue
                collected += 1
                samples.append(frame.copy())
                _save_face_sample(frame, student_name, collected)
                print(f"[Register] Captured sample {collected}/{n_samples}.")
                time.sleep(0.3)   # brief pause to encourage pose variation

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return samples


# ── Public entry point ────────────────────────────────────────────────────────

def register_new_student() -> None:
    """
    Full registration flow — prompt, capture, embed, store, QR.
    """
    print("=" * 60)
    print("   STUDENT REGISTRATION")
    print("=" * 60)

    name  = input("Enter student name  : ").strip()
    email = input("Enter student email : ").strip()

    if not name or not email:
        print("[Register] Name and email are required. Aborting.")
        return

    if student_exists_by_email(email):
        print(f"[Register] A student with email '{email}' is already registered.")
        return

    # ── Step 1: Collect face samples ─────────────────────────────────────────
    frames = _collect_face_samples(name, REGISTRATION_SAMPLES)
    if len(frames) < REGISTRATION_SAMPLES:
        print("[Register] Not enough samples collected. Registration aborted.")
        return

    # ── Step 2: Generate embeddings and average them ──────────────────────────
    print("[Register] Generating face embeddings …")
    import numpy as np
    embeddings = []
    for i, frame in enumerate(frames):
        emb = extract_embedding(frame)
        if emb is not None:
            embeddings.append(emb)
        else:
            print(f"[Register] Warning: sample {i + 1} yielded no embedding.")

    if len(embeddings) < 2:
        print("[Register] Too few valid embeddings extracted. Please re-register.")
        return

    avg_embedding = average_embedding(embeddings)

    # ── Step 3: Persist student in MySQL (placeholder QR path first) ──────────
    student_id = register_student(
        name=name,
        email=email,
        embedding=avg_embedding,
        qr_path="",        # updated after QR generation
    )

    # ── Step 4: Generate QR code ──────────────────────────────────────────────
    qr_path = generate_qr_code(student_id)

    # ── Step 5: Update QR path in DB ─────────────────────────────────────────
    from db import _get_connection
    sql = "UPDATE students SET qr_path = %s WHERE id = %s"
    with _get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(sql, (qr_path, student_id))
            conn.commit()
        finally:
            cursor.close()

    print("\n" + "=" * 60)
    print(f"  Registration complete!")
    print(f"  Name      : {name}")
    print(f"  Email     : {email}")
    print(f"  Student ID: {student_id}")
    print(f"  QR Code   : {qr_path}")
    print("=" * 60)


# ── CLI entry ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    register_new_student()
