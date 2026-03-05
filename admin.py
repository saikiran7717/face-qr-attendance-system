# =============================================================================
# admin.py — Admin console for the Face + QR Attendance System (Bonus)
#
# Usage:  python admin.py
# =============================================================================

from __future__ import annotations

import os
import shutil

from config import FACES_DIR, QR_DIR
from db import export_attendance_csv, get_all_students, delete_student, _get_connection


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_divider(char: str = "─", width: int = 60) -> None:
    print(char * width)


def _list_students() -> None:
    students = get_all_students()
    _print_divider()
    if not students:
        print("  No students registered.")
        return
    fmt = "{:<6} {:<25} {:<30}"
    print(fmt.format("ID", "Name", "Email"))
    _print_divider()
    for s in students:
        print(fmt.format(s["id"], s["name"], s["email"]))
    print(f"\n  Total: {len(students)} student(s)")


def _view_attendance() -> None:
    sql = """
        SELECT
            a.attendance_id,
            s.name,
            a.date,
            a.time,
            a.status
        FROM attendance a
        JOIN students s ON s.id = a.student_id
        ORDER BY a.date DESC, a.time DESC
        LIMIT 50
    """
    with _get_connection() as conn:
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute(sql)
            rows = cursor.fetchall()
        finally:
            cursor.close()

    _print_divider()
    if not rows:
        print("  No attendance records found.")
        return
    fmt = "{:<6} {:<25} {:<12} {:<10} {:<10}"
    print(fmt.format("ID", "Name", "Date", "Time", "Status"))
    _print_divider()
    for r in rows:
        print(fmt.format(
            r["attendance_id"],
            r["name"],
            str(r["date"]),
            str(r["time"]),
            r["status"],
        ))
    print(f"\n  Showing {len(rows)} most-recent record(s).")


def _delete_student() -> None:
    try:
        sid = int(input("Enter Student ID to delete: ").strip())
    except ValueError:
        print("[Admin] Invalid ID.")
        return

    # Fetch name + qr_path before deleting so we can clean up disk
    result = delete_student(sid)
    if result is None:
        print(f"[Admin] No student with ID={sid} found.")
        return

    name, qr_path = result["name"], result["qr_path"]

    # ── Delete QR code file ───────────────────────────────────────────────────
    if qr_path and os.path.isfile(qr_path):
        os.remove(qr_path)
        print(f"[Admin] Deleted QR file : {qr_path}")
    else:
        # Also try the default location in case qr_path is blank
        default_qr = os.path.join(QR_DIR, f"student_{sid}.png")
        if os.path.isfile(default_qr):
            os.remove(default_qr)
            print(f"[Admin] Deleted QR file : {default_qr}")

    # ── Delete faces folder ───────────────────────────────────────────────────
    safe_name  = name.replace(" ", "_").lower()
    faces_dir  = os.path.join(FACES_DIR, safe_name)
    if os.path.isdir(faces_dir):
        shutil.rmtree(faces_dir)
        print(f"[Admin] Deleted faces dir: {faces_dir}")

    print(f"[Admin] Student '{name}' (ID={sid}) fully removed — DB + QR + faces.")


# ── Menu ──────────────────────────────────────────────────────────────────────

_MENU = """
╔══════════════════════════════════════╗
║     ATTENDANCE SYSTEM — ADMIN       ║
╠══════════════════════════════════════╣
║  1. List all students               ║
║  2. View attendance records         ║
║  3. Export attendance to CSV        ║
║  4. Delete a student                ║
║  0. Exit                            ║
╚══════════════════════════════════════╝
"""


def run_admin_console() -> None:
    while True:
        print(_MENU)
        choice = input("Select option: ").strip()

        if choice == "1":
            _list_students()
        elif choice == "2":
            _view_attendance()
        elif choice == "3":
            path = input("CSV filename [attendance_export.csv]: ").strip()
            export_attendance_csv(path or "attendance_export.csv")
        elif choice == "4":
            _delete_student()
        elif choice == "0":
            print("[Admin] Goodbye.")
            break
        else:
            print("[Admin] Unknown option. Please choose 0–4.")


if __name__ == "__main__":
    run_admin_console()
