# =============================================================================
# db.py - MySQL database layer for Face + QR Attendance System
# =============================================================================

from __future__ import annotations

import csv
import pickle
from contextlib import contextmanager
from datetime import date, datetime
from typing import Generator, Optional

import mysql.connector
import numpy as np
from mysql.connector import Error as MySQLError

from config import DB_CONFIG


# -- Connection helper ---------------------------------------------------------

@contextmanager
def _get_connection() -> Generator[mysql.connector.MySQLConnection, None, None]:
    """Yield a managed MySQL connection and always close it."""
    conn: Optional[mysql.connector.MySQLConnection] = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        yield conn
    except MySQLError as exc:
        raise RuntimeError(f"[DB] Connection failed: {exc}") from exc
    finally:
        if conn and conn.is_connected():
            conn.close()


# -- Student operations --------------------------------------------------------

def register_student(
    name: str,
    email: str,
    embedding: np.ndarray,
    qr_path: str,
) -> int:
    """
    Insert a new student record and return generated student ID.
    """
    blob = pickle.dumps(embedding)
    sql = """
        INSERT INTO students (name, email, face_embedding, qr_path)
        VALUES (%s, %s, %s, %s)
    """
    with _get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(sql, (name, email, blob, qr_path))
            conn.commit()
            student_id: int = cursor.lastrowid
            print(f"[DB] Student '{name}' registered with ID={student_id}.")
            return student_id
        except MySQLError as exc:
            conn.rollback()
            raise RuntimeError(f"[DB] Failed to register student: {exc}") from exc
        finally:
            cursor.close()


def get_all_students() -> list[dict]:
    """
    Fetch all students with unpickled embeddings.
    """
    sql = "SELECT id, name, email, face_embedding, qr_path FROM students"
    with _get_connection() as conn:
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute(sql)
            rows = cursor.fetchall()
        finally:
            cursor.close()

    students: list[dict] = []
    for row in rows:
        embedding = pickle.loads(row["face_embedding"])
        students.append(
            {
                "id": row["id"],
                "name": row["name"],
                "email": row["email"],
                "embedding": embedding,
                "qr_path": row["qr_path"],
            }
        )
    return students


def get_student_by_id(student_id: int) -> Optional[dict]:
    """Return one student dict or None if not found."""
    sql = "SELECT id, name, email, qr_path FROM students WHERE id = %s"
    with _get_connection() as conn:
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute(sql, (student_id,))
            row = cursor.fetchone()
        finally:
            cursor.close()
    return row


def student_exists_by_email(email: str) -> bool:
    """Return True if student email already exists."""
    sql = "SELECT 1 FROM students WHERE email = %s LIMIT 1"
    with _get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(sql, (email,))
            return cursor.fetchone() is not None
        finally:
            cursor.close()


def delete_student(student_id: int) -> Optional[dict]:
    """
    Delete a student from DB.

    Returns dict with name and qr_path so caller can clean local files,
    or None when student does not exist.
    """
    sql_select = "SELECT name, qr_path FROM students WHERE id = %s"
    sql_delete = "DELETE FROM students WHERE id = %s"

    with _get_connection() as conn:
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute(sql_select, (student_id,))
            row = cursor.fetchone()
            if row is None:
                return None

            cursor.execute(sql_delete, (student_id,))
            conn.commit()
            print(f"[DB] Student ID={student_id} deleted from database.")
            return {"name": row["name"], "qr_path": row["qr_path"] or ""}
        except MySQLError as exc:
            conn.rollback()
            raise RuntimeError(f"[DB] Failed to delete student: {exc}") from exc
        finally:
            cursor.close()


# -- Attendance operations -----------------------------------------------------

def has_attended_today(student_id: int) -> bool:
    """Return True if attendance for today already exists."""
    today = date.today()
    sql = "SELECT 1 FROM attendance WHERE student_id = %s AND date = %s LIMIT 1"
    with _get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(sql, (student_id, today))
            return cursor.fetchone() is not None
        finally:
            cursor.close()


def mark_attendance(student_id: int, status: str = "PRESENT") -> bool:
    """
    Insert attendance for current date.

    Returns True when inserted, False when already marked.
    """
    if has_attended_today(student_id):
        return False

    now = datetime.now()
    today = now.date()
    clock = now.time().replace(microsecond=0)

    sql = """
        INSERT INTO attendance (student_id, date, time, status)
        VALUES (%s, %s, %s, %s)
    """
    with _get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(sql, (student_id, today, clock, status))
            conn.commit()
            return True
        except MySQLError as exc:
            conn.rollback()
            raise RuntimeError(f"[DB] Failed to mark attendance: {exc}") from exc
        finally:
            cursor.close()


def export_attendance_csv(filepath: str = "attendance_export.csv") -> None:
    """Export attendance joined with student info into CSV."""
    sql = """
        SELECT
            a.attendance_id,
            s.id AS student_id,
            s.name,
            s.email,
            a.date,
            a.time,
            a.status
        FROM attendance a
        JOIN students s ON s.id = a.student_id
        ORDER BY a.date DESC, a.time DESC
    """

    with _get_connection() as conn:
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute(sql)
            rows = cursor.fetchall()
        finally:
            cursor.close()

    if not rows:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "attendance_id",
                "student_id",
                "name",
                "email",
                "date",
                "time",
                "status",
            ])
        print(f"[DB] No rows found. Empty CSV created: {filepath}")
        return

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "attendance_id",
                "student_id",
                "name",
                "email",
                "date",
                "time",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DB] Exported {len(rows)} attendance row(s) to {filepath}")
