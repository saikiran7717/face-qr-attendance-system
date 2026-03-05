-- =============================================================================
-- schema.sql — Database schema for Face + QR Attendance System
-- Run:  mysql -u root -p < database/schema.sql
-- =============================================================================

CREATE DATABASE IF NOT EXISTS attendance_db
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE attendance_db;

-- ── Students ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS students (
    id              INT          NOT NULL AUTO_INCREMENT,
    name            VARCHAR(120) NOT NULL,
    email           VARCHAR(200) NOT NULL UNIQUE,
    face_embedding  LONGBLOB     NOT NULL,   -- pickled numpy array (128-d vector)
    qr_path         VARCHAR(500) NOT NULL,   -- absolute / relative path to QR PNG
    created_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (id),
    INDEX idx_email (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── Attendance ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS attendance (
    attendance_id   INT          NOT NULL AUTO_INCREMENT,
    student_id      INT          NOT NULL,
    date            DATE         NOT NULL,
    time            TIME         NOT NULL,
    status          VARCHAR(20)  NOT NULL DEFAULT 'PRESENT',

    PRIMARY KEY (attendance_id),
    UNIQUE KEY uq_student_date (student_id, date),   -- one entry per student per day
    CONSTRAINT fk_student
        FOREIGN KEY (student_id) REFERENCES students (id)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    INDEX idx_date (date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
