# Face Recognition + QR Code Attendance System

A production-ready, multimodal attendance system that combines **live face recognition**, **QR code verification**, and **blink-based liveness detection**. Attendance is marked only when all three checks pass.

---

## Project Structure

```
Project/
├── config.py          # Central config (DB creds, thresholds, paths)
├── db.py              # MySQL data layer
├── liveness.py        # EAR-based blink detection
├── utils.py           # Face embedding, QR generation & decoding
├── register.py        # Student registration CLI
├── attendance.py      # Live attendance main loop
├── admin.py           # Admin console (CSV export, view records)
├── database/
│   └── schema.sql     # MySQL table definitions
├── faces/             # Captured face samples (auto-created)
├── qr_codes/          # Generated QR PNGs (auto-created)
└── models/            # Dlib landmark model (place .dat file here)
```

---

## Prerequisites

### 1. Python 3.10+

### 2. MySQL Server
- Install and start MySQL.
- Open `config.py` and set your `DB_CONFIG` credentials.
- Run the schema:
  ```bash
  mysql -u root -p < database/schema.sql
  ```

### 3. System Libraries

**Windows:**
- Download and install **CMake** from https://cmake.org/download/
- Install **Visual Studio Build Tools** (C++ workload) from https://aka.ms/vs/17/release/vs_BuildTools.exe
- For `pyzbar`: download `libzbar-64.dll` from https://github.com/NaturalHistoryMuseum/pyzbar and place it in `C:\Windows\System32\`

**Ubuntu / Debian:**
```bash
sudo apt-get install build-essential cmake libopenblas-dev liblapack-dev
sudo apt-get install libzbar0
```

**macOS:**
```bash
brew install cmake libzbar
```

### 4. Dlib Landmark Model

Download the pre-trained model:
```bash
# Direct download
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
mv shape_predictor_68_face_landmarks.dat models/
```

Or via Python:
```python
import urllib.request, bz2, shutil
url  = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
dest = "models/shape_predictor_68_face_landmarks.dat"
urllib.request.urlretrieve(url, dest + ".bz2")
with bz2.open(dest + ".bz2") as src, open(dest, "wb") as out:
    shutil.copyfileobj(src, out)
```

---

## Installation

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux / macOS)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Note on dlib on Windows**: If `pip install dlib` fails, use a pre-built wheel:
> ```bash
> pip install https://github.com/jloh02/dlib/releases/download/v19.22/dlib-19.22.0-cp310-cp310-win_amd64.whl
> ```
> Match the wheel to your Python version (cp310, cp311, etc.).

---

## Configuration (`config.py`)

| Key | Default | Description |
|-----|---------|-------------|
| `DB_CONFIG` | `localhost:3306` | MySQL connection settings |
| `FACE_MATCH_THRESHOLD` | `0.50` | Max L2 distance to accept a face match |
| `REGISTRATION_SAMPLES` | `5` | Face captures per student |
| `EAR_THRESHOLD` | `0.25` | Eye Aspect Ratio below = blink |
| `REQUIRED_BLINKS` | `1` | Blinks needed to pass liveness |
| `QR_SCAN_TIMEOUT` | `15` | Seconds to scan QR after face match |

---

## Usage

### Register a New Student

```bash
python register.py
```

You will be prompted for name and email. The webcam opens; press **`c`** to capture each sample, **`q`** to cancel.  
A QR code PNG is saved to `qr_codes/student_<id>.png`. Print or display it on a phone.

### Run Attendance

```bash
python attendance.py
```

The system runs in three phases:

| Phase | What to do |
|-------|-----------|
| **1 — Liveness** | Look at the camera and **blink** once |
| **2 — Face match** | Stay in frame; recognition runs automatically |
| **3 — QR scan** | Show your QR code to the camera within 15 s |

Press **`r`** to reset the session, **`q`** to quit.

### Admin Console

```bash
python admin.py
```

Menu options: list students, view attendance, export CSV.

---

## Database Schema

```sql
students  (id PK, name, email, face_embedding BLOB, qr_path, created_at)
attendance(attendance_id PK, student_id FK, date, time, status)
           UNIQUE (student_id, date)   -- prevents duplicate per day
```

---

## Security Design

- QR payload contains **only the integer Student ID** — no personal data in QR.
- Face match uses configurable **L2 distance threshold** (not cosine; harder to spoof).
- **Face ID and QR ID must match** — presenting someone else's QR is rejected.
- **Duplicate guard**: `UNIQUE (student_id, date)` at the database level.
- **Liveness check**: blink requirement rejects printed photos and static videos.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `FileNotFoundError: shape_predictor…` | Place `.dat` file in `models/` |
| `dlib` build fails on Windows | Use pre-built wheel (see above) |
| `pyzbar` cannot find libzbar | Install `libzbar0` or place DLL in System32 |
| Camera not opening | Change `CAMERA_INDEX` in `config.py` to `1` or `2` |
| `Access denied` MySQL error | Check `DB_CONFIG` credentials in `config.py` |

---

## License

MIT
