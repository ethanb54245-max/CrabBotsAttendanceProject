"""Download Google's MediaPipe Face Landmarker model (~4 MB)."""
import sys
import urllib.request
from pathlib import Path

URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
DEST = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"
DEST.parent.mkdir(parents=True, exist_ok=True)
if DEST.exists() and DEST.stat().st_size > 1_000_000:
    print(f"Model already present: {DEST}")
    raise SystemExit(0)
print("Downloading MediaPipe Face Landmarker model...")
try:
    urllib.request.urlretrieve(URL, DEST)
except Exception as exc:
    print(f"Download failed: {exc}\nDownload this model manually:\n{URL}\nSave it as:\n{DEST}")
    raise SystemExit(1)
if DEST.stat().st_size < 1_000_000:
    DEST.unlink(missing_ok=True)
    print("Downloaded file was unexpectedly small; please retry.")
    raise SystemExit(1)
print(f"Saved {DEST} ({DEST.stat().st_size // 1024} KB)")
