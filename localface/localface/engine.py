"""Face detection (YuNet) + face embedding (SFace), both via OpenCV's DNN module.

Runs fully on CPU, fully offline once the models are downloaded.
"""
from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
DETECTOR_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
RECOGNIZER_PATH = MODEL_DIR / "face_recognition_sface_2021dec.onnx"

# Cosine-similarity cutoff recommended by the SFace authors. Raise it to reduce
# false matches (stricter), lower it to reduce misses.
DEFAULT_THRESHOLD = 0.363


def resize_max(img, max_side):
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


class FaceEngine:
    def __init__(self, score_threshold=0.8):
        for p in (DETECTOR_PATH, RECOGNIZER_PATH):
            if not p.exists():
                raise SystemExit(
                    f"Missing model file: {p.name}\nRun:  python scripts/download_models.py"
                )
        self.detector = cv2.FaceDetectorYN.create(
            str(DETECTOR_PATH), "", (320, 320), score_threshold, 0.3, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(RECOGNIZER_PATH), "")

    def detect(self, frame):
        """Return an (N, 15) array: x, y, w, h, 10 landmark coords, score."""
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        return faces if faces is not None else np.zeros((0, 15), dtype=np.float32)

    def embed(self, frame, face):
        aligned = self.recognizer.alignCrop(frame, face)
        return self.recognizer.feature(aligned).reshape(-1)
