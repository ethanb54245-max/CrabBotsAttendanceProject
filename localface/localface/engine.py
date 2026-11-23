"""Face detection (YuNet) + face embedding (SFace), both via OpenCV's DNN module.

Runs fully on CPU, fully offline once the models are downloaded.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:  # silence harmless backend chatter from OpenCV's DNN module
    cv2.setLogLevel(2)
except Exception:  # noqa: BLE001
    pass

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
DETECTOR_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
RECOGNIZER_PATH = MODEL_DIR / "face_recognition_sface_2021dec.onnx"

# Cosine-similarity cutoff recommended by the SFace authors. Raise it to reduce
# false matches (stricter), lower it to reduce misses.
DEFAULT_THRESHOLD = 0.363

# Black padding added around the frame before detection (top, right, bottom, left, as a
# fraction of the frame). Faces cut off by the frame edge become "whole" faces in the padded
# image, which makes detection far more reliable near the edges, especially the top.
DEFAULT_PAD = (0.30, 0.12, 0.12, 0.12)


def resize_max(img, max_side):
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


@dataclass
class Face:
    x: float
    y: float
    w: float
    h: float
    score: float          # detector confidence
    embedding: np.ndarray

    @property
    def cx(self):
        return self.x + self.w / 2

    @property
    def cy(self):
        return self.y + self.h / 2


class FaceEngine:
    def __init__(self, score_threshold=0.6):
        for p in (DETECTOR_PATH, RECOGNIZER_PATH):
            if not p.exists():
                raise SystemExit(
                    f"Missing model file: {p.name}\nRun:  python scripts/download_models.py"
                )
        self.detector = cv2.FaceDetectorYN.create(
            str(DETECTOR_PATH), "", (320, 320), score_threshold, 0.3, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(RECOGNIZER_PATH), "")

    def _detect(self, img):
        h, w = img.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(img)
        return faces if faces is not None else np.zeros((0, 15), dtype=np.float32)

    def photo_embeddings(self, img):
        """Embeddings for enrolling from ONE photo: (vectors, faces_found).

        vectors is None unless the photo has exactly one face. The mirrored image is added as a
        second sample, which makes a photo-only enrollment noticeably more robust on a live camera.
        """
        faces = self.analyze(img, max_side=1280)
        if len(faces) != 1:
            return None, len(faces)
        vecs = [faces[0].embedding]
        mirrored = self.analyze(cv2.flip(img, 1), max_side=1280)
        if len(mirrored) == 1:
            vecs.append(mirrored[0].embedding)
        return vecs, 1

    def analyze(self, img, max_side=960, pad=DEFAULT_PAD):
        """Detect and embed every face. Returned coordinates are in `img`'s own pixels.

        Detection runs on a downscaled, edge-padded copy for speed and edge robustness;
        the embedding is computed from the original full-resolution pixels for quality.
        """
        h, w = img.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        small = img if scale >= 1 else cv2.resize(
            img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        sh, sw = small.shape[:2]
        top, right = int(sh * pad[0]), int(sw * pad[1])
        bottom, left = int(sh * pad[2]), int(sw * pad[3])
        padded = cv2.copyMakeBorder(small, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))
        out = []
        for f in self._detect(padded):
            g = f.astype(np.float32).copy()
            for i in (0, 4, 6, 8, 10, 12):      # x-coordinates (box + landmarks)
                g[i] = (f[i] - left) / scale
            for i in (1, 5, 7, 9, 11, 13):      # y-coordinates
                g[i] = (f[i] - top) / scale
            g[2] /= scale
            g[3] /= scale
            emb = self.recognizer.feature(self.recognizer.alignCrop(img, g)).reshape(-1)
            out.append(Face(float(g[0]), float(g[1]), float(g[2]), float(g[3]), float(f[-1]), emb))
        return out
