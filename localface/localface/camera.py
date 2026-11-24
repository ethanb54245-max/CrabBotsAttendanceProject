"""Threaded webcam reader that always hands back the *newest* frame (no stale buffered lag)."""
import sys
import threading
import time

import cv2


class Camera:
    def __init__(self, index=0, width=1280, height=720):
        cap = None
        if sys.platform.startswith("win") and hasattr(cv2, "CAP_DSHOW"):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = None
        if cap is None:
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise SystemExit(
                f"Could not open camera {index}. Try --camera 1, or check OS camera permissions.")
        try:
            fourcc = (cv2.VideoWriter_fourcc(*"MJPG") if hasattr(cv2, "VideoWriter_fourcc")
                      else cv2.VideoWriter.fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        except Exception:  # noqa: BLE001
            pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        self._lock = threading.Lock()
        self._frame, self._seq, self._seen, self._run = None, 0, 0, True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        fails = 0
        while self._run:
            ok, frame = self._cap.read()
            if not ok:
                fails += 1
                if fails > 200:
                    break
                time.sleep(0.01)
                continue
            fails = 0
            with self._lock:
                self._frame, self._seq = frame, self._seq + 1

    def read(self, timeout=3.0):
        """Newest unseen frame (a copy you can draw on), or None if the camera stalled."""
        end = time.time() + timeout
        while time.time() < end:
            with self._lock:
                if self._frame is not None and self._seq != self._seen:
                    self._seen = self._seq
                    return self._frame.copy()
            time.sleep(0.002)
        return None

    def release(self):
        self._run = False
        self._thread.join(timeout=1.0)
        self._cap.release()
