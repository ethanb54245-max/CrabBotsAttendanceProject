"""The parts that sit between the camera/models and the rest of the app, tested without hardware or models:
face-coordinate math (scale + edge padding), the threaded camera reader, and the CLI command flows."""
import argparse
import io
import os
import sys
import tempfile
import time
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["LOCALFACE_DATA"] = tempfile.mkdtemp()

import localface.cli as cli  # noqa: E402
from localface.camera import Camera  # noqa: E402
from localface.engine import DEFAULT_PAD, Face, FaceEngine  # noqa: E402
from localface.store import Store  # noqa: E402


# ------------------------------------------------------------------ coordinate math
class FakeRecognizer:
    def __init__(self):
        self.calls = []

    def alignCrop(self, img, g):
        self.calls.append((img.shape, np.array(g, dtype=np.float32)))
        return np.zeros((112, 112, 3), np.uint8)

    def feature(self, aligned):
        return np.ones((1, 128), np.float32)


def engine_seeing(true_box, true_landmarks, W, H, max_side):
    """A FaceEngine whose detector 'finds' a face at known ORIGINAL-image coordinates, reporting it the way
    the real detector would: in the coordinates of the downscaled, edge-padded image it is given."""
    eng = FaceEngine.__new__(FaceEngine)
    eng.recognizer = FakeRecognizer()
    scale = min(1.0, max_side / max(H, W))
    sh, sw = int(H * scale), int(W * scale)
    top, right = int(sh * DEFAULT_PAD[0]), int(sw * DEFAULT_PAD[1])
    bottom, left = int(sh * DEFAULT_PAD[2]), int(sw * DEFAULT_PAD[3])

    def detect(padded):
        assert padded.shape[:2] == (sh + top + bottom, sw + left + right), "padding size is wrong"
        x, y, w, h = true_box
        row = np.zeros(15, np.float32)
        row[0], row[1], row[2], row[3] = x * scale + left, y * scale + top, w * scale, h * scale
        for i, (lx, ly) in enumerate(true_landmarks):
            row[4 + 2 * i], row[5 + 2 * i] = lx * scale + left, ly * scale + top
        row[14] = 0.97
        return np.array([row])

    eng._detect = detect
    return eng


def test_face_coordinates_map_back_to_the_original_image():
    cases = [
        (1280, 720, 960, (500, 200, 180, 220)),     # typical webcam frame, downscaled for detection
        (640, 480, 960, (250, 100, 120, 150)),      # small frame, no downscale
        (1920, 1080, 960, (900, 300, 300, 380)),    # big frame
        (1600, 1200, 1280, (100, 50, 260, 320)),    # photo upload settings
        (1280, 720, 960, (600, -60, 170, 210)),     # face cut off by the TOP edge (the case padding exists for)
        (1280, 720, 960, (-40, 300, 160, 200)),     # cut off at the left edge
    ]
    for W, H, max_side, (x, y, w, h) in cases:
        lm = [(x + 0.3 * w, y + 0.35 * h), (x + 0.7 * w, y + 0.35 * h), (x + 0.5 * w, y + 0.55 * h),
              (x + 0.35 * w, y + 0.78 * h), (x + 0.65 * w, y + 0.78 * h)]
        eng = engine_seeing((x, y, w, h), lm, W, H, max_side)
        img = np.zeros((H, W, 3), np.uint8)
        faces = eng.analyze(img, max_side=max_side)
        assert len(faces) == 1, (W, H)
        f = faces[0]
        tol = 2.0                                                       # pixels (resize rounding)
        assert abs(f.x - x) < tol and abs(f.y - y) < tol, (W, H, f.x, f.y, x, y)
        assert abs(f.w - w) < tol and abs(f.h - h) < tol, (W, H)
        assert abs(f.cx - (x + w / 2)) < tol and f.score > 0.9 and f.embedding.shape == (128,)
        shape, g = eng.recognizer.calls[0]
        assert shape == (H, W, 3), "the embedding must come from the ORIGINAL full-resolution image"
        assert np.allclose(g[:4], [x, y, w, h], atol=tol)               # box handed to the aligner
        assert np.allclose(g[4:14], np.array(lm).reshape(-1), atol=tol)  # all five landmarks, x/y in the right slots


# ------------------------------------------------------------------ threaded camera
def test_camera_reader_delivers_fresh_frames_and_shuts_down():
    path = os.path.join(tempfile.mkdtemp(), "cam.avi")
    fourcc = (cv2.VideoWriter_fourcc(*"MJPG") if hasattr(cv2, "VideoWriter_fourcc")
              else cv2.VideoWriter.fourcc(*"MJPG"))
    wr = cv2.VideoWriter(path, fourcc, 30, (320, 240))
    if not wr.isOpened():
        print("   (skipped: this OpenCV build can't write a test video)")
        return
    for i in range(90):
        wr.write(np.full((240, 320, 3), i * 2 % 255, np.uint8))
    wr.release()
    cam = Camera(path)
    f = cam.read(timeout=3)
    assert f is not None and f.shape == (240, 320, 3)
    f[:] = 7                                                            # the caller may draw on its copy...
    g = cam.read(timeout=0.5)
    assert g is None or (g is not f and not (g == 7).all())             # ...without corrupting the next frame
    t0 = time.time()
    cam.release()
    assert time.time() - t0 < 3 and not cam._thread.is_alive()
    try:
        Camera(os.path.join(tempfile.gettempdir(), "no_such_camera.avi"))
        raise AssertionError("a missing camera should raise a friendly error")
    except SystemExit as e:
        assert "Could not open camera" in str(e)


# ------------------------------------------------------------------ CLI flows with stand-in engine/camera/window
class FakeEngine:
    """Treats an image's brightness as identity, so 'photos' of different people differ."""
    def __init__(self, *a, **k):
        pass

    def _emb(self, img):
        return np.random.default_rng(int(img.mean())).normal(size=128).astype(np.float32)

    def analyze(self, img, max_side=960):
        h, w = img.shape[:2]
        return [] if img.mean() < 5 else [Face(w * .3, h * .2, w * .3, h * .4, 0.95, self._emb(img))]

    def photo_embeddings(self, img):
        f = self.analyze(img)
        return (None, 0) if not f else ([f[0].embedding, f[0].embedding], 1)


def write_photo(path, gray):
    cv2.imwrite(str(path), np.full((480, 640, 3), gray, np.uint8))


def test_cli_photo_flows():
    Store().purge()
    cli.FaceEngine = FakeEngine
    d = Path(tempfile.mkdtemp())
    out = io.StringIO()
    with redirect_stdout(out):
        write_photo(d / "Ethan Badoud.jpg", 100)
        write_photo(d / "Ethan Badoud 2.jpg", 100)
        write_photo(d / "Sam Lee.jpg", 180)
        write_photo(d / "blank.jpg", 0)                                  # no face -> skipped, not a crash
        assert cli.cmd_enroll(argparse.Namespace(name="Ethan Badoud", photo=[str(d / "Ethan Badoud.jpg")],
                                                 i_have_consent=True, samples=10)) == 0
        assert cli.cmd_enroll_folder(argparse.Namespace(folder=str(d), i_have_consent=True)) == 0
        assert cli.cmd_enroll(argparse.Namespace(name="Nobody", photo=[str(d / "blank.jpg")],
                                                 i_have_consent=True, samples=10)) == 1
        with mock.patch("builtins.input", return_value="n"):               # consent question answered "no"
            assert cli.cmd_enroll(argparse.Namespace(name="Refused", photo=[str(d / "Sam Lee.jpg")],
                                                     i_have_consent=False, samples=10)) == 1
    names = {n: c for n, c, _ in Store().people()}
    assert names["Ethan Badoud"] == 6 and names["Sam Lee"] == 2, names              # 1 + 2 photos, each also mirrored
    assert "Nobody" not in names and "Refused" not in names, names                   # no face / no consent = not saved
    assert "skip" in out.getvalue() and "blank.jpg" in out.getvalue()

    annotated = d / "result.jpg"
    with redirect_stdout(io.StringIO()) as o2:
        cli.cmd_identify(argparse.Namespace(photo=[str(d / "Sam Lee.jpg")], threshold=0.363, annotate=str(annotated)))
    assert annotated.exists() and cv2.imread(str(annotated)).shape == (480, 640, 3)
    assert "Sam Lee" in o2.getvalue() and "unknown" not in o2.getvalue()


def test_cli_identify_names_the_right_person():
    cli.FaceEngine = FakeEngine
    d = Path(tempfile.mkdtemp())
    write_photo(d / "x.jpg", 100)
    write_photo(d / "y.jpg", 180)
    write_photo(d / "z.jpg", 40)                                          # not enrolled
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli.cmd_identify(argparse.Namespace(photo=[str(d)], threshold=0.363, annotate=None))
    text = buf.getvalue()
    by_photo = {}
    current = None
    for line in text.splitlines():
        if line.startswith(str(d)):
            current = Path(line.split(":")[0]).name
        elif "face 1:" in line and current:
            by_photo[current] = line
    assert "Ethan Badoud" in by_photo["x.jpg"] and "Sam Lee" in by_photo["y.jpg"], by_photo
    assert "unknown" in by_photo["z.jpg"], by_photo                                   # a stranger stays unknown


def test_cli_webcam_enroll_loop():
    Store().purge()
    cli.CAPTURE_GAP = 0.0
    rng = np.random.default_rng(5)
    shown = []

    class Cam:
        def __init__(self, *a, **k): pass
        def read(self): time.sleep(0.002); return np.zeros((720, 1280, 3), np.uint8)
        def release(self): pass

    class Eng(FakeEngine):
        def analyze(self, img, max_side=960):
            return [Face(500, 200, 260, 320, 0.95, rng.normal(size=128).astype(np.float32))]

    class Win:
        def __init__(self, *a, **k): pass
        def show(self, img): shown.append(img.shape)
        def poll(self): return ""
        def close(self): pass

    cli.Camera, cli.FaceEngine, cli.Window = Cam, Eng, Win
    with redirect_stdout(io.StringIO()):
        code = cli.cmd_enroll(argparse.Namespace(name="Webcam Wendy", photo=None, i_have_consent=True,
                                                 samples=10, camera=0, width=1280, height=720,
                                                 det_size=960, fullscreen=False))
    assert code == 0 and dict((n, c) for n, c, _ in Store().people())["Webcam Wendy"] == 10
    assert shown and shown[0] == (720, 1280, 3)


def test_empty_database_never_crashes_the_reports():
    Store().purge()
    with redirect_stdout(io.StringIO()) as buf:
        assert cli.cmd_list(None) == 0
        assert cli.cmd_events(argparse.Namespace(limit=5)) == 0
        assert cli.cmd_attendance(argparse.Namespace(date=None, summary=False, csv=None)) == 0
        assert cli.cmd_attendance(argparse.Namespace(date=None, summary=True, csv=None)) == 0
        assert cli.cmd_remove(argparse.Namespace(name="Ghost")) == 1
    assert "Nobody enrolled" in buf.getvalue() and "No such person" in buf.getvalue()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
    print("engine/camera/cli tests passed")
