"""Model-free tests for the club features: kiosk typing, folder import naming, attendance, sign-up loop."""
import argparse
import io
import os
import sys
import tempfile
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["LOCALFACE_DATA"] = tempfile.mkdtemp()

import localface.cli as cli  # noqa: E402
from localface import ui  # noqa: E402
from localface.engine import Face  # noqa: E402
from localface.store import Store  # noqa: E402


def test_apply_key():
    buf = ""
    for ch in "Al-ice!":
        buf, act = cli.apply_key(buf, ord(ch))
        assert act is None
    assert buf == "Al-ice"                       # '!' rejected
    assert cli.apply_key(buf, 8) == ("Al-ic", None)
    assert cli.apply_key(buf, 13)[1] == "enter" and cli.apply_key(buf, 27)[1] == "esc"
    assert cli.apply_key("x" * 32, ord("y"))[0] == "x" * 32


def test_filenames_and_folder():
    f = cli.person_from_filename
    assert f("Alice Smith") == "Alice Smith"
    assert f("Alice Smith 2") == "Alice Smith"
    assert f("alice_smith-3") == "alice_smith"
    assert f("Bob (2)") == "Bob"
    assert f("123") == "123"
    d = Path(tempfile.mkdtemp())
    for n in ("Alice.jpg", "Alice 2.png", "Bob.jpeg", "notes.txt"):
        (d / n).write_bytes(b"x")
    (d / "Carol Jones").mkdir()
    (d / "Carol Jones" / "a.jpg").write_bytes(b"x")
    (d / "Carol Jones" / "b.jpg").write_bytes(b"x")
    g = cli.collect_folder(d)
    assert sorted(g) == ["Alice", "Bob", "Carol Jones"]
    assert len(g["Alice"]) == 2 and len(g["Carol Jones"]) == 2


def test_attendance_report():
    s = Store(":memory:")
    s.add_person("Alice", [np.ones(128)])
    s.add_person("Bob", [np.ones(128)])
    day1, day2 = time.mktime((2026, 11, 16, 16, 0, 0, 0, 0, -1)), time.mktime((2026, 11, 17, 16, 5, 0, 0, 0, -1))
    for ts, who in ((day1, "Alice"), (day1 + 600, "Alice"), (day1 + 60, "Bob"), (day2, "Alice")):
        s.db.execute("INSERT INTO events(ts,name,score,motion) VALUES(?,?,?,?)", (ts, who, 0.8, "walking"))
    rows = s.attendance_rows()
    assert [(r[0], r[1], r[4]) for r in rows] == [
        ("2026-11-16", "Alice", 2), ("2026-11-16", "Bob", 1), ("2026-11-17", "Alice", 1)]
    assert s.has_person("alice") and not s.has_person("Zed")


def test_attendance_command_and_csv():
    st = Store()
    st.add_person("Alice", [np.ones(128)])
    st.add_person("Bob", [np.ones(128)])
    now = time.time()
    st.db.execute("INSERT INTO events(ts,name,score,motion) VALUES(?,?,?,?)", (now, "Alice", 0.9, "standing"))
    st.db.commit()
    out = io.StringIO()
    csv_path = os.path.join(tempfile.mkdtemp(), "a.csv")
    with redirect_stdout(out):
        cli.cmd_attendance(argparse.Namespace(date=None, summary=False, csv=csv_path))
        cli.cmd_attendance(argparse.Namespace(date=None, summary=False, csv=None))
        cli.cmd_attendance(argparse.Namespace(date=None, summary=True, csv=None))
    text = out.getvalue()
    assert "1/2 present" in text and "PRESENT  Alice" in text and "absent   Bob" in text
    assert "Alice" in open(csv_path).read() and " X " in text.replace("  X  ", " X ")
    print(text)


def test_signup_kiosk_end_to_end_with_fakes():
    cli.Store().purge()
    cli.CAPTURE_GAP, cli.WELCOME_SECS = 0.0, 0.0
    rng = np.random.default_rng(3)
    person = {"v": rng.normal(size=128).astype(np.float32)}
    keys = list("Bob") + [13] + list("Bobby") + [13] + [27]
    keys = [ord(k) if isinstance(k, str) else k for k in keys]
    flags = {"name": False}

    class FakeCam:
        def __init__(self, *a, **k): pass
        def read(self): time.sleep(0.005); return np.zeros((720, 1280, 3), np.uint8)
        def release(self): pass

    class FakeEngine:
        def __init__(self, *a, **k): pass
        def analyze(self, img, max_side=960):
            return [Face(500, 200, 200, 250, 0.95, person["v"] + rng.normal(scale=0.02, size=128).astype(np.float32))]

    class FakeWin:
        def __init__(self, *a, **k): pass
        def show(self, img): pass
        def closed(self): return False
        def close(self): pass
        def raw_key(self):
            if flags["name"] and keys:
                flags["name"] = False
                return keys.pop(0)
            return 255

    real = (ui.draw_signup_name, ui.draw_enroll, ui.draw_welcome)
    def name_screen(*a, **k): flags["name"] = True; return real[0](*a, **k)
    def other(fn):
        def inner(*a, **k): flags["name"] = False; return fn(*a, **k)
        return inner
    ui.draw_signup_name, ui.draw_enroll, ui.draw_welcome = name_screen, other(real[1]), other(real[2])
    cli.Camera, cli.FaceEngine, cli.Window = FakeCam, FakeEngine, FakeWin
    out = io.StringIO()
    with redirect_stdout(out):
        cli.cmd_signup(argparse.Namespace(camera=0, width=1280, height=720, det_size=960,
                                          fullscreen=False, threshold=0.363))
    names = [p[0] for p in Store().people()]
    assert names == ["Bob"], (names, out.getvalue())          # 2nd attempt (same face, new name) was rejected
    assert Store().people()[0][1] == 5                          # 5 pose samples saved
    assert "Enrolled Bob" in out.getvalue()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
    print("all club tests passed")
