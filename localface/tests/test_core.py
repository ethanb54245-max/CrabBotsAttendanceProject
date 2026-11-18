"""Model-free tests: storage, matching, tracking, identity locking, walking/standing, UI drawing."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["LOCALFACE_DATA"] = tempfile.mkdtemp()

from localface.store import Store  # noqa: E402
from localface.tracking import Detection, Tracker  # noqa: E402


def det(cx, name="alice", score=0.8, cy=200, w=100):
    return Detection(cx, cy, w, w * 1.2, name, score)


def test_store_and_match():
    rng = np.random.default_rng(0)
    alice, bob = rng.normal(size=128), rng.normal(size=128)
    s = Store(":memory:")
    s.add_person("alice", [alice + rng.normal(scale=0.05, size=128)])
    s.add_person("bob", [bob])
    g = s.gallery()
    assert g.match(alice, 0.363)[0] == "alice"
    assert g.match(bob, 0.363)[0] == "bob"
    assert g.match(rng.normal(size=128), 0.363)[0] is None
    assert s.remove_person("bob")
    assert s.gallery().match(bob, 0.363)[0] is None


def run(positions, **kw):
    tr, t, last = Tracker(), 0.0, None
    for cx in positions:
        out = tr.update([det(cx, **kw)], t)
        last, t = out[0], t + 0.1
    return tr, last


def test_standing_vs_walking():
    assert run([300 + (i % 2) for i in range(20)])[1].motion() == "standing"
    assert run([100 + 30 * i for i in range(20)])[1].motion() == "walking"


def test_fast_walker_keeps_one_track():
    tracker, tr = run([100 + 150 * i for i in range(8)])   # 150 px per frame, face is 100 px wide
    assert len(tracker.tracks) == 1 and tr.hits == 8


def test_identity_is_fast_but_not_trigger_happy():
    t = Tracker()
    tr = t.update([det(300, score=0.7)], 0.0)[0]
    assert tr.identity(0.5)[0] == "alice"                  # one strong match is enough
    t = Tracker()
    tr = t.update([det(300, score=0.40)], 0.0)[0]
    assert tr.identity(0.5)[0] is None                     # one weak match is not
    tr = t.update([det(300, score=0.41)], 0.1)[0]
    assert tr.identity(0.5)[0] == "alice"                  # two weak matches are


def test_identity_sticks_through_failed_frames_and_switches_only_on_evidence():
    t, now = Tracker(), 0.0
    tr = t.update([det(300, score=0.7)], now)[0]
    assert tr.identity()[0] == "alice"
    for _ in range(5):                                     # recognition fails (blur, turned head)
        now += 0.1
        tr = t.update([det(300, name=None, score=0.1)], now)[0]
    assert tr.identity()[0] == "alice"
    for _ in range(3):                                     # tracks genuinely swapped to bob
        now += 0.1
        tr = t.update([det(300, name="bob", score=0.6)], now)[0]
    assert tr.identity()[0] == "bob"


def test_ui_draws_without_error_including_top_edge():
    import cv2
    from localface import ui
    for H, W in ((720, 1280), (1080, 1920), (480, 640)):
        img = np.zeros((H, W, 3), np.uint8)
        top = ui.topbar_height(img)
        ui.draw_face(img, (W // 2, 2, 120, 150), "ok", "Ethan", 0.81, "walking", top)   # face at very top
        ui.draw_face(img, (5, H - 160, 120, 150), "unknown", "Unknown", None, "standing", top)
        ui.draw_face(img, (W - 130, H // 2, 120, 150), "scanning", "Identifying...", None, None, top)
        ui.draw_topbar(img, 28.4, 2, 3)
        ui.draw_feed(img, [(__import__("time").time(), "Ethan", "walking")])
        ui.draw_hint(img, "F fullscreen   H hide HUD   Q quit")
        ui.draw_enroll(img, "Look straight at the camera", 1, 5, 3, 10, (W // 2, H // 3, 120, 150), None, 0.1)
        assert img.any()
    assert cv2 is not None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all tests passed")
