"""Live recognition: shared recognizer logic and the in-page camera endpoints."""
import os
import sys
import tempfile
import threading
import time
from http.server import ThreadingHTTPServer

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["LOCALFACE_DATA"] = tempfile.mkdtemp()

from localface import demo  # noqa: E402
from localface.engine import Face  # noqa: E402
from localface.recognizer import LiveRecognizer  # noqa: E402
from localface.store import Store  # noqa: E402
from test_demo import FakeEngine, call, photo  # noqa: E402

GOOD = {"X-LocalFace": "1"}


def vec(seed):
    return np.random.default_rng(seed).normal(size=128).astype(np.float32)


def face(x, emb, w=100):
    return Face(x, 50, w, w * 1.2, 0.95, emb)


def test_recognizer_identity_cooldown_logging_reload():
    st = Store()
    st.purge()
    st.add_person("Ava", [vec(1)])
    seq = []
    rec = LiveRecognizer(lambda img: seq.pop(0) if seq else [], 0.363, cooldown=30.0)
    assert rec.enrolled == 1 and len(rec.gallery) == 1

    seq.append([face(100, vec(1))])                                   # one strong match is enough
    results, events = rec.process(None, now=1000.0)
    assert results[0].name == "Ava" and results[0].state == "ok" and results[0].visible
    assert events == []                                               # motion still 'detecting' for < 1s
    seq.append([face(100, vec(1))])
    _, events = rec.process(None, now=1001.2)
    assert [e["name"] for e in events] == ["Ava"] and rec.present == {"Ava"}
    assert [r[1] for r in Store().recent_events()] == ["Ava"]         # logged once
    for t in (1002.0, 1010.0, 1025.0):
        seq.append([face(100, vec(1))])
        assert rec.process(None, now=t)[1] == []                      # cooldown holds

    seq.append([face(100, vec(2))])                                   # a stranger: scanning, then unknown
    r, _ = rec.process(None, now=2000.0)
    assert r[0].name is None and r[0].state == "scanning" and not r[0].visible   # single-frame flicker hidden
    seq.append([face(100, vec(2))])
    r, _ = rec.process(None, now=2001.3)
    assert r[0].state == "unknown" and r[0].visible
    assert len(Store().recent_events()) == 1                          # unknown faces are never logged

    Store().add_person("Ben", [vec(2)])                               # added elsewhere -> picked up, no restart
    seq.append([face(100, vec(2))])
    r, _ = rec.process(None, now=2002.0)
    assert rec.reloaded and rec.enrolled == 2 and r[0].name == "Ben"


def test_recognizer_log_off_writes_nothing():
    st = Store()
    st.purge()
    st.add_person("Ava", [vec(1)])
    rec = LiveRecognizer(lambda img: [face(100, vec(1))], 0.363, log=False)
    rec.process(None, now=5000.0)
    _, events = rec.process(None, now=5001.5)
    assert events and Store().recent_events() == []                   # reported to the screen, not saved


def start():
    demo.Handler.app = demo.App(0.363, engine=FakeEngine())
    srv = ThreadingHTTPServer(("127.0.0.1", 0), demo.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_live_endpoints():
    Store().purge()
    srv = start()
    assert call(srv, "POST", "/api/enroll?name=Ava+Chen&consent=1", photo(100), GOOD)[0] == 200
    assert call(srv, "POST", "/api/live/start")[0] == 403                          # needs the header
    st, j = call(srv, "POST", "/api/live/start?log=1", None, GOOD)
    assert st == 200 and j["enrolled"] == 1 and j["present"] == 0
    sid = j["sid"]
    assert call(srv, "POST", "/api/live/frame?sid=nope", photo(100), GOOD)[0] == 404
    assert call(srv, "POST", f"/api/live/frame?sid={sid}", b"junk", GOOD)[0] == 400

    st, j = call(srv, "POST", f"/api/live/frame?sid={sid}", photo(100), GOOD)       # enrolled face
    t = j["tracks"][0]
    assert st == 200 and t["name"] == "Ava Chen" and t["state"] == "ok" and t["score"] > 0.9
    assert len(t["box"]) == 4 and all(0 <= v <= 1 for v in t["box"])
    time.sleep(1.1)                                                                 # ready to log after 1s
    st, j = call(srv, "POST", f"/api/live/frame?sid={sid}", photo(100), GOOD)
    assert [e["name"] for e in j["events"]] == ["Ava Chen"] and j["present"] == 1
    assert call(srv, "POST", f"/api/live/frame?sid={sid}", photo(100), GOOD)[1]["events"] == []
    rows = Store().attendance_rows()
    assert [(r[1], r[4]) for r in rows] == [("Ava Chen", 1)]                        # attendance recorded once
    st, j = call(srv, "POST", f"/api/live/frame?sid={sid}", photo(0), GOOD)         # nobody in view
    assert j["tracks"] == []

    assert call(srv, "POST", f"/api/live/stop?sid={sid}", None, GOOD)[0] == 200
    assert call(srv, "POST", f"/api/live/frame?sid={sid}", photo(100), GOOD)[0] == 404

    # 'Record attendance' off: shown on screen, not saved
    Store().purge()
    assert call(srv, "POST", "/api/enroll?name=Ava+Chen&consent=1", photo(100), GOOD)[0] == 200
    sid = call(srv, "POST", "/api/live/start?log=0", None, GOOD)[1]["sid"]
    call(srv, "POST", f"/api/live/frame?sid={sid}", photo(100), GOOD)
    time.sleep(1.1)
    assert call(srv, "POST", f"/api/live/frame?sid={sid}", photo(100), GOOD)[1]["events"]
    assert Store().attendance_rows() == []

    # another camera feature takes over the webcam -> the page's live session ends cleanly
    assert call(srv, "POST", "/api/signup/start?name=Zed&consent=1", None, GOOD)[0] == 200
    assert call(srv, "POST", f"/api/live/frame?sid={sid}", photo(100), GOOD)[0] == 404
    srv.shutdown()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
    print("live tests passed")


def test_two_tracks_same_candidate_assigns_identity_to_strongest_face_only():
    st = Store()
    st.purge()
    known = vec(901)
    weaker = known + 0.30 * vec(902)  # still similar enough to pass, but not as strong
    st.add_person("Ava", [known])
    seq = [[face(100, known), face(500, weaker)]]
    rec = LiveRecognizer(lambda img: seq.pop(0) if seq else [face(100, known), face(500, weaker)],
                         0.363, log=False)
    results, events = rec.process(None, now=9000.0)
    assert len(results) == 2
    assert results[0].name == "Ava"
    assert results[1].name is None and results[1].state == "scanning"
    assert events == []
    st.purge()


def test_3d_pose_assistance_uses_transform_yaw_and_keeps_locked_identity_on_turn():
    from localface.face_mesh import associate_pose_yaws

    st = Store()
    st.purge()
    known = vec(1701)
    st.add_person("Ava", [known])
    rec = LiveRecognizer(lambda img: [face(100, known)], 0.363, log=False)
    frame = np.zeros((300, 600, 3), dtype=np.uint8)

    # A face mesh whose bounds overlap the YuNet detection.
    mesh = [[0.25, 0.35, 0.0] for _ in range(478)]
    for idx in (10, 152, 234, 454):
        mesh[idx] = [0.18 if idx == 234 else 0.32 if idx == 454 else 0.25,
                     0.18 if idx == 10 else 0.55 if idx == 152 else 0.35, 0.0]

    def transform_yaw(degrees):
        angle = np.radians(degrees)
        c, s = np.cos(angle), np.sin(angle)
        return [[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]]

    yaws = associate_pose_yaws([face(100, known)], [mesh], [transform_yaw(25)], frame.shape)
    assert yaws[0] is not None and abs(yaws[0] - 25) < 0.1

    # Frontal and moderate-turn frames should both recognize the enrolled person.
    results, _ = rec.process(frame, now=7000.0, face_meshes=[mesh], face_matrices=[transform_yaw(0)])
    assert results[0].name == "Ava"
    results, _ = rec.process(frame, now=7001.3, face_meshes=[mesh], face_matrices=[transform_yaw(40)])
    assert results[0].name == "Ava"

    # An extreme-pose frame may suppress a fresh match, but it must not erase the
    # identity already locked to this same continuous track.
    results, _ = rec.process(frame, now=7002.6, face_meshes=[mesh], face_matrices=[transform_yaw(75)])
    assert results[0].name == "Ava"
    assert results[0].state == "ok"
    st.purge()
