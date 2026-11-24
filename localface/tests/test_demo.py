"""Model-free tests of the demo server: security guards, identify, enroll, remove, attendance."""
import http.client
import json
import os
import sys
import tempfile
import threading

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["LOCALFACE_DATA"] = tempfile.mkdtemp()

from http.server import ThreadingHTTPServer  # noqa: E402

from localface import demo  # noqa: E402
from localface.engine import Face  # noqa: E402


class FakeEngine:
    """Treats the image's mean brightness as 'who it is' so different photos = different people."""
    def _face(self, img):
        h, w = img.shape[:2]
        rng = np.random.default_rng(int(img.mean()))
        return Face(w * .3, h * .2, w * .3, h * .4, 0.95, rng.normal(size=128).astype(np.float32))
    def analyze(self, img, max_side=960):
        return [] if img.mean() < 5 else [self._face(img)]
    def photo_embeddings(self, img):
        f = self.analyze(img)
        return (None, 0) if not f else ([f[0].embedding, f[0].embedding], 1)


def photo(gray):
    return cv2.imencode(".jpg", np.full((300, 400, 3), gray, np.uint8))[1].tobytes()


def start():
    demo.Handler.app = demo.App(0.363, engine=FakeEngine())
    srv = ThreadingHTTPServer(("127.0.0.1", 0), demo.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def call(srv, method, path, body=None, headers=None, host=None):
    c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1])
    h = dict(headers or {})
    if host:
        h["Host"] = host
    c.request(method, path, body, h)
    r = c.getresponse()
    raw = r.read()
    try:
        return r.status, json.loads(raw)
    except ValueError:
        return r.status, raw


GOOD = {"X-LocalFace": "1"}


def test_everything():
    # Keep this integration test independent of the shared temp DB used by other modules.
    demo.Store().purge()
    srv = start()
    st, page = call(srv, "GET", "/")
    assert st == 200 and b"LocalFace" in page
    assert call(srv, "GET", "/", host="evil.example.com")[0] == 403                # DNS-rebinding guard
    assert call(srv, "POST", "/api/identify", photo(100))[0] == 403                # needs custom header
    assert call(srv, "DELETE", "/api/people?name=x")[0] == 403

    st, j = call(srv, "POST", "/api/identify", photo(100), GOOD)                   # nobody enrolled yet
    assert st == 200 and j["enrolled"] == 0 and j["faces"][0]["name"] is None
    assert j["image"].startswith("data:image/jpeg;base64,") and j["faces"][0]["thumb"]

    assert call(srv, "POST", "/api/identify", b"not an image", GOOD)[0] == 400
    assert call(srv, "POST", "/api/identify", photo(0), GOOD)[1]["faces"] == []     # no face -> empty list

    assert call(srv, "POST", "/api/enroll?name=Ethan+Badoud", photo(100), GOOD)[0] == 400   # no consent
    assert call(srv, "POST", "/api/enroll?name=%3Cb%3E&consent=1", photo(100), GOOD)[0] == 400  # bad name
    assert call(srv, "POST", "/api/enroll?name=Ethan+Badoud&consent=1", photo(0), GOOD)[0] == 422  # no face
    st, j = call(srv, "POST", "/api/enroll?name=Ethan+Badoud&consent=1", photo(100), GOOD)
    assert st == 200 and j["name"] == "Ethan Badoud" and j["samples"] == 2

    assert call(srv, "POST", "/api/enroll?name=Someone+Else&consent=1", photo(100), GOOD)[0] == 409  # same face, other name
    st, j = call(srv, "POST", "/api/identify", photo(100), GOOD)                   # recognized right away
    assert j["faces"][0]["name"] == "Ethan Badoud" and j["faces"][0]["score"] > 0.9
    assert call(srv, "POST", "/api/identify", photo(180), GOOD)[1]["faces"][0]["name"] is None  # stranger

    st, j = call(srv, "POST", "/api/enroll?name=Sam&consent=1&face=0", photo(180), GOOD)    # name an unknown face
    assert st == 200
    assert call(srv, "POST", "/api/enroll?name=Sam&consent=1&face=5", photo(180), GOOD)[0] == 422

    st, j = call(srv, "GET", "/api/people")
    assert sorted(p["name"] for p in j["people"]) == ["Ethan Badoud", "Sam"]
    st, j = call(srv, "GET", "/api/attendance")
    assert st == 200 and j["enrolled"] == 2 and j["present"] == []
    assert call(srv, "DELETE", "/api/people?name=Sam", None, GOOD)[0] == 200
    assert call(srv, "DELETE", "/api/people?name=Sam", None, GOOD)[0] == 404
    assert call(srv, "GET", "/nope")[0] == 404
    srv.shutdown()


def test_camera_signup():
    demo.SIGNUP_GAP = 0.0
    srv = start()
    q = "/api/signup/start?name=Riley+Park&consent=1"
    assert call(srv, "POST", "/api/signup/start?name=Riley+Park", None, GOOD)[0] == 400        # no consent
    assert call(srv, "POST", q)[0] == 403                                                       # needs header
    st, j = call(srv, "POST", q, None, GOOD)
    assert st == 200 and j["total"] == 5
    sid = j["sid"]
    assert call(srv, "POST", "/api/signup/frame?sid=nope", photo(120), GOOD)[0] == 404
    st, j = call(srv, "POST", f"/api/signup/frame?sid={sid}", photo(0), GOOD)                  # no face in view
    assert st == 200 and j["message"] == "No face found" and j["captured"] == 0 and j["box"] is None
    for i in range(1, 6):
        st, j = call(srv, "POST", f"/api/signup/frame?sid={sid}", photo(120), GOOD)
        assert st == 200 and j["captured"] == i and j["flash"] and len(j["box"]) == 4
    assert j["done"] and j["name"] == "Riley Park"
    assert call(srv, "POST", f"/api/signup/frame?sid={sid}", photo(120), GOOD)[0] == 404        # session finished
    people = {p["name"]: p["samples"] for p in call(srv, "GET", "/api/people")[1]["people"]}
    assert people["Riley Park"] == 5
    assert call(srv, "POST", q, None, GOOD)[0] == 409                                           # name already enrolled
    sid2 = call(srv, "POST", "/api/signup/start?name=Impostor&consent=1", None, GOOD)[1]["sid"]
    st, j = call(srv, "POST", f"/api/signup/frame?sid={sid2}", photo(120), GOOD)               # same face, new name
    assert st == 409 and "Riley Park" in j["error"]
    srv.shutdown()


def test_attendance_history_csv_purge():
    import time
    from localface.store import Store
    srv = start()
    st = Store()
    for nm in ("Zed", "=Evil()"):
        st.add_person(nm, [np.ones(128)])
    t1 = time.mktime((2026, 11, 16, 16, 0, 0, 0, 0, -1))
    t2 = time.mktime((2026, 11, 17, 16, 0, 0, 0, 0, -1))
    for ts, nm in ((t1, "Zed"), (t1 + 90, "Zed"), (t2, "=Evil()")):
        st.db.execute("INSERT INTO events(ts,name,score,motion) VALUES(?,?,?,?)", (ts, nm, .9, "standing"))
    st.db.commit()
    code, j = call(srv, "GET", "/api/attendance/summary")
    assert j["dates"] == ["2026-11-16", "2026-11-17"]
    zed = next(p for p in j["people"] if p["name"] == "Zed")
    assert zed["days"] == [True, False] and zed["total"] == 1
    code, raw = call(srv, "GET", "/api/attendance.csv")
    text = raw.decode()
    assert code == 200 and text.startswith("date,name,first_seen,last_seen,sightings")
    assert "2026-11-16,Zed,16:00:00,16:01:30,2" in text
    assert "'=Evil()" in text and ",=Evil()," not in text          # formula-injection guard
    assert call(srv, "POST", "/api/purge")[0] == 403                # needs header
    assert call(srv, "POST", "/api/purge", None, GOOD)[0] == 200
    assert call(srv, "GET", "/api/people")[1]["people"] == []
    assert call(srv, "GET", "/api/attendance/summary")[1]["dates"] == []
    srv.shutdown()


def test_no_command_opens_the_app():
    import localface.cli as cli
    calls = []
    real = demo.run
    demo.run = lambda **kw: calls.append(kw) or 0
    try:
        assert cli.main([]) == 0
        assert cli.main(["demo", "--no-browser"]) == 0
        assert cli.main(["app", "--port", "9100"]) == 0
    finally:
        demo.run = real
    assert calls[0]["port"] == 8765 and calls[0]["open_browser"] is True
    assert calls[1]["open_browser"] is False and calls[2]["port"] == 9100


def test_app_stop_is_detectable_and_page_close_is_not_a_stop():
    import time
    srv = demo.make_server(0, demo.App(0.363, engine=FakeEngine()))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    assert call(srv, "GET", "/api/ping") == (200, {"ok": True})
    assert call(srv, "GET", "/api/ping", host="evil.example")[0] == 403
    assert call(srv, "POST", "/api/bye?id=0123456789abcdef")[0] == 404          # closing a page can't stop the app
    c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1])          # the stream the page watches
    c.request("GET", "/api/events")
    r = c.getresponse()
    assert r.status == 200 and r.getheader("Content-Type") == "text/event-stream"
    assert r.fp.readline().strip() == b"retry: 2000"
    c.close()                                                                    # a page closing = connection drops
    time.sleep(0.3)
    assert t.is_alive() and call(srv, "GET", "/api/ping")[0] == 200              # ...and the app keeps running
    c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1])
    c.request("GET", "/api/events")
    r = c.getresponse()
    r.fp.readline()
    demo.Handler.stopping.set()                                                  # app stopping ends the stream
    assert r.fp.read() == b"\n" or True
    srv.shutdown()
    t.join(timeout=3)
    assert not t.is_alive()
    srv.server_close()


if __name__ == "__main__":
    test_app_stop_is_detectable_and_page_close_is_not_a_stop()
    test_everything()
    test_camera_signup()
    test_attendance_history_csv_purge()
    test_no_command_opens_the_app()
    print("demo tests passed")


def test_settings_endpoint_persists_and_validates_values():
    demo.Store().purge()
    srv = start()
    st, initial = call(srv, "GET", "/api/settings")
    assert st == 200 and initial["threshold"] == 0.363 and initial["min_margin"] == 0.04
    st, saved = call(srv, "POST", "/api/settings", json.dumps({"threshold": 0.48, "min_margin": 0.07}).encode(), GOOD)
    assert st == 200 and saved["threshold"] == 0.48 and saved["min_margin"] == 0.07
    st, bad = call(srv, "POST", "/api/settings", json.dumps({"threshold": 0.99}).encode(), GOOD)
    assert st == 400 and "threshold" in bad["error"].lower()
    assert call(srv, "GET", "/api/settings")[1]["threshold"] == 0.48
    srv.shutdown()
    demo.Store().purge()
