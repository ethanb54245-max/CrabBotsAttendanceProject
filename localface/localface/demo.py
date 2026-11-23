"""Local web demo: drop a photo to see who it is, add people from photos, open the live camera.

Runs only on this computer (127.0.0.1). Nothing is uploaded anywhere; uploaded photos are
processed in memory and never written to disk.
"""
import base64
import csv
import io
import json
import os
import secrets
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from . import admin, ui
from .cli import POSES, csv_safe, face_feedback
from .recognizer import LiveRecognizer
from .engine import DEFAULT_THRESHOLD, DETECTOR_PATH, RECOGNIZER_PATH, FaceEngine, resize_max
from .store import DATA_DIR, Store

PAGE = Path(__file__).with_name("demo_page.html")
MAX_BODY = 20 * 1024 * 1024
SIGNUP_GAP = 0.8          # seconds between camera sign-up captures


class ApiError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status, self.message = status, message


def jpeg_data_url(img, quality=88):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def face_thumb(img, f, size=112):
    H, W = img.shape[:2]
    side = max(f.w, f.h) * 1.5
    x1, y1 = int(max(0, f.cx - side / 2)), int(max(0, f.cy - side / 2))
    x2, y2 = int(min(W, f.cx + side / 2)), int(min(H, f.cy + side / 2))
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return jpeg_data_url(cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA), 85)


def clean_name(raw):
    name = " ".join((raw or "").split())
    if not name or len(name) > 40 or not name.isprintable() or any(c in name for c in '<>"\\/'):
        raise ApiError(400, "Please enter a name (up to 40 characters).")
    return name


class Signup:
    def __init__(self, name):
        self.name, self.vectors, self.last, self.born = name, [], 0.0, time.time()


class App:
    def __init__(self, threshold=DEFAULT_THRESHOLD, engine=None):
        self.engine = engine or FaceEngine()
        self.threshold = threshold
        self.lock = threading.Lock()      # one inference at a time
        self.live = None
        self.sessions, self.slock = {}, threading.Lock()
        self.admin = admin.AdminSessions()
        self.live_sessions = {}           # at most one in-page live camera session at a time

    # -- helpers
    def decode(self, data):
        if not data:
            raise ApiError(400, "No photo received.")
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ApiError(400, "That file isn't a readable image. Use JPG, PNG, BMP or WEBP "
                                "(iPhone HEIC photos need converting to JPG first).")
        return resize_max(img, 1600)

    def analyze(self, img):
        with self.lock:
            return self.engine.analyze(img, max_side=1280)

    # -- API
    def identify(self, data):
        img = self.decode(data)
        faces = self.analyze(img)
        store = Store()
        gallery = store.gallery()
        out, results = img.copy(), []
        for i, f in enumerate(faces):
            name, score = gallery.match(f.embedding, self.threshold)
            ui.draw_face(out, (f.x, f.y, f.w, f.h), "ok" if name else "unknown",
                         name or "Unknown", score if name else None)
            results.append({"index": i, "name": name, "score": round(score, 3),
                            "thumb": face_thumb(img, f)})
        return {"image": jpeg_data_url(out), "faces": results, "enrolled": len(store.people())}

    def enroll(self, data, name, consent, face_index=None):
        if not consent:
            raise ApiError(400, "Please confirm this person has agreed to be enrolled.")
        name = clean_name(name)
        img = self.decode(data)
        with self.lock:
            if face_index is None:
                vecs, n = self.engine.photo_embeddings(img)
                faces = None
                if vecs is None:
                    raise ApiError(422, "No face found in that photo. Try a clearer, front-facing one."
                                   if n == 0 else
                                   f"{n} faces found. Use the photo on the left and pick the face to add.")
                faces = self.engine.analyze(img, max_side=1280)
            else:
                faces = self.engine.analyze(img, max_side=1280)
                if not 0 <= face_index < len(faces):
                    raise ApiError(422, "That face is no longer in the photo. Please try again.")
                vecs = [faces[face_index].embedding]
        face = faces[0 if face_index is None else face_index]
        store = Store()
        who, _ = store.gallery().match(vecs[0], self.threshold)
        if who and who.lower() != name.lower():
            raise ApiError(409, f"This face is already enrolled as {who}.")
        store.add_person(name, vecs)
        return {"ok": True, "name": name, "samples": len(vecs), "thumb": face_thumb(img, face)}

    def people(self):
        return {"people": [{"name": n, "samples": c} for n, c, _ in Store().people()]}

    # -- admin PIN
    def require_admin(self, token):
        if admin.pin_is_set(Store()) and not self.admin.valid(token):
            raise ApiError(401, "Admin PIN required.")

    def admin_status(self, token):
        pin_set = admin.pin_is_set(Store())
        unlocked = pin_set and self.admin.valid(token)
        return {"pin_set": pin_set, "unlocked": unlocked,
                "minutes_left": (self.admin.seconds_left(token) + 59) // 60 if unlocked else 0}

    def _attempt(self, store, pin):
        try:
            return self.admin.attempt(store, pin)
        except admin.PinError as e:
            raise ApiError(e.status, e.message)

    def admin_unlock(self, pin):
        store = Store()
        if not admin.pin_is_set(store):
            raise ApiError(400, "No PIN is set.")
        return {"token": self._attempt(store, pin or ""), "minutes": self.admin.ttl // 60}

    def admin_lock(self, token):
        self.admin.revoke(token)
        return {"ok": True}

    def admin_set_pin(self, current, new):
        """Set, change or (with new == '') remove the PIN. Changing or removing needs the current PIN."""
        store = Store()
        was_set = admin.pin_is_set(store)
        if was_set:
            self._attempt(store, current or "")
        if not new:
            if not was_set:
                raise ApiError(400, "No PIN is set.")
            admin.clear_pin(store)
            self.admin.revoke_all()
            return {"pin_set": False, "token": ""}
        try:
            admin.set_pin(store, new)
        except ValueError as e:
            raise ApiError(400, str(e))
        self.admin.revoke_all()
        return {"pin_set": True, "token": self._attempt(store, new)}

    def remove(self, name, token=None):
        self.require_admin(token)
        if not Store().remove_person(name):
            raise ApiError(404, "No such person.")
        return {"ok": True}

    def attendance(self):
        store = Store()
        today = time.strftime("%Y-%m-%d")
        rows = [r for r in store.attendance_rows() if r[0] == today]
        return {"date": today, "enrolled": len(store.people()),
                "present": [{"name": r[1], "first": time.strftime("%H:%M", time.localtime(r[2]))}
                            for r in rows]}

    def attendance_summary(self):
        store = Store()
        rows = store.attendance_rows()
        dates = sorted({r[0] for r in rows})
        attended = {(r[0], r[1]) for r in rows}
        people = [{"name": n, "days": [(d, n) in attended for d in dates],
                   "total": sum((d, n) in attended for d in dates)} for n, _, _ in store.people()]
        return {"dates": dates, "people": people}

    def attendance_csv(self):
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\r\n")
        w.writerow(["date", "name", "first_seen", "last_seen", "sightings"])
        for d, n, t0, t1, c in Store().attendance_rows():
            w.writerow([d, csv_safe(n), time.strftime("%H:%M:%S", time.localtime(t0)),
                        time.strftime("%H:%M:%S", time.localtime(t1)), c])
        return buf.getvalue().encode()

    def purge(self, token=None):
        self.require_admin(token)
        Store().purge()
        with self.slock:
            self.sessions.clear()
        return {"ok": True}

    # -- in-browser camera sign-up (same guided poses as `localface signup`)
    def signup_start(self, name, consent):
        if not consent:
            raise ApiError(400, "Please confirm you agree to be enrolled.")
        name = clean_name(name)
        if Store().has_person(name):
            raise ApiError(409, f"'{name}' is already enrolled. Remove them from the People list to start over.")
        self.stop_live()                  # the camera can only be used by one program at a time
        sid = secrets.token_hex(8)
        with self.slock:
            self.live_sessions.clear()
            now = time.time()
            for k in [k for k, v in self.sessions.items() if now - v.born > 900]:
                del self.sessions[k]
            self.sessions[sid] = Signup(name)
        return {"sid": sid, "name": name, "total": len(POSES)}

    def signup_frame(self, sid, data):
        with self.slock:
            s = self.sessions.get(sid)
        if not s:
            raise ApiError(404, "This sign-up expired. Please start again.")
        img = self.decode(data)
        H, W = img.shape[:2]
        faces = self.analyze(img)
        box, msg = face_feedback(faces, W)
        captured_now = False
        if msg is None and time.time() - s.last >= SIGNUP_GAP:
            emb = faces[0].embedding
            if not s.vectors:
                who, _ = Store().gallery().match(emb, self.threshold)
                if who:
                    with self.slock:
                        self.sessions.pop(sid, None)
                    raise ApiError(409, f"You're already enrolled as {who}.")
            s.vectors.append(emb)
            s.last, captured_now = time.time(), True
        done = len(s.vectors) >= len(POSES)
        if done:
            Store().add_person(s.name, s.vectors)
            with self.slock:
                self.sessions.pop(sid, None)
        return {"box": [box[0] / W, box[1] / H, box[2] / W, box[3] / H] if box else None,
                "message": msg, "captured": len(s.vectors), "total": len(POSES),
                "pose": POSES[min(len(s.vectors), len(POSES) - 1)],
                "flash": captured_now, "done": done, "name": s.name}

    # -- live camera inside the page
    def live_start(self, log):
        self.stop_live()                  # the separate camera window would hold the webcam
        rec = LiveRecognizer(self.analyze, self.threshold, log=log)
        sid = secrets.token_hex(8)
        with self.slock:
            self.live_sessions = {sid: rec}
        return {"sid": sid, "enrolled": rec.enrolled, "present": len(rec.present)}

    def live_frame(self, sid, data):
        with self.slock:
            rec = self.live_sessions.get(sid)
        if not rec:
            raise ApiError(404, "The live camera was stopped.")
        img = self.decode(data)
        H, W = img.shape[:2]
        results, events = rec.process(img)
        tracks = []
        for r in results:
            if not r.visible:
                continue
            x, y, w, h = r.track.view
            tracks.append({"id": r.track.id, "box": [x / W, y / H, w / W, h / H], "name": r.name,
                           "score": round(r.score, 3) if r.name else 0, "state": r.state,
                           "motion": r.motion if r.motion != "detecting" else None})
        return {"tracks": tracks, "enrolled": rec.enrolled, "present": len(rec.present),
                "events": [{"name": e["name"], "motion": e["motion"],
                            "time": time.strftime("%H:%M:%S", time.localtime(e["time"]))} for e in events]}

    def live_stop(self, sid):
        with self.slock:
            self.live_sessions.pop(sid, None)
        return {"ok": True}

    def stop_live(self):
        if self.live and self.live.poll() is None:
            self.live.terminate()
            try:
                self.live.wait(timeout=3)
            except Exception:  # noqa: BLE001
                self.live.kill()
            return True
        return False

    def open_live(self):
        if self.live and self.live.poll() is None:
            return {"ok": True, "message": "The camera window is already open."}
        with self.slock:
            self.live_sessions.clear()    # the window takes over the webcam
        env = dict(os.environ, LOCALFACE_DATA=str(Path(DATA_DIR).resolve()))
        self.live = subprocess.Popen(
            [sys.executable, "-m", "localface", "watch", "--fullscreen"],
            cwd=str(Path(__file__).resolve().parent.parent), env=env)
        return {"ok": True, "message": "Opening the camera window. Press Q in it to close."}


class Handler(BaseHTTPRequestHandler):
    app = None
    stopping = threading.Event()
    server_version = "LocalFace"

    def log_message(self, *args):
        pass

    def _send(self, status, body, ctype="application/json", extra=None):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if "json" in ctype or "html" in ctype else ""))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; "
                         "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'")
        self.end_headers()
        self.wfile.write(data)

    def _guard(self, mutating):
        host = self.headers.get("Host", "").rsplit(":", 1)[0]
        if host not in ("127.0.0.1", "localhost"):                 # blocks DNS-rebinding
            raise ApiError(403, "Forbidden host.")
        if mutating and self.headers.get("X-LocalFace") != "1":    # blocks other websites
            raise ApiError(403, "Forbidden.")

    def _run(self, mutating, fn):
        try:
            self._guard(mutating)
            self._send(200, fn())
        except ApiError as e:
            self._send(e.status, {"error": e.message})
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            self._send(500, {"error": "Something went wrong. See the console window for details."})

    def _admin_token(self):
        return self.headers.get("X-LocalFace-Admin", "")

    def _json(self):
        data = self._body()
        if len(data) > 4096:
            raise ApiError(413, "Request too large.")
        try:
            obj = json.loads(data or b"{}")
        except ValueError:
            raise ApiError(400, "Bad request.")
        return obj if isinstance(obj, dict) else {}

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise ApiError(413, "That photo is too large (20 MB max).")
        return self.rfile.read(n)

    def _events(self):
        """Open stream the page watches: if it breaks, the app has stopped and the page says so."""
        try:
            self._guard(False)
        except ApiError as e:
            return self._send(e.status, {"error": e.message})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(b"retry: 2000\n\n")
            self.wfile.flush()
            while not self.stopping.wait(10):
                self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
        except OSError:
            pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            try:
                self._guard(False)
                self._send(200, PAGE.read_bytes(), "text/html")
            except ApiError as e:
                self._send(e.status, {"error": e.message})
        elif path == "/api/people":
            self._run(False, self.app.people)
        elif path == "/api/attendance":
            self._run(False, self.app.attendance)
        elif path == "/api/events":
            self._events()
        elif path == "/api/ping":
            self._run(False, lambda: {"ok": True})
        elif path == "/api/admin/status":
            self._run(False, lambda: self.app.admin_status(self._admin_token()))
        elif path == "/api/attendance/summary":
            self._run(False, self.app.attendance_summary)
        elif path == "/api/attendance.csv":
            try:
                self._guard(False)
                self._send(200, self.app.attendance_csv(), "text/csv",
                           {"Content-Disposition": 'attachment; filename="attendance.csv"'})
            except ApiError as e:
                self._send(e.status, {"error": e.message})
        else:
            self._send(404, {"error": "Not found."})

    def do_POST(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/api/identify":
            self._run(True, lambda: self.app.identify(self._body()))
        elif u.path == "/api/enroll":
            fi = q.get("face")
            self._run(True, lambda: self.app.enroll(
                self._body(), q.get("name"), q.get("consent") == "1",
                int(fi) if fi not in (None, "") and fi.isdigit() else None))
        elif u.path == "/api/signup/start":
            self._run(True, lambda: self.app.signup_start(q.get("name"), q.get("consent") == "1"))
        elif u.path == "/api/signup/frame":
            self._run(True, lambda: self.app.signup_frame(q.get("sid", ""), self._body()))
        elif u.path == "/api/purge":
            self._run(True, lambda: self.app.purge(self._admin_token()))
        elif u.path == "/api/admin/unlock":
            self._run(True, lambda: self.app.admin_unlock(str(self._json().get("pin", ""))))
        elif u.path == "/api/admin/lock":
            self._run(True, lambda: self.app.admin_lock(self._admin_token()))
        elif u.path == "/api/admin/pin":
            def _set():
                j = self._json()
                return self.app.admin_set_pin(str(j.get("current", "")), str(j.get("new", "")))
            self._run(True, _set)
        elif u.path == "/api/live/start":
            self._run(True, lambda: self.app.live_start(q.get("log", "1") != "0"))
        elif u.path == "/api/live/frame":
            self._run(True, lambda: self.app.live_frame(q.get("sid", ""), self._body()))
        elif u.path == "/api/live/stop":
            self._run(True, lambda: self.app.live_stop(q.get("sid", "")))
        elif u.path == "/api/live":
            self._run(True, self.app.open_live)
        else:
            self._send(404, {"error": "Not found."})

    def do_DELETE(self):
        u = urlparse(self.path)
        if u.path == "/api/people":
            name = parse_qs(u.query).get("name", [""])[0]
            self._run(True, lambda: self.app.remove(name, self._admin_token()))
        else:
            self._send(404, {"error": "Not found."})


def ensure_models():
    if DETECTOR_PATH.exists() and RECOGNIZER_PATH.exists():
        return
    print("Face models not found. Downloading them now (one time, ~40 MB)...")
    script = Path(__file__).resolve().parent.parent / "scripts" / "download_models.py"
    subprocess.call([sys.executable, str(script)])
    if not (DETECTOR_PATH.exists() and RECOGNIZER_PATH.exists()):
        raise SystemExit("Could not download the models. Check your internet connection, "
                         "then run: python scripts/download_models.py")


def make_server(port, app):
    Handler.app = app
    Handler.stopping = threading.Event()
    server = None
    for p in range(port, port + 10):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", p), Handler)
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit(f"Ports {port}-{port + 9} are all busy. Close other programs or pass --port.")

    return server


def run(port=8765, threshold=DEFAULT_THRESHOLD, open_browser=True):
    ensure_models()
    app = App(threshold)
    server = make_server(port, app)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"\nLocalFace is running at {url}\n(only this computer can open it; press Ctrl+C or close this window to stop)\n")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        Handler.stopping.set()
        app.stop_live()                   # don't leave a camera window behind
        server.server_close()
    return 0
