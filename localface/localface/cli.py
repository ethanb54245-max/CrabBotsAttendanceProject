"""LocalFace command line interface."""
import argparse
import csv
import getpass
import os
import re
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from . import admin, ui
from .camera import Camera
from .engine import DEFAULT_THRESHOLD, FaceEngine, resize_max
from .store import Store
from .recognizer import LiveRecognizer

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CAPTURE_GAP = 0.9      # seconds between enrollment samples
WELCOME_SECS = 2.2     # how long the sign-up kiosk shows "Welcome" before the next person

POSES = [
    "Look straight at the camera",
    "Turn your head slightly left",
    "Turn your head slightly right",
    "Tilt your chin up a little",
    "Tilt your chin down a little",
]


# ---------------------------------------------------------------- helpers
class Window:
    """Resizable window with fullscreen toggle (F key, double-click, or --fullscreen)."""

    def __init__(self, title, fullscreen=False):
        self.title, self.full, self._shown = title, False, 0
        cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(title, 1280, 720)
        cv2.setMouseCallback(title, self._mouse)
        if fullscreen:
            self.set_full(True)

    def _mouse(self, event, *_):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.toggle()

    def set_full(self, on):
        self.full = on
        cv2.setWindowProperty(
            self.title, cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if on else cv2.WINDOW_NORMAL)

    def toggle(self):
        self.set_full(not self.full)

    def show(self, img):
        cv2.imshow(self.title, img)
        self._shown += 1

    def raw_key(self):
        """Raw key code (255 = no key). Used where letters must be typed, e.g. the sign-up kiosk."""
        return cv2.waitKey(1) & 0xFF

    def closed(self):
        if self._shown > 15:
            try:
                return cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1
            except cv2.error:
                return True
        return False

    def poll(self):
        """Handle keys. Returns the key pressed (lowercase char), or 'quit' if closed/Esc/Q."""
        k = self.raw_key()
        if k in (27, ord("q"), ord("Q")):
            return "quit"
        if k in (ord("f"), ord("F")):
            self.toggle()
        if self.closed():
            return "quit"
        return chr(k).lower() if 32 <= k < 127 else ""

    def close(self):
        cv2.destroyAllWindows()


def confirm_consent(name, flag):
    if flag:
        return True
    print("\nLocalFace only enrolls people who have agreed to it.")
    ans = input(f"Has {name} consented to having their face enrolled? [y/N] ").strip().lower()
    return ans == "y"


def load_image(path):
    p = Path(path)
    if not p.exists():
        nearby = sorted(q.name for q in Path.cwd().iterdir() if q.suffix.lower() in IMG_EXT)[:12]
        raise SystemExit(
            f"File not found: {p}\nCurrent folder: {Path.cwd()}\n"
            + (f"Image files in this folder: {', '.join(nearby)}\n" if nearby
               else "There are no image files in this folder.\n")
            + "Tip: Windows can hide extensions, so 'me.jpg' may really be 'me.jpg.jpg'. "
              "Run `dir` to see the exact name, and put names with spaces in quotes.")
    try:  # np.fromfile + imdecode also works for paths with non-English characters on Windows
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001
        img = None
    if img is None:
        raise SystemExit(
            f"Found {p.name}, but it could not be opened as an image ({p.suffix or 'no extension'}).\n"
            "Supported: JPG, PNG, BMP, WEBP. iPhone HEIC photos must be converted to JPG first "
            "(e.g. open in Photos/Paint and Save As JPG), and cloud-only OneDrive files must be downloaded.")
    return resize_max(img, 1600)


def csv_safe(value):
    """Stop spreadsheet apps from treating a name like '=1+1' as a formula."""
    s = str(value)
    return "'" + s if s[:1] in ("=", "+", "-", "@") else s


def face_feedback(faces, frame_w):
    """(box, message) for enrollment capture; message is None when the face is good to capture."""
    if not faces:
        return None, "No face found"
    if len(faces) > 1:
        return None, "Only one person in view, please"
    f = faces[0]
    box = (f.x, f.y, f.w, f.h)
    if f.w < 0.14 * frame_w:
        return box, "Move a little closer"
    if f.score < 0.85:
        return box, "Hold still, face the light"
    return box, None


def apply_key(buf, k, max_len=32):
    """Name-typing logic for the kiosk. Returns (new_buffer, action) with action in {None,'enter','esc'}."""
    if k == 27:
        return buf, "esc"
    if k in (13, 10):
        return buf, "enter"
    if k in (8, 127):
        return buf[:-1], None
    if 32 <= k < 127 and len(buf) < max_len:
        ch = chr(k)
        if ch.isalnum() or ch in " -'._":
            return buf + ch, None
    return buf, None


_TRAILING_NUM = re.compile(r"[\s_\-]*\(?\d+\)?$")


def person_from_filename(stem):
    """'Alice Smith 2' / 'Alice_Smith-2' / 'Alice Smith (2)' -> 'Alice Smith'."""
    name = _TRAILING_NUM.sub("", stem).strip()
    return name or stem


def collect_folder(root):
    """{person name: [photo paths]} from a folder of 'Name.jpg' files and/or 'Name/' sub-folders."""
    groups = defaultdict(list)
    for entry in sorted(Path(root).iterdir()):
        if entry.is_dir():
            groups[entry.name] += sorted(q for q in entry.iterdir() if q.suffix.lower() in IMG_EXT)
        elif entry.suffix.lower() in IMG_EXT:
            groups[person_from_filename(entry.stem)].append(entry)
    return dict(groups)


# ---------------------------------------------------------------- enroll
def cmd_enroll(args):
    if not confirm_consent(args.name, args.i_have_consent):
        print("Not enrolled.")
        return 1
    engine = FaceEngine()
    vectors = []

    if args.photo:
        for p in args.photo:
            vecs, n = engine.photo_embeddings(load_image(p))
            if vecs is None:
                print(f"skip  {p}: need exactly 1 face, found {n}")
                continue
            vectors += vecs
            print(f"ok    {p}")
    else:
        per_pose = max(1, args.samples // len(POSES))
        total = per_pose * len(POSES)
        cam = Camera(args.camera, args.width, args.height)
        win = Window("LocalFace - enroll", args.fullscreen)
        last = flash = 0.0
        try:
            while len(vectors) < total:
                frame = cam.read()
                if frame is None:
                    print("Camera stopped delivering frames.")
                    break
                faces = engine.analyze(frame, max_side=args.det_size)
                now = time.time()
                step = len(vectors) // per_pose
                box, msg = face_feedback(faces, frame.shape[1])
                if msg is None and now - last > CAPTURE_GAP:
                    vectors.append(faces[0].embedding)
                    last = flash = now
                ui.draw_enroll(frame, POSES[min(step, len(POSES) - 1)], min(step + 1, len(POSES)),
                               len(POSES), len(vectors), total, box, msg, now - flash)
                win.show(frame)
                if win.poll() == "quit":
                    print("Cancelled.")
                    vectors = []
                    break
        finally:
            cam.release()
            win.close()

    if not vectors:
        print("No usable face samples captured; nothing saved.")
        return 1
    Store().add_person(args.name, vectors)
    print(f"Enrolled {args.name} with {len(vectors)} sample(s).")
    return 0


# ---------------------------------------------------------------- identify a photo
def cmd_identify(args):
    store = Store()
    gallery = store.gallery()
    if not len(gallery):
        raise SystemExit("Nobody is enrolled yet. Use `enroll` first.")
    engine = FaceEngine()
    paths = []
    for p in args.photo:
        p = Path(p)
        paths += sorted(q for q in p.iterdir() if q.suffix.lower() in IMG_EXT) if p.is_dir() else [p]

    for p in paths:
        img = load_image(p)
        faces = engine.analyze(img, max_side=1280)
        print(f"\n{p}: {len(faces)} face(s)")
        for i, f in enumerate(faces, 1):
            name, score = gallery.match(f.embedding, args.threshold)
            print(f"  face {i}: {name or 'unknown':<20} similarity={score:.3f}")
            ui.draw_face(img, (f.x, f.y, f.w, f.h), "ok" if name else "unknown",
                         name or "Unknown", score if name else None)
        if args.annotate:
            out = Path(args.annotate)
            if len(paths) > 1:
                out.mkdir(parents=True, exist_ok=True)
                out = out / f"{Path(p).stem}_annotated.jpg"
            cv2.imwrite(str(out), img)
            print(f"  wrote {out}")
    return 0


# ---------------------------------------------------------------- live watch
def cmd_watch(args):
    store = Store()
    engine = FaceEngine(score_threshold=args.det_score)
    rec = LiveRecognizer(lambda img: engine.analyze(img, max_side=args.det_size), args.threshold,
                         args.cooldown, log=not args.no_log, store_factory=lambda: store)
    if not len(rec.gallery):
        print("Warning: nobody is enrolled, so every face will show as 'unknown'.")
    cam = Camera(args.camera, args.width, args.height)
    win = None if args.headless else Window("LocalFace", args.fullscreen)
    feed = deque(maxlen=8)
    fps, t_prev, show_hud = 0.0, time.time(), True
    print("Watching.  Keys: F = fullscreen (or double-click), H = hide HUD, Q/Esc = quit."
          "  Unknown faces are never stored.")
    try:
        while True:
            frame = cam.read()
            if frame is None:
                print("Camera stopped delivering frames.")
                break
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3) if fps else 1 / max(now - t_prev, 1e-3)
            t_prev = now

            results, events = rec.process(frame, now)
            if rec.reloaded:                          # picked up people added elsewhere (e.g. in the app)
                rec.reloaded = False
                print("Roster updated.")
            for ev in events:
                feed.append((ev["time"], ev["name"], ev["motion"]))
                print(time.strftime("%H:%M:%S"), f"{ev['name']}  ({ev['motion']}, similarity {ev['score']:.2f})")

            if win:
                top_safe = ui.topbar_height(frame) if show_hud else 0
                shown = 0
                for r in results:
                    if not r.visible:
                        continue                      # ignore one-frame flickers
                    shown += 1
                    label = r.name or ("Identifying..." if r.state == "scanning" else "Unknown")
                    ui.draw_face(frame, r.track.view, r.state, label, r.score if r.name else None,
                                 r.motion, top_safe)
                if show_hud:
                    ui.draw_topbar(frame, fps, rec.enrolled, shown, (len(rec.present), rec.enrolled))
                    ui.draw_feed(frame, feed, now)
                    ui.draw_hint(frame, "F fullscreen   H hide HUD   Q quit")
                    if not len(rec.gallery):
                        ui.draw_empty_notice(frame, "Nobody enrolled yet - run: python -m localface enroll NAME")
                win.show(frame)
                key = win.poll()
                if key == "quit":
                    break
                if key == "h":
                    show_hud = not show_hud
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        if win:
            win.close()
    return 0


# ---------------------------------------------------------------- bulk import
def cmd_enroll_folder(args):
    groups = collect_folder(args.folder)
    if not groups:
        raise SystemExit(f"No photos found in {args.folder}")
    print(f"Found {len(groups)} people:")
    for name, paths in groups.items():
        print(f"  {name:<28} {len(paths)} photo(s)")
    if not args.i_have_consent:
        print("\nOnly enroll people who have agreed (for under-18 members, check with your advisor/guardians).")
        if input("Has everyone listed above consented to being enrolled? [y/N] ").strip().lower() != "y":
            print("Not enrolled.")
            return 1
    engine, store = FaceEngine(), Store()
    for name, paths in groups.items():
        vecs = []
        for p in paths:
            got, n = engine.photo_embeddings(load_image(p))
            if got is not None:
                vecs += got
            else:
                print(f"  skip  {p.name}: need exactly 1 face, found {n}")
        if vecs:
            store.add_person(name, vecs)
            print(f"  ok    {name}: {len(vecs)} sample(s)")
        else:
            print(f"  FAIL  {name}: no usable photo")
    return 0


# ---------------------------------------------------------------- sign-up kiosk
def cmd_signup(args):
    """Self-service kiosk: each person types their name and does a ~6 second guided capture."""
    store, engine = Store(), FaceEngine()
    gallery = store.gallery()
    cam = Camera(args.camera, args.width, args.height)
    win = Window("LocalFace - sign up", args.fullscreen)
    state, buf, message, name = "name", "", None, ""
    vectors, last, flash, done_at = [], 0.0, 0.0, 0.0
    added = []
    print("Sign-up kiosk running. Esc on the name screen to finish.")
    try:
        while not win.closed():
            frame = cam.read()
            if frame is None:
                print("Camera stopped delivering frames.")
                break
            now = time.time()
            if state == "name":
                ui.draw_signup_name(frame, buf, message, int(now * 2) % 2 == 0)
            elif state == "capture":
                faces = engine.analyze(frame, max_side=args.det_size)
                box, msg = face_feedback(faces, frame.shape[1])
                if msg is None and now - last > CAPTURE_GAP:
                    emb = faces[0].embedding
                    if not vectors:
                        who, _ = gallery.match(emb, args.threshold)
                        if who:
                            state, buf = "name", ""
                            message = f"You're already enrolled as {who}"
                            continue
                    vectors.append(emb)
                    last = flash = now
                ui.draw_enroll(frame, POSES[min(len(vectors), len(POSES) - 1)],
                               min(len(vectors) + 1, len(POSES)), len(POSES),
                               len(vectors), len(POSES), box, msg, now - flash)
                if len(vectors) >= len(POSES):
                    store.add_person(name, vectors)
                    gallery = store.gallery()
                    added.append(name)
                    print(f"Enrolled {name}")
                    state, done_at = "done", now
            else:  # done
                ui.draw_welcome(frame, name)
                if now - done_at >= WELCOME_SECS:
                    state, buf, message = "name", "", None
            win.show(frame)
            k = win.raw_key()
            if state == "name":
                buf, action = apply_key(buf, k)
                if action == "esc":
                    break
                if k != 255:
                    message = None if action is None else message
                if action == "enter":
                    cand = buf.strip()
                    if not cand:
                        message = "Please type your name"
                    elif store.has_person(cand):
                        message = f"'{cand}' is already enrolled - see an officer"
                    else:
                        name, state, vectors, last, message = cand, "capture", [], 0.0, None
            elif state == "capture" and k == 27:
                state, buf, message = "name", "", None
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        win.close()
    print(f"Done. Enrolled this session: {', '.join(added) if added else 'nobody'}")
    return 0


# ---------------------------------------------------------------- attendance
def cmd_attendance(args):
    store = Store()
    rows = store.attendance_rows()
    roster = [p[0] for p in store.people()]
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["date", "name", "first_seen", "last_seen", "sightings"])
            for d, n, t0, t1, c in rows:
                w.writerow([d, csv_safe(n), time.strftime("%H:%M:%S", time.localtime(t0)),
                            time.strftime("%H:%M:%S", time.localtime(t1)), c])
        print(f"Wrote {len(rows)} rows to {args.csv}")
    if args.summary:
        dates = sorted({r[0] for r in rows})
        if not dates:
            print("No attendance recorded yet.")
            return 0
        attended = {(r[0], r[1]) for r in rows}
        width = max([len(n) for n in roster] + [4])
        print(f"{'':<{width}}  " + " ".join(d[5:] for d in dates) + "   total")
        for n in roster:
            marks = ["  X  " if (d, n) in attended else "  .  " for d in dates]
            print(f"{n:<{width}}  " + " ".join(marks) + f"   {sum((d, n) in attended for d in dates)}/{len(dates)}")
        return 0
    if args.csv:
        return 0
    day = args.date or time.strftime("%Y-%m-%d")
    present = [r for r in rows if r[0] == day]
    print(f"Attendance for {day}: {len(present)}/{len(roster)} present\n")
    for _, n, t0, t1, c in present:
        print(f"  PRESENT  {n:<24} first {time.strftime('%H:%M', time.localtime(t0))}"
              f"  last {time.strftime('%H:%M', time.localtime(t1))}")
    seen = {r[1] for r in present}
    for n in roster:
        if n not in seen:
            print(f"  absent   {n}")
    return 0


# ---------------------------------------------------------------- demo
def cmd_app(args):
    from .demo import run
    return run(port=args.port, threshold=args.threshold, open_browser=not args.no_browser)


# ---------------------------------------------------------------- admin
def cmd_list(_args):
    rows = Store().people()
    if not rows:
        print("Nobody enrolled.")
    for name, n, ts in rows:
        print(f"{name:<24} {n} sample(s)  enrolled {time.strftime('%Y-%m-%d', time.localtime(ts))}")
    return 0


def require_cli_pin(store):
    """Ask for the admin PIN (if one is set) before a destructive command."""
    if not admin.pin_is_set(store):
        return True
    pin = os.environ.get("LOCALFACE_PIN") or getpass.getpass("Admin PIN: ")
    if admin.verify_pin(store, pin):
        return True
    print("Wrong PIN.")
    return False


def cmd_reset_pin(args):
    store = Store()
    if not admin.pin_is_set(store):
        print("No admin PIN is set.")
        return 0
    if not args.yes and input("Remove the admin PIN? Anyone using the app will be able to delete data. [y/N] ").strip().lower() != "y":
        print("Cancelled.")
        return 1
    admin.clear_pin(store)
    print("Admin PIN removed. Set a new one in the app (People tab).")
    return 0


def cmd_remove(args):
    store = Store()
    if not require_cli_pin(store):
        return 1
    ok = store.remove_person(args.name)
    print("Removed." if ok else "No such person.")
    return 0 if ok else 1


def cmd_events(args):
    rows = Store().recent_events(args.limit)
    if not rows:
        print("No events logged.")
    for ts, name, score, motion in rows:
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)), f"{name:<20} {motion:<10} {score:.2f}")
    return 0


def cmd_purge(args):
    if not require_cli_pin(Store()):
        return 1
    if not args.yes and input("Delete ALL enrolled faces and events? Type 'delete': ") != "delete":
        print("Cancelled.")
        return 1
    Store().purge()
    print("All data deleted.")
    return 0


# ---------------------------------------------------------------- entry
def add_camera_args(p):
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280, help="requested camera width (default 1280)")
    p.add_argument("--height", type=int, default=720, help="requested camera height (default 720)")
    p.add_argument("--det-size", type=int, default=960,
                   help="longest side used for detection; higher = finds smaller/farther faces, slower")
    p.add_argument("--fullscreen", action="store_true", help="start in fullscreen")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="localface", description="Local, consent-based face recognition.",
        epilog="Run with no command to open the LocalFace app in your browser.")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("enroll", help="enroll a consenting person (guided webcam capture, or photos)")
    p.add_argument("name")
    p.add_argument("--photo", nargs="+", help="enroll from clear, single-face photo(s) instead of webcam")
    p.add_argument("--samples", type=int, default=10,
                   help="webcam samples across 5 head poses (default 10)")
    p.add_argument("--i-have-consent", action="store_true", help="skip the interactive consent prompt")
    add_camera_args(p)
    p.set_defaults(fn=cmd_enroll)

    p = sub.add_parser("identify", help="match faces in photo(s) against enrolled people")
    p.add_argument("photo", nargs="+", help="image file(s) or folder(s)")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--annotate", help="write annotated image (file, or folder for many photos)")
    p.set_defaults(fn=cmd_identify)

    p = sub.add_parser("watch", help="live webcam recognition of enrolled people")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--det-score", type=float, default=0.6,
                   help="detector confidence cutoff (lower finds partial/blurred faces; default 0.6)")
    p.add_argument("--cooldown", type=float, default=30, help="seconds between repeat log entries per person")
    p.add_argument("--headless", action="store_true", help="no preview window")
    p.add_argument("--no-log", action="store_true", help="print only; don't write sightings to the local DB")
    add_camera_args(p)
    p.set_defaults(fn=cmd_watch)

    p = sub.add_parser("signup", help="self-service sign-up kiosk: people type their name and enroll themselves")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    add_camera_args(p)
    p.set_defaults(fn=cmd_signup)

    p = sub.add_parser("enroll-folder", help="bulk-enroll from a folder of photos (Name.jpg or Name/ subfolders)")
    p.add_argument("folder")
    p.add_argument("--i-have-consent", action="store_true", help="skip the interactive consent prompt")
    p.set_defaults(fn=cmd_enroll_folder)

    p = sub.add_parser("attendance", help="who showed up (today, a date, or a summary across all meetings)")
    p.add_argument("--date", help="YYYY-MM-DD (default: today)")
    p.add_argument("--summary", action="store_true", help="people x dates grid across all recorded meetings")
    p.add_argument("--csv", help="export all attendance to a CSV file")
    p.set_defaults(fn=cmd_attendance)

    p = sub.add_parser("app", aliases=["demo"],
                       help="open the LocalFace app in your browser (this is what runs when you give no command)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--no-browser", action="store_true", help="don't open the browser automatically")
    p.set_defaults(fn=cmd_app)

    sub.add_parser("list", help="list enrolled people").set_defaults(fn=cmd_list)
    p = sub.add_parser("remove", help="delete one person's face data")
    p.add_argument("name")
    p.set_defaults(fn=cmd_remove)
    p = sub.add_parser("events", help="show recent logged sightings")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(fn=cmd_events)
    p = sub.add_parser("reset-pin", help="remove a forgotten admin PIN (needs access to this computer)")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(fn=cmd_reset_pin)
    p = sub.add_parser("purge", help="delete ALL stored faces and events")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(fn=cmd_purge)

    args = ap.parse_args(argv)
    if args.cmd is None:                         # `python -m localface` -> the app
        return cmd_app(argparse.Namespace(port=8765, threshold=DEFAULT_THRESHOLD, no_browser=False))
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
