"""LocalFace command line interface."""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

from .engine import DEFAULT_THRESHOLD, FaceEngine, resize_max
from .store import Store
from .tracking import Detection, Tracker

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------- helpers
def confirm_consent(name, flag):
    if flag:
        return True
    print(f"\nLocalFace only enrolls people who have agreed to it.")
    ans = input(f"Has {name} consented to having their face enrolled? [y/N] ").strip().lower()
    return ans == "y"


def open_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {index}. Try --camera 1, or check OS camera permissions.")
    return cap


def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"Could not read image: {path}")
    return resize_max(img, 1280)


# ---------------------------------------------------------------- enroll
def cmd_enroll(args):
    if not confirm_consent(args.name, args.i_have_consent):
        print("Not enrolled.")
        return 1
    engine = FaceEngine()
    vectors = []

    if args.photo:
        for p in args.photo:
            faces = engine.detect(load_image(p))
            if len(faces) != 1:
                print(f"skip  {p}: need exactly 1 face, found {len(faces)}")
                continue
            vectors.append(engine.embed(load_image(p), faces[0]))
            print(f"ok    {p}")
    else:
        cap = open_camera(args.camera)
        last = 0.0
        print("Look at the camera. Slightly vary your head angle between captures. Press q to abort.")
        while len(vectors) < args.samples:
            ok, frame = cap.read()
            if not ok:
                break
            frame = resize_max(frame, 640)
            faces = engine.detect(frame)
            shown = frame.copy()
            if len(faces) == 1 and faces[0][-1] > 0.9 and time.time() - last > 0.8:
                vectors.append(engine.embed(frame, faces[0]))
                last = time.time()
            msg = f"Captured {len(vectors)}/{args.samples}" if len(faces) == 1 else "Need exactly one face in view"
            cv2.putText(shown, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("LocalFace enroll", shown)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

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
        faces = engine.detect(img)
        print(f"\n{p}: {len(faces)} face(s)")
        for i, f in enumerate(faces, 1):
            name, score = gallery.match(engine.embed(img, f), args.threshold)
            label = name or "unknown"
            print(f"  face {i}: {label:<20} similarity={score:.3f}")
            x, y, w, h = map(int, f[:4])
            color = (0, 200, 0) if name else (0, 0, 220)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f"{label} {score:.2f}", (x, max(15, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
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
    gallery = store.gallery()
    if not len(gallery):
        print("Warning: nobody is enrolled, so every face will show as 'unknown'.")
    engine = FaceEngine()
    cap = open_camera(args.camera)
    tracker = Tracker()
    last_logged = {}
    print("Watching. Press q in the window (or Ctrl+C) to stop. Unknown faces are never stored.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = resize_max(frame, 640)
            now = time.time()
            dets = []
            for f in engine.detect(frame):
                x, y, w, h = f[:4]
                name, score = gallery.match(engine.embed(frame, f), args.threshold)
                dets.append(Detection(x + w / 2, y + h / 2, w, h, name, score))

            for tr in tracker.update(dets, now):
                name, score = tr.identity()
                motion = tr.motion()
                if name and now - last_logged.get(name, 0) >= args.cooldown:
                    last_logged[name] = now
                    print(time.strftime("%H:%M:%S"), f"{name}  ({motion}, similarity {score:.2f})")
                    if not args.no_log:
                        store.log_event(name, score, motion)
                if not args.headless:
                    d = tr.det
                    x, y, w, h = int(d.cx - d.w / 2), int(d.cy - d.h / 2), int(d.w), int(d.h)
                    color = (0, 200, 0) if name else (0, 0, 220)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"{name or 'unknown'} | {motion}", (x, max(15, y - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            if not args.headless:
                cv2.imshow("LocalFace", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


# ---------------------------------------------------------------- admin
def cmd_list(_args):
    rows = Store().people()
    if not rows:
        print("Nobody enrolled.")
    for name, n, ts in rows:
        print(f"{name:<24} {n} sample(s)  enrolled {time.strftime('%Y-%m-%d', time.localtime(ts))}")
    return 0


def cmd_remove(args):
    ok = Store().remove_person(args.name)
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
    if not args.yes and input("Delete ALL enrolled faces and events? Type 'delete': ") != "delete":
        print("Cancelled.")
        return 1
    Store().purge()
    print("All data deleted.")
    return 0


# ---------------------------------------------------------------- entry
def main(argv=None):
    ap = argparse.ArgumentParser(prog="localface", description="Local, consent-based face recognition.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("enroll", help="enroll a consenting person (webcam or photos)")
    p.add_argument("name")
    p.add_argument("--photo", nargs="+", help="enroll from clear, single-face photo(s) instead of webcam")
    p.add_argument("--samples", type=int, default=5, help="webcam samples to capture (default 5)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--i-have-consent", action="store_true", help="skip the interactive consent prompt")
    p.set_defaults(fn=cmd_enroll)

    p = sub.add_parser("identify", help="match faces in photo(s) against enrolled people")
    p.add_argument("photo", nargs="+", help="image file(s) or folder(s)")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--annotate", help="write annotated image (file, or folder for many photos)")
    p.set_defaults(fn=cmd_identify)

    p = sub.add_parser("watch", help="live webcam recognition of enrolled people")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--cooldown", type=float, default=30, help="seconds between repeat log entries per person")
    p.add_argument("--headless", action="store_true", help="no preview window")
    p.add_argument("--no-log", action="store_true", help="print only; don't write sightings to the local DB")
    p.set_defaults(fn=cmd_watch)

    sub.add_parser("list", help="list enrolled people").set_defaults(fn=cmd_list)
    p = sub.add_parser("remove", help="delete one person's face data")
    p.add_argument("name")
    p.set_defaults(fn=cmd_remove)
    p = sub.add_parser("events", help="show recent logged sightings")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(fn=cmd_events)
    p = sub.add_parser("purge", help="delete ALL stored faces and events")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(fn=cmd_purge)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
