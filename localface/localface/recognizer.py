"""Live recognition logic shared by the fullscreen window (`watch`) and the in-page camera.

Feed it frames; it detects faces, matches them against the enrolled roster, tracks people from frame to
frame (so identities stick and walking/standing can be told apart), logs attendance, and reloads the roster
when people are added elsewhere.
"""
import threading
import time

from .store import Store
from .tracking import Detection, Tracker


class TrackResult:
    __slots__ = ("track", "name", "score", "motion", "state", "visible")

    def __init__(self, track, name, score, motion, state, visible):
        self.track, self.name, self.score = track, name, score
        self.motion, self.state, self.visible = motion, state, visible


class LiveRecognizer:
    def __init__(self, analyze_fn, threshold, cooldown=30.0, log=True, store_factory=Store,
                 min_margin=0.04):
        """analyze_fn(frame) -> list of Face. store_factory() -> a Store usable on the calling thread."""
        self.analyze_fn, self.threshold, self.cooldown, self.log = analyze_fn, threshold, cooldown, log
        self.min_margin = min_margin
        self.store_factory = store_factory
        self.tracker = Tracker(strong_score=max(0.5, threshold + 0.12))
        store = store_factory()
        self.sig, self.gallery = store.signature(), store.gallery()
        self.enrolled, self.present = len(store.people()), set(store.names_today())
        self.last_logged, self.reloaded, self._last_check = {}, False, 0.0
        self.lock = threading.Lock()

    def _maybe_reload(self, now):
        if now - self._last_check < 1.5:
            return
        self._last_check = now
        store = self.store_factory()
        sig = store.signature()
        if sig != self.sig:                          # someone was added or removed
            self.sig, self.gallery = sig, store.gallery()
            self.enrolled, self.reloaded = len(store.people()), True

    def process(self, frame, now=None, face_meshes=None, face_matrices=None):
        """Returns (results, events); optional 3D pose estimates gate only extreme views."""
        with self.lock:
            now = now or time.time()
            self._maybe_reload(now)
            dets = []
            faces = self.analyze_fn(frame)
            pose_yaws = [None] * len(faces)
            if face_meshes and face_matrices:
                try:
                    from .face_mesh import associate_pose_yaws
                    pose_yaws = associate_pose_yaws(faces, face_meshes, face_matrices, frame.shape)
                except (AttributeError, IndexError, TypeError, ValueError):
                    # Missing/invalid pose data must not interfere with SFace matching.
                    pose_yaws = [None] * len(faces)
            for f, pose_yaw in zip(faces, pose_yaws):
                name, score, _margin = self.gallery.match_detailed(
                    f.embedding, self.threshold, self.min_margin)
                # Use MediaPipe's estimated facial transformation matrix rather than
                # the old 2D nose/eye ratio. Never override an SFace match that already
                # passed its similarity and ambiguity checks. At extreme yaw, mark only
                # an otherwise-unmatched frame as a pose dropout; an existing track can
                # retain its previously locked identity through that brief dropout.
                pose_rejected = pose_yaw is not None and pose_yaw > 65.0 and name is None
                dets.append(Detection(f.cx, f.cy, f.w, f.h, name, score, pose_rejected))
            results, events = [], []
            tracked = self.tracker.update(dets, now)
            identities = [tr.identity(self.tracker.strong_score) for tr in tracked]
            # An enrolled identity may be assigned to only one visible track at a time.
            # If multiple faces resemble the same enrolled person, give that identity to
            # the face with the strongest current-frame match; leave the other face(s)
            # Unknown. This avoids turning both faces into Unknown while preventing
            # duplicate attendance for one enrolled identity.
            winners = {}
            for idx, (tr, (name, _score)) in enumerate(zip(tracked, identities)):
                if not name:
                    continue
                current_score = tr.det.score if tr.det.name == name else _score
                previous = winners.get(name)
                if previous is None or current_score > previous[0]:
                    winners[name] = (current_score, idx)
            for idx, (tr, (name, score)) in enumerate(zip(tracked, identities)):
                if name and winners[name][1] != idx:
                    name, score = None, 0.0
                motion = tr.motion()
                state = "pose" if tr.det.pose_rejected and not name else ("ok" if name else
                         ("scanning" if now - tr.born < 1.2 else "unknown"))
                results.append(TrackResult(tr, name, score, motion, state, bool(name) or tr.hits >= 2))
                ready = motion != "detecting" or now - tr.born > 1.0
                if name and ready and now - self.last_logged.get(name, 0) >= self.cooldown:
                    self.last_logged[name] = now
                    self.present.add(name)
                    events.append({"time": now, "name": name, "motion": motion, "score": score})
                    if self.log:
                        self.store_factory().log_event(name, score, motion)
            return results, events
