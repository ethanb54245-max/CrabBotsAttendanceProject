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
    def __init__(self, analyze_fn, threshold, cooldown=30.0, log=True, store_factory=Store):
        """analyze_fn(frame) -> list of Face. store_factory() -> a Store usable on the calling thread."""
        self.analyze_fn, self.threshold, self.cooldown, self.log = analyze_fn, threshold, cooldown, log
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

    def process(self, frame, now=None):
        """Returns (results, events): a TrackResult per tracked face, and new attendance events."""
        with self.lock:
            now = now or time.time()
            self._maybe_reload(now)
            dets = []
            for f in self.analyze_fn(frame):
                name, score = self.gallery.match(f.embedding, self.threshold)
                dets.append(Detection(f.cx, f.cy, f.w, f.h, name, score))
            results, events = [], []
            for tr in self.tracker.update(dets, now):
                name, score = tr.identity(self.tracker.strong_score)
                motion = tr.motion()
                state = "ok" if name else ("scanning" if now - tr.born < 1.2 else "unknown")
                results.append(TrackResult(tr, name, score, motion, state, bool(name) or tr.hits >= 2))
                ready = motion != "detecting" or now - tr.born > 1.0
                if name and ready and now - self.last_logged.get(name, 0) >= self.cooldown:
                    self.last_logged[name] = now
                    self.present.add(name)
                    events.append({"time": now, "name": name, "motion": motion, "score": score})
                    if self.log:
                        self.store_factory().log_event(name, score, motion)
            return results, events
