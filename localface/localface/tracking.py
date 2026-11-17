"""Tiny centroid tracker that also classifies motion as walking vs standing."""
import math
from collections import Counter, deque

WINDOW_SEC = 1.5          # motion is measured over this much history
MIN_SPAN_SEC = 0.5        # need at least this much history to judge motion
WALK_SPEED = 0.6          # face-widths per second of lateral movement => "walking"
WALK_SCALE_RATE = 0.25    # relative face-size change per second (walking toward/away)
MAX_MISSING_SEC = 1.0     # drop a track after this long unseen
MIN_VOTES = 3             # identity needs this many agreeing recent frames


class Detection:
    def __init__(self, cx, cy, w, h, name, score):
        self.cx, self.cy, self.w, self.h, self.name, self.score = cx, cy, w, h, name, score


class Track:
    def __init__(self, tid, det, t):
        self.id = tid
        self.history = deque()
        self.votes = deque(maxlen=7)
        self.last_seen = t
        self.det = det
        self.update(det, t)

    def update(self, det, t):
        self.det = det
        self.last_seen = t
        self.history.append((t, det.cx, det.cy, det.w))
        while self.history and t - self.history[0][0] > WINDOW_SEC:
            self.history.popleft()
        self.votes.append((det.name, det.score))

    def identity(self):
        """Majority-vote identity over recent frames: (name or None, mean score)."""
        names = Counter(n for n, _ in self.votes if n)
        if not names:
            return None, 0.0
        name, count = names.most_common(1)[0]
        if count < MIN_VOTES:
            return None, 0.0
        scores = [s for n, s in self.votes if n == name]
        return name, sum(scores) / len(scores)

    def motion(self):
        if len(self.history) < 2:
            return "detecting"
        t0, x0, y0, w0 = self.history[0]
        t1, x1, y1, w1 = self.history[-1]
        span = t1 - t0
        if span < MIN_SPAN_SEC:
            return "detecting"
        mean_w = sum(h[3] for h in self.history) / len(self.history)
        lateral = math.hypot(x1 - x0, y1 - y0) / mean_w / span
        scale = abs(w1 - w0) / mean_w / span
        return "walking" if (lateral > WALK_SPEED or scale > WALK_SCALE_RATE) else "standing"


class Tracker:
    def __init__(self):
        self.tracks = []
        self._next = 1

    def update(self, detections, t):
        unmatched = list(self.tracks)
        for det in detections:
            best, best_d = None, None
            for tr in unmatched:
                d = math.hypot(det.cx - tr.det.cx, det.cy - tr.det.cy)
                if d < 1.2 * max(det.w, tr.det.w) and (best_d is None or d < best_d):
                    best, best_d = tr, d
            if best:
                unmatched.remove(best)
                best.update(det, t)
            else:
                self.tracks.append(Track(self._next, det, t))
                self._next += 1
        self.tracks = [tr for tr in self.tracks if t - tr.last_seen <= MAX_MISSING_SEC]
        return [tr for tr in self.tracks if tr.last_seen == t]
