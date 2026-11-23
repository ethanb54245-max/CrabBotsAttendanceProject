"""Centroid tracker with velocity prediction, sticky identities, and walking/standing detection."""
import math
from collections import Counter, deque

WINDOW_SEC = 1.5          # motion is measured over this much history
MIN_SPAN_SEC = 0.4        # need at least this much history to judge motion
WALK_SPEED = 0.6          # face-widths per second of lateral movement => "walking"
WALK_SCALE_RATE = 0.25    # relative face-size change per second (walking toward/away)
MAX_MISSING_SEC = 1.5     # keep a track alive this long without a detection
GATE = 2.0                # max match distance, in face-widths (generous for fast walkers)
SMOOTH = 0.6              # box smoothing for display (1 = no smoothing)
MIN_VOTES = 2             # weak matches need this many agreeing recent frames...
SWITCH_VOTES = 3          # ...and a locked identity only changes after this many contrary frames


class Detection:
    def __init__(self, cx, cy, w, h, name, score):
        self.cx, self.cy, self.w, self.h, self.name, self.score = cx, cy, w, h, name, score


class Track:
    def __init__(self, tid, det, t):
        self.id = tid
        self.born = t
        self.hits = 0
        self.history = deque()
        self.votes = deque(maxlen=8)
        self.locked = None
        self.locked_score = 0.0
        self.last_seen = t
        self.det = det
        self.view = (det.cx - det.w / 2, det.cy - det.h / 2, det.w, det.h)
        self.update(det, t, first=True)

    def update(self, det, t, first=False):
        self.det = det
        self.last_seen = t
        self.hits += 1
        box = (det.cx - det.w / 2, det.cy - det.h / 2, det.w, det.h)
        self.view = box if first else tuple(
            SMOOTH * n + (1 - SMOOTH) * o for n, o in zip(box, self.view))
        self.history.append((t, det.cx, det.cy, det.w))
        while self.history and t - self.history[0][0] > WINDOW_SEC:
            self.history.popleft()
        self.votes.append((det.name, det.score))

    def predict(self, t):
        """Where the face probably is now, extrapolated from its last velocity."""
        if len(self.history) < 2:
            return self.det.cx, self.det.cy
        t0, x0, y0, _ = self.history[-2]
        t1, x1, y1, _ = self.history[-1]
        dt = t1 - t0
        if dt <= 1e-6:
            return x1, y1
        lead = min(max(t - t1, 0.0), 0.5)
        return x1 + (x1 - x0) / dt * lead, y1 + (y1 - y0) / dt * lead

    def identity(self, strong=0.5):
        """(name or None, score). One strong match is enough; weaker ones need MIN_VOTES.

        Once identified, the identity sticks to the track through frames where recognition
        briefly fails (motion blur, head turned), and only changes if another enrolled
        person wins SWITCH_VOTES recent frames.
        """
        names = Counter(n for n, _ in self.votes if n)
        cand = None
        if names:
            n, count = names.most_common(1)[0]
            best = max(s for nn, s in self.votes if nn == n)
            if count >= MIN_VOTES or best >= strong:
                cand = n
        if self.locked and (cand is None or cand == self.locked
                            or names.get(cand, 0) < SWITCH_VOTES):
            cand = self.locked
        if cand:
            scores = [s for nn, s in self.votes if nn == cand]
            self.locked = cand
            if scores:
                self.locked_score = sum(scores) / len(scores)
            return cand, self.locked_score
        return None, 0.0

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
    def __init__(self, strong_score=0.5):
        self.tracks = []
        self.strong_score = strong_score
        self._next = 1

    def update(self, detections, t):
        # Drop tracks that have been gone too long *before* matching, so a stale track (and the identity
        # locked to it) can never be revived by a different person who shows up in the same place later.
        self.tracks = [tr for tr in self.tracks if t - tr.last_seen <= MAX_MISSING_SEC]
        pairs = []
        for ti, tr in enumerate(self.tracks):
            px, py = tr.predict(t)
            for di, det in enumerate(detections):
                d = min(math.hypot(det.cx - tr.det.cx, det.cy - tr.det.cy),
                        math.hypot(det.cx - px, det.cy - py))
                gate = GATE * max(det.w, tr.det.w)
                if d < gate:
                    pairs.append((d / gate, ti, di))
        pairs.sort()
        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            self.tracks[ti].update(detections[di], t)
        for di, det in enumerate(detections):
            if di not in used_d:
                self.tracks.append(Track(self._next, det, t))
                self._next += 1
        self.tracks = [tr for tr in self.tracks if t - tr.last_seen <= MAX_MISSING_SEC]
        return [tr for tr in self.tracks if tr.last_seen == t]
