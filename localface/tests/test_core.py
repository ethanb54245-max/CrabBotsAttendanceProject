"""Model-free tests for storage, matching and walking/standing classification."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["LOCALFACE_DATA"] = tempfile.mkdtemp()

from localface.store import Store  # noqa: E402
from localface.tracking import Detection, Tracker  # noqa: E402


def test_store_and_match():
    rng = np.random.default_rng(0)
    alice, bob = rng.normal(size=128), rng.normal(size=128)
    s = Store(":memory:")
    s.add_person("alice", [alice + rng.normal(scale=0.05, size=128)])
    s.add_person("bob", [bob])
    g = s.gallery()
    assert g.match(alice, 0.363)[0] == "alice"
    assert g.match(bob, 0.363)[0] == "bob"
    assert g.match(rng.normal(size=128), 0.363)[0] is None  # stranger -> unknown
    assert s.remove_person("bob")
    assert g.match(bob, 0.363)[0] == "bob"  # old gallery object unchanged
    assert s.gallery().match(bob, 0.363)[0] is None


def _run(positions):
    tr, t, last = Tracker(), 0.0, None
    for cx in positions:
        out = tr.update([Detection(cx, 200, 100, 120, "alice", 0.8)], t)
        last, t = out[0], t + 0.1
    return last


def test_standing_vs_walking():
    assert _run([300 + (i % 2) for i in range(20)]).motion() == "standing"
    assert _run([100 + 30 * i for i in range(20)]).motion() == "walking"
    assert _run([300 + (i % 2) for i in range(20)]).identity()[0] == "alice"


if __name__ == "__main__":
    test_store_and_match()
    test_standing_vs_walking()
    print("all tests passed")
