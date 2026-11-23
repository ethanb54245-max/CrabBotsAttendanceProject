"""Admin PIN: hashing, lockout, unlock tokens, server protection of destructive actions, CLI prompts."""
import argparse
import io
import json
import os
import sys
import tempfile
import threading
import time
from contextlib import redirect_stdout
from http.server import ThreadingHTTPServer
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["LOCALFACE_DATA"] = tempfile.mkdtemp()

import localface.cli as cli  # noqa: E402
from localface import admin, demo  # noqa: E402
from localface.store import Store  # noqa: E402
from test_demo import FakeEngine, call, photo  # noqa: E402

admin.ITER = 1000          # fast hashing for tests
GOOD = {"X-LocalFace": "1"}
J = {"X-LocalFace": "1", "Content-Type": "application/json"}


def test_pin_hashing():
    s = Store(":memory:")
    assert not admin.pin_is_set(s) and admin.verify_pin(s, "anything")      # no PIN: nothing to check
    admin.set_pin(s, "1234")
    assert admin.pin_is_set(s) and "1234" not in s.get_setting(admin.KEY)    # stored hashed, not in clear
    assert admin.verify_pin(s, "1234") and not admin.verify_pin(s, "1235") and not admin.verify_pin(s, "")
    for bad in ("", "123", "x" * 65, None):
        try:
            admin.set_pin(s, bad)
            raise AssertionError("accepted a bad PIN")
        except ValueError:
            pass
    first = s.get_setting(admin.KEY)
    admin.set_pin(s, "1234")
    assert s.get_setting(admin.KEY) != first                                 # fresh salt every time
    s.purge()
    assert admin.pin_is_set(s)                                               # delete-all keeps the PIN
    admin.clear_pin(s)
    assert not admin.pin_is_set(s)


def test_sessions_lockout_and_expiry():
    s = Store(":memory:")
    admin.set_pin(s, "4321")
    a = admin.AdminSessions(ttl=0.2, max_fails=3, lockout=0.3)
    for left in (2, 1):
        try:
            a.attempt(s, "0000")
        except admin.PinError as e:
            assert e.status == 403 and f"{left} attempt" in e.message
    try:
        a.attempt(s, "0000")
    except admin.PinError as e:
        assert e.status == 429
    try:
        a.attempt(s, "4321")                                                  # right PIN, but locked out
        raise AssertionError("lockout not enforced")
    except admin.PinError as e:
        assert e.status == 429
    time.sleep(0.35)
    tok = a.attempt(s, "4321")
    assert a.valid(tok) and not a.valid("nope") and not a.valid("")
    a.revoke(tok)
    assert not a.valid(tok)
    tok = a.attempt(s, "4321")
    time.sleep(0.3)
    assert not a.valid(tok)                                                   # expires when idle


def start():
    app = demo.App(0.363, engine=FakeEngine())
    demo.Handler.app = app
    srv = ThreadingHTTPServer(("127.0.0.1", 0), demo.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_server_protects_destructive_actions():
    Store().purge()
    Store().delete_setting(admin.KEY)
    srv = start()
    for nm, g in (("Ava", 100), ("Ben", 180)):
        assert call(srv, "POST", f"/api/enroll?name={nm}&consent=1", photo(g), GOOD)[0] == 200

    # no PIN set: everything works as before, and the status says so
    assert call(srv, "GET", "/api/admin/status")[1] == {"pin_set": False, "unlocked": False, "minutes_left": 0}
    assert call(srv, "DELETE", "/api/people?name=Ben", None, GOOD)[0] == 200
    assert call(srv, "POST", "/api/enroll?name=Ben&consent=1", photo(180), GOOD)[0] == 200

    # setting a PIN: needs the header, 4+ chars; returns an unlock token
    assert call(srv, "POST", "/api/admin/pin", json.dumps({"new": "2468"}), {"Content-Type": "application/json"})[0] == 403
    assert call(srv, "POST", "/api/admin/pin", json.dumps({"new": "12"}), J)[0] == 400
    st, j = call(srv, "POST", "/api/admin/pin", json.dumps({"new": "2468"}), J)
    assert st == 200 and j["pin_set"] and j["token"]
    token = j["token"]
    ADMIN = dict(GOOD, **{"X-LocalFace-Admin": token})

    # now destructive actions are locked without a token...
    assert call(srv, "DELETE", "/api/people?name=Ben", None, GOOD)[0] == 401
    assert call(srv, "POST", "/api/purge", None, GOOD)[0] == 401
    assert call(srv, "DELETE", "/api/people?name=Ben", None, dict(GOOD, **{"X-LocalFace-Admin": "bogus"}))[0] == 401
    assert len(call(srv, "GET", "/api/people")[1]["people"]) == 2
    # ...but recognizing and signing up stay open
    assert call(srv, "POST", "/api/identify", photo(100), GOOD)[0] == 200
    assert call(srv, "POST", "/api/enroll?name=Cy&consent=1", photo(30), GOOD)[0] == 200
    assert call(srv, "GET", "/api/people")[0] == 200 and call(srv, "GET", "/api/attendance.csv")[0] == 200

    # status reflects the token
    assert call(srv, "GET", "/api/admin/status", None, ADMIN)[1]["unlocked"] is True
    assert call(srv, "GET", "/api/admin/status")[1]["unlocked"] is False

    # wrong PIN, then right PIN
    st, j = call(srv, "POST", "/api/admin/unlock", json.dumps({"pin": "0000"}), J)
    assert st == 403 and "4 attempts left" in j["error"]
    st, j = call(srv, "POST", "/api/admin/unlock", json.dumps({"pin": "2468"}), J)
    assert st == 200
    ADMIN2 = dict(GOOD, **{"X-LocalFace-Admin": j["token"]})
    assert call(srv, "DELETE", "/api/people?name=Cy", None, ADMIN2)[0] == 200

    # lock now
    assert call(srv, "POST", "/api/admin/lock", None, ADMIN2)[0] == 200
    assert call(srv, "DELETE", "/api/people?name=Ben", None, ADMIN2)[0] == 401

    # changing / removing the PIN needs the current one
    assert call(srv, "POST", "/api/admin/pin", json.dumps({"current": "bad!", "new": "1357"}), J)[0] == 403
    st, j = call(srv, "POST", "/api/admin/pin", json.dumps({"current": "2468", "new": "1357"}), J)
    assert st == 200
    assert call(srv, "DELETE", "/api/people?name=Ben", None, ADMIN)[0] == 401          # old token revoked
    st, j = call(srv, "POST", "/api/admin/pin", json.dumps({"current": "1357", "new": ""}), J)
    assert st == 200 and j["pin_set"] is False
    assert call(srv, "POST", "/api/purge", None, GOOD)[0] == 200                         # no PIN again: open
    srv.shutdown()


def test_cli_prompts_for_pin_and_reset():
    s = Store()
    s.purge()
    s.add_person("Dee", [np.ones(128)])
    s.add_person("Eli", [np.ones(128)])
    admin.set_pin(s, "9999")
    out = io.StringIO()
    with redirect_stdout(out), mock.patch("getpass.getpass", return_value="nope"):
        assert cli.cmd_remove(argparse.Namespace(name="Dee")) == 1
        assert cli.cmd_purge(argparse.Namespace(yes=True)) == 1
    assert "Wrong PIN" in out.getvalue() and len(Store().people()) == 2
    with redirect_stdout(io.StringIO()), mock.patch("getpass.getpass", return_value="9999"):
        assert cli.cmd_remove(argparse.Namespace(name="Dee")) == 0
    assert [p[0] for p in Store().people()] == ["Eli"]
    with redirect_stdout(io.StringIO()), mock.patch.dict(os.environ, {"LOCALFACE_PIN": "9999"}):
        assert cli.cmd_purge(argparse.Namespace(yes=True)) == 0                          # env var for scripts
    assert Store().people() == []
    with redirect_stdout(io.StringIO()):
        assert cli.cmd_reset_pin(argparse.Namespace(yes=True)) == 0                      # forgotten PIN
    assert not admin.pin_is_set(Store())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
    print("admin tests passed")
