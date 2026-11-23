"""Starts the app exactly the way `python -m localface` does (separate process, real port, real Ctrl+C),
with a stand-in face engine, and checks it serves the page, falls back to a free port, and exits cleanly."""
import os
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WRAPPER = textwrap.dedent("""
    import sys
    sys.path.insert(0, {root!r})
    from localface import demo
    demo.ensure_models = lambda: None
    class Eng:
        def analyze(self, img, max_side=960): return []
        def photo_embeddings(self, img): return None, 0
    demo.FaceEngine = Eng
    sys.exit(demo.run(port={port}, open_browser=False))
""")


def test_real_process_start_port_fallback_and_ctrl_c():
    if sys.platform.startswith("win"):
        print("   (skipped on Windows: signals differ)")
        return
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    busy = blocker.getsockname()[1]                         # a port that is already taken
    env = dict(os.environ, LOCALFACE_DATA=tempfile.mkdtemp(), PYTHONUNBUFFERED="1")
    proc = subprocess.Popen([sys.executable, "-c", WRAPPER.format(root=ROOT, port=busy)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    try:
        url = None
        for _ in range(100):
            line = proc.stdout.readline()
            if "running at" in line:
                url = line.split("running at")[1].strip()
                break
        assert url, "the app never reported its address"
        assert url != f"http://127.0.0.1:{busy}"            # it moved on to the next free port
        page = urllib.request.urlopen(url + "/", timeout=5).read().decode()
        assert "LocalFace" in page and "Start live camera" in page
        assert urllib.request.urlopen(url + "/api/ping", timeout=5).status == 200
        proc.send_signal(signal.SIGINT)                      # what pressing Ctrl+C does
        assert proc.wait(timeout=10) == 0                    # clean exit, no traceback
        try:
            urllib.request.urlopen(url + "/api/ping", timeout=2)
            raise AssertionError("the server kept running after Ctrl+C")
        except OSError:
            pass
    finally:
        if proc.poll() is None:
            proc.kill()
        blocker.close()


if __name__ == "__main__":
    test_real_process_start_port_fallback_and_ctrl_c()
    print("server process tests passed")
