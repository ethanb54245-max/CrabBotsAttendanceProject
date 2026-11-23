"""Admin PIN: protects destructive actions (remove a person, delete everything).

The PIN is stored only as a salted PBKDF2 hash in the local database. It protects the app's page and the
command-line `remove`/`purge`; it can't stop someone who can open the computer's files, so keep the laptop's own
login secure. `python -m localface reset-pin` clears a forgotten PIN (it needs access to this computer).
"""
import hashlib
import hmac
import os
import secrets
import threading
import time

ITER = 200_000
MIN_LEN, MAX_LEN = 4, 64
KEY = "admin_pin"


class PinError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status, self.message = status, message


def _hash(pin, salt, iters):
    return hashlib.pbkdf2_hmac("sha256", pin.encode(), salt, iters)


def pin_is_set(store):
    return store.get_setting(KEY) is not None


def validate(pin):
    if not isinstance(pin, str) or not (MIN_LEN <= len(pin) <= MAX_LEN):
        raise ValueError(f"The PIN must be {MIN_LEN} to {MAX_LEN} characters.")
    return pin


def set_pin(store, pin):
    validate(pin)
    salt = os.urandom(16)
    store.set_setting(KEY, f"pbkdf2${ITER}${salt.hex()}${_hash(pin, salt, ITER).hex()}")


def clear_pin(store):
    store.delete_setting(KEY)


def verify_pin(store, pin):
    """True if the PIN matches, or if no PIN is set (nothing to check)."""
    rec = store.get_setting(KEY)
    if rec is None:
        return True
    try:
        _, iters, salt, digest = rec.split("$")
        got = _hash(pin or "", bytes.fromhex(salt), int(iters))
        return hmac.compare_digest(got, bytes.fromhex(digest))
    except (ValueError, TypeError):
        return False


class AdminSessions:
    """Short-lived unlock tokens held in memory, plus a lockout against PIN guessing."""

    def __init__(self, ttl=600, max_fails=5, lockout=60):
        self.ttl, self.max_fails, self.lockout = ttl, max_fails, lockout
        self.tokens, self.fails, self.locked_until = {}, 0, 0.0
        self.lock = threading.Lock()

    def attempt(self, store, pin):
        """Check a PIN. Returns a fresh unlock token, or raises PinError (403 wrong, 429 locked out)."""
        now = time.time()
        with self.lock:
            if now < self.locked_until:
                raise PinError(429, f"Too many wrong attempts. Try again in {int(self.locked_until - now) + 1} seconds.")
        ok = verify_pin(store, pin)
        with self.lock:
            if ok:
                self.fails = 0
                tok = secrets.token_hex(16)
                self.tokens[tok] = time.time() + self.ttl
                return tok
            self.fails += 1
            if self.fails >= self.max_fails:
                self.fails, self.locked_until = 0, time.time() + self.lockout
                raise PinError(429, f"Too many wrong attempts. Locked for {self.lockout} seconds.")
            left = self.max_fails - self.fails
            raise PinError(403, f"Wrong PIN. {left} attempt{'s' if left != 1 else ''} left.")

    def valid(self, token):
        """Is this token unlocked? Using it keeps it alive (it expires after ttl seconds of inactivity)."""
        if not token:
            return False
        with self.lock:
            exp = self.tokens.get(token)
            if exp and exp > time.time():
                self.tokens[token] = time.time() + self.ttl
                return True
            self.tokens.pop(token, None)
            return False

    def seconds_left(self, token):
        with self.lock:
            return max(0, int(self.tokens.get(token, 0) - time.time()))

    def revoke(self, token):
        with self.lock:
            self.tokens.pop(token, None)

    def revoke_all(self):
        with self.lock:
            self.tokens.clear()
