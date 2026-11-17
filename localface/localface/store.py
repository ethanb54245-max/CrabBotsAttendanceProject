"""Local SQLite storage for enrolled people, their face embeddings, and sighting events.

Nothing here touches the network. Unknown faces are never stored.
"""
import os
import sqlite3
import time
from pathlib import Path

import numpy as np

DATA_DIR = Path(os.environ.get("LOCALFACE_DATA", Path.cwd() / "data"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS people (
    id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY,
    person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    vec BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY, ts REAL NOT NULL, name TEXT NOT NULL,
    score REAL NOT NULL, motion TEXT NOT NULL);
"""


def _unit(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class Gallery:
    """In-memory set of enrolled embeddings for fast matching."""

    def __init__(self, names, matrix):
        self.names = names
        self.matrix = matrix

    def __len__(self):
        return len(self.names)

    def match(self, embedding, threshold):
        """Return (name or None, best cosine similarity)."""
        if not self.names:
            return None, 0.0
        sims = self.matrix @ _unit(embedding)
        i = int(np.argmax(sims))
        score = float(sims[i])
        return (self.names[i] if score >= threshold else None), score


class Store:
    def __init__(self, path=None):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path or DATA_DIR / "localface.db"))
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA)

    def add_person(self, name, vectors):
        row = self.db.execute("SELECT id FROM people WHERE name=?", (name,)).fetchone()
        if row:
            pid = row[0]
        else:
            pid = self.db.execute(
                "INSERT INTO people(name, created_at) VALUES(?,?)", (name, time.time())
            ).lastrowid
        for v in vectors:
            blob = np.asarray(v, dtype=np.float32).reshape(-1).tobytes()
            self.db.execute("INSERT INTO embeddings(person_id, vec) VALUES(?,?)", (pid, blob))
        self.db.commit()
        return pid

    def people(self):
        return self.db.execute(
            "SELECT p.name, COUNT(e.id), p.created_at FROM people p "
            "LEFT JOIN embeddings e ON e.person_id=p.id GROUP BY p.id ORDER BY p.name"
        ).fetchall()

    def remove_person(self, name):
        cur = self.db.execute("DELETE FROM people WHERE name=?", (name,))
        self.db.commit()
        return cur.rowcount > 0

    def purge(self):
        for t in ("embeddings", "events", "people"):
            self.db.execute(f"DELETE FROM {t}")
        self.db.commit()

    def gallery(self):
        rows = self.db.execute(
            "SELECT p.name, e.vec FROM embeddings e JOIN people p ON p.id=e.person_id"
        ).fetchall()
        names = [r[0] for r in rows]
        if not rows:
            return Gallery([], np.zeros((0, 128), dtype=np.float32))
        mat = np.stack([_unit(np.frombuffer(r[1], dtype=np.float32)) for r in rows])
        return Gallery(names, mat)

    def log_event(self, name, score, motion):
        self.db.execute(
            "INSERT INTO events(ts, name, score, motion) VALUES(?,?,?,?)",
            (time.time(), name, float(score), motion),
        )
        self.db.commit()

    def recent_events(self, limit=20):
        return self.db.execute(
            "SELECT ts, name, score, motion FROM events ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
