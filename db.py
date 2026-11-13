"""Database SQLite: nguoi dung, phien, mau du lieu, su kien canh bao."""
import os
import sqlite3
import time

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "monitor.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL,
  base_ear REAL, base_pitch REAL, created REAL, base_ver INTEGER DEFAULT 0,
  base_blink REAL DEFAULT 0);
CREATE TABLE IF NOT EXISTS sessions(
  id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, kind TEXT NOT NULL,
  started REAL, ended REAL, FOREIGN KEY(user_id) REFERENCES users(id));
CREATE TABLE IF NOT EXISTS samples(
  id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL, t REAL,
  ear_n REAL, mar REAL, dpitch REAL, roll REAL, label INTEGER NOT NULL,
  yaw REAL DEFAULT 0, blink REAL, jaw REAL,
  FOREIGN KEY(session_id) REFERENCES sessions(id));
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL, t REAL,
  type TEXT, level INTEGER, score REAL,
  FOREIGN KEY(session_id) REFERENCES sessions(id));
CREATE INDEX IF NOT EXISTS idx_samples_session ON samples(session_id);
"""

# Cot moi them dan qua cac phien ban: (bang, cot, kieu)
_NEW_COLS = [
    ("samples", "yaw", "REAL DEFAULT 0"),
    ("samples", "blink", "REAL"),      # diem nham mat (MediaPipe blendshape), NULL = du lieu cu
    ("samples", "jaw", "REAL"),        # diem ha mieng (MediaPipe blendshape)
    ("users", "base_ver", "INTEGER DEFAULT 0"),
    ("users", "base_blink", "REAL DEFAULT 0"),
]


def _migrate(con):
    """Nang cap DB cu: drivers -> users, sessions.driver_id -> user_id, them cot moi."""
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "drivers" in tables and "users" not in tables:
        con.execute("ALTER TABLE drivers RENAME TO users")
        tables.discard("drivers"); tables.add("users")
    if "sessions" in tables:
        cols = [r[1] for r in con.execute("PRAGMA table_info(sessions)")]
        if "driver_id" in cols:
            con.execute("ALTER TABLE sessions RENAME COLUMN driver_id TO user_id")
    for table, col, typ in _NEW_COLS:
        if table in tables:
            cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
            if col not in cols:
                con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
    con.commit()


def connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    _migrate(con)
    con.executescript(SCHEMA)
    return con


def get_user(con, name):
    row = con.execute("SELECT * FROM users WHERE name=?", (name,)).fetchone()
    if row is None:
        con.execute("INSERT INTO users(name, created) VALUES(?,?)", (name, time.time()))
        con.commit()
        row = con.execute("SELECT * FROM users WHERE name=?", (name,)).fetchone()
    return row


BASE_VER = 3  # tang khi doi cach tinh dac trung -> baseline cu tu dong bi bo


def get_baseline(user_row):
    if user_row["base_ear"] is None or (user_row["base_ver"] or 0) != BASE_VER:
        return None
    return {"ear": user_row["base_ear"], "pitch": user_row["base_pitch"],
            "blink": user_row["base_blink"] or 0.0}


def save_baseline(con, user_id, base):
    con.execute("UPDATE users SET base_ear=?, base_pitch=?, base_blink=?, base_ver=? WHERE id=?",
                (base["ear"], base["pitch"], base.get("blink", 0.0), BASE_VER, user_id))
    con.commit()


def new_session(con, user_id, kind):
    cur = con.execute("INSERT INTO sessions(user_id, kind, started) VALUES(?,?,?)",
                      (user_id, kind, time.time()))
    con.commit()
    return cur.lastrowid


def end_session(con, sid):
    con.execute("UPDATE sessions SET ended=? WHERE id=?", (time.time(), sid))
    con.commit()


def add_samples(con, rows):
    """rows: (session_id, t, ear_n, mar, dpitch, roll, yaw, blink, jaw, label)"""
    con.executemany("INSERT INTO samples(session_id,t,ear_n,mar,dpitch,roll,yaw,blink,jaw,label) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()


def sample_row(sid, t, n, label):
    """Tao 1 dong samples tu vector da normalize n (7 phan tu)."""
    return (sid, t, *[float(v) for v in n[:7]], int(label))


def add_event(con, sid, etype, level, score=0.0):
    con.execute("INSERT INTO events(session_id,t,type,level,score) VALUES(?,?,?,?,?)",
                (sid, time.time(), etype, level, float(score)))
    con.commit()
