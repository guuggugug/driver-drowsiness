"""Database SQLite: tai xe, phien, mau du lieu, su kien canh bao."""
import os
import sqlite3
import time

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "driver.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS drivers(
  id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL,
  base_ear REAL, base_pitch REAL, created REAL);
CREATE TABLE IF NOT EXISTS sessions(
  id INTEGER PRIMARY KEY, driver_id INTEGER NOT NULL, kind TEXT NOT NULL,
  started REAL, ended REAL, FOREIGN KEY(driver_id) REFERENCES drivers(id));
CREATE TABLE IF NOT EXISTS samples(
  id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL, t REAL,
  ear_n REAL, mar REAL, dpitch REAL, roll REAL, label INTEGER NOT NULL,
  FOREIGN KEY(session_id) REFERENCES sessions(id));
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL, t REAL,
  type TEXT, level INTEGER, score REAL,
  FOREIGN KEY(session_id) REFERENCES sessions(id));
CREATE INDEX IF NOT EXISTS idx_samples_session ON samples(session_id);
"""


def connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def get_driver(con, name):
    row = con.execute("SELECT * FROM drivers WHERE name=?", (name,)).fetchone()
    if row is None:
        con.execute("INSERT INTO drivers(name, created) VALUES(?,?)", (name, time.time()))
        con.commit()
        row = con.execute("SELECT * FROM drivers WHERE name=?", (name,)).fetchone()
    return row


def get_baseline(driver_row):
    if driver_row["base_ear"] is None:
        return None
    return {"ear": driver_row["base_ear"], "pitch": driver_row["base_pitch"]}


def save_baseline(con, driver_id, base):
    con.execute("UPDATE drivers SET base_ear=?, base_pitch=? WHERE id=?",
                (base["ear"], base["pitch"], driver_id))
    con.commit()


def new_session(con, driver_id, kind):
    cur = con.execute("INSERT INTO sessions(driver_id, kind, started) VALUES(?,?,?)",
                      (driver_id, kind, time.time()))
    con.commit()
    return cur.lastrowid


def end_session(con, sid):
    con.execute("UPDATE sessions SET ended=? WHERE id=?", (time.time(), sid))
    con.commit()


def add_samples(con, rows):
    """rows: (session_id, t, ear_n, mar, dpitch, roll, label)"""
    con.executemany("INSERT INTO samples(session_id,t,ear_n,mar,dpitch,roll,label) "
                    "VALUES(?,?,?,?,?,?,?)", rows)
    con.commit()


def add_event(con, sid, etype, level, score=0.0):
    con.execute("INSERT INTO events(session_id,t,type,level,score) VALUES(?,?,?,?,?)",
                (sid, time.time(), etype, level, float(score)))
    con.commit()
