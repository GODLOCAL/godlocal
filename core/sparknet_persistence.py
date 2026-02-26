"""
SparkNet SQLite Persistence
===========================
Survives VPS reboots. Q-Learning state is saved after every update.
Drop-in: call load_sparks() at startup, save_spark() after each judge() call.
"""
import sqlite3
import os
import logging
from pathlib import Path
from typing import Dict

log = logging.getLogger(__name__)

DB_PATH = os.getenv("SPARKNET_DB", "/tmp/sparknet.db")


def _conn() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sparks (
            name           TEXT PRIMARY KEY,
            trial_count    INTEGER NOT NULL DEFAULT 0,
            success_count  INTEGER NOT NULL DEFAULT 0,
            ema_score      REAL    NOT NULL DEFAULT 0.5,
            updated_at     TEXT    NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    return conn


def save_spark(name: str, trial_count: int, success_count: int, ema_score: float) -> None:
    """Upsert one Spark's stats to SQLite."""
    try:
        with _conn() as conn:
            conn.execute("""
                INSERT INTO sparks (name, trial_count, success_count, ema_score, updated_at)
                VALUES (?, ?, ?, ?, datetime('now'))
                ON CONFLICT(name) DO UPDATE SET
                    trial_count   = excluded.trial_count,
                    success_count = excluded.success_count,
                    ema_score     = excluded.ema_score,
                    updated_at    = excluded.updated_at
            """, (name, trial_count, success_count, ema_score))
    except Exception as e:
        log.warning(f"[SparkPersist] save_spark({name}) failed: {e}")


def load_sparks() -> Dict[str, dict]:
    """Load all Spark stats from SQLite. Returns {name: {trial_count, success_count, ema_score}}."""
    try:
        with _conn() as conn:
            rows = conn.execute(
                "SELECT name, trial_count, success_count, ema_score FROM sparks"
            ).fetchall()
        result = {
            row[0]: {
                "trial_count": row[1],
                "success_count": row[2],
                "ema_score": row[3],
            }
            for row in rows
        }
        log.info(f"[SparkPersist] Loaded {len(result)} sparks from DB.")
        return result
    except Exception as e:
        log.warning(f"[SparkPersist] load_sparks() failed: {e}")
        return {}


def wipe_db() -> None:
    """Nuclear option — reset all Q-Learning state. Use only in tests."""
    try:
        with _conn() as conn:
            conn.execute("DELETE FROM sparks")
        log.warning("[SparkPersist] DB wiped.")
    except Exception as e:
        log.warning(f"[SparkPersist] wipe_db() failed: {e}")
