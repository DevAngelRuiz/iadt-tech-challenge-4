"""Receptor local persistente, com deduplicação por episódio e confirmação de leitura."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import time
import uuid
from contextlib import contextmanager


class AlertStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY, run_id TEXT, episode_key TEXT, source_epoch REAL,
                severity TEXT, payload TEXT, received_at TEXT, acknowledged_at TEXT)""")
            db.execute("CREATE INDEX IF NOT EXISTS episode_idx ON alerts(run_id, episode_key, source_epoch)")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(str(self.path), timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def publish(self, alert, run_id, started=None, cooldown_seconds=60):
        now = datetime.now(timezone.utc)
        source = datetime.fromisoformat(alert.get("timestamp", now.isoformat()).replace("Z", "+00:00"))
        episode = alert.get("episode_key") or f"{alert.get('patient_id', 'demo')}:{alert['type']}"
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute("SELECT source_epoch, severity FROM alerts WHERE run_id=? AND episode_key=? ORDER BY source_epoch DESC LIMIT 1",
                                  (run_id, episode)).fetchone()
            if previous and 0 <= source.timestamp() - previous["source_epoch"] < cooldown_seconds:
                if not (alert["severity"] == "CRITICO" and previous["severity"] != "CRITICO"):
                    return None
            payload = {**alert, "id": uuid.uuid4().hex, "run_id": run_id, "received_at": now.isoformat(),
                       "processing_latency_ms": round((time.perf_counter() - started) * 1000, 3) if started is not None else None}
            db.execute("INSERT INTO alerts VALUES (?, ?, ?, ?, ?, ?, ?, NULL)",
                       (payload["id"], run_id, episode, source.timestamp(), alert["severity"],
                        json.dumps(payload, ensure_ascii=False), now.isoformat()))
        return payload

    def list(self, limit=200):
        with self.connect() as db:
            rows = db.execute("SELECT payload, acknowledged_at FROM alerts ORDER BY received_at DESC LIMIT ?", (limit,)).fetchall()
        return [{**json.loads(row["payload"]), "acknowledged_at": row["acknowledged_at"]} for row in rows]

    def acknowledge(self, alert_id):
        with self.connect() as db:
            result = db.execute("UPDATE alerts SET acknowledged_at=COALESCE(acknowledged_at, ?) WHERE id=?",
                                (datetime.now(timezone.utc).isoformat(), alert_id))
        return result.rowcount > 0
