"""
Initializes the SQLite database with our two tables.
Run this once before starting the system.
Safe to run multiple times — uses CREATE TABLE IF NOT EXISTS.
"""
import sqlite3
import os
import fcntl

DB_PATH = '/workspace/data/metrics.db'
DB_LOCK_PATH = '/workspace/data/metrics.db.lock'


class db_write_lock:
    def __enter__(self):
        os.makedirs(os.path.dirname(DB_LOCK_PATH), exist_ok=True)
        self.lock_file = open(DB_LOCK_PATH, 'w')
        fcntl.flock(self.lock_file, fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, tb):
        fcntl.flock(self.lock_file, fcntl.LOCK_UN)
        self.lock_file.close()

def init_database():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with db_write_lock():
        conn = sqlite3.connect(DB_PATH, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        cur  = conn.cursor()

        # Raw time-series metrics from telemetry agent
        cur.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       REAL    NOT NULL,
                node_id         TEXT    NOT NULL,
                job_id          TEXT    NOT NULL,
                nic_bytes_sent  INTEGER,
                nic_bytes_recv  INTEGER,
                all_reduce_ms   REAL,
                rtt_ms          REAL,
                epoch           INTEGER
            )
        """)

        # Indexes to make dashboard queries fast
        cur.execute("CREATE INDEX IF NOT EXISTS idx_metrics_node ON metrics(node_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_metrics_job  ON metrics(job_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_metrics_time ON metrics(timestamp)")

        # Computed health scores (one row per node, updated over time)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS health_scores (
                node_id        TEXT    PRIMARY KEY,
                current_score  REAL    NOT NULL,
                last_updated   REAL    NOT NULL,
                total_jobs     INTEGER DEFAULT 0
            )
        """)

        # Per-peer RTT measurements captured during each epoch idle window.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS rtt_metrics (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       REAL    NOT NULL,
                job_id          TEXT    NOT NULL,
                node_id         TEXT    NOT NULL,
                peer_node_id    TEXT    NOT NULL,
                rtt_ms          REAL,
                epoch           INTEGER
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rtt_job  ON rtt_metrics(job_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rtt_node ON rtt_metrics(node_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rtt_peer ON rtt_metrics(peer_node_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rtt_time ON rtt_metrics(timestamp)")

        conn.commit()
        conn.close()
    print(f"Database initialized at {DB_PATH}")

if __name__ == "__main__":
    init_database()
