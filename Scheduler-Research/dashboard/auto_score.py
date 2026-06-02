"""
Automatic health score updater.

Runs in the scorer container and periodically recomputes health_scores from
the shared metrics database. The recomputation is idempotent: each cycle clears
health_scores and rebuilds the table from all metrics jobs using the existing
health_scores.py formula.
"""

import os
import sqlite3
import sys
import time

sys.path.insert(0, "/workspace/dashboard")

from health_scores import DB_PATH, recompute_health_scores

SLEEP_SECONDS = float(os.environ.get("SCORER_INTERVAL_SECONDS", "3"))


def log(message):
    print(f"[scorer] {message}", flush=True)


def wait_reason():
    if not os.path.exists(DB_PATH):
        return "waiting for metrics.db"

    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return f"waiting for readable database ({exc})"

    missing = [name for name in ("metrics", "health_scores") if name not in tables]
    if missing:
        return f"waiting for tables: {', '.join(missing)}"
    return None


def main():
    last_status = None

    while True:
        reason = wait_reason()
        if reason:
            if reason != last_status:
                log(reason)
                last_status = reason
            time.sleep(SLEEP_SECONDS)
            continue

        try:
            result = recompute_health_scores(DB_PATH, verbose=False)
            status = (
                f"processed {result['jobs_processed']} jobs; "
                f"updated {result['scores_written']} scores"
            )
            if status != last_status:
                log(status)
                last_status = status
        except (FileNotFoundError, RuntimeError, sqlite3.Error) as exc:
            status = f"waiting for scoreable metrics ({exc})"
            if status != last_status:
                log(status)
                last_status = status

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()
