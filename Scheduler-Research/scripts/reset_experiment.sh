#!/usr/bin/env bash
set -euo pipefail

clear_logs=0
if [[ "${1:-}" == "--logs" ]]; then
  clear_logs=1
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--logs]" >&2
  exit 2
fi

restart_scorer=0
if docker compose ps --status running --services 2>/dev/null | grep -qx "scorer"; then
  restart_scorer=1
fi

restart_scorer_if_needed() {
  if [[ "$restart_scorer" -eq 1 ]]; then
    echo "Restarting scorer..."
    docker compose start scorer >/dev/null 2>&1 || true
    restart_scorer=0
  fi
}

trap restart_scorer_if_needed EXIT

recover_db() {
  local ts backup_dir
  ts="$(date +%Y%m%d_%H%M%S)"
  backup_dir="data/bad_db_backups/${ts}"
  mkdir -p "$backup_dir"

  echo "Recovering malformed database. Backing up DB files to ${backup_dir}/"
  for file in data/metrics.db data/metrics.db-wal data/metrics.db-shm; do
    if [[ -e "$file" ]]; then
      cp "$file" "$backup_dir/"
      rm -f "$file"
    fi
  done

  docker exec dashboard python3 dashboard/init_db.py
  echo "Database recreated from dashboard/init_db.py."
}

echo "Temporarily stopping scorer during reset..."
docker compose stop scorer >/dev/null 2>&1 || true

if [[ ! -e data/metrics.db ]]; then
  echo "metrics.db is missing; initializing a fresh database."
  docker exec dashboard python3 dashboard/init_db.py
elif ! ./scripts/check_db.sh >/tmp/telemetry_db_check.out 2>/tmp/telemetry_db_check.err; then
  cat /tmp/telemetry_db_check.out || true
  cat /tmp/telemetry_db_check.err >&2 || true
  recover_db
else
  cat /tmp/telemetry_db_check.out
fi

docker exec dashboard python3 dashboard/init_db.py >/dev/null

docker exec -i dashboard python3 - <<'PY'
import os
import fcntl
import sqlite3

db_path = "/workspace/data/metrics.db"
lock_path = "/workspace/data/metrics.db.lock"
sync_dir = "/workspace/data/sync"

with open(lock_path, "w") as lock_file:
    fcntl.flock(lock_file, fcntl.LOCK_EX)
    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    for table in ("metrics", "health_scores", "rtt_metrics"):
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        if exists:
            conn.execute(f"DELETE FROM {table}")
    conn.commit()
    conn.close()
    fcntl.flock(lock_file, fcntl.LOCK_UN)

os.makedirs(sync_dir, exist_ok=True)
for name in os.listdir(sync_dir):
    path = os.path.join(sync_dir, name)
    if os.path.isfile(path) or os.path.islink(path):
        os.unlink(path)

print("Cleared metrics, health_scores, and sync files.")
PY

restart_scorer_if_needed

if [[ "$clear_logs" -eq 1 ]]; then
  mkdir -p results
  find results -maxdepth 1 -type f -name "*.log" -delete
  echo "Cleared results/*.log."
fi

echo "Experiment reset complete. MNIST data was left untouched."
