#!/usr/bin/env bash
set -euo pipefail

docker exec -i dashboard python3 - <<'PY'
import os
import sqlite3
import sys

db_path = "/workspace/data/metrics.db"

if not os.path.exists(db_path):
    print("metrics.db is missing", file=sys.stderr)
    raise SystemExit(1)

try:
    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute("PRAGMA busy_timeout=5000")
    row = conn.execute("PRAGMA integrity_check").fetchone()
    conn.close()
except sqlite3.Error as exc:
    print(f"integrity_check failed: {exc}", file=sys.stderr)
    raise SystemExit(1)

result = row[0] if row else "no result"
print(f"integrity_check: {result}")
raise SystemExit(0 if result == "ok" else 1)
PY
