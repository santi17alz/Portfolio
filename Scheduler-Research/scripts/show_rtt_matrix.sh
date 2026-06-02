#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <job_id>" >&2
  exit 2
fi

job_id="$1"

docker exec -i -e JOB_ID="$job_id" dashboard python3 - <<'PY'
import os
import sqlite3

job_id = os.environ["JOB_ID"]
conn = sqlite3.connect("/workspace/data/metrics.db", timeout=10)
conn.execute("PRAGMA busy_timeout=5000")

exists = conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name='rtt_metrics'"
).fetchone()
if not exists:
    raise SystemExit("rtt_metrics table does not exist. Re-run dashboard/init_db.py.")

rows = conn.execute(
    """
    SELECT node_id,
           peer_node_id,
           COUNT(*) AS rows_seen,
           AVG(rtt_ms) AS avg_rtt_ms,
           MAX(rtt_ms) AS max_rtt_ms,
           SUM(CASE WHEN rtt_ms IS NULL THEN 1 ELSE 0 END) AS timeout_count
    FROM rtt_metrics
    WHERE job_id = ?
    GROUP BY node_id, peer_node_id
    ORDER BY node_id, peer_node_id
    """,
    (job_id,),
).fetchall()
conn.close()

if not rows:
    raise SystemExit(f"No per-peer RTT rows found for job {job_id!r}.")

max_avg = max((row[3] or 0.0) for row in rows)
warn_threshold = max(10.0, max_avg * 0.75)

print(f"RTT matrix for job {job_id}")
print()
print(f"{'from':<10} {'to':<10} {'rows':>5} {'avg_rtt_ms':>12} {'max_rtt_ms':>12} {'timeouts':>9} {'status':>10}")
for node_id, peer_node_id, rows_seen, avg_rtt, max_rtt, timeouts in rows:
    def fmt(value):
        return "timeout" if value is None else f"{value:.2f}"

    if timeouts:
        status = "timeout"
    elif avg_rtt is not None and avg_rtt >= warn_threshold:
        status = "high"
    else:
        status = "ok"

    print(
        f"{node_id:<10} {peer_node_id:<10} {rows_seen:>5} "
        f"{fmt(avg_rtt):>12} {fmt(max_rtt):>12} {timeouts:>9} {status:>10}"
    )
PY
