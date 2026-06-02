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
conn = sqlite3.connect("/workspace/data/metrics.db")
rows = conn.execute(
    """
    SELECT node_id,
           COUNT(*) AS row_count,
           AVG(all_reduce_ms) AS avg_all_reduce_ms,
           AVG(rtt_ms) AS avg_rtt_ms,
           AVG(nic_bytes_sent) / 1000000.0 AS avg_sent_mb,
           AVG(nic_bytes_recv) / 1000000.0 AS avg_recv_mb,
           AVG(nic_bytes_sent + nic_bytes_recv) / 1000000.0 AS avg_total_mb
    FROM metrics
    WHERE job_id = ?
    GROUP BY node_id
    ORDER BY node_id
    """,
    (job_id,),
).fetchall()
conn.close()

if not rows:
    raise SystemExit(f"No metrics found for job {job_id!r}.")

print(f"Raw averages for job {job_id}")
print(
    f"{'node_id':<10} {'rows':>6} {'all_reduce_ms':>15} "
    f"{'rtt_ms':>10} {'sent_MB/s':>12} {'recv_MB/s':>12} {'total_MB/s':>12}"
)
for node_id, count, ar, rtt, sent, recv, total in rows:
    def fmt(value):
        return "-" if value is None else f"{value:.2f}"

    print(
        f"{node_id:<10} {count:>6} {fmt(ar):>15} "
        f"{fmt(rtt):>10} {fmt(sent):>12} {fmt(recv):>12} {fmt(total):>12}"
    )
PY
