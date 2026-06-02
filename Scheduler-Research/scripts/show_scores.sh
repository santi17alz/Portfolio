#!/usr/bin/env bash
set -euo pipefail

docker exec -i dashboard python3 - <<'PY'
import sqlite3

conn = sqlite3.connect("/workspace/data/metrics.db", timeout=10)
conn.execute("PRAGMA busy_timeout=5000")

rows = conn.execute(
    """
    SELECT node_id, current_score, last_updated, total_jobs
    FROM health_scores
    ORDER BY current_score DESC, node_id
    """
).fetchall()

if not rows:
    print("No health scores yet.")
else:
    print("Current health scores")
    print(f"{'node_id':<10} {'score':>8} {'total_jobs':>10} {'last_updated':>14}")
    for node_id, score, last_updated, total_jobs in rows:
        print(f"{node_id:<10} {score:>8.3f} {total_jobs:>10} {last_updated:>14.0f}")

jobs = conn.execute(
    """
    SELECT job_id,
           COUNT(*) AS rows,
           COUNT(DISTINCT node_id) AS nodes,
           AVG(all_reduce_ms) AS avg_ar,
           AVG(rtt_ms) AS avg_rtt,
           AVG(nic_bytes_sent + nic_bytes_recv) / 1000000.0 AS avg_nic
    FROM metrics
    GROUP BY job_id
    ORDER BY MAX(timestamp) DESC
    LIMIT 10
    """
).fetchall()

if jobs:
    print()
    print("Recent job averages")
    print(f"{'job_id':<28} {'rows':>5} {'nodes':>5} {'all_reduce':>12} {'rtt':>9} {'nic_MB/s':>10}")
    for job_id, rows, nodes, avg_ar, avg_rtt, avg_nic in jobs:
        def fmt(value):
            return "-" if value is None else f"{value:.2f}"

        print(
            f"{job_id:<28} {rows:>5} {nodes:>5} "
            f"{fmt(avg_ar):>12} {fmt(avg_rtt):>9} {fmt(avg_nic):>10}"
        )

conn.close()
PY
