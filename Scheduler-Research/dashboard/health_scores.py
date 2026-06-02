"""
Health score computation.

Higher score means healthier. Each job produces a per-node score from:
  - all_reduce_ms: lower is better, but DDP all-reduce is collective, so every
    node can slow down together when one path is degraded.
  - rtt_ms: lower is better and is more useful for identifying communication
    path problems.
  - NIC total bytes/sec: higher useful training throughput is generally better,
    but background traffic can make raw NIC bytes misleading.

The old pure min-max scoring was too harsh: tiny baseline differences could
force one healthy node to 0.0. This version applies deadbands before min-max
normalization. If nodes are effectively tied on a signal, every node receives
full credit for that signal.

Historical scores are still aggregated with EMA:
  H_hist(node) = alpha * H(job) + (1-alpha) * H_hist_prev(node)
"""
import os
import fcntl
import sqlite3
import time

DB_PATH = '/workspace/data/metrics.db'
DB_LOCK_PATH = '/workspace/data/metrics.db.lock'

# Weights for the three signals (must sum to 1.0).
W_ALL_REDUCE = 0.5
W_NIC        = 0.3
W_RTT        = 0.2

# Backward-compatible aliases for older notes/scripts.
W_THROUGHPUT = W_NIC
W_LATENCY    = W_RTT

# Deadbands prevent baseline noise from creating fake stragglers.
ALL_REDUCE_DEADBAND_MS = 2.0
RTT_DEADBAND_MS        = 1.0
NIC_DEADBAND_FRAC      = 0.10

ALPHA   = 0.3    # EMA: 30% new job, 70% history
EPSILON = 1e-9


class db_write_lock:
    def __enter__(self):
        os.makedirs(os.path.dirname(DB_LOCK_PATH), exist_ok=True)
        self.lock_file = open(DB_LOCK_PATH, 'w')
        fcntl.flock(self.lock_file, fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, tb):
        fcntl.flock(self.lock_file, fcntl.LOCK_UN)
        self.lock_file.close()


def normalize(value, min_v, max_v, lower_is_better=True):
    """Min-max normalize to [0,1]. Higher output = healthier node."""
    if max_v - min_v < EPSILON:
        return 1.0  # no variance — all nodes equal on this signal
    score = (value - min_v) / (max_v - min_v)
    return 1.0 - score if lower_is_better else score


def median(values):
    sorted_values = sorted(values)
    n = len(sorted_values)
    mid = n // 2
    if n == 0:
        return 0.0
    if n % 2:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def score_metric(values, lower_is_better, deadband_abs=None, deadband_frac=None):
    """
    Score one metric across nodes.

    If the spread is inside the deadband, every node gets 1.0 for this signal.
    Otherwise we fall back to min-max normalization.
    """
    raw_values = list(values.values())
    min_v = min(raw_values)
    max_v = max(raw_values)
    spread = max_v - min_v
    med = median(raw_values)

    if deadband_abs is not None:
        threshold = deadband_abs
    else:
        threshold = abs(med) * (deadband_frac or 0.0)

    in_deadband = spread <= threshold + EPSILON
    if in_deadband:
        scores = {node_id: 1.0 for node_id in values}
    else:
        scores = {
            node_id: normalize(value, min_v, max_v, lower_is_better=lower_is_better)
            for node_id, value in values.items()
        }

    return scores, {
        "min": min_v,
        "max": max_v,
        "median": med,
        "spread": spread,
        "threshold": threshold,
        "in_deadband": in_deadband,
    }


def compute_job_score(conn, job_id, include_details=False):
    """Compute health score for each node for a single job."""
    rows = conn.execute("""
        SELECT node_id,
               AVG(all_reduce_ms),
               AVG(nic_bytes_sent + nic_bytes_recv),
               AVG(rtt_ms)
        FROM metrics
        WHERE job_id = ?
        GROUP BY node_id
    """, (job_id,)).fetchall()

    if not rows:
        return ({}, None) if include_details else {}

    measured_ar = [r[1] for r in rows if r[1] is not None]
    measured_tp = [r[2] for r in rows if r[2] is not None]
    measured_rtts = [r[3] for r in rows if r[3] is not None]

    ar_penalty = (max(measured_ar) * 2.0) if measured_ar else 1000.0
    tp_penalty = 0.0
    rtt_penalty = (max(measured_rtts) * 2.0) if measured_rtts else 100.0

    node_values = {}
    ar_values = {}
    nic_values = {}
    rtt_values = {}

    for node_id, ar, tp, rtt in rows:
        ar_clean = ar if ar is not None else ar_penalty
        tp_clean = tp if tp is not None else tp_penalty
        rtt_clean = rtt if rtt is not None else rtt_penalty

        node_values[node_id] = {
            "avg_all_reduce_ms": ar,
            "avg_rtt_ms": rtt,
            "avg_nic_total_bps": tp,
            "scored_all_reduce_ms": ar_clean,
            "scored_rtt_ms": rtt_clean,
            "scored_nic_total_bps": tp_clean,
        }
        ar_values[node_id] = ar_clean
        nic_values[node_id] = tp_clean
        rtt_values[node_id] = rtt_clean

    ar_scores, ar_meta = score_metric(
        ar_values,
        lower_is_better=True,
        deadband_abs=ALL_REDUCE_DEADBAND_MS,
    )
    rtt_scores, rtt_meta = score_metric(
        rtt_values,
        lower_is_better=True,
        deadband_abs=RTT_DEADBAND_MS,
    )
    nic_scores, nic_meta = score_metric(
        nic_values,
        lower_is_better=False,
        deadband_frac=NIC_DEADBAND_FRAC,
    )

    scores = {}
    details = {
        "metrics": {
            "all_reduce": ar_meta,
            "rtt": rtt_meta,
            "nic": nic_meta,
        },
        "nodes": {},
    }

    for node_id in node_values:
        score = (
            W_ALL_REDUCE * ar_scores[node_id]
            + W_NIC * nic_scores[node_id]
            + W_RTT * rtt_scores[node_id]
        )
        scores[node_id] = round(score, 3)
        details["nodes"][node_id] = {
            **node_values[node_id],
            "all_reduce_component": ar_scores[node_id],
            "nic_component": nic_scores[node_id],
            "rtt_component": rtt_scores[node_id],
            "score": scores[node_id],
        }

    if include_details:
        return scores, details
    return scores


def update_historical_scores(conn, job_scores, commit=True):
    """Apply EMA to update each node's long-term score."""
    now = time.time()
    for node_id, new_score in job_scores.items():
        row = conn.execute(
            "SELECT current_score, total_jobs FROM health_scores WHERE node_id = ?",
            (node_id,)
        ).fetchone()

        if row is None:
            conn.execute("""
                INSERT INTO health_scores (node_id, current_score, last_updated, total_jobs)
                VALUES (?, ?, ?, 1)
            """, (node_id, new_score, now))
        else:
            old_score, total_jobs = row
            updated = ALPHA * new_score + (1 - ALPHA) * old_score
            conn.execute("""
                UPDATE health_scores
                SET current_score = ?, last_updated = ?, total_jobs = ?
                WHERE node_id = ?
            """, (updated, now, total_jobs + 1, node_id))

    if commit:
        conn.commit()


def fmt_value(value, decimals=2):
    if value is None:
        return "-"
    return f"{value:.{decimals}f}"


def fmt_deadband(meta, unit, decimals=2):
    state = "inside" if meta["in_deadband"] else "outside"
    return (
        f"{state} deadband "
        f"(spread={meta['spread']:.{decimals}f}{unit}, "
        f"threshold={meta['threshold']:.{decimals}f}{unit})"
    )


def print_job_details(job_id, scores, details):
    print(f"Job {job_id}:")
    print(f"  all_reduce: {fmt_deadband(details['metrics']['all_reduce'], 'ms')}")
    print(f"  rtt:        {fmt_deadband(details['metrics']['rtt'], 'ms')}")
    nic_meta = {
        **details["metrics"]["nic"],
        "spread": details["metrics"]["nic"]["spread"] / 1e6,
        "threshold": details["metrics"]["nic"]["threshold"] / 1e6,
    }
    print(f"  nic total:  {fmt_deadband(nic_meta, ' MB/s')}")
    print(
        "  "
        f"{'node':<8} {'score':>6} {'all_reduce':>12} {'rtt':>9} "
        f"{'nic_total':>11} {'ar_c':>6} {'rtt_c':>6} {'nic_c':>6}"
    )
    for node_id in sorted(scores):
        node = details["nodes"][node_id]
        nic_mb = None
        if node["avg_nic_total_bps"] is not None:
            nic_mb = node["avg_nic_total_bps"] / 1e6
        print(
            "  "
            f"{node_id:<8} {scores[node_id]:>6.3f} "
            f"{fmt_value(node['avg_all_reduce_ms']):>12} "
            f"{fmt_value(node['avg_rtt_ms']):>9} "
            f"{fmt_value(nic_mb):>11} "
            f"{node['all_reduce_component']:>6.3f} "
            f"{node['rtt_component']:>6.3f} "
            f"{node['nic_component']:>6.3f}"
        )
    print()


def recompute_health_scores(db_path=DB_PATH, verbose=True):
    """Recompute health_scores from all jobs using the existing formula."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(db_path)

    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        metrics_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='metrics'"
        ).fetchone()
        health_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='health_scores'"
        ).fetchone()

        if metrics_exists is None:
            raise RuntimeError("metrics table does not exist")
        if health_exists is None:
            raise RuntimeError("health_scores table does not exist")

        job_ids = [r[0] for r in conn.execute(
            "SELECT job_id FROM metrics GROUP BY job_id ORDER BY MIN(timestamp)"
        ).fetchall()]

        if verbose:
            print(f"Processing {len(job_ids)} jobs...\n")

        with db_write_lock():
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM health_scores")
            for job_id in job_ids:
                if verbose:
                    scores, details = compute_job_score(conn, job_id, include_details=True)
                else:
                    scores = compute_job_score(conn, job_id)
                    details = None
                update_historical_scores(conn, scores, commit=False)
                if verbose and details:
                    print_job_details(job_id, scores, details)
            conn.commit()

        rows = conn.execute("""
            SELECT node_id, current_score, total_jobs
            FROM health_scores
            ORDER BY current_score DESC
        """).fetchall()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    if verbose:
        print("=" * 50)
        print("FINAL HISTORICAL HEALTH SCORES")
        print("=" * 50)

        for node_id, score, jobs in rows:
            bar = '█' * int(score * 30)
            print(f"  {node_id}: {score:.3f}  {bar}  ({jobs} jobs)")

    return {
        "jobs_processed": len(job_ids),
        "scores_written": len(rows),
    }


def process_all_jobs():
    recompute_health_scores(DB_PATH, verbose=True)


if __name__ == "__main__":
    process_all_jobs()
