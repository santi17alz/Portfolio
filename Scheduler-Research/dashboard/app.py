"""
Real-time Flask dashboard for the telemetry system.

Runs inside the dashboard container:
  python3 dashboard/app.py

Open:
  http://localhost:5050
"""

import os
import sqlite3
import time

from flask import Flask, jsonify, render_template_string, request

try:
    from recommend import EMPTY_RECOMMENDATION, compute_recommendation
except ImportError:
    from dashboard.recommend import EMPTY_RECOMMENDATION, compute_recommendation

app = Flask(__name__)
DB_PATH = "/workspace/data/metrics.db"
POLL_SECONDS = 2
HEALTH_STRAGGLER_THRESHOLD = 0.75
RTT_ELEVATED_THRESHOLD_MS = 10.0
ALL_REDUCE_SLOW_THRESHOLD_MS = 100.0
ALL_REDUCE_BASELINE_MULTIPLIER = 3.0


def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def table_exists(conn, table_name):
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def fmt_time(ts):
    if ts is None:
        return "-"
    try:
        return time.strftime("%H:%M:%S", time.localtime(float(ts)))
    except Exception:
        return "-"


def fmt_datetime(ts):
    if ts is None:
        return "-"
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(ts)))
    except Exception:
        return "-"


def fmt_num(value, decimals=2):
    if value is None:
        return "-"
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return "-"


def score_class(score):
    if score is None:
        return "muted"
    if score >= 0.75:
        return "good"
    if score >= 0.50:
        return "warn"
    return "bad"


def score_label(score):
    if score is None:
        return "No score"
    if score >= 0.75:
        return "Healthy"
    if score >= 0.50:
        return "Watch"
    return "Straggler risk"


def metric_status(rtt_ms, all_reduce_ms):
    if rtt_ms is None:
        return "bad", "timeout"
    if rtt_ms > 10:
        return "warn", "slow RTT"
    if all_reduce_ms is not None and all_reduce_ms > 100:
        return "warn", "slow sync"
    return "good", "healthy"


def metric_payload(row):
    nic_sent_mb = (row["nic_bytes_sent"] or 0) / 1e6
    nic_recv_mb = (row["nic_bytes_recv"] or 0) / 1e6
    total_mb = nic_sent_mb + nic_recv_mb
    cls, label = metric_status(row["rtt_ms"], row["all_reduce_ms"])
    return {
        "timestamp": row["timestamp"],
        "timestamp_fmt": fmt_time(row["timestamp"]),
        "node_id": row["node_id"],
        "job_id": row["job_id"],
        "epoch": row["epoch"],
        "epoch_fmt": row["epoch"] if row["epoch"] is not None else "-",
        "all_reduce_ms": row["all_reduce_ms"],
        "all_reduce_ms_fmt": fmt_num(row["all_reduce_ms"], 2),
        "rtt_ms": row["rtt_ms"],
        "rtt_ms_fmt": fmt_num(row["rtt_ms"], 2) if row["rtt_ms"] is not None else "timeout",
        "nic_bytes_sent": row["nic_bytes_sent"],
        "nic_bytes_recv": row["nic_bytes_recv"],
        "nic_sent_mb": nic_sent_mb,
        "nic_recv_mb": nic_recv_mb,
        "nic_total_mb": total_mb,
        "nic_sent_mb_fmt": fmt_num(nic_sent_mb, 2),
        "nic_recv_mb_fmt": fmt_num(nic_recv_mb, 2),
        "nic_total_mb_fmt": fmt_num(total_mb, 2),
        "status_class": cls,
        "status_label": label,
    }


def empty_dashboard_payload(error=None, hint=None):
    return {
        "ready": error is None,
        "error": error,
        "hint": hint,
        "jobs": [],
        "selected_job": None,
        "summary": {
            "job_count": 0,
            "node_count": 0,
            "metric_rows": 0,
            "last_seen": None,
            "last_seen_fmt": "-",
        },
        "selected_summary": None,
        "latest_by_node": [],
        "recent_metrics": [],
        "health_scores": [],
        "health_available": False,
        "recommendation": EMPTY_RECOMMENDATION,
        "diagnosis": None,
        "current_job_status": None,
        "generated_at": time.time(),
        "generated_at_fmt": fmt_time(time.time()),
    }


def build_current_job_status(latest_by_node, baseline_all_reduce_ms=None):
    if not latest_by_node:
        return {
            "status_class": "neutral",
            "messages": ["No live node metrics for the selected job yet."],
            "max_rtt_ms": None,
            "max_all_reduce_ms": None,
            "all_reduce_threshold_ms": ALL_REDUCE_SLOW_THRESHOLD_MS,
        }

    rtts = [row["rtt_ms"] for row in latest_by_node if row["rtt_ms"] is not None]
    all_reduces = [row["all_reduce_ms"] for row in latest_by_node if row["all_reduce_ms"] is not None]
    max_rtt = max(rtts) if rtts else None
    max_all_reduce = max(all_reduces) if all_reduces else None
    ar_threshold = ALL_REDUCE_SLOW_THRESHOLD_MS
    if baseline_all_reduce_ms is not None:
        ar_threshold = min(ar_threshold, baseline_all_reduce_ms * ALL_REDUCE_BASELINE_MULTIPLIER)

    messages = []
    status_class = "good"
    if max_rtt is not None and max_rtt > RTT_ELEVATED_THRESHOLD_MS:
        messages.append("Current job is experiencing elevated RTT.")
        status_class = "warn"
    if max_all_reduce is not None and max_all_reduce > ar_threshold:
        messages.append("Current job is experiencing collective all-reduce slowdown.")
        status_class = "warn"

    if not messages:
        messages.append("Current job metrics are within configured RTT and all-reduce thresholds.")

    return {
        "status_class": status_class,
        "messages": messages,
        "max_rtt_ms": max_rtt,
        "max_rtt_ms_fmt": fmt_num(max_rtt, 2),
        "max_all_reduce_ms": max_all_reduce,
        "max_all_reduce_ms_fmt": fmt_num(max_all_reduce, 2),
        "all_reduce_threshold_ms": ar_threshold,
        "all_reduce_threshold_ms_fmt": fmt_num(ar_threshold, 2),
    }


def load_dashboard_data(selected_job=None, target_node_count=None):
    if not os.path.exists(DB_PATH):
        return empty_dashboard_payload(
            "Database file not found.",
            "Run: docker exec -it dashboard python3 dashboard/init_db.py",
        )

    try:
        conn = get_conn()
    except sqlite3.Error as exc:
        return empty_dashboard_payload(f"Could not open database: {exc}", None)

    try:
        if not table_exists(conn, "metrics"):
            return empty_dashboard_payload(
                "The metrics table is missing.",
                "Run: docker exec -it dashboard python3 dashboard/init_db.py",
            )

        health_available = table_exists(conn, "health_scores")

        job_rows = conn.execute(
            """
            SELECT job_id,
                   COUNT(*) AS metric_rows,
                   MIN(timestamp) AS started,
                   MAX(timestamp) AS ended,
                   COUNT(DISTINCT node_id) AS node_count,
                   MAX(epoch) AS max_epoch
            FROM metrics
            GROUP BY job_id
            ORDER BY ended DESC
            """
        ).fetchall()

        jobs = []
        for row in job_rows:
            jobs.append({
                "job_id": row["job_id"],
                "metric_rows": row["metric_rows"],
                "started": row["started"],
                "started_fmt": fmt_datetime(row["started"]),
                "ended": row["ended"],
                "ended_fmt": fmt_datetime(row["ended"]),
                "node_count": row["node_count"],
                "max_epoch": row["max_epoch"],
            })

        job_ids = [job["job_id"] for job in jobs]
        if selected_job not in job_ids:
            selected_job = job_ids[0] if job_ids else None

        summary_row = conn.execute(
            """
            SELECT COUNT(*) AS metric_rows,
                   COUNT(DISTINCT job_id) AS job_count,
                   COUNT(DISTINCT node_id) AS node_count,
                   MAX(timestamp) AS last_seen
            FROM metrics
            """
        ).fetchone()

        selected_summary = None
        latest_by_node = []
        recent_metrics = []

        if selected_job:
            selected_summary_row = conn.execute(
                """
                SELECT job_id,
                       COUNT(*) AS metric_rows,
                       COUNT(DISTINCT node_id) AS node_count,
                       MIN(timestamp) AS started,
                       MAX(timestamp) AS ended,
                       MAX(epoch) AS max_epoch,
                       AVG(all_reduce_ms) AS avg_all_reduce_ms,
                       AVG(rtt_ms) AS avg_rtt_ms,
                       AVG(nic_bytes_sent + nic_bytes_recv) / 1000000.0 AS avg_nic_total_mb
                FROM metrics
                WHERE job_id = ?
                GROUP BY job_id
                """,
                (selected_job,),
            ).fetchone()

            if selected_summary_row:
                selected_summary = dict(selected_summary_row)
                selected_summary.update({
                    "started_fmt": fmt_datetime(selected_summary["started"]),
                    "ended_fmt": fmt_datetime(selected_summary["ended"]),
                    "avg_all_reduce_ms_fmt": fmt_num(selected_summary["avg_all_reduce_ms"], 2),
                    "avg_rtt_ms_fmt": fmt_num(selected_summary["avg_rtt_ms"], 2),
                    "avg_nic_total_mb_fmt": fmt_num(selected_summary["avg_nic_total_mb"], 2),
                })

            latest_rows = conn.execute(
                """
                SELECT m.*
                FROM metrics m
                JOIN (
                    SELECT node_id, MAX(timestamp) AS max_ts
                    FROM metrics
                    WHERE job_id = ?
                    GROUP BY node_id
                ) latest
                ON m.node_id = latest.node_id AND m.timestamp = latest.max_ts
                WHERE m.job_id = ?
                ORDER BY m.node_id
                """,
                (selected_job, selected_job),
            ).fetchall()
            latest_by_node = [metric_payload(row) for row in latest_rows]

            recent_rows = conn.execute(
                """
                SELECT timestamp, node_id, job_id, epoch,
                       all_reduce_ms, rtt_ms, nic_bytes_sent, nic_bytes_recv
                FROM metrics
                WHERE job_id = ?
                ORDER BY timestamp DESC, epoch DESC, node_id ASC
                LIMIT 300
                """,
                (selected_job,),
            ).fetchall()
            recent_metrics = [metric_payload(row) for row in recent_rows]

        baseline_row = conn.execute(
            """
            SELECT MIN(avg_all_reduce_ms) AS baseline_all_reduce_ms
            FROM (
                SELECT job_id, AVG(all_reduce_ms) AS avg_all_reduce_ms
                FROM metrics
                WHERE all_reduce_ms IS NOT NULL
                GROUP BY job_id
            )
            """
        ).fetchone()
        baseline_all_reduce_ms = baseline_row["baseline_all_reduce_ms"] if baseline_row else None
        current_job_status = build_current_job_status(latest_by_node, baseline_all_reduce_ms)

        health_scores = []
        if health_available:
            health_rows = conn.execute(
                """
                SELECT node_id, current_score, last_updated, total_jobs
                FROM health_scores
                ORDER BY current_score DESC
                """
            ).fetchall()
            for row in health_rows:
                score = float(row["current_score"])
                health_scores.append({
                    "node_id": row["node_id"],
                    "current_score": score,
                    "current_score_fmt": fmt_num(score, 3),
                    "score_pct": int(max(0, min(score, 1)) * 100),
                    "status_class": score_class(score),
                    "status_label": score_label(score),
                    "last_updated": row["last_updated"],
                    "last_updated_fmt": fmt_time(row["last_updated"]),
                    "total_jobs": row["total_jobs"],
                })

        weakest = health_scores[-1] if health_scores else None
        strongest = health_scores[0] if health_scores else None
        diagnosis = None
        if weakest and strongest:
            is_unhealthy = weakest["current_score"] < HEALTH_STRAGGLER_THRESHOLD
            recommendation_avoid = []
            diagnosis = {
                "has_unhealthy_node": is_unhealthy,
                "weakest_node": weakest["node_id"],
                "weakest_score": weakest["current_score"],
                "weakest_score_fmt": weakest["current_score_fmt"],
                "strongest_node": strongest["node_id"],
                "strongest_score": strongest["current_score"],
                "strongest_score_fmt": strongest["current_score_fmt"],
                "threshold": HEALTH_STRAGGLER_THRESHOLD,
                "threshold_fmt": fmt_num(HEALTH_STRAGGLER_THRESHOLD, 2),
                "message": (
                    f"Most likely straggler: {weakest['node_id']}."
                    if is_unhealthy
                    else f"No unhealthy node detected. Lowest relative score: {weakest['node_id']}."
                ),
                "recommendation_avoid_nodes": recommendation_avoid,
            }

        try:
            recommendation = compute_recommendation(
                conn,
                selected_job_id=selected_job,
                target_node_count=target_node_count,
            )
        except Exception as exc:
            recommendation = {
                "recommended_nodes": [],
                "avoid_nodes": [],
                "confidence": 0.0,
                "reason": f"Recommendation temporarily unavailable: {exc}",
                "mode": "no_data",
                "selected_job_id": selected_job,
                "signals": {},
            }
        if diagnosis:
            diagnosis["recommendation_avoid_nodes"] = recommendation.get("avoid_nodes", [])

        now = time.time()
        last_seen = summary_row["last_seen"] if summary_row else None
        return {
            "ready": True,
            "error": None,
            "hint": None,
            "jobs": jobs,
            "selected_job": selected_job,
            "summary": {
                "job_count": summary_row["job_count"] if summary_row else 0,
                "node_count": summary_row["node_count"] if summary_row else 0,
                "metric_rows": summary_row["metric_rows"] if summary_row else 0,
                "last_seen": last_seen,
                "last_seen_fmt": fmt_datetime(last_seen),
            },
            "selected_summary": selected_summary,
            "latest_by_node": latest_by_node,
            "recent_metrics": recent_metrics,
            "health_scores": health_scores,
            "health_available": health_available,
            "recommendation": recommendation,
            "diagnosis": diagnosis,
            "current_job_status": current_job_status,
            "generated_at": now,
            "generated_at_fmt": fmt_time(now),
        }
    except sqlite3.Error as exc:
        return empty_dashboard_payload(f"Database temporarily unavailable: {exc}", None)
    finally:
        conn.close()


TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Cluster Telemetry Dashboard</title>
  <style>
    :root {
      --bg: #090b10;
      --panel: rgba(22, 27, 38, 0.92);
      --panel-2: #111723;
      --line: rgba(148, 163, 184, 0.16);
      --text: #e5edf7;
      --muted: #8b98aa;
      --faint: #5e6a7d;
      --good: #33d17a;
      --warn: #facc15;
      --bad: #fb7185;
      --blue: #60a5fa;
      --shadow: 0 18px 60px rgba(0, 0, 0, 0.35);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(96,165,250,0.16), transparent 34rem),
        radial-gradient(circle at top right, rgba(51,209,122,0.10), transparent 28rem),
        var(--bg);
      color: var(--text);
      min-height: 100vh;
    }

    .wrap { max-width: 1320px; margin: 0 auto; padding: 28px; }
    .topbar {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 24px;
    }

    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 7px 10px;
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--muted);
      background: rgba(15, 23, 42, 0.6);
      font-size: 0.78rem;
      margin-bottom: 12px;
    }
    .pulse {
      width: 8px; height: 8px; border-radius: 999px; background: var(--good);
      box-shadow: 0 0 0 6px rgba(51, 209, 122, 0.12);
    }
    .pulse.stale { background: var(--warn); box-shadow: 0 0 0 6px rgba(250, 204, 21, 0.12); }
    .pulse.error { background: var(--bad); box-shadow: 0 0 0 6px rgba(251, 113, 133, 0.12); }

    h1 { margin: 0; font-size: clamp(2rem, 4vw, 3.6rem); line-height: 1; letter-spacing: -0.035em; }
    .subtitle { margin-top: 12px; max-width: 760px; color: var(--muted); line-height: 1.55; }
    .actions { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; justify-content: flex-end; }
    .select, .button {
      border: 1px solid var(--line);
      color: var(--text);
      background: rgba(17, 24, 39, 0.78);
      border-radius: 10px;
      padding: 11px 12px;
      font-size: 0.9rem;
      outline: none;
    }
    .button { text-decoration: none; cursor: pointer; }
    .button:hover, .select:hover { border-color: rgba(96, 165, 250, 0.55); }

    .grid { display: grid; gap: 16px; }
    .summary-grid { grid-template-columns: repeat(4, minmax(0, 1fr)); margin-bottom: 16px; }
    .health-grid { grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); margin-bottom: 16px; }
    .card, .section {
      background: linear-gradient(180deg, rgba(30, 41, 59, 0.78), rgba(15, 23, 42, 0.92));
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .card { padding: 18px; min-height: 112px; }
    .section { padding: 20px; margin-bottom: 16px; }

    .label { color: var(--muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700; }
    .big { font-size: 2rem; letter-spacing: -0.035em; font-weight: 800; margin-top: 8px; overflow-wrap: anywhere; }
    .hint { color: var(--faint); font-size: 0.82rem; margin-top: 8px; }
    .node-head { display: flex; justify-content: space-between; align-items: center; gap: 10px; margin-bottom: 14px; }
    .node-name { font-size: 1.05rem; font-weight: 800; }
    .score { font-size: 2.5rem; font-weight: 900; letter-spacing: -0.04em; }
    .good { color: var(--good); }
    .warn { color: var(--warn); }
    .bad { color: var(--bad); }
    .muted { color: var(--muted); }

    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 10px;
      border-radius: 999px;
      font-size: 0.74rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      border: 1px solid transparent;
      white-space: nowrap;
    }
    .pill.good { background: rgba(51,209,122,0.11); border-color: rgba(51,209,122,0.25); }
    .pill.warn { background: rgba(250,204,21,0.11); border-color: rgba(250,204,21,0.25); }
    .pill.bad { background: rgba(251,113,133,0.11); border-color: rgba(251,113,133,0.25); }
    .pill.neutral { color: var(--muted); background: rgba(148,163,184,0.08); border-color: var(--line); }
    .bar { width: 100%; height: 10px; border-radius: 999px; overflow: hidden; background: rgba(148, 163, 184, 0.14); margin: 12px 0; }
    .fill { height: 100%; border-radius: 999px; }
    .fill.good { background: linear-gradient(90deg, #16a34a, var(--good)); }
    .fill.warn { background: linear-gradient(90deg, #ca8a04, var(--warn)); }
    .fill.bad { background: linear-gradient(90deg, #e11d48, var(--bad)); }
    .recommendation-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); margin-bottom: 14px; }
    .mini-list { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }

    .section-title { display: flex; justify-content: space-between; align-items: end; gap: 16px; margin-bottom: 14px; }
    h2 { margin: 0; font-size: 1.05rem; letter-spacing: -0.02em; }
    .section-subtitle { color: var(--muted); font-size: 0.85rem; margin-top: 5px; }
    table { width: 100%; border-collapse: collapse; overflow: hidden; }
    th {
      text-align: left;
      color: var(--muted);
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      border-bottom: 1px solid var(--line);
      padding: 12px 10px;
      white-space: nowrap;
    }
    td {
      padding: 13px 10px;
      border-bottom: 1px solid rgba(148, 163, 184, 0.08);
      color: #dce7f5;
      font-size: 0.9rem;
    }
    tr:hover td { background: rgba(96, 165, 250, 0.045); }
    tr:last-child td { border-bottom: none; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
    .right { text-align: right; }
    .empty {
      padding: 30px;
      color: var(--muted);
      background: rgba(15, 23, 42, 0.5);
      border: 1px dashed var(--line);
      border-radius: 8px;
      text-align: center;
    }
    .callout {
      border: 1px solid rgba(96, 165, 250, 0.28);
      background: rgba(96, 165, 250, 0.09);
      color: #cfe3ff;
      padding: 14px 16px;
      border-radius: 8px;
      line-height: 1.45;
    }
    code {
      background: rgba(0,0,0,0.28);
      border: 1px solid rgba(255,255,255,0.08);
      padding: 2px 6px;
      border-radius: 7px;
      color: #eaf2ff;
    }
    .hidden { display: none !important; }

    @media (max-width: 920px) {
      .topbar { flex-direction: column; }
      .summary-grid, .health-grid { grid-template-columns: 1fr; }
      .actions { justify-content: flex-start; }
      .wrap { padding: 18px; }
      table { display: block; overflow-x: auto; white-space: nowrap; }
    }
  </style>
</head>
<body>
  <main class="wrap">
    <div id="not-ready" class="hidden">
      <div class="topbar">
        <div>
          <div class="eyebrow"><span class="pulse error"></span> Telemetry Dashboard</div>
          <h1>Database not ready</h1>
          <p id="error-message" class="subtitle"></p>
        </div>
      </div>
      <div id="error-hint" class="callout"></div>
    </div>

    <div id="dashboard" class="hidden">
      <div class="topbar">
        <div>
          <div class="eyebrow"><span id="live-pulse" class="pulse"></span><span id="live-label">Live</span> · <span id="last-polled">waiting</span></div>
          <h1>Training Cluster Health</h1>
          <p class="subtitle">
            Node-level view of all-reduce timing, inter-node RTT, and NIC throughput from the shared SQLite telemetry database.
          </p>
        </div>
        <div class="actions">
          <select id="job-select" class="select" aria-label="Select job"></select>
          <button id="latest-job" class="button" type="button">Latest job</button>
        </div>
      </div>

      <section class="grid summary-grid">
        <div class="card">
          <div class="label">Jobs tracked</div>
          <div id="global-jobs" class="big">-</div>
          <div class="hint">Unique training job IDs</div>
        </div>
        <div class="card">
          <div class="label">Nodes reporting</div>
          <div id="global-nodes" class="big">-</div>
          <div class="hint">Distinct node IDs in metrics</div>
        </div>
        <div class="card">
          <div class="label">Metric rows</div>
          <div id="global-rows" class="big">-</div>
          <div class="hint">Raw telemetry samples</div>
        </div>
        <div class="card">
          <div class="label">Last update</div>
          <div id="global-last-update" class="big" style="font-size:1.25rem; letter-spacing:0;">-</div>
          <div class="hint">From metrics table</div>
        </div>
      </section>

      <section id="selected-overview" class="section hidden">
        <div class="section-title">
          <div>
            <h2>Selected Job Overview</h2>
            <div id="selected-subtitle" class="section-subtitle"></div>
          </div>
          <span id="selected-epochs" class="pill neutral">0 epochs</span>
        </div>
        <div class="grid summary-grid">
          <div class="card">
            <div class="label">Avg all-reduce</div>
            <div id="avg-all-reduce" class="big">-</div>
            <div class="hint">milliseconds</div>
          </div>
          <div class="card">
            <div class="label">Avg RTT</div>
            <div id="avg-rtt" class="big">-</div>
            <div class="hint">milliseconds</div>
          </div>
          <div class="card">
            <div class="label">Avg NIC total</div>
            <div id="avg-nic" class="big">-</div>
            <div class="hint">MB/s sent + received</div>
          </div>
          <div class="card">
            <div class="label">Samples</div>
            <div id="selected-samples" class="big">-</div>
            <div class="hint">Rows for this job</div>
          </div>
        </div>
      </section>

      <section id="empty-state" class="section hidden">
        <div class="empty">No metrics found yet. Start a training job and this page will update automatically.</div>
      </section>

      <section id="health-section" class="grid health-grid"></section>

      <section class="section">
        <div class="section-title">
          <div>
            <h2>Scheduler Recommendation</h2>
            <div class="section-subtitle">Advisory only. Uses health scores and recent per-peer RTT paths.</div>
          </div>
          <span id="recommendation-confidence" class="pill neutral">0% confidence</span>
        </div>
        <div id="recommendation-section"></div>
      </section>

      <section id="diagnosis-section" class="section hidden">
        <div class="section-title">
          <div>
            <h2>Cluster Diagnosis</h2>
            <div class="section-subtitle">Health-score interpretation plus current selected-job status.</div>
          </div>
        </div>
        <div id="diagnosis" class="callout"></div>
        <div id="current-job-status" class="callout" style="margin-top: 12px;"></div>
      </section>

      <section class="section">
        <div class="section-title">
          <div>
            <h2>Latest Node Metrics</h2>
            <div class="section-subtitle">Most recent telemetry row per node for the selected job.</div>
          </div>
        </div>
        <div id="latest-node-metrics"></div>
      </section>

      <section class="section">
        <div class="section-title">
          <div>
            <h2>Recent Per-Epoch Metrics</h2>
            <div class="section-subtitle">Newest samples first. Use the job selector above to switch runs.</div>
          </div>
        </div>
        <div id="recent-metrics"></div>
      </section>
    </div>
  </main>

  <script>
    const POLL_MS = {{ poll_seconds * 1000 }};
    const initialJob = new URLSearchParams(window.location.search).get("job");
    let selectedJob = initialJob || "";
    let pollTimer = null;
    let isPolling = false;

    const $ = (id) => document.getElementById(id);
    const text = (id, value) => { $(id).textContent = value ?? "-"; };
    const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[char]));

    function pill(cls, label) {
      return `<span class="pill ${escapeHtml(cls || "neutral")}">${escapeHtml(label || "-")}</span>`;
    }

    function setLiveState(state, label) {
      $("live-pulse").className = `pulse ${state === "ok" ? "" : state}`;
      text("live-label", label);
    }

    function updateUrl(jobId) {
      const url = new URL(window.location.href);
      if (jobId) url.searchParams.set("job", jobId);
      else url.searchParams.delete("job");
      window.history.replaceState({}, "", url);
    }

    function renderJobs(jobs, currentJob) {
      const select = $("job-select");
      const existing = select.value;
      select.innerHTML = "";
      if (!jobs.length) {
        select.innerHTML = `<option value="">No jobs yet</option>`;
        select.disabled = true;
        return;
      }
      select.disabled = false;
      jobs.forEach((job) => {
        const option = document.createElement("option");
        option.value = job.job_id;
        option.textContent = `Job ${job.job_id}`;
        if (job.job_id === currentJob) option.selected = true;
        select.appendChild(option);
      });
      if (existing && jobs.some((job) => job.job_id === existing)) {
        select.value = currentJob;
      }
    }

    function renderHealth(scores) {
      const section = $("health-section");
      if (!scores.length) {
        section.innerHTML = `<div class="card"><div class="empty">No health scores yet. Live raw metrics will continue to update while the job is running.</div></div>`;
        return;
      }
      section.innerHTML = scores.map((score) => `
        <div class="card">
          <div class="node-head">
            <div class="node-name">${escapeHtml(score.node_id)}</div>
            ${pill(score.status_class, score.status_label)}
          </div>
          <div class="score ${escapeHtml(score.status_class)}">${escapeHtml(score.current_score_fmt)}</div>
          <div class="bar"><div class="fill ${escapeHtml(score.status_class)}" style="width: ${score.score_pct || 0}%;"></div></div>
          <div class="hint">${escapeHtml(score.total_jobs)} scored job${score.total_jobs === 1 ? "" : "s"} · updated ${escapeHtml(score.last_updated_fmt)}</div>
        </div>
      `).join("");
    }

    function renderRecommendation(rec) {
      const target = $("recommendation-section");
      const confidence = Math.round((rec?.confidence || 0) * 100);
      text("recommendation-confidence", `${confidence}% confidence`);

      if (!rec || (!rec.recommended_nodes?.length && !rec.avoid_nodes?.length && !rec.signals?.node_risk)) {
        target.innerHTML = `<div class="empty">Not enough data to recommend nodes yet.</div>`;
        return;
      }

      const recommended = rec.recommended_nodes || [];
      const avoid = rec.avoid_nodes || [];
      const notEvaluated = rec.not_evaluated_nodes || rec.signals?.not_evaluated_nodes || [];
      const schedulerChoice = rec.scheduler_choice || rec.signals?.scheduler_choice;
      const nodeRisk = rec.signals?.node_risk || {};
      const modeLabel = rec.mode === "selected_job"
        ? `Selected-job RTT recommendation${rec.selected_job_id ? ` for ${rec.selected_job_id}` : ""}`
        : rec.mode === "recent_history"
          ? "Recent-history RTT and health recommendation"
          : "Recommendation unavailable";
      const riskRows = Object.entries(nodeRisk)
        .sort((a, b) => (b[1].risk_score || 0) - (a[1].risk_score || 0))
        .map(([node, stats]) => `
          <tr>
            <td><strong>${escapeHtml(node)}</strong></td>
            <td class="right mono">${escapeHtml(stats.risk_score ?? "-")}</td>
            <td class="right mono">${escapeHtml(stats.health_score?.toFixed ? stats.health_score.toFixed(3) : stats.health_score ?? "-")}</td>
            <td class="right mono">${escapeHtml(stats.avg_rtt_ms ?? "-")}</td>
            <td class="right mono">${escapeHtml(stats.high_rtt_path_count ?? 0)}</td>
            <td class="right mono">${escapeHtml(stats.timeout_count ?? 0)}</td>
          </tr>
        `).join("");

      target.innerHTML = `
        <div class="grid recommendation-grid">
          <div class="card">
            <div class="label">Recommended nodes</div>
            <div class="mini-list">
              ${recommended.length ? recommended.map((node) => pill("good", node)).join("") : pill("neutral", "none yet")}
            </div>
            <div class="hint">Preferred for future jobs if capacity is available</div>
          </div>
          <div class="card">
            <div class="label">Avoid nodes</div>
            <div class="mini-list">
              ${avoid.length ? avoid.map((node) => pill("bad", node)).join("") : pill("neutral", "none")}
            </div>
            <div class="hint">Advisory only; does not launch or block jobs</div>
          </div>
          <div class="card">
            <div class="label">Reason</div>
            <div class="hint">${escapeHtml(modeLabel)}</div>
            <div class="hint" style="color: var(--text); line-height: 1.45;">${escapeHtml(rec.reason || "Not enough RTT data yet.")}</div>
          </div>
        </div>
        ${notEvaluated.length ? `
          <div class="callout" style="margin-bottom: 14px;">
            <strong>Not evaluated in selected job:</strong>
            ${notEvaluated.map((node) => pill("neutral", node)).join(" ")}
          </div>
        ` : ""}
        ${schedulerChoice ? `
          <div class="section" style="margin-bottom: 14px; box-shadow: none;">
            <div class="section-title">
              <div>
                <h2>Best ${escapeHtml(schedulerChoice.target_node_count)}-node scheduler choice</h2>
                <div class="section-subtitle">Top-K advisory selection by lowest node risk. This does not launch jobs automatically.</div>
              </div>
            </div>
            <div class="grid recommendation-grid">
              <div class="card">
                <div class="label">Selected nodes</div>
                <div class="mini-list">
                  ${(schedulerChoice.selected_nodes || []).length ? schedulerChoice.selected_nodes.map((node) => pill("good", node)).join("") : pill("neutral", "none")}
                </div>
              </div>
              <div class="card">
                <div class="label">Excluded nodes</div>
                <div class="mini-list">
                  ${(schedulerChoice.excluded_nodes || []).length ? schedulerChoice.excluded_nodes.map((node) => pill("warn", node)).join("") : pill("neutral", "none")}
                </div>
              </div>
              <div class="card">
                <div class="label">Top-K reason</div>
                <div class="hint" style="color: var(--text); line-height: 1.45;">${escapeHtml(schedulerChoice.reason || "")}</div>
              </div>
            </div>
          </div>
        ` : ""}
        ${riskRows ? `
          <table>
            <thead>
              <tr>
                <th>Node</th><th class="right">Risk</th><th class="right">Health</th><th class="right">Avg RTT ms</th><th class="right">High paths</th><th class="right">Timeouts</th>
              </tr>
            </thead>
            <tbody>${riskRows}</tbody>
          </table>
        ` : ""}
      `;
    }

    function renderDiagnosis(diagnosis, currentStatus, recommendation) {
      if (!diagnosis && !currentStatus) {
        $("diagnosis-section").classList.add("hidden");
        return;
      }

      $("diagnosis-section").classList.remove("hidden");
      if (diagnosis) {
        const recAvoid = recommendation?.avoid_nodes || diagnosis.recommendation_avoid_nodes || [];
        const recNote = recAvoid.length
          ? ` Scheduler Recommendation currently suggests avoiding: <strong>${escapeHtml(recAvoid.join(", "))}</strong>.`
          : " Scheduler Recommendation is the source of node-avoidance guidance when confidence is high.";
        const leading = diagnosis.has_unhealthy_node
          ? `Most likely straggler: <strong>${escapeHtml(diagnosis.weakest_node)}</strong> with score <strong>${escapeHtml(diagnosis.weakest_score_fmt)}</strong>.`
          : `No unhealthy node detected. Lowest relative score: <strong>${escapeHtml(diagnosis.weakest_node)}</strong> with score <strong>${escapeHtml(diagnosis.weakest_score_fmt)}</strong>.`;
        $("diagnosis").innerHTML = `
          ${leading}
          Healthiest node: <strong>${escapeHtml(diagnosis.strongest_node)}</strong>
          with score <strong>${escapeHtml(diagnosis.strongest_score_fmt)}</strong>.
          Straggler threshold: <strong>${escapeHtml(diagnosis.threshold_fmt)}</strong>.
          ${recNote}
        `;
      } else {
        $("diagnosis").innerHTML = "No health-score diagnosis is available yet.";
      }

      if (currentStatus) {
        const messages = currentStatus.messages || [];
        $("current-job-status").innerHTML = `
          <strong>Current job status:</strong>
          ${messages.map((message) => `<div>${escapeHtml(message)}</div>`).join("")}
          <div class="hint">
            Max latest RTT: ${escapeHtml(currentStatus.max_rtt_ms_fmt || "-")} ms ·
            Max latest all-reduce: ${escapeHtml(currentStatus.max_all_reduce_ms_fmt || "-")} ms ·
            All-reduce threshold: ${escapeHtml(currentStatus.all_reduce_threshold_ms_fmt || "-")} ms
          </div>
        `;
      } else {
        $("current-job-status").innerHTML = "No current-job status is available yet.";
      }
    }

    function renderLatest(rows) {
      const target = $("latest-node-metrics");
      if (!rows.length) {
        target.innerHTML = `<div class="empty">No node metrics for this job yet.</div>`;
        return;
      }
      target.innerHTML = `
        <table>
          <thead>
            <tr>
              <th>Node</th><th>Epoch</th><th class="right">All-reduce ms</th>
              <th class="right">RTT ms</th><th class="right">NIC total MB/s</th><th>Seen</th><th>Status</th>
            </tr>
          </thead>
          <tbody>
            ${rows.map((row) => `
              <tr>
                <td><strong>${escapeHtml(row.node_id)}</strong></td>
                <td class="mono">${escapeHtml(row.epoch_fmt)}</td>
                <td class="right mono">${escapeHtml(row.all_reduce_ms_fmt)}</td>
                <td class="right mono">${escapeHtml(row.rtt_ms_fmt)}</td>
                <td class="right mono">${escapeHtml(row.nic_total_mb_fmt)}</td>
                <td class="mono">${escapeHtml(row.timestamp_fmt)}</td>
                <td>${pill(row.status_class, row.status_label)}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      `;
    }

    function renderRecent(rows) {
      const target = $("recent-metrics");
      if (!rows.length) {
        target.innerHTML = `<div class="empty">No metrics found. Run a training job or generate fake metrics.</div>`;
        return;
      }
      target.innerHTML = `
        <table>
          <thead>
            <tr>
              <th>Time</th><th>Job</th><th>Node</th><th>Epoch</th>
              <th class="right">All-reduce ms</th><th class="right">RTT ms</th>
              <th class="right">Sent MB/s</th><th class="right">Recv MB/s</th>
              <th class="right">Total MB/s</th><th>Status</th>
            </tr>
          </thead>
          <tbody>
            ${rows.map((row) => `
              <tr>
                <td class="mono">${escapeHtml(row.timestamp_fmt)}</td>
                <td class="mono">${escapeHtml(row.job_id)}</td>
                <td><strong>${escapeHtml(row.node_id)}</strong></td>
                <td class="mono">${escapeHtml(row.epoch_fmt)}</td>
                <td class="right mono">${escapeHtml(row.all_reduce_ms_fmt)}</td>
                <td class="right mono">${escapeHtml(row.rtt_ms_fmt)}</td>
                <td class="right mono">${escapeHtml(row.nic_sent_mb_fmt)}</td>
                <td class="right mono">${escapeHtml(row.nic_recv_mb_fmt)}</td>
                <td class="right mono">${escapeHtml(row.nic_total_mb_fmt)}</td>
                <td>${pill(row.status_class, row.status_label)}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      `;
    }

    function renderDashboard(data) {
      $("not-ready").classList.add("hidden");
      $("dashboard").classList.remove("hidden");

      const hasMetrics = data.summary.metric_rows > 0;
      $("empty-state").classList.toggle("hidden", hasMetrics);
      $("selected-overview").classList.toggle("hidden", !data.selected_summary);

      selectedJob = data.selected_job || selectedJob || "";
      renderJobs(data.jobs, selectedJob);
      updateUrl(selectedJob);

      text("global-jobs", data.summary.job_count);
      text("global-nodes", data.summary.node_count);
      text("global-rows", data.summary.metric_rows);
      text("global-last-update", data.summary.last_seen_fmt);

      if (data.selected_summary) {
        const summary = data.selected_summary;
        text("selected-subtitle", `Job ${summary.job_id} · ${summary.started_fmt} to ${summary.ended_fmt}`);
        text("selected-epochs", `${summary.max_epoch ?? 0} epochs`);
        text("avg-all-reduce", summary.avg_all_reduce_ms_fmt);
        text("avg-rtt", summary.avg_rtt_ms_fmt);
        text("avg-nic", summary.avg_nic_total_mb_fmt);
        text("selected-samples", summary.metric_rows);
      }

      renderHealth(data.health_scores);
      renderRecommendation(data.recommendation);
      renderDiagnosis(data.diagnosis, data.current_job_status, data.recommendation);

      renderLatest(data.latest_by_node);
      renderRecent(data.recent_metrics);
      text("last-polled", `Last updated ${data.generated_at_fmt}`);
      setLiveState("ok", "Live");
    }

    function renderNotReady(data) {
      $("dashboard").classList.add("hidden");
      $("not-ready").classList.remove("hidden");
      text("error-message", data.error || "Database is not ready.");
      $("error-hint").innerHTML = data.hint ? `<strong>Next step:</strong> <code>${escapeHtml(data.hint)}</code>` : "";
    }

    async function fetchDashboard() {
      if (isPolling) return;
      isPolling = true;
      try {
        const params = new URLSearchParams();
        if (selectedJob) params.set("job", selectedJob);
        const response = await fetch(`/api/dashboard?${params.toString()}`, { cache: "no-store" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        if (data.ready) renderDashboard(data);
        else renderNotReady(data);
      } catch (error) {
        setLiveState("error", "Disconnected");
        text("last-polled", error.message || "poll failed");
      } finally {
        isPolling = false;
      }
    }

    $("job-select").addEventListener("change", (event) => {
      selectedJob = event.target.value;
      updateUrl(selectedJob);
      fetchDashboard();
    });
    $("latest-job").addEventListener("click", () => {
      selectedJob = "";
      updateUrl("");
      fetchDashboard();
    });

    fetchDashboard();
    pollTimer = window.setInterval(fetchDashboard, POLL_MS);
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(TEMPLATE, poll_seconds=POLL_SECONDS)


@app.route("/api/dashboard")
def api_dashboard():
    selected_job = request.args.get("job")
    target_nodes = request.args.get("target_nodes")
    target_node_count = None
    if target_nodes:
        try:
            target_node_count = int(target_nodes)
        except ValueError:
            target_node_count = None
    elif not target_nodes:
        target_node_count = 4
    return jsonify(load_dashboard_data(selected_job, target_node_count))


if __name__ == "__main__":
    print("Dashboard running at http://localhost:5050")
    app.run(host="0.0.0.0", port=5000, debug=False)
