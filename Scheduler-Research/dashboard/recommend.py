"""
Advisory scheduler recommendation logic.

This module does not launch jobs or schedule containers. It reads current
health scores and recent per-peer RTT paths, then returns an explainable
recommendation about which nodes look safer to prefer or avoid.
"""

import json
import sqlite3
import sys

DB_PATH = "/workspace/data/metrics.db"
RECENT_JOB_LIMIT = 5
RTT_HIGH_THRESHOLD_MS = 10.0
RTT_SEVERE_THRESHOLD_MS = 25.0
MIN_HEALTH_SCORE = 0.75
TIMEOUT_RISK_WEIGHT = 0.15
HIGH_RTT_RISK_WEIGHT = 0.10
HEALTH_RISK_WEIGHT = 0.55
AVG_RTT_RISK_WEIGHT = 0.20
RISK_SEPARATION_THRESHOLD = 0.15
SELECTED_JOB_RISK_SEPARATION = 0.20
SELECTED_JOB_MIN_HIGH_PATHS = 2


EMPTY_RECOMMENDATION = {
    "recommended_nodes": [],
    "avoid_nodes": [],
    "confidence": 0.0,
    "reason": "Not enough RTT data yet.",
    "signals": {},
    "selected_job_id": None,
    "high_rtt_paths": [],
    "per_node_risk": {},
    "not_evaluated_nodes": [],
    "scheduler_choice": None,
    "mode": "no_data",
}


def _empty_recommendation(reason="Not enough RTT data yet.", selected_job_id=None):
    rec = dict(EMPTY_RECOMMENDATION)
    rec["reason"] = reason
    rec["selected_job_id"] = selected_job_id
    return rec


def get_conn(db_path=DB_PATH):
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _table_exists(conn, table_name):
    return conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone() is not None


def _clamp(value, lo=0.0, hi=1.0):
    return max(lo, min(hi, value))


def _recent_jobs(conn):
    if not _table_exists(conn, "metrics"):
        return []

    return [
        row[0]
        for row in conn.execute(
            """
            SELECT job_id
            FROM metrics
            GROUP BY job_id
            ORDER BY MAX(timestamp) DESC
            LIMIT ?
            """,
            (RECENT_JOB_LIMIT,),
        ).fetchall()
    ]


def _health_scores(conn):
    if not _table_exists(conn, "health_scores"):
        return {}

    health_rows = conn.execute(
        """
        SELECT node_id, current_score
        FROM health_scores
        """
    ).fetchall()
    return {row["node_id"]: float(row["current_score"]) for row in health_rows}


def _known_nodes(conn):
    nodes = set(_health_scores(conn))

    if _table_exists(conn, "metrics"):
        rows = conn.execute("SELECT DISTINCT node_id FROM metrics").fetchall()
        nodes.update(row["node_id"] for row in rows)

    if _table_exists(conn, "rtt_metrics"):
        rows = conn.execute(
            """
            SELECT node_id FROM rtt_metrics
            UNION
            SELECT peer_node_id AS node_id FROM rtt_metrics
            """
        ).fetchall()
        nodes.update(row["node_id"] for row in rows)

    return nodes


def _selected_job_participants(conn, selected_job_id, rtt_rows):
    participants = set()
    for row in rtt_rows:
        participants.add(row["node_id"])
        participants.add(row["peer_node_id"])

    if _table_exists(conn, "metrics"):
        rows = conn.execute(
            "SELECT DISTINCT node_id FROM metrics WHERE job_id = ?",
            (selected_job_id,),
        ).fetchall()
        participants.update(row["node_id"] for row in rows)

    return participants


def _load_rtt_rows(conn, job_ids):
    if not job_ids:
        return []
    if not _table_exists(conn, "rtt_metrics"):
        return []

    placeholders = ",".join("?" for _ in job_ids)
    return conn.execute(
        f"""
        SELECT node_id, peer_node_id, rtt_ms, job_id
        FROM rtt_metrics
        WHERE job_id IN ({placeholders})
        """,
        job_ids,
    ).fetchall()


def _build_path_and_node_stats(rtt_rows, health_scores, participating_nodes=None):
    if participating_nodes is None:
        nodes = set(health_scores)
        for row in rtt_rows:
            nodes.add(row["node_id"])
            nodes.add(row["peer_node_id"])
    else:
        nodes = set(participating_nodes)

    node_stats = {
        node: {
            "health_score": health_scores.get(node),
            "rtt_values": [],
            "timeout_count": 0,
            "high_rtt_path_count": 0,
            "severe_rtt_path_count": 0,
            "path_count": 0,
        }
        for node in nodes
    }

    path_scores = {}
    for row in rtt_rows:
        src = row["node_id"]
        peer = row["peer_node_id"]
        if src not in nodes or peer not in nodes:
            continue
        rtt = row["rtt_ms"]
        key = f"{src}->{peer}"
        path = path_scores.setdefault(key, {
            "from": src,
            "to": peer,
            "rows": 0,
            "avg_rtt_ms": None,
            "max_rtt_ms": None,
            "timeout_count": 0,
            "is_high_rtt": False,
            "is_severe_rtt": False,
            "_rtts": [],
        })
        path["rows"] += 1

        if rtt is None:
            path["timeout_count"] += 1
            node_stats[src]["timeout_count"] += 1
            node_stats[peer]["timeout_count"] += 1
            continue

        rtt = float(rtt)
        path["_rtts"].append(rtt)
        node_stats[src]["rtt_values"].append(rtt)
        node_stats[peer]["rtt_values"].append(rtt)

    high_rtt_paths = []
    for path in path_scores.values():
        values = path.pop("_rtts")
        if values:
            path["avg_rtt_ms"] = round(sum(values) / len(values), 2)
            path["max_rtt_ms"] = round(max(values), 2)
            path["is_high_rtt"] = path["avg_rtt_ms"] >= RTT_HIGH_THRESHOLD_MS
            path["is_severe_rtt"] = path["max_rtt_ms"] >= RTT_SEVERE_THRESHOLD_MS
        if path["timeout_count"] > 0:
            path["is_high_rtt"] = True
            path["is_severe_rtt"] = True

        src = path["from"]
        peer = path["to"]
        for node in (src, peer):
            node_stats[node]["path_count"] += 1
            if path["is_high_rtt"]:
                node_stats[node]["high_rtt_path_count"] += 1
            if path["is_severe_rtt"]:
                node_stats[node]["severe_rtt_path_count"] += 1

        if path["is_high_rtt"]:
            high_rtt_paths.append({
                "from": src,
                "to": peer,
                "avg_rtt_ms": path["avg_rtt_ms"],
                "max_rtt_ms": path["max_rtt_ms"],
                "timeout_count": path["timeout_count"],
                "is_severe_rtt": path["is_severe_rtt"],
            })

    return nodes, path_scores, node_stats, high_rtt_paths


def _finalize_node_risk(node_stats, selected_job_mode=False):
    max_avg_rtt = 0.0
    for stats in node_stats.values():
        values = stats["rtt_values"]
        stats["avg_rtt_ms"] = (sum(values) / len(values)) if values else None
        stats["max_rtt_ms"] = max(values) if values else None
        if stats["avg_rtt_ms"] is not None:
            max_avg_rtt = max(max_avg_rtt, stats["avg_rtt_ms"])

    max_avg_rtt = max(max_avg_rtt, RTT_HIGH_THRESHOLD_MS)
    risks = {}
    for node, stats in node_stats.items():
        health = stats["health_score"]
        health_risk = 0.5 if health is None else _clamp((MIN_HEALTH_SCORE - health) / MIN_HEALTH_SCORE)
        avg_rtt = stats["avg_rtt_ms"] or 0.0
        avg_rtt_risk = _clamp(avg_rtt / max_avg_rtt)
        path_count = max(stats["path_count"], 1)
        timeout_risk = _clamp(stats["timeout_count"] / path_count)
        high_rtt_risk = _clamp(stats["high_rtt_path_count"] / path_count)
        severe_rtt_risk = _clamp(stats["severe_rtt_path_count"] / path_count)

        if selected_job_mode:
            # For the selected job, per-peer RTT shape should dominate. Health
            # scores are historical and useful context, but they should not hide
            # a current job where one node appears on most high-latency paths.
            risk = (
                0.42 * high_rtt_risk
                + 0.23 * severe_rtt_risk
                + 0.20 * avg_rtt_risk
                + 0.10 * timeout_risk
                + 0.05 * health_risk
            )
        else:
            risk = (
                HEALTH_RISK_WEIGHT * health_risk
                + AVG_RTT_RISK_WEIGHT * avg_rtt_risk
                + TIMEOUT_RISK_WEIGHT * timeout_risk
                + HIGH_RTT_RISK_WEIGHT * high_rtt_risk
            )

        risks[node] = risk
        stats["risk_score"] = round(risk, 3)
        stats["avg_rtt_ms"] = round(stats["avg_rtt_ms"], 2) if stats["avg_rtt_ms"] is not None else None
        stats["max_rtt_ms"] = round(stats["max_rtt_ms"], 2) if stats["max_rtt_ms"] is not None else None
        stats["high_rtt_path_fraction"] = round(stats["high_rtt_path_count"] / path_count, 3)
        stats["severe_rtt_path_fraction"] = round(stats["severe_rtt_path_count"] / path_count, 3)
        stats.pop("rtt_values", None)

    return risks


def _scheduler_sort_key(item):
    node, stats = item
    health = stats.get("health_score")
    avg_rtt = stats.get("avg_rtt_ms")
    return (
        stats.get("risk_score", 0.0),
        -(health if health is not None else -1.0),
        avg_rtt if avg_rtt is not None else float("inf"),
        stats.get("high_rtt_path_count", 0),
        node,
    )


def _build_scheduler_choice(node_stats, target_node_count, not_evaluated_nodes=None):
    if target_node_count is None:
        return None

    try:
        target_node_count = int(target_node_count)
    except (TypeError, ValueError):
        return {
            "target_node_count": target_node_count,
            "selected_nodes": [],
            "excluded_nodes": [],
            "reason": "Invalid target node count.",
            "selection_mode": "top_k_by_lowest_risk",
        }

    if target_node_count <= 0:
        return {
            "target_node_count": target_node_count,
            "selected_nodes": [],
            "excluded_nodes": sorted(node_stats),
            "reason": "Target node count must be greater than zero.",
            "selection_mode": "top_k_by_lowest_risk",
        }

    ranked = sorted(node_stats.items(), key=_scheduler_sort_key)
    ranked_nodes = [node for node, _ in ranked]
    selected_nodes = ranked_nodes[:target_node_count]
    excluded_nodes = ranked_nodes[target_node_count:]

    if len(ranked_nodes) < target_node_count:
        warning = (
            f"Only {len(ranked_nodes)} participating nodes are available for a "
            f"{target_node_count}-node request; selecting all participating nodes."
        )
    else:
        warning = None

    high_nodes = [
        node
        for node, stats in ranked
        if stats.get("high_rtt_path_count", 0) > 0
        or stats.get("severe_rtt_path_count", 0) > 0
        or stats.get("timeout_count", 0) > 0
        or (stats.get("health_score") is not None and stats["health_score"] < MIN_HEALTH_SCORE)
    ]

    if warning:
        reason = warning
    elif len(ranked_nodes) == target_node_count:
        reason = f"Selected all {target_node_count} participating nodes; no node needed to be excluded."
    elif not high_nodes:
        reason = (
            f"All evaluated nodes look healthy; excluded {', '.join(excluded_nodes) or 'none'} "
            f"only because the requested job size is {target_node_count}."
        )
    elif len(excluded_nodes) == 1 and excluded_nodes[0] in high_nodes:
        excluded_risk = node_stats[excluded_nodes[0]].get("risk_score", 0.0)
        selected_max_risk = max((node_stats[node].get("risk_score", 0.0) for node in selected_nodes), default=0.0)
        if excluded_risk - selected_max_risk >= SELECTED_JOB_RISK_SEPARATION:
            reason = (
                f"Selected the {target_node_count} lowest-risk participating nodes and excluded "
                f"{excluded_nodes[0]}, the highest-risk node."
            )
        else:
            reason = (
                f"Multiple nodes show degradation signals ({', '.join(high_nodes)}); selected the "
                f"best {target_node_count} available nodes by lowest risk."
            )
    elif len(high_nodes) == 1:
        reason = (
            f"Selected the {target_node_count} lowest-risk participating nodes and excluded "
            f"{', '.join(excluded_nodes) or 'none'}; {high_nodes[0]} is the clearest degraded node."
        )
    else:
        reason = (
            f"Multiple nodes show degradation signals ({', '.join(high_nodes)}); selected the "
            f"best {target_node_count} available nodes by lowest risk."
        )

    if not_evaluated_nodes:
        reason += f" Not evaluated in this selected job: {', '.join(not_evaluated_nodes)}."

    return {
        "target_node_count": target_node_count,
        "selected_nodes": selected_nodes,
        "excluded_nodes": excluded_nodes,
        "reason": reason,
        "selection_mode": "top_k_by_lowest_risk",
    }


def _not_evaluated_note(not_evaluated_nodes):
    if not not_evaluated_nodes:
        return ""
    if len(not_evaluated_nodes) == 1:
        return f" {not_evaluated_nodes[0]} was not evaluated in this selected job."
    return f" {', '.join(not_evaluated_nodes)} were not evaluated in this selected job."


def _selected_job_recommendation(
    selected_job_id,
    rtt_rows,
    health_scores,
    participating_nodes,
    known_nodes,
    target_node_count=None,
):
    not_evaluated_nodes = sorted(known_nodes - participating_nodes)
    nodes, path_scores, node_stats, high_rtt_paths = _build_path_and_node_stats(
        rtt_rows,
        health_scores,
        participating_nodes=participating_nodes,
    )
    risks = _finalize_node_risk(node_stats, selected_job_mode=True)
    ranked = sorted(risks.items(), key=lambda item: item[1], reverse=True)

    if not ranked:
        return None

    if not high_rtt_paths:
        recommended_nodes = [node for node, _ in sorted(risks.items(), key=lambda item: item[1])]
        reason = (
            f"No degraded node detected among participating nodes for selected job {selected_job_id}; "
            f"no RTT paths are above {RTT_HIGH_THRESHOLD_MS:.0f} ms."
            f"{_not_evaluated_note(not_evaluated_nodes)}"
        )
        return _recommendation_payload(
            recommended_nodes=recommended_nodes,
            avoid_nodes=[],
            confidence=0.0,
            reason=reason,
            health_scores=health_scores,
            path_scores=path_scores,
            node_stats=node_stats,
            high_rtt_paths=high_rtt_paths,
            recent_jobs=[selected_job_id],
            selected_job_id=selected_job_id,
            mode="selected_job",
            not_evaluated_nodes=not_evaluated_nodes,
            target_node_count=target_node_count,
        )

    highest_node, highest_risk = ranked[0]
    second_risk = ranked[1][1] if len(ranked) > 1 else 0.0
    top_stats = node_stats[highest_node]
    high_path_gap = top_stats["high_rtt_path_count"] - max(
        (node_stats[node]["high_rtt_path_count"] for node in nodes if node != highest_node),
        default=0,
    )
    all_high_paths_involve_top = all(
        path["from"] == highest_node or path["to"] == highest_node
        for path in high_rtt_paths
    )
    top_has_multiple_high_paths = top_stats["high_rtt_path_count"] >= SELECTED_JOB_MIN_HIGH_PATHS

    should_avoid = (
        top_has_multiple_high_paths
        and high_path_gap > 0
        and all_high_paths_involve_top
    )

    if should_avoid:
        avoid_nodes = [highest_node]
        recommended_nodes = [
            node
            for node, _ in sorted(risks.items(), key=lambda item: item[1])
            if node not in avoid_nodes
        ]
        separation = max(highest_risk - second_risk, high_path_gap / max(top_stats["path_count"], 1))
        confidence = round(_clamp(separation / SELECTED_JOB_RISK_SEPARATION), 2)
        severe_count = top_stats["severe_rtt_path_count"]
        reason = (
            f"Selected job {selected_job_id}: {highest_node} appears on "
            f"{top_stats['high_rtt_path_count']} high-latency RTT paths"
            f"{f' including {severe_count} severe paths' if severe_count else ''}. "
            "The RTT paths between the remaining nodes look healthy, so this is the clearest node to avoid."
            f"{_not_evaluated_note(not_evaluated_nodes)}"
        )
    else:
        avoid_nodes = []
        recommended_nodes = [node for node, _ in sorted(risks.items(), key=lambda item: item[1])]
        separation = highest_risk - second_risk
        confidence = round(_clamp(separation / SELECTED_JOB_RISK_SEPARATION), 2)
        reason = (
            f"Selected job {selected_job_id}: high RTT exists, but it is not concentrated on one node "
            "strongly enough to recommend avoiding a specific node."
            f"{_not_evaluated_note(not_evaluated_nodes)}"
        )

    return _recommendation_payload(
        recommended_nodes=recommended_nodes,
        avoid_nodes=avoid_nodes,
        confidence=confidence,
        reason=reason,
        health_scores=health_scores,
        path_scores=path_scores,
        node_stats=node_stats,
        high_rtt_paths=high_rtt_paths,
        recent_jobs=[selected_job_id],
        selected_job_id=selected_job_id,
        mode="selected_job",
        not_evaluated_nodes=not_evaluated_nodes,
        target_node_count=target_node_count,
    )


def _recommendation_payload(
    recommended_nodes,
    avoid_nodes,
    confidence,
    reason,
    health_scores,
    path_scores,
    node_stats,
    high_rtt_paths,
    recent_jobs,
    selected_job_id=None,
    mode="historical",
    not_evaluated_nodes=None,
    target_node_count=None,
):
    per_node_risk = dict(sorted(node_stats.items()))
    not_evaluated_nodes = sorted(not_evaluated_nodes or [])
    scheduler_choice = _build_scheduler_choice(per_node_risk, target_node_count, not_evaluated_nodes)
    return {
        "recommended_nodes": recommended_nodes,
        "avoid_nodes": avoid_nodes,
        "confidence": confidence,
        "reason": reason,
        "selected_job_id": selected_job_id,
        "high_rtt_paths": high_rtt_paths,
        "per_node_risk": per_node_risk,
        "not_evaluated_nodes": not_evaluated_nodes,
        "scheduler_choice": scheduler_choice,
        "mode": mode,
        "signals": {
            "health_scores": {node: round(score, 3) for node, score in health_scores.items()},
            "high_rtt_paths": high_rtt_paths,
            "per_node_risk": per_node_risk,
            "node_risk": per_node_risk,
            "not_evaluated_nodes": not_evaluated_nodes,
            "scheduler_choice": scheduler_choice,
            "rtt_path_scores": path_scores,
            "recent_jobs_considered": recent_jobs,
            "selected_job_id": selected_job_id,
            "mode": mode,
            "thresholds": {
                "rtt_high_ms": RTT_HIGH_THRESHOLD_MS,
                "rtt_severe_ms": RTT_SEVERE_THRESHOLD_MS,
                "min_health_score": MIN_HEALTH_SCORE,
            },
        },
    }


def compute_recommendation(db_path=DB_PATH, selected_job_id=None, target_node_count=None):
    """
    Return an advisory node recommendation.

    `db_path` may be a SQLite path or an existing sqlite3 connection. The
    connection form keeps dashboard/app.py efficient; the path form makes this
    module easy to call from scripts.
    """
    close_conn = False
    if hasattr(db_path, "execute"):
        conn = db_path
    else:
        conn = get_conn(db_path)
        close_conn = True

    try:
        if not _table_exists(conn, "rtt_metrics"):
            return _empty_recommendation("Not enough RTT data yet.", selected_job_id)

        health_scores = _health_scores(conn)

        if selected_job_id:
            selected_rows = _load_rtt_rows(conn, [selected_job_id])
            if selected_rows:
                participating_nodes = _selected_job_participants(conn, selected_job_id, selected_rows)
                known_nodes = _known_nodes(conn)
                return _selected_job_recommendation(
                    selected_job_id,
                    selected_rows,
                    health_scores,
                    participating_nodes,
                    known_nodes,
                    target_node_count=target_node_count,
                )

        recent_jobs = _recent_jobs(conn)
        if not recent_jobs:
            return _empty_recommendation("Not enough RTT data yet.", selected_job_id)

        rtt_rows = _load_rtt_rows(conn, recent_jobs)
        if not rtt_rows:
            return _empty_recommendation("Not enough RTT data yet.", selected_job_id)

        nodes, path_scores, node_stats, high_rtt_paths = _build_path_and_node_stats(rtt_rows, health_scores)
        risks = _finalize_node_risk(node_stats, selected_job_mode=False)

        ranked = sorted(risks.items(), key=lambda item: item[1], reverse=True)
        if not ranked:
            return _empty_recommendation("Not enough RTT data yet.", selected_job_id)

        highest_node, highest_risk = ranked[0]
        second_risk = ranked[1][1] if len(ranked) > 1 else 0.0
        separation = highest_risk - second_risk
        confidence = _clamp(separation / max(RISK_SEPARATION_THRESHOLD, 0.01))

        if highest_risk < RISK_SEPARATION_THRESHOLD or separation < RISK_SEPARATION_THRESHOLD:
            avoid_nodes = []
            recommended_nodes = [node for node, _ in sorted(risks.items(), key=lambda item: item[1])]
            reason = "No clearly degraded node detected from recent health and RTT path data."
            confidence = round(confidence, 2)
        else:
            avoid_nodes = [highest_node]
            recommended_nodes = [
                node
                for node, _ in sorted(risks.items(), key=lambda item: item[1])
                if node not in avoid_nodes
            ]
            confidence = round(confidence, 2)
            reason = (
                f"Recent-history view: {highest_node} has the highest recent risk from "
                "health score and per-peer RTT paths. Treat this as advisory, not an "
                "automatic scheduling decision."
            )

        return _recommendation_payload(
            recommended_nodes=recommended_nodes,
            avoid_nodes=avoid_nodes,
            confidence=confidence,
            reason=reason,
            health_scores=health_scores,
            path_scores=path_scores,
            node_stats=node_stats,
            high_rtt_paths=high_rtt_paths,
            recent_jobs=recent_jobs,
            selected_job_id=selected_job_id,
            mode="recent_history",
            target_node_count=target_node_count,
        )
    finally:
        if close_conn:
            conn.close()


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    selected_job_id = argv[0] if argv else None
    target_node_count = argv[1] if len(argv) > 1 else None
    print(
        json.dumps(
            compute_recommendation(
                selected_job_id=selected_job_id,
                target_node_count=target_node_count,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
