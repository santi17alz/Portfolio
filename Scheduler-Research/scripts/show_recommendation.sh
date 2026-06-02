#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 2 ]]; then
  echo "Usage: $0 [job_id] [target_node_count]" >&2
  exit 2
fi

job_id="${1:-}"
target_node_count="${2:-}"

if [[ -n "$job_id" ]]; then
  docker exec -i -e JOB_ID="$job_id" -e TARGET_NODE_COUNT="$target_node_count" dashboard python3 - <<'PY'
import os
from dashboard.recommend import compute_recommendation

job_id = os.environ["JOB_ID"]
target = os.environ.get("TARGET_NODE_COUNT") or None
rec = compute_recommendation(selected_job_id=job_id, target_node_count=target)

def join_nodes(nodes):
    return ", ".join(nodes) if nodes else "none"

print(f"Recommendation for job {job_id}")
print(f"Mode: {rec.get('mode', 'no_data')}")
print(f"Recommended nodes: {join_nodes(rec.get('recommended_nodes', []))}")
print(f"Avoid nodes: {join_nodes(rec.get('avoid_nodes', []))}")
print(f"Not evaluated nodes: {join_nodes(rec.get('not_evaluated_nodes', []))}")
print(f"Confidence: {float(rec.get('confidence', 0.0)):.2f}")
print(f"Reason: {rec.get('reason', 'Not enough RTT data yet.')}")

choice = rec.get("scheduler_choice")
if choice:
    print()
    print(f"Scheduler choice ({choice.get('selection_mode', 'top_k_by_lowest_risk')}):")
    print(f"  Target node count: {choice.get('target_node_count')}")
    print(f"  Selected nodes: {join_nodes(choice.get('selected_nodes', []))}")
    print(f"  Excluded nodes: {join_nodes(choice.get('excluded_nodes', []))}")
    print(f"  Reason: {choice.get('reason', '')}")

per_node = rec.get("signals", {}).get("per_node_risk", {}) or rec.get("per_node_risk", {})
if per_node:
    print()
    print(f"{'node':<10} {'risk':>7} {'health':>8} {'avg_rtt':>9} {'high':>6} {'severe':>7} {'timeouts':>8}")
    for node, stats in sorted(per_node.items(), key=lambda item: item[1].get("risk_score", 0), reverse=True):
        health = stats.get("health_score")
        avg_rtt = stats.get("avg_rtt_ms")
        health_text = "-" if health is None else f"{float(health):.3f}"
        avg_rtt_text = "-" if avg_rtt is None else f"{float(avg_rtt):.2f}"
        print(
            f"{node:<10} "
            f"{stats.get('risk_score', 0):>7.3f} "
            f"{health_text:>8} "
            f"{avg_rtt_text:>9} "
            f"{stats.get('high_rtt_path_count', 0):>6} "
            f"{stats.get('severe_rtt_path_count', 0):>7} "
            f"{stats.get('timeout_count', 0):>8}"
        )
PY
else
  docker exec -i -e TARGET_NODE_COUNT="$target_node_count" dashboard python3 - <<'PY'
import os
from dashboard.recommend import compute_recommendation

target = os.environ.get("TARGET_NODE_COUNT") or None
rec = compute_recommendation(target_node_count=target)

def join_nodes(nodes):
    return ", ".join(nodes) if nodes else "none"

print("Recommendation from recent history")
print(f"Mode: {rec.get('mode', 'no_data')}")
print(f"Recommended nodes: {join_nodes(rec.get('recommended_nodes', []))}")
print(f"Avoid nodes: {join_nodes(rec.get('avoid_nodes', []))}")
print(f"Not evaluated nodes: {join_nodes(rec.get('not_evaluated_nodes', []))}")
print(f"Confidence: {float(rec.get('confidence', 0.0)):.2f}")
print(f"Reason: {rec.get('reason', 'Not enough RTT data yet.')}")

choice = rec.get("scheduler_choice")
if choice:
    print()
    print(f"Scheduler choice ({choice.get('selection_mode', 'top_k_by_lowest_risk')}):")
    print(f"  Target node count: {choice.get('target_node_count')}")
    print(f"  Selected nodes: {join_nodes(choice.get('selected_nodes', []))}")
    print(f"  Excluded nodes: {join_nodes(choice.get('excluded_nodes', []))}")
    print(f"  Reason: {choice.get('reason', '')}")

per_node = rec.get("signals", {}).get("per_node_risk", {}) or rec.get("per_node_risk", {})
if per_node:
    print()
    print(f"{'node':<10} {'risk':>7} {'health':>8} {'avg_rtt':>9} {'high':>6} {'severe':>7} {'timeouts':>8}")
    for node, stats in sorted(per_node.items(), key=lambda item: item[1].get("risk_score", 0), reverse=True):
        health = stats.get("health_score")
        avg_rtt = stats.get("avg_rtt_ms")
        health_text = "-" if health is None else f"{float(health):.3f}"
        avg_rtt_text = "-" if avg_rtt is None else f"{float(avg_rtt):.2f}"
        print(
            f"{node:<10} "
            f"{stats.get('risk_score', 0):>7.3f} "
            f"{health_text:>8} "
            f"{avg_rtt_text:>9} "
            f"{stats.get('high_rtt_path_count', 0):>6} "
            f"{stats.get('severe_rtt_path_count', 0):>7} "
            f"{stats.get('timeout_count', 0):>8}"
        )
PY
fi
