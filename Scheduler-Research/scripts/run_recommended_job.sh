#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 <new_job_id> [source_job_id] [target_node_count]" >&2
  exit 2
fi

new_job_id="$1"
source_job_id="${2:-}"
target_node_count="${3:-}"

if [[ -n "$source_job_id" ]]; then
  recommended_nodes="$(
    docker exec -i -e JOB_ID="$source_job_id" -e TARGET_NODE_COUNT="$target_node_count" dashboard python3 - <<'PY'
import os
from dashboard.recommend import compute_recommendation

target = os.environ.get("TARGET_NODE_COUNT") or None
rec = compute_recommendation(selected_job_id=os.environ["JOB_ID"], target_node_count=target)
choice = rec.get("scheduler_choice") or {}
print(" ".join(choice.get("selected_nodes") or rec.get("recommended_nodes", [])))
PY
  )"
else
  recommended_nodes="$(
    docker exec -i -e TARGET_NODE_COUNT="$target_node_count" dashboard python3 - <<'PY'
import os
from dashboard.recommend import compute_recommendation

target = os.environ.get("TARGET_NODE_COUNT") or None
rec = compute_recommendation(target_node_count=target)
choice = rec.get("scheduler_choice") or {}
print(" ".join(choice.get("selected_nodes") or rec.get("recommended_nodes", [])))
PY
  )"
fi

if [[ -z "$recommended_nodes" ]]; then
  echo "No recommended nodes are available. Run more telemetry or inspect recommendation details first." >&2
  exit 1
fi

echo "Running advisory recommended job ${new_job_id}"
echo "Using nodes: ${recommended_nodes}"
NODES="$recommended_nodes" ./scripts/run_job.sh "$new_job_id"
