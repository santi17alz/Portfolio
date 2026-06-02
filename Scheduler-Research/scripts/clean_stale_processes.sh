#!/usr/bin/env bash
set -euo pipefail

read -r -a nodes <<< "${NODES:-node0 node1 node2 node3 node4}"
patterns=(
  "telemetry/launch.py"
  "telemetry/agent.py"
  "training/train.py"
)

echo "Cleaning stale training and telemetry processes..."

for node in "${nodes[@]}"; do
  echo "[$node] checking for stale processes"
  for pattern in "${patterns[@]}"; do
    echo "[$node] pkill -TERM -f $pattern"
    docker exec "$node" pkill -TERM -f "$pattern" 2>/dev/null || true
  done
done

sleep 2

for node in "${nodes[@]}"; do
  for pattern in "${patterns[@]}"; do
    docker exec "$node" pkill -KILL -f "$pattern" 2>/dev/null || true
  done
done

echo "Stale process cleanup complete."
