#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <node0|node1|node2|node3|node4>" >&2
  exit 2
fi

node="$1"
read -r -a valid_nodes <<< "${NODES:-node0 node1 node2 node3 node4}"
valid=0
for valid_node in "${valid_nodes[@]}"; do
  if [[ "$node" == "$valid_node" ]]; then
    valid=1
    break
  fi
done
if [[ "$valid" -ne 1 ]]; then
  echo "Argument must be one of: ${valid_nodes[*]}." >&2
  exit 2
fi

docker exec -i "$node" sh -s <<'SH'
set -eu

iface="$(ip route show default 2>/dev/null | awk '{print $5; exit}')"
if [ -z "${iface:-}" ]; then
  iface="eth0"
fi

echo "Current qdisc on $iface:"
tc qdisc show dev "$iface"
SH
