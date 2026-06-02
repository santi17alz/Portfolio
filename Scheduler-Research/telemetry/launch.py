"""
launch.py — starts the telemetry agent and training together on one node.

Run this inside each node container instead of calling train.py directly.
It starts the agent as a background process, then runs training, then
ensures the agent exits cleanly when training is done.

Usage (inside a node container):
  python3 telemetry/launch.py --node-id node0 --peers node1 node2 node3 node4

Environment variables expected (set by docker-compose.yml):
  MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK

Optional:
  JOB_ID — if not set, launch.py generates one on rank 0 and shares it.
"""

import argparse
import os
import subprocess
import sys
import time
import uuid


SYNC_DIR = '/workspace/data/sync'


def resolve_job_id(arg_job_id):
    if arg_job_id:
        return arg_job_id

    os.makedirs(SYNC_DIR, exist_ok=True)
    job_id_path = os.path.join(SYNC_DIR, 'current_job_id')
    rank = int(os.environ.get('RANK', '0'))
    started_at = time.time()
    freshness_window = 30

    if rank == 0:
        job_id = str(uuid.uuid4())[:8]
        tmp_path = job_id_path + '.tmp'
        with open(tmp_path, 'w') as f:
            f.write(job_id)
        os.replace(tmp_path, job_id_path)
        print(f"[Launch] Generated job_id={job_id}")
        return job_id

    deadline = started_at + 60
    while time.time() < deadline:
        try:
            if os.path.getmtime(job_id_path) >= started_at - freshness_window:
                with open(job_id_path, 'r') as f:
                    job_id = f.read().strip()
                if job_id:
                    print(f"[Launch] Resolved job_id={job_id}")
                    return job_id
        except FileNotFoundError:
            pass
        time.sleep(0.2)

    raise TimeoutError(f"Timed out waiting for {job_id_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--node-id', required=True,
                        help='This node\'s name, e.g. node0')
    parser.add_argument('--peers',   nargs='*', default=[],
                        help='Peer hostnames to probe, e.g. node1 node2 node3 node4')
    parser.add_argument('--job-id',  default=None,
                        help='Optional job ID — auto-generated if not provided')
    args = parser.parse_args()

    env = os.environ.copy()
    job_id = resolve_job_id(args.job_id)
    env['JOB_ID'] = job_id

    # Build agent command
    agent_cmd = [
        sys.executable, 'telemetry/agent.py',
        '--node-id', args.node_id,
        '--job-id',  job_id,
        '--peers',   *args.peers
    ]

    # Start the agent first so the RTT server is up before training begins
    print(f"[Launch] Starting telemetry agent on {args.node_id}...")
    agent_proc = subprocess.Popen(agent_cmd, env=env)
    time.sleep(1.0)   # give the RTT server a moment to bind
    if agent_proc.poll() is not None:
        print(
            f"[Launch] Telemetry agent exited early on {args.node_id} "
            f"with code {agent_proc.returncode}."
        )
        return agent_proc.returncode or 1

    # Start training
    print(f"[Launch] Starting training on {args.node_id}...")
    train_cmd  = [sys.executable, 'training/train.py']
    train_proc = subprocess.Popen(train_cmd, env=env)

    # Wait for training to finish
    train_returncode = train_proc.wait()
    print(f"[Launch] Training finished on {args.node_id}.")

    # Agent will exit on its own via the done signal, but give it a moment
    try:
        agent_proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        print("[Launch] Agent did not exit cleanly — terminating.")
        agent_proc.terminate()
        try:
            agent_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("[Launch] Agent still running — killing.")
            agent_proc.kill()
            agent_proc.wait()

    print(f"[Launch] Done.")
    return train_returncode or agent_proc.returncode or 0


if __name__ == '__main__':
    raise SystemExit(main())
