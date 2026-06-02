"""
Fake metrics generator — pretends to be Kyle's telemetry agent.
Writes realistic-looking metrics into the database every few seconds.

This lets us develop the dashboard immediately without waiting for
Kyle's agent. When his real agent is ready, we just stop running this
script and his data will flow into the same tables.

Usage:
  python3 fake_metrics.py                 # healthy cluster
  python3 fake_metrics.py --slow node3    # simulate node3 as a straggler
"""
import sqlite3
import time
import random
import argparse
import uuid

DB_PATH = '/workspace/data/metrics.db'
NODES   = ['node0', 'node1', 'node2', 'node3', 'node4']

# Realistic baseline values (what a healthy node looks like)
BASELINE = {
    'all_reduce_ms':  20.0,   # ~20ms per all-reduce in our small model
    'rtt_ms':         0.5,    # very fast, same docker network
    'throughput_bps': 50_000_000,  # 50 MB/sec
}

def jitter(value, pct=0.1):
    """Add +/- pct random variation to simulate real measurement noise"""
    return value * (1 + random.uniform(-pct, pct))

def generate_metric(node_id, job_id, epoch, slow_node=None):
    """Generate one metric row for one node at one point in time"""
    is_slow    = (node_id == slow_node)
    multiplier = 2.5 if is_slow else 1.0  # slow node takes 2.5x longer

    return {
        'timestamp':      time.time(),
        'node_id':        node_id,
        'job_id':         job_id,
        'nic_bytes_sent': int(jitter(BASELINE['throughput_bps']) / (multiplier if is_slow else 1)),
        'nic_bytes_recv': int(jitter(BASELINE['throughput_bps']) / (multiplier if is_slow else 1)),
        'all_reduce_ms':  jitter(BASELINE['all_reduce_ms']) * multiplier,
        'rtt_ms':         jitter(BASELINE['rtt_ms']) * (multiplier if is_slow else 1),
        'epoch':          epoch,
    }

def insert_metric(conn, m):
    conn.execute("""
        INSERT INTO metrics
            (timestamp, node_id, job_id, nic_bytes_sent, nic_bytes_recv,
             all_reduce_ms, rtt_ms, epoch)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (m['timestamp'], m['node_id'], m['job_id'],
          m['nic_bytes_sent'], m['nic_bytes_recv'],
          m['all_reduce_ms'], m['rtt_ms'], m['epoch']))

def simulate_job(slow_node=None, num_epochs=10, epoch_interval=2.0):
    """Simulate one full training job — N epochs, all configured nodes reporting per epoch"""
    conn   = sqlite3.connect(DB_PATH)
    job_id = str(uuid.uuid4())[:8]

    label = f"healthy" if slow_node is None else f"slow node = {slow_node}"
    print(f"\n[Job {job_id}] Starting simulated job ({label})")

    for epoch in range(1, num_epochs + 1):
        for node in NODES:
            metric = generate_metric(node, job_id, epoch, slow_node)
            insert_metric(conn, metric)
        conn.commit()

        slow_marker = "  ⚠️  STRAGGLER" if slow_node else ""
        print(f"  Epoch {epoch:2d} | inserted metrics for all nodes{slow_marker}")
        time.sleep(epoch_interval)

    conn.close()
    print(f"[Job {job_id}] Complete\n")
    return job_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--slow',     type=str, default=None,
                        choices=NODES,
                        help='Simulate this node as a straggler')
    parser.add_argument('--epochs',   type=int, default=10)
    parser.add_argument('--interval', type=float, default=2.0,
                        help='Seconds between epochs')
    args = parser.parse_args()

    simulate_job(slow_node=args.slow, num_epochs=args.epochs,
                 epoch_interval=args.interval)
