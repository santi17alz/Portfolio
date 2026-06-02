"""
Telemetry Agent — collects the three metric categories described in the paper:
  1. NIC counters      — bytes sent/received from /proc/net/dev
  2. Inter-node RTT    — TCP round-trip time to each peer node
  3. All-Reduce timing — wall-clock time of the gradient sync phase

Design decisions that match the paper's architecture:
  - RTT probes are gated to run ONLY during the idle window between epochs,
    detected by watching the epoch_sync file written by the training hook.
    This prevents probe traffic from competing with all-reduce bandwidth.
  - All metrics write to the shared SQLite database that the dashboard reads.
  - The agent runs as a separate process alongside training (not inside it),
    so it requires no changes to train.py. The all-reduce timing is collected
    via the timing hook injected by hook.py (imported in train.py).
  - No elevated privileges required — /proc/net/dev is world-readable,
    and RTT uses plain TCP sockets.

Usage (inside the telemetry container):
  python3 telemetry/agent.py --node-id node0 --job-id <job_id> --peers node1 node2 node3 node4
"""

import argparse
import errno
import fcntl
import os
import socket
import sqlite3
import sys
import time
import json
import threading
import psutil

# ── config ────────────────────────────────────────────────────────────────────
DB_PATH        = '/workspace/data/metrics.db'
DB_LOCK_PATH   = '/workspace/data/metrics.db.lock'
SYNC_DIR       = '/workspace/data/sync'       # shared dir for epoch signals
NIC_INTERVAL   = 2.0    # seconds between NIC counter samples during training
RTT_PORT       = 19876  # port the RTT probe server listens on
RTT_TIMEOUT    = 2.0    # seconds before an RTT probe times out
RTT_PROBES     = 5      # number of pings per peer per measurement round
FINAL_RTT_SERVER_GRACE_SEC = 5.0  # keep echo server up for peers' final probes
PROBE_PAYLOAD  = b'PING'


# ── database ──────────────────────────────────────────────────────────────────

def get_conn():
    """Return a SQLite connection. Called per-thread to avoid sharing."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")   # allows concurrent readers/writers
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


class db_write_lock:
    def __enter__(self):
        os.makedirs(os.path.dirname(DB_LOCK_PATH), exist_ok=True)
        self.lock_file = open(DB_LOCK_PATH, 'w')
        fcntl.flock(self.lock_file, fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, tb):
        fcntl.flock(self.lock_file, fcntl.LOCK_UN)
        self.lock_file.close()


def insert_metric(node_id, job_id, epoch,
                  nic_sent, nic_recv, all_reduce_ms, rtt_ms, peer_rtts=None):
    for attempt in range(3):
        conn = None
        try:
            with db_write_lock():
                conn = get_conn()
                now = time.time()
                conn.execute("""
                    INSERT INTO metrics
                        (timestamp, node_id, job_id, nic_bytes_sent, nic_bytes_recv,
                         all_reduce_ms, rtt_ms, epoch)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (now, node_id, job_id,
                      nic_sent, nic_recv, all_reduce_ms, rtt_ms, epoch))

                for peer_node_id, peer_rtt_ms in (peer_rtts or {}).items():
                    conn.execute("""
                        INSERT INTO rtt_metrics
                            (timestamp, job_id, node_id, peer_node_id, rtt_ms, epoch)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (now, job_id, node_id, peer_node_id, peer_rtt_ms, epoch))
                conn.commit()

                row = conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM metrics
                    WHERE job_id = ? AND node_id = ? AND epoch = ?
                    """,
                    (job_id, node_id, epoch),
                ).fetchone()
            if row and row[0] > 0:
                return
            raise sqlite3.OperationalError("insert verification failed")
        except sqlite3.OperationalError as exc:
            retryable = "locked" in str(exc).lower() or "verification failed" in str(exc).lower()
            if not retryable or attempt == 2:
                raise
            print(f"[Agent] SQLite write for epoch {epoch} did not stick; retrying...", flush=True)
            time.sleep(0.5 * (attempt + 1))
        finally:
            if conn is not None:
                conn.close()


# ── NIC counters ──────────────────────────────────────────────────────────────

def read_nic_counters(iface=None):
    """
    Read cumulative bytes sent/received from /proc/net/dev.
    Returns (bytes_sent, bytes_recv) as a snapshot (not a rate).
    If iface is None, sums across all non-loopback interfaces.
    """
    stats = psutil.net_io_counters(pernic=True)
    total_sent = 0
    total_recv = 0
    for name, counters in stats.items():
        if name == 'lo':
            continue
        if iface and name != iface:
            continue
        total_sent += counters.bytes_sent
        total_recv += counters.bytes_recv
    return total_sent, total_recv


def compute_nic_rate(snap1, snap2, elapsed):
    """
    Convert two cumulative snapshots into bytes/sec rates.
    snap = (bytes_sent, bytes_recv)
    """
    sent_rate = (snap2[0] - snap1[0]) / elapsed if elapsed > 0 else 0
    recv_rate = (snap2[1] - snap1[1]) / elapsed if elapsed > 0 else 0
    return int(sent_rate), int(recv_rate)


# ── RTT probe server ──────────────────────────────────────────────────────────

def create_rtt_server_socket():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind(('0.0.0.0', RTT_PORT))
        server.listen(10)
        server.settimeout(1.0)
    except OSError:
        server.close()
        raise
    return server


def start_rtt_server(server, stop_event):
    """
    Lightweight TCP echo server. Runs in a background thread.
    Peers connect, send PING, we echo it back immediately.
    This is what makes RTT measurement possible without ICMP privileges.
    """
    print(f"[RTT server] Listening on port {RTT_PORT}")

    while not stop_event.is_set():
        try:
            conn, _ = server.accept()
            data = conn.recv(64)
            conn.sendall(data)   # echo immediately — minimizes server processing time
            conn.close()
        except socket.timeout:
            continue
        except Exception as e:
            if not stop_event.is_set():
                print(f"[RTT server] Error: {e}")

    server.close()


def measure_rtt(peer_host, num_probes=RTT_PROBES):
    """
    Measure round-trip time to a peer by opening a TCP connection,
    sending a small payload, and timing the echo response.
    Returns median RTT in milliseconds, or None if unreachable.
    
    Using TCP (not ICMP ping) so no special privileges are needed,
    consistent with the paper's no-elevated-privileges requirement.
    """
    rtts = []
    for _ in range(num_probes):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(RTT_TIMEOUT)
            t0 = time.perf_counter()
            sock.connect((peer_host, RTT_PORT))
            sock.sendall(PROBE_PAYLOAD)
            sock.recv(64)
            t1 = time.perf_counter()
            sock.close()
            rtts.append((t1 - t0) * 1000)   # convert to ms
        except Exception:
            pass

    if not rtts:
        return None

    rtts.sort()
    return rtts[len(rtts) // 2]   # median — more robust than mean


def measure_all_rtts(peers):
    """
    Measure RTT to every peer and return the worst (max) RTT.
    We use max because all-reduce performance is bounded by the slowest
    communication path — matching the T_comm = max(T_comm_node_i) model
    from the paper's insights section.
    """
    worst_rtt = None
    results = {}
    for peer in peers:
        rtt = measure_rtt(peer)
        if rtt is not None:
            results[peer] = rtt
            print(f"  [RTT] → {peer}: {rtt:.2f} ms")
        else:
            print(f"  [RTT] → {peer}: unreachable")

    if results:
        worst_rtt = max(results.values())
    return worst_rtt, results


# ── epoch sync ────────────────────────────────────────────────────────────────
# The training hook (hook.py) writes a small JSON file at the end of each
# epoch containing the all-reduce timing for that epoch. The agent watches
# this file to know:
#   (a) when an epoch has completed (safe to run RTT probes)
#   (b) what the all-reduce time was for that epoch

def wait_for_epoch_signal(job_id, node_id, expected_epoch, timeout=300):
    """
    Block until the training hook signals that the next expected epoch completed.
    Reads the per-node signal file so each agent gets its own node's
    actual measured all-reduce time, not rank 0's shared value.
    Returns (epoch_number, all_reduce_ms) or None on timeout/done-before-file.
    """
    signal_path = os.path.join(SYNC_DIR, f'{job_id}_{node_id}_epoch_{expected_epoch}.json')
    deadline = time.time() + timeout

    while time.time() < deadline:
        if os.path.exists(signal_path):
            try:
                with open(signal_path, 'r') as f:
                    data = json.load(f)
                epoch = data.get('epoch', 0)
                if epoch == expected_epoch:
                    print(f"[Agent] Read epoch signal file: {signal_path}")
                    return epoch, data.get('all_reduce_ms', None)
            except (json.JSONDecodeError, KeyError):
                pass
        elif read_final_signal(job_id, node_id):
            print(
                f"[Agent] Done signal exists before epoch {expected_epoch} file; "
                "assuming training stopped early"
            )
            return None
        time.sleep(0.2)

    return None


def read_final_signal(job_id, node_id):
    """Check if this node's training has signalled completion."""
    done_path = os.path.join(SYNC_DIR, f'{job_id}_{node_id}_done')
    return os.path.exists(done_path)


def resolve_job_id(job_id, timeout=60):
    """
    Resolve the shared job ID when launch.py was started without --job-id.

    Rank 0 writes current_job_id before the first epoch starts. The agent runs
    before training, so it has to wait briefly for that file.
    """
    if job_id != 'auto':
        return job_id

    job_id_path = os.path.join(SYNC_DIR, 'current_job_id')
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with open(job_id_path, 'r') as f:
                resolved = f.read().strip()
            if resolved:
                print(f"[Agent] Resolved auto job ID: {resolved}", flush=True)
                return resolved
        except FileNotFoundError:
            pass
        time.sleep(0.2)

    raise TimeoutError(f"Timed out waiting for {job_id_path}")


# ── main agent loop ───────────────────────────────────────────────────────────

def run_agent(node_id, job_id, peers):
    """
    Main collection loop. Runs for the duration of one training job.

    Per-epoch flow:
      1. Collect NIC rate (sampled continuously in background thread)
      2. Wait for epoch-complete signal from training hook
      3. In the idle window after the epoch: run RTT probes to all peers
      4. Read all-reduce timing from the epoch signal
      5. Write one metric row to the database
    """
    os.makedirs(SYNC_DIR, exist_ok=True)
    job_id = resolve_job_id(job_id)

    stop_event = threading.Event()

    # Start RTT echo server so peers can probe us
    try:
        server_socket = create_rtt_server_socket()
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            print(
                f"[RTT server] ERROR: port {RTT_PORT} is already in use. "
                "Run ./scripts/clean_stale_processes.sh and retry.",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(f"[RTT server] ERROR: failed to bind port {RTT_PORT}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)

    server_thread = threading.Thread(
        target=start_rtt_server,
        args=(server_socket, stop_event,),
        daemon=True
    )
    server_thread.start()

    # NIC sampling state — we track running averages between epoch signals
    nic_snap_prev = read_nic_counters()
    nic_time_prev = time.time()
    nic_sent_rate = 0
    nic_recv_rate = 0

    # Background NIC sampling thread updates rates continuously
    nic_lock = threading.Lock()

    def nic_sampler():
        nonlocal nic_snap_prev, nic_time_prev, nic_sent_rate, nic_recv_rate
        while not stop_event.is_set():
            time.sleep(NIC_INTERVAL)
            snap = read_nic_counters()
            now  = time.time()
            with nic_lock:
                elapsed = now - nic_time_prev
                s, r = compute_nic_rate(nic_snap_prev, snap, elapsed)
                nic_sent_rate = s
                nic_recv_rate = r
                nic_snap_prev = snap
                nic_time_prev = now

    nic_thread = threading.Thread(target=nic_sampler, daemon=True)
    nic_thread.start()

    print(f"[Agent] Started | node={node_id} | job={job_id} | peers={peers}")

    expected_epoch = 1
    final_grace_needed = False

    while True:
        # Wait for training to signal epoch complete
        result = wait_for_epoch_signal(job_id, node_id, expected_epoch)

        if result is None:
            print("[Agent] Timeout waiting for epoch signal — assuming training finished")
            break

        epoch, all_reduce_ms = result
        expected_epoch = epoch + 1
        ar_label = f"{all_reduce_ms:.1f}ms" if all_reduce_ms is not None else "unknown"
        print(f"[Agent] Epoch {epoch} complete | all_reduce={ar_label}")

        # RTT probes run NOW — in the idle window between epochs
        # This is the gating mechanism described in the challenges section
        peer_rtts = {}
        if peers:
            print(f"[Agent] Running RTT probes (idle window)...")
            worst_rtt, measured_rtts = measure_all_rtts(peers)
            peer_rtts = {peer: measured_rtts.get(peer) for peer in peers}
        else:
            worst_rtt = None

        # Snapshot current NIC rates
        with nic_lock:
            sent = nic_sent_rate
            recv = nic_recv_rate

        insert_metric(
            node_id=node_id,
            job_id=job_id,
            epoch=epoch,
            nic_sent=sent,
            nic_recv=recv,
            all_reduce_ms=all_reduce_ms,
            rtt_ms=worst_rtt,
            peer_rtts=peer_rtts,
        )

        print(f"[Agent] Wrote metrics | nic_sent={sent/1e6:.1f}MB/s "
              f"nic_recv={recv/1e6:.1f}MB/s rtt={worst_rtt}ms "
              f"peer_rtt_rows={len(peer_rtts)}")

        # Check if training is done
        if read_final_signal(job_id, node_id):
            final_grace_needed = True
            print("[Agent] Training done; keeping RTT server alive for final peer probes...")
            break

    if final_grace_needed:
        time.sleep(FINAL_RTT_SERVER_GRACE_SEC)
        print("[Agent] Final RTT grace period complete. Shutting down.")

    stop_event.set()
    print("[Agent] Exiting.")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Telemetry agent for distributed training')
    parser.add_argument('--node-id', required=True,
                        help='This node\'s identifier, e.g. node0')
    parser.add_argument('--job-id',  required=True,
                        help='Job identifier shared across all nodes for this run')
    parser.add_argument('--peers',   nargs='*', default=[],
                        help='Hostnames of peer nodes to probe, e.g. node1 node2 node3 node4')
    args = parser.parse_args()

    run_agent(args.node_id, args.job_id, args.peers)
