"""
Training hook — measures all-reduce timing and signals the telemetry agent.

Key change from v1:
  Every rank now writes its OWN signal file keyed by node_id, so each
  node's agent reads that node's actual measured all-reduce time rather
  than rank 0's value broadcast to everyone.

  Signal file per node and epoch:
    /workspace/data/sync/<job_id>_<node_id>_epoch_<epoch>.json
    { "epoch": 2, "all_reduce_ms": 31.4, "node_id": "node1", "timestamp": ... }

  Why per-node timing differs in practice:
    In DDP all-reduce, each node measures from when it calls backward()
    to when the optimizer can step. A node with higher RTT to its peers
    will spend more wall-clock time waiting for gradients to arrive,
    so its measured all-reduce time is genuinely longer. These differences
    are small on a fast local network but become significant under congestion,
    which is exactly the straggler effect we are trying to detect.
"""

import os
import json
import time
import socket
import threading

SYNC_DIR = '/workspace/data/sync'

_job_id    = None
_rank      = None
_node_id   = None
_ar_times  = []
_lock      = threading.Lock()
_installed = False


def install(job_id, rank, node_id=None):
    """
    Call once after dist.init_process_group().
    node_id defaults to the container hostname if not provided.
    """
    global _job_id, _rank, _node_id, _installed
    _job_id    = job_id
    _rank      = rank
    _node_id   = node_id or socket.gethostname()
    _installed = True
    os.makedirs(SYNC_DIR, exist_ok=True)
    print(f"[Hook] Installed | job={job_id} rank={rank} node={_node_id}")


def before_allreduce():
    """Call immediately before loss.backward()."""
    return time.perf_counter()


def after_allreduce(t_start):
    """Call immediately after optimizer.step()."""
    if not _installed:
        return
    elapsed_ms = (time.perf_counter() - t_start) * 1000
    with _lock:
        _ar_times.append(elapsed_ms)


def end_epoch(epoch_num):
    """
    Call at the end of each epoch.
    Every rank writes its own signal file — the agent on this node
    reads this node's file specifically.
    """
    with _lock:
        times = list(_ar_times)
        _ar_times.clear()

    mean_ar = (sum(times) / len(times)) if times else None

    if mean_ar is not None:
        _write_epoch_signal(epoch_num, mean_ar)


def training_done():
    """Call after the last epoch on all ranks."""
    done_path = os.path.join(SYNC_DIR, f'{_job_id}_{_node_id}_done')
    open(done_path, 'w').close()
    print(f"[Hook] Written done signal → {done_path}")

    if _rank == 0:
        global_done = os.path.join(SYNC_DIR, f'{_job_id}_done')
        open(global_done, 'w').close()


def _write_epoch_signal(epoch, all_reduce_ms):
    """
    Atomic write to a per-node, per-epoch signal file.
    Agent reads /workspace/data/sync/<job_id>_<node_id>_epoch_<epoch>.json
    """
    signal_path = os.path.join(SYNC_DIR, f'{_job_id}_{_node_id}_epoch_{epoch}.json')
    tmp_path    = signal_path + '.tmp'

    payload = {
        'epoch':         epoch,
        'all_reduce_ms': round(all_reduce_ms, 3),
        'node_id':       _node_id,
        'timestamp':     time.time()
    }

    with open(tmp_path, 'w') as f:
        json.dump(payload, f)

    os.replace(tmp_path, signal_path)
    print(f"[Hook] Epoch {epoch} | node={_node_id} | all_reduce={all_reduce_ms:.1f}ms")
