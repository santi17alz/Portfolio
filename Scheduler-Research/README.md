# Telemetry Agent Simulation

A distributed AI training monitor that detects node-level network degradation in real time and recommends which nodes to include or avoid — so the scheduler doesn't waste compute time training on a slow cluster.

**Cloud Computing Research · University of Massachusetts Amherst · Spring 2026**

---

## The Problem

Distributed deep learning training with PyTorch Distributed Data Parallel (DDP) requires every node to synchronize gradients at the end of each epoch — the **all-reduce** step. When one node in the cluster has degraded network performance (high latency, packet loss, or bandwidth limits), it bottlenecks the entire training job. Every other node waits for it. The slower the weakest node, the slower the whole run.

Standard schedulers allocate jobs to nodes without real-time visibility into network health. This project builds that visibility layer and turns it into actionable scheduling recommendations.

---

## System Architecture

 ┌─────────────────────── Docker cluster (training-net) ────────────────────────┐

 │                                                                                │

 │   node0 ─── node1 ─── node2 ─── node3 ─── node4   (PyTorch DDP / MNIST)     │

 │       └─────────┴─────────┴────────┘                                          │

 │              all-reduce gradient sync (each epoch)                             │

 │                          │                                                     │

 │                   hook.py writes epoch                                         │

 │                   timing signals to disk                                       │

 │                          │                                                     │

 │              telemetry/agent.py (per node)                                     │

 │              ├── reads NIC counters from /proc/net/dev                         │

 │              ├── probes RTT to all peers (TCP, gated between epochs)           │

 │              └── records all-reduce timing from hook signals                   │

 │                          │                                                     │

 │                    data/metrics.db  (SQLite WAL)                               │

 │                          │                                                     │

 │              ┌───────────┴──────────────┐                                      │

 │              │                          │                                       │

 │       auto\_score.py              dashboard/app.py                              │

 │       (health scores)            (Flask, :5050)                                │

 │              │                          │                                       │

 │       recommend.py  ◄──────────── API calls                                    │

 │       (node advisory)                                                           │

 └────────────────────────────────────────────────────────────────────────────────┘

---

## What the Agent Measures

| Metric | Source | Purpose |
| :---- | :---- | :---- |
| **All-reduce time** (ms) | `hook.py` epoch timing files | Measures gradient sync cost per epoch |
| **Inter-node RTT** (ms) | TCP probes, gated between epochs | Detects which node-to-node paths are slow |
| **NIC throughput** (bytes/s) | `/proc/net/dev` counters | Measures actual network utilization per node |

### Key Design Decision: Epoch-Gated RTT Probing

RTT probes cannot run during training — probe traffic competes with DDP all-reduce bandwidth and inflates the very metric we're trying to measure. The agent detects the idle window between epochs (via `epoch_sync` files written by the training hook) and **only probes RTT during that window**. This keeps measurement traffic from interfering with training.

---

## Health Scoring & Recommendation Engine

Each node gets a **health score** computed from its recent telemetry. The recommendation engine uses weighted risk signals to rank nodes:

| Signal | Weight | What it captures |
| :---- | :---- | :---- |
| Health score | 55% | Overall node quality (higher \= healthier) |
| Average RTT | 20% | General network latency to peers |
| High-RTT paths | 10% | Count of peer paths above 10ms threshold |
| Timeout risk | 15% | RTT probe timeouts (severe degradation) |

The engine identifies nodes whose risk score is separated from the cluster average by a configurable margin, then recommends avoiding them and running the next job on the remaining subset.

---

## Experiments

Network degradation is injected with `tc/netem` inside the Docker containers (`NET_ADMIN` capability). Three primary degradation types tested:

| Experiment | Fault Injected | Expected Signal |
| :---- | :---- | :---- |
| Baseline | None | Clean RTT matrix, stable all-reduce |
| Delay | `delay 20ms` on one node | Elevated RTT for all paths through that node |
| Packet loss | `loss 5%` on one node | RTT instability, occasional timeouts |
| Bandwidth limit | `rate 10mbit` on one node | High all-reduce time, NIC saturation |
| Recommended subset | Previous degraded job → drop bad node → re-run on 4 nodes | Reduced all-reduce time |

The core experimental loop is:

1. Run a clean baseline (all 5 nodes)  
2. Inject degradation on one node  
3. Run the same job — observe telemetry signal  
4. Query recommendation engine — it should advise dropping the degraded node  
5. Run a 4-node job on the recommended subset — observe improvement

---

## Project Structure

telemetry/

  agent.py        Telemetry collector: NIC, RTT probes, all-reduce timing

  hook.py         PyTorch training hook that writes epoch timing signals

  launch.py       Starts agent \+ training on each node

training/

  train.py        PyTorch DDP MNIST training script (the workload)

dashboard/

  app.py          Flask dashboard and JSON API (http://127.0.0.1:5050)

  health\_scores.py  Health score formula

  auto\_score.py   Background process that recomputes scores continuously

  recommend.py    Advisory node recommendation engine

  init\_db.py      SQLite schema

scripts/

  run\_job.sh              Launch a distributed job across all/subset of nodes

  run\_recommended\_job.sh  Launch a job using the recommendation engine's output

  apply\_netem.sh          Inject delay / loss / rate limit on a node

  clear\_netem.sh          Remove injected degradation

  show\_recommendation.sh  Print current node recommendation

  show\_rtt\_matrix.sh      Print directed RTT between all node pairs

  reset\_experiment.sh     Clear metrics and sync files between experiments

reports/figures/          Generated plots (all-reduce time, RTT matrices, recovery)

results/                  Per-node log files from each experiment run

---

## Setup & Quick Start

**Requires:** Docker Desktop or Docker Engine with Compose plugin.

\# Create the network and start the cluster

docker network create training-net 2\>/dev/null || true

docker compose up \-d \--build

\# Dashboard at http://127.0.0.1:5050

**Run the core experiment sequence:**

\# Baseline

./scripts/run\_job.sh five\_baseline\_001

./scripts/show\_rtt\_matrix.sh five\_baseline\_001

\# Inject degradation and observe

./scripts/apply\_netem.sh node3 delay 20ms

./scripts/run\_job.sh five\_node3\_delay\_20ms\_001

./scripts/clear\_netem.sh node3

\# Query the recommendation

./scripts/show\_recommendation.sh five\_node3\_delay\_20ms\_001 4

\# Run on recommended subset and compare

./scripts/run\_recommended\_job.sh recommended\_without\_node3\_001 five\_node3\_delay\_20ms\_001 4

---

## Tech Stack

`Python` · `PyTorch DDP` · `Flask` · `SQLite (WAL)` · `Docker` · `tc/netem` · `psutil` · `/proc/net/dev`

---

## Related

This project is the implementation behind the cloud computing research paper submitted to ACM (Spring 2026\) on scheduling framework optimizations for distributed AI model training.  
