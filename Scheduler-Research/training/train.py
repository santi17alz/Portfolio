"""
train.py — distributed training with telemetry hook integrated.

Changes from the original:
  1. Import telemetry.hook
  2. Call hook.install() after dist.init_process_group()
  3. Wrap loss.backward() + optimizer.step() with timing calls
  4. Call hook.end_epoch() at the end of each epoch
  5. Call hook.training_done() after the last epoch

Everything else is identical to the original train.py.
"""

import os
import time
import uuid
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler

import sys
sys.path.insert(0, '/workspace')
import telemetry.hook as telemetry_hook


def setup():
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
        world_size=int(os.environ['WORLD_SIZE']),
        rank=int(os.environ['RANK'])
    )


def cleanup():
    dist.destroy_process_group()


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.net(x)


def train():
    rank       = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    epochs     = int(os.environ.get("EPOCHS", "3"))

    setup()

    # ── telemetry hook ────────────────────────────────────────────────────────
    # Job ID is shared across all nodes via an environment variable so the
    # agent on each node writes to the same job partition in the database.
    # If not set, generate one on rank 0 and broadcast it.
    job_id_env = os.environ.get('JOB_ID', None)
    if job_id_env:
        job_id = job_id_env
    else:
        # Generate on rank 0, broadcast to all ranks via a simple barrier trick
        if rank == 0:
            job_id = str(uuid.uuid4())[:8]
            # Write to a shared file so other ranks can read it
            os.makedirs('/workspace/data/sync', exist_ok=True)
            with open('/workspace/data/sync/current_job_id', 'w') as f:
                f.write(job_id)
        dist.barrier()   # wait for rank 0 to write
        if rank != 0:
            with open('/workspace/data/sync/current_job_id', 'r') as f:
                job_id = f.read().strip()

    telemetry_hook.install(job_id=job_id, rank=rank, node_id=os.environ.get('HOSTNAME', f'node{rank}'))
    # ─────────────────────────────────────────────────────────────────────────

    if rank == 0:
        print(f"[Master] Cluster ready — {world_size} nodes connected | job_id={job_id} | epochs={epochs}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(
        '/workspace/data',
        train=True,
        download=True,
        transform=transform
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader  = DataLoader(dataset, batch_size=64, sampler=sampler)

    model     = DDP(SimpleNet())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        epoch_start = time.time()
        total_loss  = 0

        for batch_idx, (data, target) in enumerate(loader):
            optimizer.zero_grad()
            output = model(data)
            loss   = criterion(output, target)

            # ── time the all-reduce (backward pass triggers DDP gradient sync)
            t_ar = telemetry_hook.before_allreduce()
            loss.backward()    # DDP all-reduce happens here
            optimizer.step()
            telemetry_hook.after_allreduce(t_ar)
            # ─────────────────────────────────────────────────────────────────

            total_loss += loss.item()

        epoch_time = time.time() - epoch_start

        # ── signal end of epoch to telemetry agent ────────────────────────────
        telemetry_hook.end_epoch(epoch + 1)
        # ─────────────────────────────────────────────────────────────────────

        if rank == 0:
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")

            with open('/workspace/results/training_log.txt', 'a') as f:
                f.write(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s\n")

    # ── signal training complete ──────────────────────────────────────────────
    telemetry_hook.training_done()
    # ─────────────────────────────────────────────────────────────────────────

    cleanup()
    if rank == 0:
        print("Training complete!")


if __name__ == "__main__":
    train()
