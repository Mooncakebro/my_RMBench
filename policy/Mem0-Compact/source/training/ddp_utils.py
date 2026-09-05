"""
DDP helpers shared by the training loop and debug tests.

torchrun-compatible: reads RANK / WORLD_SIZE / LOCAL_RANK from the environment
(as set by torchrun / torch.distributed.run). When absent, falls back to
single-process mode (rank 0, world_size 1, no process group).

Backend/device selection:
  - nccl + one cuda device per rank when enough physical GPUs exist
    (world_size <= torch.cuda.device_count()).
  - Otherwise falls back to gloo + cpu — lets multi-rank DDP be tested on
    single-GPU machines (NCCL refuses two ranks on one physical device).
"""
from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.distributed as dist


def _backend_for(world_size: int) -> str:
    if torch.cuda.is_available() and world_size <= torch.cuda.device_count():
        return "nccl"
    return "gloo"


def setup_distributed() -> Tuple[int, int, int, bool]:
    """Init the process group if launched under torchrun.

    Returns (rank, world_size, local_rank, ddp_enabled).
    """
    if "WORLD_SIZE" in os.environ and "RANK" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        backend = _backend_for(world_size)
        dist.init_process_group(backend=backend)
        if backend == "nccl":
            torch.cuda.set_device(local_rank % torch.cuda.device_count())
        return rank, world_size, local_rank, True
    return 0, 1, 0, False


def get_ddp_device(local_rank: int, world_size: int) -> torch.device:
    """Device for this rank, matching setup_distributed's backend choice."""
    if world_size > 1:
        if torch.cuda.is_available() and world_size <= torch.cuda.device_count():
            return torch.device(f"cuda:{local_rank}")
        return torch.device("cpu")  # gloo fallback (single-GPU multi-rank)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_rank0() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def rank0_print(*args, **kwargs):
    if is_rank0():
        print(*args, **kwargs)


def all_reduce_avg(t: torch.Tensor) -> torch.Tensor:
    """Average a tensor across ranks (in place). No-op for single process."""
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t.div_(dist.get_world_size())
    return t


def destroy_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
