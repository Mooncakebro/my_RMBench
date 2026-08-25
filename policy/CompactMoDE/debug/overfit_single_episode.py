"""
Single-episode overfit test for BOTH CompactMoDE variants.

Purpose: wiring correctness. A correctly wired policy should drive the
diffusion loss down clearly when trained on a tiny fixed set of frames
(one episode). This is NOT a performance benchmark.

  - baseline: random minibatches over the first N frames of episode 0
  - compact:  sequential TBPTT chunks over the same frames (memory carried,
              detached at chunk boundaries, episode loops)

Runs on the 8GB dev GPU: frozen base (bf16) + small DiT by default.

Example (from policy/CompactMoDE/):
    python debug/overfit_single_episode.py --variant baseline --steps 200
    python debug/overfit_single_episode.py --variant compact --steps 100
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))

from compact_mode.config import CompactMoDEConfig
from compact_mode.dataset import DEFAULT_DATA_ROOT, FrameDataset, collate_frames


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=("baseline", "compact", "both"), default="both")
    p.add_argument("--task", type=str, default="swap_blocks")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--n-frames", type=int, default=100)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8, help="baseline minibatch size")
    p.add_argument("--chunk-size", type=int, default=8, help="compact TBPTT chunk length")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--mem-opt", choices=("sgd", "adamw"), default="sgd",
                   help="optimizer for the 300M side-memory params (compact only). "
                        "Default sgd: AdamW's m/v states (2.4GB) do not fit on the "
                        "8GB dev GPU next to the 4.4GB frozen base. Server training "
                        "(train_compact.py) uses AdamW for everything.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def cfg_from_args(args) -> CompactMoDEConfig:
    return CompactMoDEConfig(
        freeze_base=True, bf16=True, gradient_checkpointing=True,
        embed_dim=args.embed_dim, n_layers=args.n_layers, n_heads=args.n_heads,
        attn_pdrop=0.0, mlp_pdrop=0.0, goal_drop=0.0,
        lr=args.lr, chunk_size=args.chunk_size,
    )


def episode_frame_indices(dataset: FrameDataset, episode: int, n_frames: int):
    rows = dataset.episode_valid_rows[episode]
    rows = rows[:n_frames]
    row_to_index = {row: i for i, row in enumerate(dataset.valid_rows)}
    return [row_to_index[r] for r in rows]


def run_baseline(args, dataset, indices, device):
    from compact_mode.model_baseline import BaselineQwenMoDEPolicy

    print(f"=== Baseline overfit: {len(indices)} frames, {args.steps} steps, "
          f"batch {args.batch_size} ===")
    model = BaselineQwenMoDEPolicy(cfg_from_args(args)).to(device)
    model.set_normalizers(dataset.action_normalizer, dataset.state_normalizer)
    opt = torch.optim.AdamW(
        model.get_optim_groups(args.lr, args.lr, args.lr, 0.01), lr=args.lr)
    model.train()

    first, last, losses = None, None, []
    rng = random.Random(args.seed)
    for step in range(args.steps):
        batch = collate_frames([dataset[i] for i in rng.sample(indices, args.batch_size)])
        loss = model.diffusion_loss(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        losses.append(loss.item())
        if first is None:
            first = loss.item()
        if (step + 1) % args.log_every == 0:
            last = sum(losses[-args.log_every:]) / args.log_every
            print(f"  step {step + 1}/{args.steps}: loss {last:.4f}")
    print(f"  RESULT baseline: first {first:.4f} -> last {last:.4f} "
          f"({last / first:.1%} of initial)")
    del model, opt
    torch.cuda.empty_cache()
    return first, last


def run_compact(args, dataset, indices, device):
    from compact_mode.model_compact import CompactMoDEPolicy

    n_chunks = max(1, len(indices) // args.chunk_size)
    print(f"=== COMPACT overfit: {len(indices)} frames, {args.steps} chunks of "
          f"{args.chunk_size} ({n_chunks} chunks/episode pass) ===")
    model = CompactMoDEPolicy(cfg_from_args(args)).to(device)
    model.set_normalizers(dataset.action_normalizer, dataset.state_normalizer)
    groups = model.get_optim_groups(args.lr, args.lr, args.lr, 0.01)
    mem_groups = [g for g in groups if g["name"] == "memory"]
    other_groups = [g for g in groups if g["name"] != "memory"]
    opt = torch.optim.AdamW(other_groups, lr=args.lr)
    if args.mem_opt == "adamw":
        opt_mem = torch.optim.AdamW(mem_groups, lr=args.lr)
    else:
        opt_mem = torch.optim.SGD(
            [p for g in mem_groups for p in g["params"]], lr=args.lr)
    opts = [opt, opt_mem]
    model.train()

    first, last = None, None
    diff_losses = []
    cursor = 0
    for step in range(args.steps):
        if cursor + args.chunk_size > len(indices):
            cursor = 0  # loop the episode
        chunk = indices[cursor:cursor + args.chunk_size]
        cursor += args.chunk_size

        memory = model.init_memory(1, device)
        chunk_loss = 0.0
        last_dict = {}
        for s in range(args.chunk_size):
            batch = collate_frames([dataset[chunk[s]]])
            loss, loss_dict, memory = model.diffusion_loss(batch, memory)
            chunk_loss = chunk_loss + loss / args.chunk_size
            last_dict = loss_dict

        for o in opts:
            o.zero_grad(set_to_none=True)
        chunk_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        for o in opts:
            o.step()
        memory = model.detach_memory(memory)

        diff_losses.append(float(last_dict["diffusion"]))
        if first is None:
            first = float(last_dict["diffusion"])
        if (step + 1) % args.log_every == 0:
            last = sum(diff_losses[-args.log_every:]) / args.log_every
            print(f"  chunk {step + 1}/{args.steps}: diffusion {last:.4f} "
                  f"(total {float(last_dict['total']):.4f} "
                  f"obs {float(last_dict['obs']):.3f} nll {float(last_dict['nll']):.3f})")
    print(f"  RESULT compact: first {first:.4f} -> last {last:.4f} "
          f"({last / first:.1%} of initial)")
    del model, opt, opt_mem
    torch.cuda.empty_cache()
    return first, last


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda")

    dataset = FrameDataset(Path(args.data_root) / args.task,
                           action_horizon=10, normalize=True)
    indices = episode_frame_indices(dataset, args.episode, args.n_frames)
    print(f"Overfit on {args.task} episode {args.episode}: {len(indices)} frames; "
          f"instruction: {dataset[indices[0]]['instruction'][:80]!r}")

    results = {}
    t0 = time.time()
    if args.variant in ("baseline", "both"):
        results["baseline"] = run_baseline(args, dataset, indices, device)
    if args.variant in ("compact", "both"):
        results["compact"] = run_compact(args, dataset, indices, device)

    print(f"\n==== overfit summary ({time.time() - t0:.0f}s) ====")
    ok = True
    for name, (first, last) in results.items():
        dropped = last < 0.5 * first
        ok = ok and dropped
        print(f"  {name}: {first:.4f} -> {last:.4f} "
              f"({'OK' if dropped else 'WEAK — investigate'})")
    print("OVERFIT TEST " + ("PASSED" if ok else "INCONCLUSIVE"))


if __name__ == "__main__":
    main()
