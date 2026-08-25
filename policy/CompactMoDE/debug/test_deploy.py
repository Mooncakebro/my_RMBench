"""
Deploy round-trip test: save_pretrained dir -> CompactMoDEDeployer -> fake
observation -> action chunk. Verifies the train->deploy chain (checkpoint
loading, normalizer decode, memory carry) WITHOUT the SAPIEN simulator.

Uses the final/ dirs written by the mini debug training runs:
    runs/debug_baseline/final   (variant=baseline)
    runs/debug_compact/final    (variant=compact)

Example (from policy/CompactMoDE/):
    python debug/test_deploy.py
    python debug/test_deploy.py --variant compact --n-calls 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))

from compact_mode.dataset import DEFAULT_DATA_ROOT, FrameDataset


class FakeTaskEnv:
    """Minimal stand-in for the RMBench TASK_ENV (instruction only)."""

    def __init__(self, instruction: str):
        self._instruction = instruction

    def get_instruction(self):
        return self._instruction


def fake_observation(sample: dict) -> dict:
    """Build an observation dict in the RMBench get_obs() layout from a
    FrameDataset sample. NOTE: sample['image'] is uint8 HWC like the sim's
    decoded rgb; the joint vector is reconstructed by DE-normalizing the
    stored state (deploy encodes it again, so this round-trips)."""
    return {
        "observation": {"head_camera": {"rgb": sample["image"]}},
        "joint_action": {"vector": sample["raw_state"]},
    }


def run(variant: str, ckpt: Path, dataset: FrameDataset, n_calls: int):
    from deploy_policy import CompactMoDEDeployer

    print(f"=== deploy test: variant={variant} ckpt={ckpt} ===")
    deployer = CompactMoDEDeployer({
        "variant": variant,
        "checkpoint_path": str(ckpt),
        "num_sampling_steps": 3,  # keep the test fast
    })
    deployer.reset()

    # attach raw (un-normalized) state for fake observations
    table = dataset.table
    for call in range(n_calls):
        sample = dataset[call]
        sample["raw_state"] = np.asarray(
            table.at[dataset.valid_rows[call], "observation.state"], dtype=np.float32)
        env = FakeTaskEnv(sample["instruction"])
        actions = deployer.get_action_chunk(fake_observation(sample),
                                            env.get_instruction())
        assert actions.shape == (10, 14), f"chunk shape {actions.shape}"
        assert np.isfinite(actions).all(), "non-finite actions"
        # plausible raw range: joint targets roughly within dataset bounds
        lo, hi = dataset.action_normalizer.min, dataset.action_normalizer.max
        margin = (hi - lo).abs().numpy() * 0.5 + 0.1
        assert (actions > (lo.numpy() - margin)).all() and \
               (actions < (hi.numpy() + margin)).all(), \
            f"actions wildly out of range: min {actions.min():.2f} max {actions.max():.2f}"
        print(f"  call {call}: chunk {actions.shape}, "
              f"range [{actions.min():.3f}, {actions.max():.3f}], "
              f"prev_action set: {deployer.prev_action is not None}")
    if variant == "compact":
        assert deployer.memory is not None, "compact memory was not carried"
        m0 = deployer.memory["m"][0]
        assert torch.isfinite(m0).all(), "non-finite memory"
        print(f"  memory carried: layer0 {tuple(m0.shape)}")
    deployer.reset()
    assert deployer.memory is None and deployer.prev_action is None
    print("  reset OK")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=("baseline", "compact", "both"), default="both")
    p.add_argument("--task", type=str, default="swap_blocks")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--runs-dir", type=Path, default=THIS_DIR.parent / "runs")
    p.add_argument("--n-calls", type=int, default=3)
    args = p.parse_args()

    dataset = FrameDataset(Path(args.data_root) / args.task,
                           action_horizon=10, normalize=True)

    variants = {"baseline": args.runs_dir / "debug_baseline" / "final",
                "compact": args.runs_dir / "debug_compact" / "final"}
    for name, ckpt in variants.items():
        if args.variant not in (name, "both"):
            continue
        if not ckpt.exists():
            print(f"[skip] {name}: {ckpt} does not exist "
                  f"(run the mini training first)")
            continue
        run(name, ckpt, dataset, args.n_calls)
    print("DEPLOY TEST DONE")


if __name__ == "__main__":
    main()
