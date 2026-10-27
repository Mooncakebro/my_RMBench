#!/usr/bin/env python3
"""Teacher-forced open-loop evaluation for the released official Mem-0 policy.

The policy sees each recorded observation in chronological order and updates
its MemoryBank, but its predicted actions are never applied to the trajectory.
The outputs match open_loop_eval_compact.py: per-episode joint plots, JSON
metrics, and an aggregate summary.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
MEM0_ROOT = REPO_ROOT / "policy" / "Mem-0"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=MEM0_ROOT / "source/config/execution_module_train.yaml")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="LeRobot dataset directory for one M1 task")
    parser.add_argument("--norm-stats", type=Path, default=None,
                        help="optional training normalization JSON")
    parser.add_argument("--base-model", type=str, default=None,
                        help="override execution_module.qwen_vl.model_path")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--episode-ids", type=int, nargs="+", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--start-frame", type=int, default=0,
                        help="earlier frames still update memory")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None,
                        help="prediction interval; default is action horizon")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--instruction", default=None,
                        help="override the per-frame instruction stored in the dataset")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "openloop_eval_result")
    return parser.parse_args()


def _load_model(args, device):
    config_path = args.config.expanduser().resolve()
    cfg = OmegaConf.load(config_path)
    if args.base_model:
        cfg.execution_module.qwen_vl.model_path = args.base_model
    # The upstream executor uses FlashAttention2 unconditionally and creates
    # the VLM in bf16. Keep its original architecture/inference implementation.
    sys.path.insert(0, str(MEM0_ROOT))
    from source.models.execution_module.memorymatters_executor import MemoryMattersExecutor

    model = MemoryMattersExecutor(cfg, device=device).to(device)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Expected a Mem-0 checkpoint mapping")
    state = payload.get("model_state_dict", payload.get("model", payload.get("state_dict")))
    if state is None:
        state = payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/model mismatch: missing={len(missing)} {missing[:8]}, "
            f"unexpected={len(unexpected)} {unexpected[:8]}"
        )
    model.eval()
    return model, cfg


@torch.inference_mode()
def _eval_episode(args, model, dataset, raw_actions, raw_states, bounds,
                  episode_id, device, output_dir, horizon, helpers, names):
    start, end = bounds[episode_id]
    episode_length = end - start
    first_anchor = args.start_frame
    valid_last = max(0, episode_length - horizon + 1)
    if first_anchor >= valid_last:
        raise ValueError(f"Episode {episode_id} has no complete {horizon}-action horizon")
    stop = episode_length if args.steps is None else min(episode_length, first_anchor + args.steps)
    stop = min(stop, valid_last)
    stride = args.stride
    anchors, preds, targets, preds_raw, targets_raw = [], [], [], [], []
    memory_id = int(episode_id)
    model.memory_bank.reset([memory_id])

    action_stats = dataset.dataset.meta.stats["action"]
    action_min = (dataset.action_min if hasattr(dataset, "action_min")
                  else torch.as_tensor(action_stats["min"]))
    action_max = (dataset.action_max if hasattr(dataset, "action_max")
                  else torch.as_tensor(action_stats["max"]))
    action_min = helpers._vector16(action_min)
    action_max = helpers._vector16(action_max)

    for offset in range(stop):
        sample = dataset[start + offset]
        image = sample["image"][0]
        instruction = args.instruction if args.instruction is not None else str(sample["lang"])
        qwen_inputs = model.qwen_model.build_qwenvl_inputs(
            [[image]], [instruction], system_prompt=None,
            add_summary_token=False, add_generation_prompt=False, max_length=128,
        )
        outputs = model.qwen_model(
            **qwen_inputs, output_attentions=False,
            output_hidden_states=True, return_dict=True,
        )
        image_feature, text_feature = model.qwen_model.extract_features(
            qwen_inputs.input_ids, outputs.hidden_states[-1]
        )
        fused, anchor, _ = model.memory_bank.update_on_eval(
            image_feature, text_feature, model.classifier, episode_id=memory_id
        )
        summary = torch.cat((fused, anchor, text_feature), dim=1)
        state = torch.as_tensor(sample["state"], dtype=fused.dtype, device=device).reshape(1, 1, 16)

        if offset < first_anchor or (offset - first_anchor) % stride:
            continue
        prediction = model.action_model.predict_action(summary, state)[0].float().cpu().numpy()
        target = torch.as_tensor(sample["action"], dtype=torch.float32).cpu().numpy()
        prediction, target = prediction[:horizon, :16], target[:horizon, :16]
        if prediction.shape != target.shape:
            raise ValueError(f"Prediction/target mismatch: {prediction.shape} vs {target.shape}")
        anchors.append(offset)
        preds.append(prediction)
        targets.append(target)
        preds_raw.append(helpers._mem0_denormalize(prediction, action_min, action_max))
        targets_raw.append(raw_actions[start + offset:start + offset + horizon, :16])

    return helpers._save_episode_result(
        output_dir, episode_id, anchors, preds, targets, preds_raw, targets_raw,
        raw_states, start, stop, stride, names,
    )


def main():
    args = _parse_args()
    if args.start_frame < 0 or (args.steps is not None and args.steps < 1):
        raise ValueError("start-frame must be nonnegative and steps must be positive")
    if args.stride is not None and args.stride < 1:
        raise ValueError("stride must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    model, cfg = _load_model(args, device)
    horizon = int(model.action_horizon)
    args.stride = args.stride or horizon
    if args.stride > horizon:
        raise ValueError("stride cannot exceed action horizon because that leaves gaps")

    sys.path.insert(0, str(MEM0_ROOT))
    from source.dataloader.dataset_min_max import LeRobot_Dataset
    from script import open_loop_eval_compact as helpers

    dataset_root = args.dataset.expanduser().resolve()
    norm_stats = args.norm_stats.expanduser().resolve() if args.norm_stats else None
    dataset = LeRobot_Dataset(
        repo_id=dataset_root.name, root=str(dataset_root),
        features_to_load=["observation.state", "action", "subtask", "subtask_end",
                          "episode_id", "observation.image.head_camera"],
        action_horizon=horizon,
        image_scale=None,
        norm_stats_path=str(norm_stats) if norm_stats else None,
        video_backend=args.video_backend,
    )
    raw_dataset = dataset.dataset
    bounds = helpers._episode_bounds(raw_dataset, dataset_root)
    raw_actions, raw_states = helpers._preload_trajectory_vectors(raw_dataset, dataset_root)
    raw_actions, raw_states = helpers._vectors16(raw_actions), helpers._vectors16(raw_states)
    names = helpers._action_names(raw_dataset, 16)

    output_dir = args.output_dir.expanduser().resolve() / (
        f"mem0_{args.task}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    episode_ids = sorted(bounds) if args.episode_ids is None else args.episode_ids
    if args.max_episodes is not None:
        episode_ids = episode_ids[:args.max_episodes]
    print(f"[open-loop] Mem-0 checkpoint={args.checkpoint.resolve()} device={device}", flush=True)
    print(f"[open-loop] dataset={dataset_root} episodes={len(episode_ids)} horizon={horizon} stride={args.stride}", flush=True)

    results = []
    for episode_id in episode_ids:
        if episode_id not in bounds:
            print(f"[open-loop] skipping absent episode {episode_id}", flush=True)
            continue
        result = _eval_episode(
            args, model, dataset, raw_actions, raw_states, bounds, episode_id,
            device, output_dir, horizon, helpers, names,
        )
        results.append(result)
        metric = result["trajectory_metrics"]
        print(f"[open-loop] episode={episode_id} frames={metric['frames']} raw_mse={metric['raw_action_mse']:.6g} raw_mae={metric['raw_action_mae']:.6g} plot={result['plot']}", flush=True)
    if not results:
        raise RuntimeError("None of the requested episode IDs exist in the dataset")

    arrays = [np.load(output_dir / f"episode_{item['episode_id']:05d}.npz") for item in results]
    concat = lambda key: np.concatenate([arr[key] for arr in arrays], axis=0)
    summary = {
        "policy": "Mem-0",
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "dataset": str(dataset_root),
        "episodes": [item["episode_id"] for item in results],
        "action_horizon": horizon,
        "stride": args.stride,
        "trajectory_metrics": helpers._trajectory_metrics(
            concat("trajectory_predictions_model_space"),
            concat("trajectory_targets_model_space"),
            concat("trajectory_predictions_raw_actions"),
            concat("trajectory_targets_raw_actions"),
            concat("trajectory_states_raw"), names,
        ),
        "horizon_metrics": helpers._metrics(
            concat("predictions_model_space"), concat("targets_model_space"),
            concat("predictions_raw_actions"), concat("targets_raw_actions"),
        ),
        "per_episode": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for array in arrays:
        array.close()
    print(f"[open-loop] summary saved to {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
