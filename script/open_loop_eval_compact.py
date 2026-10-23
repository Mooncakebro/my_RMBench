#!/usr/bin/env python3
"""Teacher-forced open-loop evaluation for Mem0-Compact and Pi05-Compact.

The evaluator walks each selected LeRobot episode in chronological order,
updates the policy's recurrent COMPACT state from dataset observations, and
occasionally predicts an action chunk. Predicted chunks are compared with the
corresponding ground-truth action horizon; predicted actions are never applied
to the observations. Results include model-space and, when available,
denormalized action errors, per-lead errors, and compressed prediction arrays.

Use the policy's own environment: Mem0-Compact commonly uses the RMBench
environment, while Pi05-Compact may need its separate Transformers-patched
environment.

Examples (from the RMBench repository root):
    python script/open_loop_eval_compact.py --policy mem0 \
      --checkpoint policy/Mem0-Compact/runs/compact_swap_blocks/ckpt_best.pt \
      --task swap_blocks --episode-ids 0 1 --steps 200

    python script/open_loop_eval_compact.py --policy pi05 \
      --checkpoint policy/Pi05-Compact/runs/pi05_compact_swap_blocks/ckpt_final.pt \
      --task swap_blocks --tokenizer-path /path/to/paligemma_tokenizer.model \
      --episode-ids 0 --steps 200 --stride 10
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_ROOTS = {
    "mem0": REPO_ROOT / "policy" / "Mem0-Compact",
    "pi05": REPO_ROOT / "policy" / "Pi05-Compact",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, choices=("mem0", "pi05"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None,
                        help="training YAML; Mem0 checkpoint config is preferred for model architecture")
    parser.add_argument("--dataset", type=Path, default=None,
                        help="LeRobot dataset directory; defaults to the selected policy's training config")
    parser.add_argument("--norm-stats", type=Path, default=None,
                        help="optional normalization JSON; defaults to the task asset file if present")
    parser.add_argument("--base-model", type=str, default=None,
                        help="override the Qwen model path for Mem0 or use the Pi05 base checkpoint")
    parser.add_argument("--tokenizer-path", type=Path, default=None,
                        help="local PaliGemma tokenizer model (Pi05; avoids GCS lookup)")
    parser.add_argument("--task", type=str, default=None,
                        help="task override; otherwise inferred from the training config")
    parser.add_argument("--episode-ids", type=int, nargs="+", default=[0])
    parser.add_argument("--start-frame", type=int, default=0,
                        help="first frame eligible to be an evaluation anchor; earlier frames warm recurrent memory")
    parser.add_argument("--steps", type=int, default=100,
                        help="number of frame positions to evaluate per episode, starting at --start-frame")
    parser.add_argument("--stride", type=int, default=1,
                        help="predict at every Nth frame; recurrent state is still updated on intervening frames")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampling-steps", type=int, default=10,
                        help="Pi05 Euler sampling steps; Mem0 uses its configured inference steps")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--head-camera-key", default="observation.image.head_camera")
    parser.add_argument("--left-camera-key", default="observation.image.left_camera")
    parser.add_argument("--right-camera-key", default="observation.image.right_camera")
    parser.add_argument("--instruction", default=None,
                        help="fixed instruction override; by default use the dataset's training instruction")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def _resolve_path(path: Path, base: Path = REPO_ROOT) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _read_yaml(path: Path) -> dict:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return values


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def _checkpoint_state(payload: Any) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must be a mapping containing a model state dict")
    state = payload.get("model", payload.get("model_state_dict", payload.get("state_dict")))
    if state is None and payload and all(torch.is_tensor(v) for v in payload.values()):
        state = payload
    if not isinstance(state, dict):
        raise ValueError("Could not find model/model_state_dict/state_dict in checkpoint")
    return state


def _add_policy_import_roots(policy: str) -> Path:
    policy_root = POLICY_ROOTS[policy]
    # This script imports only one policy per process, so a single `source`
    # package root is unambiguous.
    for item in (policy_root, policy_root / "source", policy_root / "vendor"):
        if item.is_dir() and str(item) not in sys.path:
            sys.path.insert(0, str(item))
    return policy_root


def _load_mem0(args, policy_root: Path, device: torch.device):
    from omegaconf import OmegaConf

    default_config = policy_root / "source" / "config" / "mem0_compact_train.yaml"
    config_path = _resolve_path(args.config, REPO_ROOT) if args.config else default_config
    config_values = _read_yaml(config_path)
    payload = _load_torch(_resolve_path(args.checkpoint))
    checkpoint_config = payload.get("config") if isinstance(payload, dict) else None
    if checkpoint_config is None:
        checkpoint_config = config_values
    cfg = OmegaConf.create(checkpoint_config)
    if args.base_model:
        cfg.execution_module.qwen_vl.model_path = args.base_model
    elif args.config:
        # Checkpoints retain the training machine's Qwen path. When a caller
        # supplies a local YAML, use its model path while keeping checkpoint
        # architecture values (mem_dim/layers/etc.) authoritative.
        configured_model = config_values.get("execution_module", {}).get("qwen_vl", {}).get("model_path")
        if configured_model:
            cfg.execution_module.qwen_vl.model_path = configured_model

    from source.models.execution_module.mem0_compact_executor import Mem0CompactExecutor

    model = Mem0CompactExecutor(cfg, device=device).to(device)
    missing, unexpected = model.load_state_dict(_checkpoint_state(payload), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Mem0 checkpoint/model mismatch: "
            f"missing={len(missing)} ({missing[:8]}), "
            f"unexpected={len(unexpected)} ({unexpected[:8]})"
        )
    model.eval()
    return model, cfg, payload, config_values


def _load_pi05(args, policy_root: Path, device: torch.device):
    default_config = policy_root / "source" / "config" / "pi05_compact_train.yaml"
    config_path = _resolve_path(args.config, REPO_ROOT) if args.config else default_config
    values = _read_yaml(config_path)
    model_values = {**values.get("model", {}), **values.get("compact", {})}
    if device.type == "cpu" and model_values.get("dtype") == "bfloat16":
        # CPU reference runs are more portable in FP32; checkpoint tensors are
        # cast by load_state_dict and architecture/shapes stay identical.
        model_values["dtype"] = "float32"
    cfg = SimpleNamespace(**model_values)

    from source.models.pi05_compact_model import Pi05CompactModel

    model = Pi05CompactModel(
        cfg,
        freeze_base=bool(getattr(cfg, "freeze_base", False)),
        mem_dim=int(getattr(cfg, "mem_dim", 128)),
        num_mem_tokens=int(getattr(cfg, "num_mem_tokens", 16)),
        num_heads=int(getattr(cfg, "num_heads", 8)),
        num_obs_tokens=int(getattr(cfg, "num_obs_tokens", 4)),
    )
    if args.base_model:
        model.load_base_checkpoint(args.base_model, strict=False)
    payload = _load_torch(_resolve_path(args.checkpoint))
    missing, unexpected = model.load_state_dict(_checkpoint_state(payload), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Pi05-Compact checkpoint/model mismatch: "
            f"missing={len(missing)} ({missing[:8]}), "
            f"unexpected={len(unexpected)} ({unexpected[:8]})"
        )
    model.to(device).eval()
    return model, values, payload


def _find_norm_path(args, policy_root: Path, task: str, config_values: dict) -> Path | None:
    if args.norm_stats:
        path = _resolve_path(args.norm_stats, REPO_ROOT)
        if not path.is_file():
            raise FileNotFoundError(f"Normalization stats not found: {path}")
        return path
    if args.policy == "pi05":
        configured = config_values.get("data", {}).get("norm_stats")
        candidates = []
        if configured:
            candidates.append(_resolve_path(Path(configured), policy_root))
        candidates.append(policy_root / "assets" / task / "norm_stats.json")
    else:
        configured = config_values.get("trainer", {}).get("norm_stats_path")
        candidates = []
        if configured:
            candidates.append(_resolve_path(Path(configured), policy_root))
        candidates.append(policy_root / "assets" / task / "norm_stats.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _dataset_root(args, policy_root: Path, config_values: dict, task: str) -> Path:
    if args.dataset:
        root = _resolve_path(args.dataset, REPO_ROOT)
    elif args.policy == "pi05":
        value = Path(config_values.get("data", {}).get("root", f"lerobot_datasets/{task}"))
        root = _resolve_path(value, policy_root)
        if args.task:
            root = root.parent / task
    else:
        # Mem0's trainer resolves --task under policy/Mem0-Compact/lerobot_datasets.
        data_cfg = config_values.get("vla_dataset", {}).get("RMBench", {})
        configured = data_cfg.get("repo_id")
        if args.task:
            root = policy_root / "lerobot_datasets" / task
        elif configured and not str(configured).startswith(("http", "hf://")):
            root = _resolve_path(Path(configured), policy_root)
        else:
            root = policy_root / "lerobot_datasets" / task
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {root}. Pass --dataset /path/to/LeRobot/dataset."
        )
    return root.resolve()


def _episode_bounds(dataset, root: Path) -> dict[int, tuple[int, int]]:
    """Read authoritative LeRobot episode ranges as [start, end_exclusive)."""
    meta = getattr(dataset, "meta", None)
    episodes = getattr(meta, "episodes", None)
    required = ("episode_index", "dataset_from_index", "dataset_to_index")
    rows = None
    if episodes is not None:
        columns = getattr(episodes, "column_names", None)
        if columns and all(key in columns for key in required):
            rows = list(episodes)
    if rows is None:
        files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
        if files:
            import pyarrow.parquet as pq

            rows = []
            for path in files:
                table = pq.read_table(path, columns=list(required))
                rows.extend(table.to_pylist())
    if rows is None:
        raise RuntimeError(
            f"Could not read LeRobot episode bounds under {root}/meta/episodes; "
            "expected episode_index/dataset_from_index/dataset_to_index metadata."
        )
    bounds = {
        int(row["episode_index"]): (int(row["dataset_from_index"]), int(row["dataset_to_index"]))
        for row in rows
    }
    if not bounds or any(end <= start for start, end in bounds.values()):
        raise ValueError("LeRobot episode metadata is empty or contains invalid index ranges")
    return bounds


def _load_raw_lerobot(root: Path, video_backend: str, image_keys: set[str]):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError("Open-loop evaluation requires the LeRobot package") from exc
    dataset = LeRobotDataset(repo_id=root.name, root=root, video_backend=video_backend)
    meta = getattr(dataset, "meta", None)
    if meta is not None and hasattr(meta, "info"):
        features = meta.info.get("features", {})
        meta.info["features"] = {
            key: value for key, value in features.items()
            if value.get("dtype") not in {"video", "image"} or key in image_keys
        }
    return dataset


def _preload_actions(dataset, root: Path) -> np.ndarray:
    """Load action rows from parquet once, in LeRobot's global index order."""
    import pyarrow.parquet as pq

    files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No data/chunk-*/file-*.parquet files under {root}")
    actions, indices = [], []
    has_index = True
    for path in files:
        schema_names = set(pq.read_schema(path).names)
        if "action" not in schema_names:
            raise KeyError(f"No 'action' column in {path}")
        cols = ["action"] + (["index"] if "index" in schema_names else [])
        table = pq.read_table(path, columns=cols)
        values = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"Expected 2D action rows in {path}, got shape={values.shape}")
        actions.append(values)
        if "index" in cols:
            indices.append(np.asarray(table["index"].to_pylist(), dtype=np.int64))
        else:
            has_index = False
    action_rows = np.concatenate(actions, axis=0)
    if has_index and len(indices) == len(actions):
        row_indices = np.concatenate(indices, axis=0)
        if len(row_indices) != len(dataset) or not np.array_equal(np.sort(row_indices), np.arange(len(dataset))):
            raise ValueError("Parquet 'index' values do not form the expected LeRobot dataset range")
        action_rows = action_rows[np.argsort(row_indices, kind="stable")]
    elif len(action_rows) != len(dataset):
        raise ValueError(
            f"Parquet actions have {len(action_rows)} rows, LeRobot reports {len(dataset)}; "
            "cannot align action targets safely."
        )
    return action_rows


def _as_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _vector16(value) -> np.ndarray:
    vector = np.asarray(_as_numpy(value), dtype=np.float32).reshape(-1)
    if vector.size > 16:
        vector = vector[:16]
    if vector.size < 16:
        vector = np.pad(vector, (0, 16 - vector.size))
    return vector


def _image_nchw(value) -> torch.Tensor:
    image = torch.as_tensor(_as_numpy(value))
    if image.ndim != 3:
        raise ValueError(f"Expected one RGB image with 3 dimensions, got {tuple(image.shape)}")
    if image.shape[-1] == 3:
        image = image.permute(2, 0, 1)
    if image.shape[0] != 3:
        raise ValueError(f"Expected RGB image, got {tuple(image.shape)}")
    if image.dtype == torch.uint8:
        image = image.float() / 127.5 - 1.0
    else:
        image = image.float()
        if image.max().item() > 1.5:
            image = image / 127.5 - 1.0
    return image.contiguous()


def _mem0_denormalize(value: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32).copy()
    mask = np.ones(value.shape[-1], dtype=bool)
    mask[[6, 13, 14, 15]] = False
    span = np.asarray(maximum, dtype=np.float32) - np.asarray(minimum, dtype=np.float32)
    value[..., mask] = 0.5 * (value[..., mask] + 1.0) * span[mask] + minimum[mask]
    return value


def _pi05_normalize(value: np.ndarray, stats: dict | None, key: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if not stats:
        return value.copy()
    block = stats.get(key)
    if not block or "q01" not in block or "q99" not in block:
        raise KeyError(f"Pi05 stats must provide {key}.q01 and {key}.q99")
    q01 = np.asarray(block["q01"], dtype=np.float32)
    q99 = np.asarray(block["q99"], dtype=np.float32)
    return 2.0 * (value - q01) / (q99 - q01 + 1e-6) - 1.0


def _pi05_denormalize(value: np.ndarray, stats: dict | None, key: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if not stats:
        return value.copy()
    q01 = np.asarray(stats[key]["q01"], dtype=np.float32)
    q99 = np.asarray(stats[key]["q99"], dtype=np.float32)
    return 0.5 * (value + 1.0) * (q99 - q01 + 1e-6) + q01


def _instruction(sample: dict, policy: str, override: str | None) -> str:
    if override is not None:
        return override
    candidates = ("subtask", "global_task", "task", "lang") if policy == "mem0" else (
        "global_task", "subtask", "task", "lang"
    )
    for key in candidates:
        value = sample.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = value[0] if value else ""
        if torch.is_tensor(value) and value.numel() == 1:
            value = value.item()
        value = str(value).strip()
        if value:
            return value
    return ""


def _load_stats(path: Path | None) -> dict | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(pred: np.ndarray, target: np.ndarray, pred_raw: np.ndarray,
             target_raw: np.ndarray) -> dict:
    err = pred - target
    raw_err = pred_raw - target_raw
    per_lead = []
    for lead in range(pred.shape[1]):
        per_lead.append({
            "lead": lead,
            "model_space_mse": float(np.mean(err[:, lead] ** 2)),
            "model_space_mae": float(np.mean(np.abs(err[:, lead]))),
            "raw_action_mse": float(np.mean(raw_err[:, lead] ** 2)),
            "raw_action_mae": float(np.mean(np.abs(raw_err[:, lead]))),
        })
    return {
        "anchors": int(pred.shape[0]),
        "horizon": int(pred.shape[1]),
        "action_dim": int(pred.shape[2]),
        "model_space_mse": float(np.mean(err ** 2)),
        "model_space_mae": float(np.mean(np.abs(err))),
        "raw_action_mse": float(np.mean(raw_err ** 2)),
        "raw_action_mae": float(np.mean(np.abs(raw_err))),
        "per_lead": per_lead,
    }


@torch.inference_mode()
def _evaluate_mem0(args, model, dataset, bounds, stats, episode_id: int,
                   device: torch.device, output_dir: Path, horizon: int) -> dict:
    start, end = bounds[episode_id]
    episode_length = end - start
    first_anchor = max(0, args.start_frame)
    last_requested = min(episode_length, first_anchor + args.steps)
    # Full action horizons only; no target is allowed to cross an episode edge.
    valid_last = max(0, episode_length - horizon + 1)
    if first_anchor >= valid_last:
        raise ValueError(f"Episode {episode_id} has no full {horizon}-action window after frame {first_anchor}")
    last_frame = min(last_requested, valid_last)
    if last_frame <= 0:
        raise ValueError(f"Episode {episode_id} has no eligible frames")

    memory = model.init_memory(1, device)
    previous_episode_start = True
    anchors, predictions, targets, predictions_raw, targets_raw = [], [], [], [], []
    action_min = getattr(dataset, "action_min", None)
    action_max = getattr(dataset, "action_max", None)
    if action_min is None or action_max is None:
        raw_stats = dataset.dataset.meta.stats["action"]
        action_min = torch.as_tensor(raw_stats["min"], dtype=torch.float32)
        action_max = torch.as_tensor(raw_stats["max"], dtype=torch.float32)
    action_min = _vector16(action_min)
    action_max = _vector16(action_max)

    for offset in range(last_frame):
        sample = dataset[start + offset]
        state = torch.as_tensor(sample["state"], dtype=torch.float32).reshape(1, 16).to(device)
        image = sample["image"][0]
        instruction = _instruction({"subtask": sample.get("lang", "")}, "mem0", args.instruction)
        # Match Mem0 training: prev_action is the current normalized state,
        # with a zero vector only on the first frame of each episode.
        prev_action = torch.zeros_like(state) if previous_episode_start else state.clone()
        previous_episode_start = False
        batch = {
            "image": [[image]],
            "lang": [instruction],
            "state": state[:, None, :],
            "prev_action": prev_action,
        }
        summary, state_out, memory = model.update_obs_inference(batch, memory)
        if offset < first_anchor or (offset - first_anchor) % args.stride:
            continue

        pred = model.action_model.predict_action(summary, state_out)[0].detach().float().cpu().numpy()
        target = torch.as_tensor(sample["action"], dtype=torch.float32).cpu().numpy()
        pred = pred[:horizon, :16]
        target = target[:horizon, :16]
        if pred.shape != target.shape:
            raise ValueError(f"Mem0 prediction/target shape mismatch: {pred.shape} vs {target.shape}")
        anchors.append(offset)
        predictions.append(pred)
        targets.append(target)
        predictions_raw.append(_mem0_denormalize(pred, action_min, action_max))
        targets_raw.append(_mem0_denormalize(target, action_min, action_max))

    return _save_episode_result(
        output_dir, episode_id, anchors, predictions, targets,
        predictions_raw, targets_raw,
    )


@torch.inference_mode()
def _evaluate_pi05(args, model, dataset, actions_raw, bounds, stats, tokenizer,
                   episode_id: int, device: torch.device, output_dir: Path,
                   horizon: int) -> dict:
    start, end = bounds[episode_id]
    episode_length = end - start
    first_anchor = max(0, args.start_frame)
    last_requested = min(episode_length, first_anchor + args.steps)
    valid_last = max(0, episode_length - horizon + 1)
    if first_anchor >= valid_last:
        raise ValueError(f"Episode {episode_id} has no full {horizon}-action window after frame {first_anchor}")
    last_frame = min(last_requested, valid_last)
    if last_frame <= 0:
        raise ValueError(f"Episode {episode_id} has no eligible frames")

    memory = None
    prev_action = torch.zeros(1, 16, device=device, dtype=torch.float32)
    anchors, predictions, targets, predictions_raw, targets_raw = [], [], [], [], []
    camera_keys = {
        "base_0_rgb": args.head_camera_key,
        "left_wrist_0_rgb": args.left_camera_key,
        "right_wrist_0_rgb": args.right_camera_key,
    }

    for offset in range(last_frame):
        raw_sample = dataset[start + offset]
        state_raw = _vector16(raw_sample["observation.state"])
        state_norm = _pi05_normalize(state_raw, stats, "state")
        state_tensor = torch.from_numpy(state_norm).reshape(1, 16).to(device)
        # Match Pi05 training: after the first frame, feed the current state
        # as prev_action (the trainer assigns this immediately after each
        # frame), rather than the preceding frame's state.
        prev_action = torch.zeros_like(state_tensor) if offset == 0 else state_tensor
        instruction = _instruction(raw_sample, "pi05", args.instruction)
        tokens, token_mask = tokenizer.tokenize(instruction, state_norm)
        images = {}
        for output_key, source_key in camera_keys.items():
            if source_key not in raw_sample:
                raise KeyError(
                    f"Dataset sample is missing camera feature {source_key!r}; "
                    f"sample keys include {sorted(raw_sample.keys())}"
                )
            images[output_key] = _image_nchw(raw_sample[source_key])[None].to(device)
        observation = SimpleNamespace(
            images=images,
            image_masks={key: torch.ones(1, dtype=torch.bool, device=device) for key in images},
            state=state_tensor,
            tokenized_prompt=torch.from_numpy(tokens).reshape(1, -1).to(device=device, dtype=torch.long),
            tokenized_prompt_mask=torch.from_numpy(token_mask).reshape(1, -1).to(device=device, dtype=torch.bool),
            token_ar_mask=None,
            token_loss_mask=None,
        )

        is_anchor = offset >= first_anchor and (offset - first_anchor) % args.stride == 0
        if is_anchor:
            pred_tensor, memory = model.sample_actions(
                observation, memory=memory, prev_action=prev_action,
                num_steps=args.sampling_steps,
            )
            pred = pred_tensor[0].detach().float().cpu().numpy()[:, :16]
            target_raw = actions_raw[start + offset:start + offset + horizon]
            target = _pi05_normalize(target_raw, stats, "actions")
            if pred.shape != target.shape:
                raise ValueError(f"Pi05 prediction/target shape mismatch: {pred.shape} vs {target.shape}")
            anchors.append(offset)
            predictions.append(pred)
            targets.append(target)
            predictions_raw.append(_pi05_denormalize(pred, stats, "actions"))
            targets_raw.append(target_raw.astype(np.float32, copy=False))
        else:
            # The official sampler's prefix prepass is exactly this forward
            # with t=1 and zero noise/actions. Do it without the Euler denoise
            # loop on frames which aren't scored anchors, keeping recurrent
            # history identical while saving most compute.
            zeros = torch.zeros(1, horizon, 16, device=device)
            _, memory = model._forward_impl(
                observation, zeros, memory, prev_action,
                noise=torch.zeros_like(zeros),
                time=torch.ones(1, device=device), train=False,
            )
    return _save_episode_result(
        output_dir, episode_id, anchors, predictions, targets,
        predictions_raw, targets_raw,
    )


def _save_episode_result(output_dir: Path, episode_id: int, anchors: list,
                         predictions: list, targets: list,
                         predictions_raw: list, targets_raw: list) -> dict:
    if not anchors:
        raise RuntimeError(f"No evaluation anchors were generated for episode {episode_id}")
    pred = np.stack(predictions).astype(np.float32)
    target = np.stack(targets).astype(np.float32)
    pred_raw = np.stack(predictions_raw).astype(np.float32)
    target_raw = np.stack(targets_raw).astype(np.float32)
    metrics = _metrics(pred, target, pred_raw, target_raw)
    metrics["episode_id"] = int(episode_id)
    np.savez_compressed(
        output_dir / f"episode_{episode_id:05d}.npz",
        anchor_frames=np.asarray(anchors, dtype=np.int64),
        predictions_model_space=pred,
        targets_model_space=target,
        predictions_raw_actions=pred_raw,
        targets_raw_actions=target_raw,
    )
    return metrics


def main() -> None:
    args = _parser().parse_args()
    if args.steps < 1 or args.stride < 1 or args.start_frame < 0:
        raise ValueError("--steps and --stride must be positive; --start-frame must be nonnegative")
    if args.sampling_steps < 1:
        raise ValueError("--sampling-steps must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    policy_root = _add_policy_import_roots(args.policy)

    if args.policy == "mem0":
        model, model_cfg, payload, yaml_values = _load_mem0(args, policy_root, device)
        config_values = payload.get("config", yaml_values)
        horizon = int(model.action_horizon)
    else:
        model, config_values, payload = _load_pi05(args, policy_root, device)
        horizon = int(model.config.action_horizon)

    if args.task:
        task = args.task
    elif args.policy == "pi05":
        task = Path(config_values.get("data", {}).get("root", "swap_blocks")).name
    else:
        configured_repo = config_values.get("vla_dataset", {}).get("RMBench", {}).get("repo_id")
        task = Path(str(configured_repo)).name if configured_repo else "swap_blocks"
    stats_path = _find_norm_path(args, policy_root, task, config_values)
    stats = _load_stats(stats_path)
    dataset_root = _dataset_root(args, policy_root, config_values, task)
    if args.policy == "mem0":
        from source.dataloader.dataset_min_max import LeRobot_Dataset

        data_cfg = config_values.get("vla_dataset", {}).get("RMBench", {})
        features = data_cfg.get("features_to_load", [
            "observation.state", "action", "subtask", "subtask_end",
            "episode_id", "observation.image.head_camera",
        ])
        # Use the resolved local stats file, otherwise let the dataset wrapper
        # use its training-time LeRobot metadata stats. Checkpoint paths may
        # point to a different machine and must not be passed through blindly.
        norm_path = str(stats_path) if stats_path else None
        horizon = int(model.action_horizon)
        dataset = LeRobot_Dataset(
            repo_id=dataset_root.name,
            root=str(dataset_root),
            features_to_load=list(features),
            action_horizon=horizon,
            norm_stats_path=norm_path,
            image_augment=False,
            video_backend=args.video_backend,
        )
        raw_dataset = dataset.dataset
        bounds = _episode_bounds(raw_dataset, dataset_root)
        action_stats = raw_dataset.meta.stats["action"]
        if stats_path:
            stats = _load_stats(stats_path)
            action_min = np.asarray(stats["action_min"], dtype=np.float32)
            action_max = np.asarray(stats["action_max"], dtype=np.float32)
            # The dataset wrapper owns the exact stats it used to normalize.
            dataset.action_min = torch.as_tensor(action_min)
            dataset.action_max = torch.as_tensor(action_max)
        else:
            dataset.action_min = torch.as_tensor(action_stats["min"], dtype=torch.float32)
            dataset.action_max = torch.as_tensor(action_stats["max"], dtype=torch.float32)
        action_min = _vector16(dataset.action_min)
        action_max = _vector16(dataset.action_max)
        # Mem0's dataset wrapper already returns exactly the training-normalized
        # states/actions. Preserve metadata stats for reporting and inverse map.
        _ = action_min, action_max
        tokenizer = None
        actions_raw = None
    else:
        image_keys = {args.head_camera_key, args.left_camera_key, args.right_camera_key}
        dataset = _load_raw_lerobot(dataset_root, args.video_backend, image_keys)
        bounds = _episode_bounds(dataset, dataset_root)
        actions_raw = _preload_actions(dataset, dataset_root)
        if stats_path is None:
            stats = None
        from openpi.models.tokenizer import PaligemmaTokenizer

        tokenizer = PaligemmaTokenizer(
            int(getattr(model.config, "max_token_len", 200)),
            args.tokenizer_path,
        )

    output_base = args.output_dir or (REPO_ROOT / "eval_result" / "open_loop_eval")
    output_dir = _resolve_path(output_base, REPO_ROOT) / (
        f"{args.policy}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"[open-loop] policy={args.policy} checkpoint={_resolve_path(args.checkpoint)}", flush=True)
    print(f"[open-loop] dataset={dataset_root} episodes={len(bounds)} horizon={horizon} device={device}", flush=True)
    print(f"[open-loop] norm_stats={stats_path or '<dataset stats / unnormalized>'}; output={output_dir}", flush=True)

    results = []
    for episode_id in args.episode_ids:
        if episode_id not in bounds:
            print(f"[open-loop] skipping absent episode {episode_id}", flush=True)
            continue
        if args.policy == "mem0":
            metrics = _evaluate_mem0(
                args, model, dataset, bounds, stats, episode_id,
                device, output_dir, horizon,
            )
        else:
            metrics = _evaluate_pi05(
                args, model, dataset, actions_raw, bounds, stats, tokenizer,
                episode_id, device, output_dir, horizon,
            )
        results.append(metrics)
        print(json.dumps(metrics, indent=2), flush=True)

    if not results:
        raise RuntimeError("None of the requested episode IDs exist in the dataset")
    pred_arrays = [np.load(output_dir / f"episode_{item['episode_id']:05d}.npz") for item in results]
    all_pred = np.concatenate([array["predictions_model_space"] for array in pred_arrays], axis=0)
    all_target = np.concatenate([array["targets_model_space"] for array in pred_arrays], axis=0)
    all_pred_raw = np.concatenate([array["predictions_raw_actions"] for array in pred_arrays], axis=0)
    all_target_raw = np.concatenate([array["targets_raw_actions"] for array in pred_arrays], axis=0)
    summary = {
        "policy": args.policy,
        "checkpoint": str(_resolve_path(args.checkpoint)),
        "dataset": str(dataset_root),
        "episodes": [item["episode_id"] for item in results],
        "action_horizon": horizon,
        "stride": args.stride,
        "seed": args.seed,
        "metrics": _metrics(all_pred, all_target, all_pred_raw, all_target_raw),
        "per_episode": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for array in pred_arrays:
        array.close()
    print(f"[open-loop] summary saved to {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
