"""Sequential TBPTT trainer for Pi05-Compact.

This is deliberately a small PyTorch entry point.  It keeps the COMPACT state
outside the module, resets rows when episode ids change, and calls the wrapper
through its public ``forward`` so the same code works under DDP.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
import yaml

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "source"
for import_root in (SOURCE_ROOT, ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))
try:
    from source.models.pi05_compact_model import Pi05CompactModel
    from source.dataloader.random_episode_dataloader import RandomEpisodeIterableDataset
    from source.training.ddp_utils import destroy_distributed, get_ddp_device, rank0_print, setup_distributed
except ModuleNotFoundError as exc:
    if exc.name not in {"source", "source.models", "source.dataloader", "source.training"}:
        raise
    from models.pi05_compact_model import Pi05CompactModel
    from dataloader.random_episode_dataloader import RandomEpisodeIterableDataset
    from training.ddp_utils import destroy_distributed, get_ddp_device, rank0_print, setup_distributed


def _cfg(values):
    model = values["model"]
    compact = values.get("compact", {})
    return SimpleNamespace(**model, **compact)


class WindowDataset:
    """Expose one frame plus a contiguous action horizon from LeRobot."""

    IMAGE_KEYS = (
        "observation.image.head_camera",
        "observation.image.left_camera",
        "observation.image.right_camera",
    )

    def __init__(self, dataset, horizon):
        self.dataset, self.horizon = dataset, horizon
        self.episode_to_indices = getattr(dataset, "episode_to_indices", None)
        if self.episode_to_indices is None:
            # LeRobot stores authoritative contiguous global index ranges in
            # meta.episodes.  Use them directly; scanning dataset[i] would
            # decode every video frame just to reconstruct this mapping.
            episodes = getattr(getattr(dataset, "meta", None), "episodes", None)
            if episodes is not None and all(
                key in episodes.column_names
                for key in ("episode_index", "dataset_from_index", "dataset_to_index")
            ):
                self.episode_to_indices = {
                    int(row["episode_index"]): list(
                        range(int(row["dataset_from_index"]), int(row["dataset_to_index"]))
                    )
                    for row in episodes
                }
            else:
                self.episode_to_indices = {}
                for i in range(len(dataset)):
                    episode = int(dataset[i]["episode_id"])
                    self.episode_to_indices.setdefault(episode, []).append(i)
        # Pre-load the action columns once.  Building a horizon of H actions
        # through dataset[i] costs H video-frame decodes per training frame
        # (LeRobotDataset decodes the image on every access even when only
        # "action" is used); the parquet action columns are tiny, so read
        # them directly and slice horizons in memory instead.
        self.actions = self._preload_actions(dataset)

    @staticmethod
    def _preload_actions(dataset):
        """Return a [total_frames, action_dim] float32 tensor of all actions,
        or None to fall back to per-row dataset lookups.

        LeRobot v3.0 stores rows in global-index order across
        data/chunk-*/file-*.parquet, so concatenating the action columns in
        sorted file order reproduces dataset indices exactly.
        """
        try:
            import pyarrow.parquet as pq

            root = Path(dataset.root)
            files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
            if not files:
                return None
            action_arrays, index_arrays = [], []
            for path in files:
                table = pq.read_table(path, columns=["action", "index"])
                action_arrays.append(np.asarray(table["action"].to_pylist(), dtype=np.float32))
                index_arrays.append(np.asarray(table["index"].to_numpy(), dtype=np.int64))
            actions_np = np.concatenate(action_arrays, axis=0)
            indices_np = np.concatenate(index_arrays, axis=0)
            if not np.array_equal(np.sort(indices_np), np.arange(len(dataset))):
                return None
            # Usually files are already in global-index order. Sort defensively
            # so appended/repacked parquet files cannot silently misalign the
            # action horizon with the decoded observation at ``dataset[index]``.
            order = np.argsort(indices_np, kind="stable")
            return torch.from_numpy(actions_np[order].copy())
        except Exception as exc:
            print(f"[data] action pre-load failed ({exc}); "
                  "falling back to per-row dataset lookups", flush=True)
            return None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        episode = int(item["episode_id"])
        indices = self.episode_to_indices[episode]
        item = dict(item)
        if self.actions is not None:
            # episode rows are contiguous global indices; clamp at episode end
            end = indices[-1]
            stop = min(index + self.horizon, end + 1)
            action = self.actions[index:stop]
            if action.shape[0] < self.horizon:
                action = torch.cat([action, action[-1:].expand(self.horizon - action.shape[0], -1)], dim=0)
            item["action"] = action
            return item
        # Fallback: per-row dataset lookups (decodes a video frame per row).
        pos = indices.index(index)
        action = []
        for j in range(self.horizon):
            source = self.dataset[indices[min(pos + j, len(indices) - 1)]]
            action.append(torch.as_tensor(source["action"], dtype=torch.float32))
        item["action"] = torch.stack(action)
        return item


def _image_tensor(value):
    x = torch.as_tensor(value)
    if x.ndim == 3 and x.shape[-1] == 3:
        x = x.permute(2, 0, 1)
    if x.dtype == torch.uint8:
        x = x.float() / 127.5 - 1.0
    else:
        x = x.float()
        if x.max() > 1.5:
            x = x / 127.5 - 1.0
    return x


def _restrict_video_features(dataset):
    """Prevent LeRobot from decoding cameras that Pi05 never consumes."""
    meta = getattr(dataset, "meta", None)
    if meta is None or not hasattr(meta, "info"):
        return
    features = meta.info.get("features", {})
    keep = set(WindowDataset.IMAGE_KEYS)
    filtered = {
        key: value
        for key, value in features.items()
        if value.get("dtype") not in {"video", "image"} or key in keep
    }
    meta.info["features"] = filtered


def make_collate(tokenizer, stats):
    def collate(samples):
        states = torch.stack([torch.as_tensor(s["observation.state"], dtype=torch.float32) for s in samples])
        actions = torch.stack([torch.as_tensor(s["action"], dtype=torch.float32) for s in samples])
        if stats:
            state_stats = stats["state"]
            action_stats = stats["actions"]
            states = (states - torch.tensor(state_stats["q01"])) / (torch.tensor(state_stats["q99"]) - torch.tensor(state_stats["q01"]) + 1e-6) * 2 - 1
            actions = (actions - torch.tensor(action_stats["q01"])) / (torch.tensor(action_stats["q99"]) - torch.tensor(action_stats["q01"]) + 1e-6) * 2 - 1
        prompts = [str(s.get("global_task", s.get("subtask", ""))) for s in samples]
        tok, mask = zip(*(tokenizer.tokenize(p, st.numpy()) for p, st in zip(prompts, states)))
        images = {}
        for name, key in (("base_0_rgb", "observation.image.head_camera"), ("left_wrist_0_rgb", "observation.image.left_camera"), ("right_wrist_0_rgb", "observation.image.right_camera")):
            images[name] = torch.stack([_image_tensor(s[key]) for s in samples])
        observation = SimpleNamespace(
            images=images,
            image_masks={k: torch.ones(len(samples), dtype=torch.bool) for k in images},
            state=states,
            tokenized_prompt=torch.as_tensor(np.stack(tok), dtype=torch.long),
            tokenized_prompt_mask=torch.as_tensor(np.stack(mask), dtype=torch.bool),
            token_ar_mask=None,
            token_loss_mask=None,
        )
        return {"observation": observation, "actions": actions,
                "episode_id": torch.tensor([int(s["episode_id"]) for s in samples]),
                "state": states}
    return collate


def _move_to_device(value, device):
    """Move tensors in nested observation dictionaries to the model device."""
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).parents[1] / "config/pi05_compact_train.yaml")
    parser.add_argument("--task", default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--base-checkpoint", type=Path, default=None,
                        help="official pi05_base PyTorch checkpoint for warm start")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--grad-accum-windows", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=None,
                        help="local PaliGemma tokenizer.model; avoids per-rank GCS download")
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--freeze-base", action="store_true", default=None)
    parser.add_argument("--train-base", action="store_true", help="override config and train the base model")
    args = parser.parse_args()
    values = yaml.safe_load(args.config.read_text())
    if args.batch_size is not None:
        values["data"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        values["data"]["num_workers"] = args.num_workers
    if args.window_size is not None:
        values["training"]["window_size"] = args.window_size
    if args.grad_accum_windows is not None:
        values["training"]["grad_accum_windows"] = args.grad_accum_windows
    if args.output_dir is not None:
        values["training"]["output_dir"] = str(args.output_dir)
    if args.log_interval is not None:
        values["training"]["log_interval"] = args.log_interval
    model_cfg = _cfg(values)
    if args.freeze_base:
        model_cfg.freeze_base = True
    elif args.train_base:
        model_cfg.freeze_base = False
    rank, world_size, local_rank, ddp_enabled = setup_distributed()
    device = get_ddp_device(local_rank, world_size)
    print(f"[rank {rank}] constructing Pi05-Compact model on {device}", flush=True)
    raw_model = Pi05CompactModel(model_cfg, freeze_base=model_cfg.freeze_base).to(device)
    print(f"[rank {rank}] model constructed", flush=True)
    if args.checkpoint:
        if rank == 0:
            start = time.perf_counter()
            print(f"[rank {rank}] loading trainer checkpoint: {args.checkpoint}", flush=True)
            state = torch.load(args.checkpoint, map_location="cpu")
            raw_model.load_state_dict(state.get("model", state), strict=False)
            print(f"[rank {rank}] trainer checkpoint loaded in {time.perf_counter() - start:.1f}s", flush=True)
        else:
            print(f"[rank {rank}] waiting for DDP checkpoint broadcast", flush=True)
    if args.base_checkpoint:
        if rank == 0:
            start = time.perf_counter()
            print(f"[rank {rank}] loading base checkpoint: {args.base_checkpoint}", flush=True)
            raw_model.load_base_checkpoint(args.base_checkpoint, strict=False)
            print(f"[rank {rank}] base checkpoint loaded in {time.perf_counter() - start:.1f}s", flush=True)
        else:
            print(f"[rank {rank}] waiting for DDP base-checkpoint broadcast", flush=True)
    rank0_print("[init] Pi05-Compact model and checkpoint loaded", flush=True)
    model = DDP(raw_model, device_ids=[local_rank] if device.type == "cuda" else None,
                find_unused_parameters=True) if ddp_enabled else raw_model

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from openpi.models.tokenizer import PaligemmaTokenizer
    except ImportError as exc:
        raise RuntimeError("training requires lerobot and openpi tokenizer dependencies") from exc
    root = Path(values["data"]["root"])
    if not root.is_absolute():
        root = ROOT / root
    if args.task:
        root = root.parent / args.task
    lerobot_dataset = LeRobotDataset(repo_id=root.name, root=root)
    _restrict_video_features(lerobot_dataset)
    dataset = WindowDataset(lerobot_dataset, model_cfg.action_horizon)
    rank0_print(
        f"[data] cameras={len(lerobot_dataset.meta.video_keys)} "
        f"action_horizon={'parquet-preloaded' if dataset.actions is not None else 'dataset-lookups'}",
        flush=True,
    )
    stats_value = values["data"].get("norm_stats")
    stats_path = Path(stats_value) if stats_value else None
    if stats_path is not None and not stats_path.is_absolute():
        stats_path = ROOT / stats_path
    stats = json.loads(stats_path.read_text()) if stats_path is not None and stats_path.exists() else None
    tokenizer = PaligemmaTokenizer(model_cfg.max_token_len, args.tokenizer_path)
    rank0_print("[init] dataset metadata and tokenizer loaded", flush=True)
    loader = torch.utils.data.DataLoader(
        RandomEpisodeIterableDataset(dataset, rank=rank, world_size=world_size, shuffle=True, infinite=True),
        batch_size=int(values["data"].get("batch_size", 1)),
        collate_fn=make_collate(tokenizer, stats),
        num_workers=int(values["data"].get("num_workers", 0)),
    )
    rank0_print("[init] dataloader ready; beginning sequential TBPTT", flush=True)
    optimizer = AdamW((p for p in raw_model.parameters() if p.requires_grad),
                      lr=float(values["training"].get("learning_rate", 1e-4)),
                      weight_decay=float(values["training"].get("weight_decay", 0.01)))
    max_steps = args.max_steps or int(values["training"].get("max_steps", 30000))
    window = int(values["training"].get("window_size", 8))
    accum_windows = int(values["training"].get("grad_accum_windows", 1))
    memory = None
    prev_action = None
    window_loss = None
    windows_since_update = 0
    optimizer_step = 0
    optimizer.zero_grad(set_to_none=True)
    for frame_step, batch in enumerate(loader, 1):
        obs, actions = batch["observation"], batch["actions"]
        obs = SimpleNamespace(**{k: _move_to_device(v, device) for k, v in vars(obs).items()})
        actions = actions.to(device)
        episode_ids = batch["episode_id"].to(device)
        if memory is None:
            memory = raw_model.init_memory(actions.shape[0], device)
            prev_action = torch.zeros(actions.shape[0], 16, device=device)
        reset = [True] * len(episode_ids) if frame_step == 1 else (episode_ids != last_episode).tolist()
        raw_model.reset_memory_rows(memory, reset, device)
        # DDP.no_sync() must cover BOTH the forward and backward of every
        # non-final accumulation window; otherwise each window's backward
        # triggers a full gradient all-reduce (accum_windows syncs per
        # optimizer step instead of one).
        last_accum_window = windows_since_update == accum_windows - 1
        sync_ctx = (
            nullcontext()
            if (not ddp_enabled) or last_accum_window
            else model.no_sync()
        )
        with sync_ctx:
            out, memory = model(obs, actions, memory=memory, prev_action=prev_action)
            window_loss = out["total"] if window_loss is None else window_loss + out["total"]
            if frame_step % window == 0:
                (window_loss / (window * accum_windows)).backward()
                window_loss = None
        prev_action = batch["state"].to(device)
        last_episode = episode_ids
        if frame_step % window == 0:
            windows_since_update += 1
            memory = raw_model.detach_memory(memory)
            prev_action = prev_action.detach()
            if windows_since_update == accum_windows:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), float(values["training"].get("max_grad_norm", 1.0)))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                windows_since_update = 0
                optimizer_step += 1
                if optimizer_step % int(values["training"].get("log_interval", 10)) == 0:
                    rank0_print(f"[train] step {optimizer_step}/{max_steps} frame {frame_step} total {out['total'].item():.5f} flow {out['flow'].mean().item():.5f} obs {out['aux']['obs'].item():.5f} nll {out['aux']['nll'].item():.5f}", flush=True)
                if optimizer_step >= max_steps:
                    break

    output_dir = Path(values["training"].get("output_dir", "checkpoints/pi05_compact"))
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(), "step": optimizer_step}, output_dir / "ckpt_final.pt")
    destroy_distributed()


if __name__ == "__main__":
    main()
