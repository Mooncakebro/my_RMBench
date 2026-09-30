"""Convert one RMBench raw task to a LeRobot v3 dataset.

The converter intentionally keeps the raw episode order.  ``action[t]`` is
the next observed joint state, which is the convention used by the existing
RMBench policies and makes the previous-action control input unambiguous.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# LeRobot/Hugging Face creates dataset lock files even for local reads. Keep
# those ephemeral files out of a potentially read-only home cache.
os.environ.setdefault("HF_HOME", "/tmp/pi05_compact_hf")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/pi05_compact_hf/datasets")

import cv2
import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

TASKS = {
    "observe_and_pickup": "Initially, there is one target object on the shelf and five random objects on the table. Then, a screen obscures the target object. Pick up the corresponding target object from the table and lift it up.",
    "put_back_block": "There are four mats, one block, and a button on the table. First, put the block to the center, then press the button. Then, put the block back in its original position.",
    "rearrange_blocks": "Move the block between the two mats onto the empty mat, press the button, then move the other block to the space between the two mats.",
    "swap_blocks": "There are three trays on the table, and two blocks are placed in two different trays. Swap the positions of the two blocks. Finally press the button.",
    "swap_T": "Swap the poses of the two T-blocks, including both position and orientation.",
    "battery_try": "There are two batteries and a battery slot on the table. Combining the two batteries in different orientations causes the dashboard needle to rotate.",
    "blocks_ranking_try": "There is a button and three colored cubes arranged in a random row on the table. Rearrange the cubes until the arrangement is successful.",
    "classify_blocks": "There are two colors of blocks and two baskets on the table. Collect blocks of the same color into the same basket.",
    "cover_blocks": "Cover the blocks from left to right using the lids, and then uncover them in the sequence red, green, and blue.",
    "press_button": "Observe the two numbers on the table. Press each button the required number of times, then press the right button once to confirm.",
    "place_block_mat": "Pick up the blocks from the blue mat and place them on the green mat, then put them back on the original mat.",
    "storage_blocks": "There are blocks and a basket on the table. Store all the blocks on the table into the basket.",
}

NAMES = [
    "left_joint_0", "left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4", "left_joint_5", "left_joint_6",
    "right_joint_0", "right_joint_1", "right_joint_2", "right_joint_3", "right_joint_4", "right_joint_5", "right_joint_6",
    "left_gripper", "right_gripper",
]


def _feature(dtype, shape, names):
    return {"dtype": dtype, "shape": shape, "names": names}


FEATURES = {
    "observation.state": _feature("float32", (16,), NAMES),
    "action": _feature("float32", (16,), NAMES),
    "observation.image.head_camera": _feature("video", (240, 320, 3), ["height", "width", "channels"]),
    "observation.image.left_camera": _feature("video", (240, 320, 3), ["height", "width", "channels"]),
    "observation.image.right_camera": _feature("video", (240, 320, 3), ["height", "width", "channels"]),
    "subtask": _feature("string", (1,), ["subtask_annotation"]),
    "global_task": _feature("string", (1,), ["global_task_annotation"]),
    "subtask_end": _feature("bool", (1,), ["subtask_end_flag"]),
    "episode_id": _feature("int32", (1,), ["episode_id"]),
}


def _state(f, i: int) -> np.ndarray:
    ja = f["joint_action"]
    return np.asarray(np.concatenate([
        ja["left_arm"][i], [0.0], ja["right_arm"][i], [0.0],
        [ja["left_gripper"][i], ja["right_gripper"][i]],
    ]), dtype=np.float32)


def _image(f, camera: str, i: int) -> np.ndarray:
    encoded = f["observation"][f"{camera}_camera"]["rgb"][i]
    bgr = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"could not decode {camera} frame {i}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _subtasks(path: Path, episode: int, length: int, global_task: str):
    if not path.exists():
        return [(0, length - 1, global_task)]
    annotations = json.loads(path.read_text(encoding="utf-8"))
    values = annotations.get(f"episode_{episode}", annotations.get(str(episode)))
    if not values:
        return [(0, length - 1, global_task)]
    out, start = [], 0
    for text, duration in values:
        end = start + int(duration) - 1
        if end >= length:
            raise ValueError(f"annotation for episode {episode} exceeds {length} frames")
        out.append((start, end, str(text)))
        start = end + 1
    if start != length:
        raise ValueError(f"annotation for episode {episode} has {start} frames, expected {length}")
    return out


def convert(args) -> None:
    raw_root = Path(args.raw_root) / args.task / args.demo_root
    output_root = Path(args.output_root) / args.task
    if args.append:
        dataset = LeRobotDataset(repo_id=args.task, root=output_root)
    else:
        if output_root.exists() and any(output_root.iterdir()):
            raise FileExistsError(f"{output_root} is non-empty; use --append or choose another output root")
        dataset = LeRobotDataset.create(repo_id=args.task, root=output_root, fps=30,
                                        features=FEATURES, use_videos=True)
    task_text = TASKS.get(args.task, args.task)
    annotation_path = raw_root / "language_annotation.json"
    for episode in range(args.episodes):
        path = raw_root / "data" / f"episode{episode}.hdf5"
        if not path.exists():
            if args.skip_missing:
                continue
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as f:
            length = len(f["joint_action"]["left_arm"])
            states = [_state(f, i) for i in range(length)]
            subtasks = _subtasks(annotation_path, episode, length, task_text)
            for i in range(length):
                subtask = next(text for start, end, text in subtasks if start <= i <= end)
                end = next(end for start, end, text in subtasks if start <= i <= end)
                frame = {
                    "observation.state": states[i],
                    "action": states[min(i + 1, length - 1)],
                    "observation.image.head_camera": _image(f, "head", i),
                    "observation.image.left_camera": _image(f, "left", i),
                    "observation.image.right_camera": _image(f, "right", i),
                    "subtask": subtask,
                    "global_task": task_text,
                    "subtask_end": np.asarray([end - i <= 8], dtype=bool),
                    "episode_id": np.asarray([args.episode_id_offset + episode], dtype=np.int32),
                    "task": args.task,
                }
                dataset.add_frame(frame)
        dataset.save_episode()
        print(f"saved {args.task}/episode{episode} ({length} frames)")
    dataset.finalize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--demo-root", default="demo_clean")
    parser.add_argument("--episode-id-offset", type=int, default=0)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--raw-root", default="/media/spc/新加卷/RMBench_dataset/data")
    parser.add_argument("--output-root", default=str(Path(__file__).resolve().parents[1] / "lerobot_datasets"))
    convert(parser.parse_args())


if __name__ == "__main__":
    main()
