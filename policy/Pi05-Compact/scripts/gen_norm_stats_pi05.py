"""Write openpi quantile normalization statistics for a LeRobot dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _collect(dataset_root: Path, key: str, limit: int | None):
    values = []
    remaining = limit
    for path in sorted((dataset_root / "data").glob("chunk-*/*.parquet")):
        table = pq.read_table(path, columns=[key])
        rows = table[key].to_pylist()
        if remaining is not None:
            rows = rows[:remaining]
            remaining -= len(rows)
        values.extend(np.asarray(value, dtype=np.float64).reshape(-1) for value in rows)
        if remaining is not None and remaining <= 0:
            break
    if not values:
        raise ValueError(f"dataset has no values for {key}")
    return np.stack(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()
    result = {}
    for out_key, data_key in (("state", "observation.state"), ("actions", "action")):
        x = _collect(args.dataset, data_key, args.max_frames)
        result[out_key] = {
            "mean": x.mean(axis=0).tolist(),
            "std": np.maximum(x.std(axis=0), 1e-6).tolist(),
            "q01": np.quantile(x, 0.01, axis=0).tolist(),
            "q99": np.quantile(x, 0.99, axis=0).tolist(),
        }
    output = args.output or args.dataset / "meta" / "norm_stats.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
