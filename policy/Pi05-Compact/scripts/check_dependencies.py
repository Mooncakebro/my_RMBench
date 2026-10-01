#!/usr/bin/env python3
"""Audit Pi05-Compact conversion/training dependencies without installing them.

Examples:
  python policy/Pi05-Compact/scripts/check_dependencies.py --openpi-root /mnt/lvm_storage/songyuebing/openpi
  python policy/Pi05-Compact/scripts/check_dependencies.py --mode conversion --openpi-root /path/to/openpi
  python policy/Pi05-Compact/scripts/check_dependencies.py --mode training --openpi-root /path/to/openpi

The script intentionally reports an install command instead of mutating the
environment. OpenPI is a workspace checkout, so ``openpi_client`` is checked
from its source directory as well as from site-packages.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import importlib.metadata
import os
import platform
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Requirement:
    module: str
    package: str
    purpose: str
    modes: frozenset[str] = frozenset({"conversion", "training", "data"})


REQUIREMENTS = (
    Requirement("numpy", "numpy<2", "arrays"),
    Requirement("torch", "torch", "PyTorch model/training"),
    Requirement("yaml", "PyYAML", "training YAML"),
    Requirement("safetensors", "safetensors", "converted checkpoint loading"),
    Requirement("transformers", "transformers==4.53.2", "OpenPI PyTorch model"),
    Requirement("tyro", "tyro>=0.9.5", "OpenPI converter CLI", frozenset({"conversion"})),
    Requirement("jax", "jax", "Orbax/OpenPI checkpoint restore", frozenset({"conversion"})),
    Requirement("flax", "flax==0.10.2", "OpenPI checkpoint/config restore", frozenset({"conversion"})),
    Requirement("orbax.checkpoint", "orbax-checkpoint==0.11.13", "Orbax checkpoint restore", frozenset({"conversion"})),
    Requirement("jaxtyping", "jaxtyping==0.2.36", "OpenPI array typing", frozenset({"conversion", "training"})),
    Requirement("beartype", "beartype==0.19.0", "OpenPI runtime typing", frozenset({"conversion", "training"})),
    Requirement("openpi_client", "openpi-client", "OpenPI workspace client", frozenset({"conversion", "training"})),
    Requirement("einops", "einops>=0.8.0", "OpenPI tensor reshaping", frozenset({"conversion", "training"})),
    Requirement("fsspec", "fsspec", "OpenPI checkpoint/filesystem helpers", frozenset({"conversion", "training"})),
    Requirement("filelock", "filelock", "OpenPI download/cache helpers", frozenset({"conversion", "training"})),
    Requirement("tqdm_loggable", "tqdm-loggable", "OpenPI restore progress", frozenset({"conversion", "training"})),
    Requirement("numpydantic", "numpydantic", "OpenPI normalization types", frozenset({"conversion", "training"})),
    Requirement("pydantic", "pydantic", "OpenPI normalization types", frozenset({"conversion", "training"})),
    Requirement("optax", "optax", "OpenPI training utilities", frozenset({"conversion", "training"})),
    Requirement("etils", "etils", "OpenPI checkpoint utilities", frozenset({"conversion", "training"})),
    Requirement("termcolor", "termcolor", "RMBench dataloader logging", frozenset({"training"})),
    Requirement("lerobot", "lerobot", "RMBench LeRobot dataset", frozenset({"training", "data"})),
    Requirement("cv2", "opencv-python", "HDF5 camera decoding", frozenset({"data"})),
    Requirement("h5py", "h5py", "RMBench HDF5 conversion", frozenset({"data"})),
    Requirement("pyarrow", "pyarrow", "LeRobot parquet/statistics", frozenset({"data", "training"})),
    Requirement("sentencepiece", "sentencepiece>=0.2.0", "PaliGemma tokenizer", frozenset({"training"})),
)


def _version(module: str) -> str:
    try:
        loaded = importlib.import_module(module)
    except Exception:
        return ""
    return str(getattr(loaded, "__version__", "installed"))


def _distribution_name(spec: str) -> str:
    return re.split(r"[<>=!~]", spec, maxsplit=1)[0].strip()


def _installed_version(spec: str) -> str | None:
    try:
        return importlib.metadata.version(_distribution_name(spec))
    except importlib.metadata.PackageNotFoundError:
        return None


def _version_matches(spec: str, installed: str | None) -> bool:
    constraint = spec[len(_distribution_name(spec)):]
    if not constraint:
        # Workspace packages such as openpi-client may be importable from a
        # source checkout without an installed distribution metadata record.
        return True
    if installed is None:
        return False
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version
        return Version(installed) in SpecifierSet(constraint)
    except Exception:
        # Keep the audit useful even if packaging itself is unavailable.
        return not constraint.startswith("==") or constraint == f"=={installed}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-root", type=Path, default=None,
                        help="OpenPI checkout; also read OPENPI_ROOT")
    parser.add_argument("--mode", choices=("all", "conversion", "training", "data"), default="all")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    script = Path(__file__).resolve()
    policy_root = script.parents[1]
    repo_root = policy_root.parents[1]
    openpi_root = args.openpi_root or (Path(os.environ["OPENPI_ROOT"]) if os.environ.get("OPENPI_ROOT") else None)

    import_roots = [
        policy_root,
        policy_root / "source",
        policy_root / "vendor" / "openpi_torch",
    ]
    if openpi_root:
        import_roots.extend([
            openpi_root / "src",
            openpi_root / "packages" / "openpi-client" / "src",
        ])
    for root in reversed(import_roots):
        if root.is_dir():
            sys.path.insert(0, str(root))

    mode = "conversion" if args.mode == "all" else args.mode
    selected = [r for r in REQUIREMENTS if args.mode == "all" or mode in r.modes]
    missing: list[Requirement] = []
    print(f"Python: {sys.executable} ({platform.python_version()})")
    print(f"Policy root: {policy_root}")
    print(f"OpenPI root: {openpi_root or '<not supplied>'}")
    if openpi_root and not (openpi_root / "examples" / "convert_jax_model_to_pytorch.py").is_file():
        print("ERROR: OpenPI converter script not found under --openpi-root/examples")
    if openpi_root and not (openpi_root / "src").is_dir():
        print("ERROR: OpenPI src directory not found under --openpi-root/src")
    print("\nPackage checks:")
    for requirement in selected:
        installed = _installed_version(requirement.package)
        try:
            importlib.import_module(requirement.module)
        except Exception as exc:
            missing.append(requirement)
            print(f"MISSING {requirement.module:<22} {requirement.package:<28} ({requirement.purpose}): {type(exc).__name__}: {exc}")
        else:
            if not _version_matches(requirement.package, installed):
                missing.append(requirement)
                print(f"MISMATCH {requirement.module:<21} required {requirement.package}, found {installed or _version(requirement.module)} ({requirement.purpose})")
            else:
                print(f"OK      {requirement.module:<22} {installed or _version(requirement.module):<28} ({requirement.purpose})")

    if openpi_root:
        patch_root = openpi_root / "src" / "openpi" / "models_pytorch" / "transformers_replace"
        try:
            import transformers
            transformers_version = transformers.__version__
            patch_ok = False
            try:
                from transformers.models.siglip import check
                patch_ok = bool(check.check_whether_transformers_replace_is_installed_correctly())
            except Exception:
                pass
            print(f"\nTransformers patch: {'OK' if patch_ok else 'MISSING'} (version {transformers_version})")
            if not patch_ok:
                print(f"  cp -r {patch_root}/* "
                      '"$($PYTHON -c \'import pathlib, transformers; print(pathlib.Path(transformers.__file__).parent)\')/"')
        except Exception:
            pass

    print("\nSummary:")
    if not missing:
        print("All requested Python imports are available.")
        return 0
    packages = shlex.join(list(dict.fromkeys(item.package for item in missing)))
    print(f"Missing {len(missing)} package(s). Install in this exact environment with:")
    print(f"  {sys.executable} -m pip install {packages}")
    print("Then rerun this audit; it is safe to run repeatedly.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
