"""
Convert an rx2_blackbox LeRobot v2.1 dataset (produced by the blackbox2lerobot
pipeline) into the metadata layout DreamZero expects, without touching the
parquet files or videos.

The rx2_blackbox source layout differs from the DreamZero converter's default
in two ways:

  * State is spread across three columns (observation.state(27),
    observation.left_gripper(1), observation.right_gripper(1)) instead of one
    packed 29-dim column.
  * Action is also spread across three columns (action.wbc(27),
    action.left_gripper(1), action.right_gripper(1)); other action.* columns
    that blackbox2lerobot emits (action.motion_token, action.robot_motion_*)
    are SONIC-internal buffers that the VLA policy does not learn.

The DreamZero data loader supports per-sub-key ``original_key`` in
``modality.json``, so we do not need to repack the parquet files. This script
just writes the correct metadata.

Existing ``meta/tasks.jsonl`` and ``meta/episodes.jsonl`` from blackbox2lerobot
are already in the right format and are left untouched. The DreamZero loader
resolves ``annotation.task`` from the numeric ``task_index`` column via
``tasks.jsonl`` automatically (see groot/vla/data/dataset/lerobot.py:1693).

Usage:
    python scripts/data/convert_rx2_blackbox_to_gear.py \\
        --dataset-path /data/.../rx2_blackbox/<task>/sortie_XXXX_YYYY_train.../
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

EMBODIMENT_TAG = "rx2_blackbox"

# The 6 parquet columns we actually use. Everything else in the parquet
# (action.motion_token, action.robot_motion_*, observation.root_orientation,
# observation.projected_gravity, observation.init_base_quat, blackbox.*) is
# ignored by the policy.
STATE_COLUMNS: dict[str, tuple[str, int]] = {
    "joint_pos":     ("observation.state",          27),
    "left_gripper":  ("observation.left_gripper",    1),
    "right_gripper": ("observation.right_gripper",   1),
}
ACTION_COLUMNS: dict[str, tuple[str, int]] = {
    "joint_pos":     ("action.wbc",           27),
    "left_gripper":  ("action.left_gripper",   1),
    "right_gripper": ("action.right_gripper",  1),
}
# Only joint_pos is relative-to-current-state; grippers stay absolute
# because they are already normalized to [0, 1] and the delta space is small.
RELATIVE_ACTION_SUBKEYS: tuple[str, ...] = ("joint_pos",)
VIDEO_KEY = "observation.images.ego_view"
TASK_INDEX_COLUMN = "task_index"
DEFAULT_ACTION_HORIZON = 24


def load_info(dataset_path: Path) -> dict:
    with (dataset_path / "meta" / "info.json").open() as f:
        return json.load(f)


def get_parquet_paths(dataset_path: Path, info: dict) -> list[Path]:
    pattern = info.get(
        "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    chunks_size = info.get("chunks_size", 1000)
    paths = []
    for ep_idx in range(info["total_episodes"]):
        chunk_idx = ep_idx // chunks_size
        p = dataset_path / pattern.format(episode_chunk=chunk_idx, episode_index=ep_idx)
        if p.exists():
            paths.append(p)
    return sorted(paths)


def build_modality_json(info: dict) -> dict:
    features = info["features"]

    def spec(original_key: str, start: int, end: int) -> dict:
        return {
            "original_key": original_key,
            "start": start,
            "end": end,
            "rotation_type": None,
            "absolute": True,
            "dtype": features[original_key].get("dtype", "float64"),
            "range": None,
        }

    return {
        "state": {
            name: spec(col, 0, dim) for name, (col, dim) in STATE_COLUMNS.items()
        },
        "action": {
            name: spec(col, 0, dim) for name, (col, dim) in ACTION_COLUMNS.items()
        },
        "video": {
            "ego_view": {"original_key": VIDEO_KEY},
        },
        # task text is stored in tasks.jsonl and resolved via task_index by the
        # loader (lerobot.py:1693 auto-lookup for numeric annotation columns).
        "annotation": {
            "task": {"original_key": TASK_INDEX_COLUMN},
        },
    }


def _column_as_2d(series: pd.Series) -> np.ndarray:
    """Stack a parquet column into shape (N, D). Handles both scalar-per-row
    (blackbox2lerobot writes shape (1,) as scalar) and list-per-row layouts."""
    arr = np.stack(series.to_numpy())
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr.astype(np.float64, copy=False)


def compute_stats(parquet_paths: list[Path]) -> dict:
    columns = sorted({col for col, _ in STATE_COLUMNS.values()} | {col for col, _ in ACTION_COLUMNS.values()})
    buffers: dict[str, list[np.ndarray]] = {c: [] for c in columns}
    for pp in tqdm(parquet_paths, desc="stats"):
        df = pd.read_parquet(pp, columns=columns)
        for c in columns:
            buffers[c].append(_column_as_2d(df[c]))
    stats = {}
    for c, chunks in buffers.items():
        data = np.concatenate(chunks, axis=0)
        stats[c] = {
            "mean": data.mean(axis=0).tolist(),
            "std":  data.std(axis=0).tolist(),
            "min":  data.min(axis=0).tolist(),
            "max":  data.max(axis=0).tolist(),
            "q01":  np.quantile(data, 0.01, axis=0).tolist(),
            "q99":  np.quantile(data, 0.99, axis=0).tolist(),
        }
    return stats


def compute_relative_stats(parquet_paths: list[Path], action_horizon: int) -> dict:
    """Relative-action stats for keys present in both state and action modalities.

    Mirrors the logic in groot/vla/data/dataset/lerobot.py
    _calculate_relative_stats_for_key: for each frame i in [0, T - horizon),
    take (action[i:i+horizon] - state[i]) and accumulate.
    """
    stats: dict = {}
    for subkey in RELATIVE_ACTION_SUBKEYS:
        state_col, state_dim = STATE_COLUMNS[subkey]
        action_col, action_dim = ACTION_COLUMNS[subkey]
        if state_dim != action_dim:
            log.error("relative subkey %r has state dim %d != action dim %d", subkey, state_dim, action_dim)
            sys.exit(1)

        all_relative: list[np.ndarray] = []
        for pp in tqdm(parquet_paths, desc=f"relative[{subkey}]"):
            df = pd.read_parquet(pp, columns=[state_col, action_col])
            state = _column_as_2d(df[state_col])
            action = _column_as_2d(df[action_col])
            T = len(df)
            usable = T - action_horizon
            for i in range(max(usable, 0)):
                ref = state[i]
                chunk_end = min(i + action_horizon, T)
                all_relative.append(action[i:chunk_end] - ref)
        if not all_relative:
            log.warning("no relative frames computed for %r", subkey)
            continue
        data = np.concatenate(all_relative, axis=0)
        stats[subkey] = {
            "max":  data.max(axis=0).tolist(),
            "min":  data.min(axis=0).tolist(),
            "mean": data.mean(axis=0).tolist(),
            "std":  data.std(axis=0).tolist(),
            "q01":  np.quantile(data, 0.01, axis=0).tolist(),
            "q99":  np.quantile(data, 0.99, axis=0).tolist(),
        }
    return stats


def validate(dataset_path: Path, info: dict) -> list[str]:
    warnings = []
    features = info.get("features", {})
    for col, dim in list(STATE_COLUMNS.values()) + list(ACTION_COLUMNS.values()):
        if col not in features:
            warnings.append(f"missing feature: {col}")
            continue
        shape = features[col].get("shape") or [1]
        if shape[0] != dim:
            warnings.append(f"{col} shape[0]={shape[0]}, expected {dim}")
    if VIDEO_KEY not in features:
        warnings.append(f"missing video feature: {VIDEO_KEY}")
    if TASK_INDEX_COLUMN not in features:
        warnings.append(f"missing task_index column: {TASK_INDEX_COLUMN}")
    for name in ("tasks.jsonl", "episodes.jsonl"):
        if not (dataset_path / "meta" / name).exists():
            warnings.append(f"missing meta/{name}")
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to one blackbox2lerobot output (e.g. .../sortie_XXXX_YYYY_train.../)")
    parser.add_argument("--action-horizon", type=int, default=DEFAULT_ACTION_HORIZON,
                        help=f"Horizon for relative-action stats (default: {DEFAULT_ACTION_HORIZON})")
    parser.add_argument("--force", action="store_true", help="Overwrite existing GEAR metadata files")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path).resolve()
    if not dataset_path.is_dir():
        log.error("dataset path is not a directory: %s", dataset_path)
        sys.exit(1)

    info = load_info(dataset_path)
    warns = validate(dataset_path, info)
    if warns:
        for w in warns:
            log.error("validation: %s", w)
        log.error("dataset does not match rx2_blackbox schema; aborting.")
        sys.exit(1)

    meta_dir = dataset_path / "meta"
    parquet_paths = get_parquet_paths(dataset_path, info)
    if not parquet_paths:
        log.error("no parquet files found under %s", dataset_path / "data")
        sys.exit(1)

    log.info("dataset: %s", dataset_path.name)
    log.info("  episodes: %d  fps: %s", info["total_episodes"], info.get("fps"))
    log.info("  parquet files: %d", len(parquet_paths))

    modality = build_modality_json(info)
    modality_path = meta_dir / "modality.json"
    if modality_path.exists() and not args.force:
        log.info("  modality.json exists; skipping (use --force to overwrite)")
    else:
        modality_path.write_text(json.dumps(modality, indent=4))
        log.info("  wrote modality.json (%d state / %d action / %d video / %d annotation keys)",
                 len(modality["state"]), len(modality["action"]),
                 len(modality["video"]), len(modality["annotation"]))

    embodiment_path = meta_dir / "embodiment.json"
    if embodiment_path.exists() and not args.force:
        log.info("  embodiment.json exists; skipping")
    else:
        embodiment_path.write_text(json.dumps(
            {"robot_type": info.get("robot_type", EMBODIMENT_TAG),
             "embodiment_tag": EMBODIMENT_TAG},
            indent=4,
        ))
        log.info("  wrote embodiment.json (tag=%s)", EMBODIMENT_TAG)

    stats_path = meta_dir / "stats.json"
    if stats_path.exists() and not args.force:
        log.info("  stats.json exists; skipping")
    else:
        stats_path.write_text(json.dumps(compute_stats(parquet_paths), indent=4))
        log.info("  wrote stats.json")

    rel_stats_path = meta_dir / "relative_stats_dreamzero.json"
    if rel_stats_path.exists() and not args.force:
        log.info("  relative_stats_dreamzero.json exists; skipping")
    else:
        rel = compute_relative_stats(parquet_paths, args.action_horizon)
        rel_stats_path.write_text(json.dumps(rel, indent=4))
        log.info("  wrote relative_stats_dreamzero.json (keys=%s)", list(rel))

    print("\n" + "=" * 60)
    print(f"OK: {dataset_path}")
    print(f"  embodiment_tag = {EMBODIMENT_TAG}")
    print(f"  state sub-keys = {list(STATE_COLUMNS)}")
    print(f"  action sub-keys = {list(ACTION_COLUMNS)}")
    print(f"  relative sub-keys = {list(RELATIVE_ACTION_SUBKEYS)}")
    print(f"  video key = {VIDEO_KEY}")
    print("=" * 60)


if __name__ == "__main__":
    main()
