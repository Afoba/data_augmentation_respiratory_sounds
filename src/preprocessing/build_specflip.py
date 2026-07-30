"""
build_balanced_specflip.py
----------------------------
CLI entry point for generating an OFFLINE, class-balanced Spectrogram-flipped
dataset, following the "Spectrogram flipping" method described in Wang et al.
2025 (Sci Rep 15:39268): each real cycle's cached spectrogram is reflected
along one axis, producing a synthetic same-label copy.

Terminology mapping — paper vs. this codebase
------------------------------------------------
The paper defines flips on an image of size (width, height), where width is
the time axis and height is the frequency axis (see their Fig. 3):

    "horizontal" flip: (x, y) -> (width - x - 1, y)   — reflects the TIME axis
    "vertical"   flip: (x, y) -> (x, height - y - 1)  — reflects the FREQ axis

augmentation/spectrogram.py's SpectrogramFlip names these the other way
round (by axis, not by screen direction): time_flip_p / freq_flip_p. So:

    --mode vertical    <->  freq_flip_p=1.0  (paper's "vertical flipping")
    --mode horizontal  <->  time_flip_p=1.0  (paper's "horizontal flipping")
    --mode both         <->  both at once — NOT in the paper (see below)

The paper never flips both axes on the same sample — it only ever generates
vertical-only or horizontal-only copies (Table 1, Table 3: their augmented
pool is 3x the original — original + all-vertical + all-horizontal). `--mode
both` is provided here as an explicit exception/extension beyond the paper,
for when you want to compare against a "180-degree rotation" variant too.

Reuses the exact same class-balancing planner as build_balanced_augmented.py
/ build_balanced_specaugment.py (build_plan): existing real + previously-
generated (matched by generator-prefix) counts toward each class's target;
re-running tops up any remaining shortfall. Each invocation handles ONE mode
— call it up to three times (vertical / horizontal / both) with distinct
--generator-prefix values (or a shared prefix plus the mode is already
folded into the generator name — see note below) to build up each pool
independently.

Note on the paper's "flip every real cycle" step vs. this script's
per-class TARGET: the paper's step 1 flips every real cycle in the entire
dataset (uniformly tripling every class's pool), then subsamples per class
at train/epoch time — the quantity that matters is a flat "N new copies per
class," independent of how many real cycles that class has. By default this
script instead works like the other two offline scripts here (top up to a
combined real+synthetic total), which is the wrong semantics for that goal —
n_needed comes out to 0 for any class whose real count already exceeds
--target. Pass --exact to switch to the flat-count semantics instead: with
--exact, --target/--target-map means "generate exactly this many new
samples," ignoring n_real entirely — e.g. --mode vertical --target 500
--exact gives 500 new vertical-flipped copies of every class regardless of
real count, no real_count + N arithmetic required, and it stays correct
even if your real counts change later (a different train/val split, etc.),
since it was never a function of them to begin with.

Usage
-----
    # Vertical-only (paper's "vertical flipping"), 500/class
    python build_balanced_specflip.py --mode vertical --target 500 \\
        --generator-prefix specflip_v --workers 4

    # Horizontal-only, 500/class
    python build_balanced_specflip.py --mode horizontal --target 500 \\
        --generator-prefix specflip_h --workers 4

    # Exception: both axes at once, 500/class
    python build_balanced_specflip.py --mode both --target 500 \\
        --generator-prefix specflip_b --workers 4
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))

from restructure.features import param_tag, EXTRACTORS
from augmentation.spectrogram import SpectrogramFlip
from build_augmented import (
    load_yaml, feature_npy_path, build_plan, _parse_target_map,
)

logger = logging.getLogger(__name__)

# paper's flip name -> (time_flip_p, freq_flip_p) for augmentation/spectrogram.py's SpectrogramFlip
_MODE_TO_FLIP_PS = {
    "vertical":   {"time_flip_p": 0.0, "freq_flip_p": 1.0},   # paper: vertical flip
    "horizontal": {"time_flip_p": 1.0, "freq_flip_p": 0.0},   # paper: horizontal flip
    "both":       {"time_flip_p": 1.0, "freq_flip_p": 1.0},   # exception, not in the paper
}


def process_flip_cycle(
    task: dict,
    flipper: SpectrogramFlip,
    mode: str,
    feature_type: str,
    ptag: str,
    base_dir: Path,
    generator_prefix: str,
    skip_existing: bool,
) -> dict | None:
    record = task["base_record"]
    new_cycle_id = task["new_cycle_id"]

    dataset = record["source_dataset"]
    original_cycle_id = record["cycle_id"]
    generator_name = f"{generator_prefix}__specflip_{mode}"

    src_path = feature_npy_path(base_dir, dataset, feature_type, ptag, original_cycle_id)
    out_path = feature_npy_path(base_dir, dataset, feature_type, ptag, new_cycle_id)

    if skip_existing and out_path.exists():
        return _build_meta_row(record, new_cycle_id, original_cycle_id, generator_name, mode, feature_type, out_path)

    try:
        if not src_path.exists():
            raise FileNotFoundError(f"Cached real feature not found: {src_path}")

        feature = np.load(src_path).astype(np.float32)
        spec = torch.from_numpy(feature)
        flipped = flipper(spec).numpy().astype(np.float32)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), flipped)

    except Exception as e:
        logger.error("Failed to flip cycle from base=%s (mode=%s, %s): %s",
                     original_cycle_id, mode, feature_type, e)
        return None

    return _build_meta_row(record, new_cycle_id, original_cycle_id, generator_name, mode, feature_type, out_path)


def _build_meta_row(record, new_cycle_id, original_cycle_id, generator_name, mode, feature_type, out_path) -> dict:
    new_row = dict(record)
    new_row["cycle_id"] = new_cycle_id
    new_row["original_cycle_id"] = original_cycle_id
    new_row["source_type"] = "augmented"
    new_row["generator"] = generator_name
    new_row["aug_params"] = json.dumps({"technique": f"specflip_{mode}", "params": _MODE_TO_FLIP_PS[mode]})
    new_row["feature_type"] = feature_type
    new_row["feature_path"] = str(out_path)
    return new_row


def _parallel_map(fn, items: list, n_workers: int) -> list:
    results = [None] * len(items)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(fn, item): idx for idx, item in enumerate(items)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error("Worker raised exception for item %d: %s", idx, e)
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate an offline, class-balanced Spectrogram-flipped dataset.")
    parser.add_argument("--mode", required=True, choices=["vertical", "horizontal", "both"],
                        help="vertical/horizontal follow the paper's naming (see module docstring for the "
                             "mapping to this codebase's time_flip/freq_flip). 'both' flips both axes on the "
                             "same sample — an exception beyond what the paper does.")
    parser.add_argument("--datasets", nargs="+", default=["icbhi", "sprsound"], choices=["icbhi", "sprsound"])
    parser.add_argument("--features", nargs="+", default=list(EXTRACTORS.keys()), choices=list(EXTRACTORS.keys()))
    parser.add_argument("--features-cfg", default="config/features.yaml")
    parser.add_argument("--label-col", default="label_4class")
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--target", type=int, default=None,
                        help="Uniform target sample count per class. Meaning depends on --exact.")
    parser.add_argument("--target-map", nargs="+", default=None)
    parser.add_argument("--exact", action="store_true",
                        help="Generate exactly --target/--target-map new flipped copies per class, "
                             "ignoring real count — e.g. --mode vertical --target 500 --exact gives "
                             "500 vertical-flipped copies of every class regardless of how many real "
                             "cycles exist, no real_count + N arithmetic needed. Without this flag, "
                             "--target is a top-up total instead (existing default behavior).")
    parser.add_argument("--split", default="train")
    parser.add_argument("--generator-prefix", default="specflip",
                        help="Give each --mode run its own prefix (not a prefix of the others) so their "
                             "per-class targets are tracked independently on re-runs.")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    target_map = _parse_target_map(args.target_map)
    if args.target is None and not target_map:
        logger.error("Specify --target (uniform) and/or --target-map (per-class overrides).")
        sys.exit(1)

    feat_cfg = load_yaml(args.features_cfg)
    base_dir = Path(feat_cfg["output"]["base_dir"]).expanduser().resolve()
    meta_file = Path(feat_cfg["output"]["metadata_file"]).expanduser().resolve()
    target_sr = feat_cfg.get("target_sr", 16000)

    if not meta_file.exists():
        logger.error("metadata.parquet not found at %s. Run build_features.py first.", meta_file)
        sys.exit(1)

    df = pd.read_parquet(meta_file)

    tasks, plan_df = build_plan(
        df=df, datasets=args.datasets, label_col=args.label_col, classes=args.classes,
        target=args.target, target_map=target_map, split=args.split,
        generator_prefix=f"{args.generator_prefix}__specflip_{args.mode}", seed=args.seed,
        techniques=[args.mode], compose_min=1, compose_max=1, exact=args.exact,
    )

    logger.info("Generation plan (mode=%s, split=%s, label_col=%s):\n%s", args.mode, args.split, args.label_col,
                plan_df.to_string(index=False) if not plan_df.empty else "(empty)")

    if args.dry_run:
        logger.info("--dry-run set: no files written.")
        return
    if not tasks:
        logger.info("Nothing to generate — every class already meets its target.")
        return

    # Construct once (not per-task/worker-call): SpectrogramFlip warns once at
    # construction time when a flip probability is nonzero — building it here
    # keeps that to a single warning instead of one per generated sample.
    with warnings.catch_warnings():
        warnings.simplefilter("once")
        flipper = SpectrogramFlip(**_MODE_TO_FLIP_PS[args.mode])

    logger.info("Generating %d new cycles × %d feature types (mode=%s)…", len(tasks), len(args.features), args.mode)
    all_new_rows: list[dict] = []
    for feature_type in args.features:
        ptag = param_tag(feature_type, feat_cfg, target_sr)
        worker_fn = partial(
            process_flip_cycle, flipper=flipper, mode=args.mode, feature_type=feature_type, ptag=ptag,
            base_dir=base_dir, generator_prefix=args.generator_prefix, skip_existing=args.skip_existing,
        )
        results = _parallel_map(worker_fn, tasks, args.workers) if args.workers > 1 else [worker_fn(t) for t in tasks]
        n_ok = sum(1 for r in results if r is not None)
        logger.info("%s: %d/%d generated successfully.", feature_type, n_ok, len(tasks))
        all_new_rows.extend(r for r in results if r is not None)

    if all_new_rows:
        df_new = pd.DataFrame(all_new_rows)
        df_combined = (
            pd.concat([df, df_new], ignore_index=True)
            .drop_duplicates(subset=["cycle_id", "feature_type"], keep="last")
        )
        df_combined.to_parquet(meta_file, index=False)
        logger.info("Metadata updated at %s — added %d rows (%d total rows now).",
                    meta_file, len(df_new), len(df_combined))
    else:
        logger.warning("No rows produced — check errors above.")


if __name__ == "__main__":
    main()