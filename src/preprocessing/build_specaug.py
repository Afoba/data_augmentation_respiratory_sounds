"""
build_balanced_specaugment.py
-------------------------------
CLI entry point for generating an OFFLINE, class-balanced SpecAugment'd
dataset. This is the spectrogram-domain counterpart to
build_balanced_augmented.py.

WHY A SEPARATE SCRIPT: SpecAugment (augmentation/spectrogram.py) masks an
already-extracted 2-D feature array (freq x time) — it has no waveform
input, so none of build_balanced_augmented.py's load/filter/pad/extract
machinery applies. This script instead loads the REAL cached .npy feature
array directly, applies frequency/time masking, and saves the result as a
new .npy + metadata row with source_type="augmented".

Baking vs. on-the-fly — a real tradeoff, not just a technicality: the whole
point of SpecAugment during training is normally that a NEW random mask is
drawn every epoch, so the same underlying example never looks identical
twice — that's what makes it a regulariser rather than just more data. This
script bakes ONE fixed mask per generated copy, so what you get is closer to
"N extra static training examples" than "epoch-varying noise". Both are
legitimate; just know the effect on training is different from
augmentation.spec_augment in config/experiments/*.yaml, which still runs
on-the-fly with a fresh mask every epoch regardless of whether you use this
script.

Class balancing reuses the exact same planner as build_balanced_augmented.py
(build_plan) — existing real + previously-generated (matched by
generator-prefix) counts toward each class's target; re-running tops up any
remaining shortfall.

Output is written into the SAME metadata.parquet as everything else,
distinguished by generator="<prefix>__specaugment". Select it at train time
via data.generators in your experiment config, exactly like pitch-shifted or
other offline-augmented rows.

Usage
-----
    python build_balanced_specaugment.py --target 2000 --dry-run
    python build_balanced_specaugment.py --target 2000 --workers 4
    python build_balanced_specaugment.py --target 2000 \\
        --freq-mask-param 15 --time-mask-param 25 \\
        --n-freq-masks 2 --n-time-masks 2 --workers 4
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
import uuid
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))

from restructure.features import param_tag, EXTRACTORS
from augmentation.spectrogram import SpecAugment
from build_augmented import (
    load_yaml, feature_npy_path, build_plan, _parse_target_map,
)

logger = logging.getLogger(__name__)


def process_specaugment_cycle(
    task: dict,
    feature_type: str,
    ptag: str,
    base_dir: Path,
    generator_prefix: str,
    skip_existing: bool,
    spec_cfg: dict,
) -> dict | None:
    record = task["base_record"]
    new_cycle_id = task["new_cycle_id"]
    task_seed = task["task_seed"]

    dataset = record["source_dataset"]
    original_cycle_id = record["cycle_id"]
    generator_name = f"{generator_prefix}__specaugment"

    src_path = feature_npy_path(base_dir, dataset, feature_type, ptag, original_cycle_id)
    out_path = feature_npy_path(base_dir, dataset, feature_type, ptag, new_cycle_id)

    if skip_existing and out_path.exists():
        return _build_meta_row(record, new_cycle_id, original_cycle_id, generator_name, {}, feature_type, out_path)

    try:
        if not src_path.exists():
            raise FileNotFoundError(f"Cached real feature not found: {src_path}")

        feature = np.load(src_path).astype(np.float32)

        # SpecAugment draws from the GLOBAL torch RNG (torch.rand/randint),
        # not an injectable Generator — seed it locally per task so results
        # are reproducible given the same --seed, without disturbing any
        # other process's RNG state (each worker is a separate process).
        torch.manual_seed(task_seed)

        augmenter = SpecAugment(
            freq_mask_param=spec_cfg["freq_mask_param"],
            time_mask_param=spec_cfg["time_mask_param"],
            n_freq_masks=spec_cfg["n_freq_masks"],
            n_time_masks=spec_cfg["n_time_masks"],
            fill_value=spec_cfg["fill_value"],
            p=1.0,  # always apply — this task exists specifically to bake a masked copy
        )
        spec = torch.from_numpy(feature)
        masked = augmenter(spec).numpy().astype(np.float32)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), masked)

    except Exception as e:
        logger.error(
            "Failed to SpecAugment cycle from base=%s (%s): %s",
            original_cycle_id, feature_type, e,
        )
        return None

    return _build_meta_row(record, new_cycle_id, original_cycle_id, generator_name, spec_cfg, feature_type, out_path)


def _build_meta_row(record, new_cycle_id, original_cycle_id, generator_name, spec_cfg, feature_type, out_path) -> dict:
    new_row = dict(record)
    new_row["cycle_id"] = new_cycle_id
    new_row["original_cycle_id"] = original_cycle_id
    new_row["source_type"] = "augmented"
    new_row["generator"] = generator_name
    new_row["aug_params"] = json.dumps({"technique": "specaugment", "params": spec_cfg})
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
    parser = argparse.ArgumentParser(description="Generate an offline, class-balanced SpecAugment'd dataset.")
    parser.add_argument("--datasets", nargs="+", default=["icbhi", "sprsound"], choices=["icbhi", "sprsound"])
    parser.add_argument("--features", nargs="+", default=list(EXTRACTORS.keys()), choices=list(EXTRACTORS.keys()))
    parser.add_argument("--features-cfg", default="config/features.yaml")
    parser.add_argument("--label-col", default="label_4class")
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--target", type=int, default=None)
    parser.add_argument("--target-map", nargs="+", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--generator-prefix", default="offline_aug")
    parser.add_argument("--freq-mask-param", type=int, default=15)
    parser.add_argument("--time-mask-param", type=int, default=25)
    parser.add_argument("--n-freq-masks", type=int, default=2)
    parser.add_argument("--n-time-masks", type=int, default=2)
    parser.add_argument("--fill-value", default="mean", help='"mean" or a float constant')
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    fill_value = args.fill_value
    if fill_value != "mean":
        fill_value = float(fill_value)

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

    # Reuse the exact same planner as build_balanced_augmented.py. It is
    # agnostic to WHAT augmentation will be applied — it just tracks real
    # vs. already-generated counts per class and picks base cycles — so
    # generator_prefix + a distinct techniques marker keeps this script's
    # rows and the waveform script's rows from being double-counted against
    # each other's targets (they use different generator suffixes below).
    tasks, plan_df = build_plan(
        df=df, datasets=args.datasets, label_col=args.label_col, classes=args.classes,
        target=args.target, target_map=target_map, split=args.split,
        generator_prefix=f"{args.generator_prefix}__specaugment", seed=args.seed,
        techniques=["specaugment"], compose_min=1, compose_max=1,
    )
    # build_plan's generator_prefix filter above matched against the FULL
    # generator string, so it already looked for rows whose generator
    # startswith "<prefix>__specaugment" — consistent with what this script
    # writes in _build_meta_row.

    logger.info("Generation plan (split=%s, label_col=%s):\n%s", args.split, args.label_col,
                plan_df.to_string(index=False) if not plan_df.empty else "(empty)")

    if args.dry_run:
        logger.info("--dry-run set: no files written.")
        return
    if not tasks:
        logger.info("Nothing to generate — every class already meets its target.")
        return

    spec_cfg = {
        "freq_mask_param": args.freq_mask_param, "time_mask_param": args.time_mask_param,
        "n_freq_masks": args.n_freq_masks, "n_time_masks": args.n_time_masks,
        "fill_value": fill_value,
    }

    logger.info("Generating %d new cycles × %d feature types…", len(tasks), len(args.features))
    all_new_rows: list[dict] = []
    for feature_type in args.features:
        ptag = param_tag(feature_type, feat_cfg, target_sr)
        worker_fn = partial(
            process_specaugment_cycle, feature_type=feature_type, ptag=ptag, base_dir=base_dir,
            generator_prefix=args.generator_prefix, skip_existing=args.skip_existing, spec_cfg=spec_cfg,
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