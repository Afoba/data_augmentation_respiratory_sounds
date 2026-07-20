"""
build_mixup_pairs.py
----------------------
CLI entry point for generating an OFFLINE "spectrogram mix-up" pool, following
the specific strategy in Wang et al. 2025 (Sci Rep 15:39268): pair a Crackle
cycle and a Wheeze cycle, linearly interpolate their cached spectrograms
(their eq. 3), and treat the result as a synthetic Crackle+Wheeze ("both")
sample — used specifically to relieve C&W's severe minority-class shortage.

This is intentionally NOT a general-purpose mixup script (it doesn't mix
arbitrary pairs/classes) — it implements the one specific C+W -> "both"
recipe the paper describes.

IMPORTANT SIMPLIFICATION — hard label, not soft label
-------------------------------------------------------
The paper's own math (eq. 3) produces a SOFT label: mixing a Crackle (label
[1,0]) and Wheeze (label [0,1]) sample with factor beta gives target
[beta, 1-beta] — a genuinely continuous target requiring a loss function
that accepts float targets (e.g. soft cross-entropy / KL-divergence), not
hard integer class indices.

Your losses.py currently only builds nn.CrossEntropyLoss with hard integer
targets — there's no soft-label loss wired up in the training pipeline. The
paper itself sidesteps exactly this by choosing to treat every C+W mix as a
hard "both" record for bookkeeping purposes ("it is natural to regard it as
a newly C&W record" — see their mix-up section), even though during their
own actual training they say soft labels were used. Given you've opted for
the practical-approximation path, this script follows their SIMPLER
bookkeeping choice: every generated pair gets the hard label "both" (or
whatever --output-class you specify), not a soft target. The exact mixing
factor is still recorded in aug_params for reference/debugging, and if you
later want true soft-label training, the natural extension is a
soft-cross-entropy loss in losses.py plus a float-label path in dataset.py —
happy to build that if you decide you want full fidelity here later.

Usage
-----
    python build_mixup_pairs.py --n-samples 500 --workers 4
    python build_mixup_pairs.py --n-samples 500 --alpha 1.0 --dry-run
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

sys.path.insert(0, str(Path(__file__).parent))

from restructure.features import param_tag, EXTRACTORS
from build_augmented import load_yaml, feature_npy_path

logger = logging.getLogger(__name__)


def plan_pairs(
    df: pd.DataFrame,
    datasets: list[str],
    label_col: str,
    class_a: str,
    class_b: str,
    split: str,
    n_samples: int,
    alpha: float,
    seed: int,
) -> list[dict]:
    df_real = df[(df["source_type"] == "real")
                 & (df["source_dataset"].isin(datasets))
                 & (df["split"] == split)]
    df_real = df_real.drop_duplicates(subset=["cycle_id"])

    pool_a = df_real[df_real[label_col] == class_a].drop(
        columns=["feature_type", "feature_path"], errors="ignore").to_dict("records")
    pool_b = df_real[df_real[label_col] == class_b].drop(
        columns=["feature_type", "feature_path"], errors="ignore").to_dict("records")

    if not pool_a:
        raise ValueError(f"No real {class_a!r} cycles found for datasets={datasets}, split={split!r}.")
    if not pool_b:
        raise ValueError(f"No real {class_b!r} cycles found for datasets={datasets}, split={split!r}.")

    rng = np.random.default_rng(seed)
    tasks = []
    for i in range(n_samples):
        rec_a = pool_a[rng.integers(0, len(pool_a))]
        rec_b = pool_b[rng.integers(0, len(pool_b))]
        beta = float(rng.beta(alpha, alpha))
        suffix = uuid.uuid4().hex[:8]
        new_cycle_id = f"mixup__{rec_a['cycle_id']}__{rec_b['cycle_id']}__{i:05d}_{suffix}"
        tasks.append({
            "record_a": rec_a, "record_b": rec_b, "beta": beta, "new_cycle_id": new_cycle_id,
        })
    return tasks


def process_mixup_pair(
    task: dict,
    feature_type: str,
    ptag: str,
    base_dir: Path,
    generator_prefix: str,
    output_class: str,
    label_cols: list[str],
    skip_existing: bool,
) -> dict | None:
    rec_a, rec_b, beta = task["record_a"], task["record_b"], task["beta"]
    new_cycle_id = task["new_cycle_id"]
    generator_name = f"{generator_prefix}__mixup"

    # Both records must be from the same dataset for their cached paths to be
    # directly comparable/combinable — this is enforced by construction since
    # plan_pairs draws each class's pool from the same --datasets filter, but
    # a pair CAN legitimately span datasets (e.g. crackle from icbhi, wheeze
    # from sprsound) if --datasets includes both. Use record_a's dataset for
    # the output path in that case.
    dataset = rec_a["source_dataset"]
    src_a = feature_npy_path(base_dir, rec_a["source_dataset"], feature_type, ptag, rec_a["cycle_id"])
    src_b = feature_npy_path(base_dir, rec_b["source_dataset"], feature_type, ptag, rec_b["cycle_id"])
    out_path = feature_npy_path(base_dir, dataset, feature_type, ptag, new_cycle_id)

    if skip_existing and out_path.exists():
        return _build_meta_row(rec_a, rec_b, beta, new_cycle_id, generator_name,
                                output_class, label_cols, feature_type, out_path)

    try:
        if not src_a.exists():
            raise FileNotFoundError(f"Cached real feature not found: {src_a}")
        if not src_b.exists():
            raise FileNotFoundError(f"Cached real feature not found: {src_b}")

        xa = np.load(src_a).astype(np.float32)
        xb = np.load(src_b).astype(np.float32)
        mixed = beta * xa + (1.0 - beta) * xb

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), mixed)

    except Exception as e:
        logger.error("Failed to mix %s + %s (%s): %s", rec_a["cycle_id"], rec_b["cycle_id"], feature_type, e)
        return None

    return _build_meta_row(rec_a, rec_b, beta, new_cycle_id, generator_name,
                            output_class, label_cols, feature_type, out_path)


def _build_meta_row(rec_a, rec_b, beta, new_cycle_id, generator_name, output_class, label_cols, feature_type, out_path) -> dict:
    new_row = dict(rec_a)  # template: split/device/etc. inherited from record_a
    new_row["cycle_id"] = new_cycle_id
    new_row["original_cycle_id"] = f"{rec_a['cycle_id']}+{rec_b['cycle_id']}"
    new_row["source_type"] = "augmented"
    new_row["generator"] = generator_name
    for col in label_cols:
        new_row[col] = output_class
    new_row["aug_params"] = json.dumps({
        "technique": "mixup", "beta": beta,
        "record_a": rec_a["cycle_id"], "record_b": rec_b["cycle_id"],
    })
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
    parser = argparse.ArgumentParser(
        description="Generate an offline Crackle+Wheeze -> 'both' mix-up pool (hard-label simplification)."
    )
    parser.add_argument("--datasets", nargs="+", default=["icbhi", "sprsound"], choices=["icbhi", "sprsound"])
    parser.add_argument("--features", nargs="+", default=list(EXTRACTORS.keys()), choices=list(EXTRACTORS.keys()))
    parser.add_argument("--features-cfg", default="config/features.yaml")
    parser.add_argument("--label-col", default="label_4class")
    parser.add_argument("--class-a", default="crackle")
    parser.add_argument("--class-b", default="wheeze")
    parser.add_argument("--output-class", default="both",
                        help="Hard label assigned to every generated pair (paper's own simplification).")
    parser.add_argument("--n-samples", type=int, required=True,
                        help="Exact number of new C+W pairs to generate (paper used 500).")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Beta(alpha, alpha) mixing-factor distribution. The paper specifies a beta "
                             "distribution but not a specific alpha; 1.0 = uniform on [0,1]. "
                             "augmentation/spectrogram.py's SpectrogramMixup defaults to alpha=0.4 if you'd "
                             "rather match that repo convention instead.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--generator-prefix", default="specmixup")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    feat_cfg = load_yaml(args.features_cfg)
    base_dir = Path(feat_cfg["output"]["base_dir"]).expanduser().resolve()
    meta_file = Path(feat_cfg["output"]["metadata_file"]).expanduser().resolve()
    target_sr = feat_cfg.get("target_sr", 16000)

    if not meta_file.exists():
        logger.error("metadata.parquet not found at %s. Run build_features.py first.", meta_file)
        sys.exit(1)

    df = pd.read_parquet(meta_file)
    label_cols = [c for c in ["label_4class", "label_fine", "label_coarse"] if c in df.columns and c == args.label_col] \
        or [args.label_col]

    tasks = plan_pairs(
        df=df, datasets=args.datasets, label_col=args.label_col,
        class_a=args.class_a, class_b=args.class_b, split=args.split,
        n_samples=args.n_samples, alpha=args.alpha, seed=args.seed,
    )
    logger.info("Plan: %d new %s+%s -> %s pairs (alpha=%.2f)",
                len(tasks), args.class_a, args.class_b, args.output_class, args.alpha)

    if args.dry_run:
        logger.info("--dry-run set: no files written.")
        return

    all_new_rows: list[dict] = []
    for feature_type in args.features:
        ptag = param_tag(feature_type, feat_cfg, target_sr)
        worker_fn = partial(
            process_mixup_pair, feature_type=feature_type, ptag=ptag, base_dir=base_dir,
            generator_prefix=args.generator_prefix, output_class=args.output_class,
            label_cols=label_cols, skip_existing=args.skip_existing,
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