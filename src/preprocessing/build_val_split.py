"""
build_val_split.py
-------------------
Carves a validation set out of the existing "train" split in metadata.parquet,
WITHOUT touching "test" rows at all (so official-benchmark comparability is
preserved).

Splitting strategy
-------------------
Grouped + approximately stratified:

  - Grouping unit: ICBHI -> patient_id, SPRSound -> recording_id.
    All cycles belonging to the same group are assigned to the SAME split
    (train or val), never split across — this prevents leakage (a model
    should never see one cycle from a patient at train time and a different
    cycle from the SAME patient at validation time).

  - Stratification target: per-group label distribution is used to greedily
    assign groups to train/val such that the AGGREGATE class balance (and,
    where available, device balance) across the two splits stays close to
    the requested val_fraction. This is an approximate/greedy stratified
    split, not an exact one — exact joint group+label balancing is not
    generally solvable (e.g. a single large single-label patient cannot be
    balanced away). Achieved per-split class proportions are printed/logged
    so you can inspect how close the approximation got.

This script is IDEMPOTENT and SAFE to re-run: each run recomputes the
train/val assignment from scratch using ALL non-"test" real rows currently
in metadata.parquet (i.e. rows currently marked "train" OR "val" — not just
"train"), so a previous run's val assignment doesn't shrink the pool seen by
the next run. Rows already marked "test" are never touched. Re-running with
the same --seed and --val-fraction reproduces the same split exactly.

CRITICAL — augmented/synthetic row propagation: derivative rows (anything
with source_type != "real", e.g. pitch-shift augmented variants from
build_augmented.py or synthetic rows from write_synthetic.py) are NOT
included in the train/val assignment decision itself, but every derivative
row's split is forced to match its parent real cycle's CURRENT split after
that decision is made (matched via original_cycle_id). Without this, a real
cycle moved into "val" would leave its augmented siblings behind in "train"
with a stale split value inherited from whenever build_augmented.py was run
— the model would then train on a near-duplicate (e.g. pitch-shifted by one
semitone) of its own validation data, producing inflated val metrics that
don't reflect real generalisation. A consistency check runs before writing
to verify no cycle "family" (a real cycle + all its derivatives) ends up
spanning more than one split; the script aborts without writing if it does.

Usage
-----
    python build_val_split.py --val-fraction 0.2 --seed 42

    # Inspect the resulting balance without writing anything
    python build_val_split.py --val-fraction 0.2 --seed 42 --dry-run

Options
-------
    --features-cfg   Path to features config (for metadata_file location). Default: config/features.yaml
    --val-fraction   Fraction of TRAIN groups to move to validation. Default: 0.2
    --seed           Random seed for the greedy assignment / tie-breaking. Default: 42
    --label-col      Which label column to stratify on. Default: label_4class
    --stratify-device  Also balance by `device` column (ICBHI only) in addition to label. Default: off
    --datasets       Restrict to specific datasets. Default: icbhi sprsound (both, independently split)
    --dry-run        Compute and print the split summary but do not write metadata.parquet
    --log-level      Default: INFO
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Maps each dataset to its grouping column (the unit that must stay whole)
_GROUP_COLUMN = {
    "icbhi": "patient_id",
    "sprsound": "recording_id",
}


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _group_label_counts(df: pd.DataFrame, group_col: str, label_col: str) -> pd.DataFrame:
    """
    One row per group: group id, total cycle count, and per-label counts.
    Operates on DISTINCT cycles (deduplicated across feature_type rows),
    since a cycle appears once per feature_type in metadata.parquet and we
    don't want to triple-count it when balancing.
    """
    cycles = df.drop_duplicates(subset=["cycle_id"])
    counts = (
        cycles.groupby([group_col, label_col])
        .size()
        .unstack(fill_value=0)
    )
    counts["__total__"] = counts.sum(axis=1)
    return counts


def _greedy_stratified_group_split(
    group_counts: pd.DataFrame,
    val_fraction: float,
    seed: int,
) -> tuple[set, set]:
    """
    Greedily assign groups to train/val to approximately match val_fraction
    of total cycles AND approximately preserve the per-label proportions
    that exist across all groups combined.

    Algorithm: largest-remainder greedy bin-packing.
      1. Shuffle groups (seeded) to avoid deterministic bias from input order.
      2. Sort by total group size descending (place big groups first — this
         is the standard heuristic for balanced bin-packing problems, since
         small groups are easier to use later for fine-grained correction).
      3. For each group, tentatively compute the resulting val-side class
         proportions if assigned to val vs. to train; assign to whichever
         keeps the running val-side label distribution closer to the global
         label distribution, subject to not overshooting val_fraction of
         total cycles by too much.

    Returns
    -------
    (val_group_ids, train_group_ids)
    """
    rng = np.random.default_rng(seed)

    label_cols = [c for c in group_counts.columns if c != "__total__"]
    total_cycles = group_counts["__total__"].sum()
    target_val_cycles = total_cycles * val_fraction

    # Global label proportions — what we want val (and train) to resemble
    global_label_totals = group_counts[label_cols].sum(axis=0)
    global_label_props = global_label_totals / global_label_totals.sum()

    # Shuffle group order, then sort by size descending (stable shuffle-then-sort
    # means ties in size are broken randomly, not by original/group-id order)
    groups = group_counts.sample(frac=1.0, random_state=seed).copy()
    groups = groups.iloc[np.argsort(-groups["__total__"].values, kind="stable")]

    val_ids: list = []
    train_ids: list = []
    val_label_running = pd.Series(0, index=label_cols, dtype=float)
    val_cycles_running = 0

    for group_id, row in groups.iterrows():
        group_total = row["__total__"]
        group_labels = row[label_cols]

        # If adding this group to val would overshoot target by more than
        # one group's worth, prefer train (prevents val from ballooning when
        # iterating through groups, since we sorted by size descending).
        room_left = target_val_cycles - val_cycles_running

        if room_left <= 0:
            train_ids.append(group_id)
            continue

        # Compute the label-distribution distance (L1) if this group goes to val
        hypothetical_val = val_label_running + group_labels
        hypothetical_val_props = hypothetical_val / hypothetical_val.sum() if hypothetical_val.sum() > 0 else hypothetical_val
        dist_if_val = float((hypothetical_val_props - global_label_props).abs().sum())

        # Compute the label-distribution distance if this group goes to train instead
        # (val distribution stays as-is; we only check whether skipping hurts val's progress)
        dist_if_skip = float(
            ((val_label_running / val_label_running.sum() if val_label_running.sum() > 0 else val_label_running)
             - global_label_props).abs().sum()
        )

        # Assign to val if doing so does not overshoot too much AND keeps
        # (or improves) label balance; else train. A small randomised
        # tolerance avoids a fully deterministic greedy artifact.
        overshoot_tolerance = group_total * 1.5  # allow modest overshoot for large groups
        if group_total <= room_left + overshoot_tolerance and (
            dist_if_val <= dist_if_skip or rng.random() < 0.15
        ):
            val_ids.append(group_id)
            val_label_running = hypothetical_val
            val_cycles_running += group_total
        else:
            train_ids.append(group_id)

    return set(val_ids), set(train_ids)


def _summarise_split(
    df: pd.DataFrame,
    group_col: str,
    label_col: str,
    val_ids: set,
    train_ids: set,
    device_col: str | None = None,
) -> str:
    cycles = df.drop_duplicates(subset=["cycle_id"]).copy()
    cycles["__assigned_split__"] = np.where(
        cycles[group_col].isin(val_ids), "val",
        np.where(cycles[group_col].isin(train_ids), "train", "unassigned"),
    )

    lines = []
    n_train_groups, n_val_groups = len(train_ids), len(val_ids)
    lines.append(f"Groups: {n_train_groups} train, {n_val_groups} val "
                 f"({n_val_groups / (n_train_groups + n_val_groups) * 100:.1f}% of groups)")

    for split_name in ("train", "val"):
        subset = cycles[cycles["__assigned_split__"] == split_name]
        n = len(subset)
        if n == 0:
            lines.append(f"  {split_name}: 0 cycles")
            continue
        props = (subset[label_col].value_counts(normalize=True) * 100).round(1)
        lines.append(f"  {split_name}: {n} cycles — label %: {props.to_dict()}")
        if device_col and device_col in subset.columns and subset[device_col].notna().any():
            dev_props = (subset[device_col].value_counts(normalize=True) * 100).round(1)
            lines.append(f"             device %: {dev_props.to_dict()}")

    return "\n".join(lines)


def build_val_split_for_dataset(
    df: pd.DataFrame,
    dataset: str,
    label_col: str,
    val_fraction: float,
    seed: int,
    stratify_device: bool,
) -> pd.Series:
    """
    Returns a pandas Series indexed like df (for rows of this dataset with
    split=="train") giving the NEW split value ("train" or "val") for each row.
    """
    group_col = _GROUP_COLUMN[dataset]

    df_dataset = df[(df["source_dataset"] == dataset) & (df["source_type"] == "real")]
    # IMPORTANT: pool from everything that is NOT "test", not just rows
    # currently marked "train". This script is meant to be re-run repeatedly
    # (e.g. with a different seed or val_fraction) — if we only looked at
    # split=="train", a previous run's "val" rows would be excluded from the
    # pool on the next run, silently shrinking what's available and breaking
    # idempotency / reproducibility across re-runs.
    df_train = df_dataset[df_dataset["split"] != "test"]

    if df_train.empty:
        logger.warning("No non-test real rows found for dataset=%s — skipping.", dataset)
        return pd.Series(dtype=object)

    if df_train[group_col].isna().any():
        n_missing = df_train[group_col].isna().sum()
        logger.warning(
            "%d rows in dataset=%s have a missing %s — these will be treated "
            "as their own singleton groups (cannot verify no-leakage grouping for them).",
            n_missing, dataset, group_col,
        )
        # Fill missing group ids with their own cycle_id so they at least don't
        # get silently merged into a shared "None" group.
        df_train = df_train.copy()
        df_train[group_col] = df_train[group_col].fillna(df_train["cycle_id"])

    strat_label_col = label_col
    if stratify_device and dataset == "icbhi" and "device" in df_train.columns:
        # Combine label + device into a compound stratification key by
        # concatenating group-level counts on a synthetic combined column.
        df_train = df_train.copy()
        df_train["__strat_key__"] = df_train[label_col].astype(str) + "__" + df_train["device"].astype(str)
        strat_label_col = "__strat_key__"

    group_counts = _group_label_counts(df_train, group_col, strat_label_col)
    val_ids, train_ids = _greedy_stratified_group_split(group_counts, val_fraction, seed)

    summary = _summarise_split(
        df_train, group_col, label_col, val_ids, train_ids,
        device_col="device" if dataset == "icbhi" else None,
    )
    logger.info("Split summary for dataset=%s (group_col=%s):\n%s", dataset, group_col, summary)

    new_split = df_train[group_col].map(
        lambda g: "val" if g in val_ids else ("train" if g in train_ids else "train")
    )
    return new_split


def main():
    parser = argparse.ArgumentParser(description="Carve a validation split out of existing train data.")
    parser.add_argument("--features-cfg", default="config/features.yaml")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-col", default="label_4class")
    parser.add_argument("--stratify-device", action="store_true",
                        help="Also balance by device column for ICBHI")
    parser.add_argument("--datasets", nargs="+", default=["icbhi", "sprsound"],
                        choices=["icbhi", "sprsound"])
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and print the split but do not write metadata.parquet")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    feat_cfg = load_yaml(args.features_cfg)
    meta_file = Path(feat_cfg["output"]["metadata_file"]).expanduser().resolve()

    if not meta_file.exists():
        logger.error("metadata.parquet not found at %s. Run build_features.py first.", meta_file)
        sys.exit(1)

    df = pd.read_parquet(meta_file)

    if args.label_col not in df.columns:
        logger.error("label_col=%r not found in metadata columns: %s", args.label_col, df.columns.tolist())
        sys.exit(1)

    df = df.set_index(df.index)  # keep original index for safe assignment
    updated_split = df["split"].copy()

    # Track, per (dataset, original real cycle_id), what its NEW split is —
    # used below to propagate the reassignment to derivative rows.
    reassigned_real_cycle_split: dict[str, str] = {}

    for dataset in args.datasets:
        new_split_for_train_rows = build_val_split_for_dataset(
            df, dataset, args.label_col, args.val_fraction, args.seed, args.stratify_device,
        )
        if not new_split_for_train_rows.empty:
            updated_split.loc[new_split_for_train_rows.index] = new_split_for_train_rows
            # Map cycle_id -> new split, for propagation to augmented/synthetic
            # rows below. Real rows can repeat across feature_type, but
            # cycle_id -> split is consistent across those repeats, so a
            # plain dict update (last-write-wins on duplicates) is safe.
            reassigned_cycle_ids = df.loc[new_split_for_train_rows.index, "cycle_id"]
            for cid, new_split in zip(reassigned_cycle_ids, new_split_for_train_rows):
                reassigned_real_cycle_split[cid] = new_split

    # ── Propagate to augmented/synthetic rows derived from reassigned cycles ──
    # CRITICAL: without this step, a real cycle moved into "val" still has its
    # pitch-shifted (or other derivative) copies sitting in "train" with a
    # stale split value inherited at the time build_augmented.py ran — the
    # model then trains on a near-duplicate of validation data, producing
    # inflated val metrics that don't reflect real generalisation. Every
    # derivative row's split must always match its source real cycle's
    # CURRENT split, not whatever split existed when the derivative was created.
    if "original_cycle_id" in df.columns:
        has_parent = df["original_cycle_id"].notna()
        n_propagated = 0
        for idx in df.index[has_parent]:
            parent_id = df.at[idx, "original_cycle_id"]
            if parent_id in reassigned_real_cycle_split:
                new_split = reassigned_real_cycle_split[parent_id]
                if updated_split.at[idx] != new_split:
                    updated_split.at[idx] = new_split
                    n_propagated += 1
        if n_propagated:
            logger.info(
                "Propagated split reassignment to %d augmented/synthetic row(s) "
                "whose original_cycle_id was moved to a different split — "
                "without this, derivative rows would leak across train/val.",
                n_propagated,
            )

    n_changed = (updated_split != df["split"]).sum()
    logger.info(
        "Total rows reassigned (including propagated derivatives): %d (out of %d total rows, %d datasets processed)",
        n_changed, len(df), len(args.datasets),
    )

    # ── Sanity check: assert no cycle FAMILY (real cycle + its derivatives)
    # ends up split across train/val/test after propagation. This is the
    # exact invariant this script exists to guarantee — verify it holds
    # before writing, rather than trusting the logic silently.
    if "original_cycle_id" in df.columns:
        check_df = df.copy()
        check_df["_new_split"] = updated_split
        check_df["_family_id"] = check_df["original_cycle_id"].fillna(check_df["cycle_id"])
        family_split_counts = check_df.groupby("_family_id")["_new_split"].nunique()
        leaking_families = family_split_counts[family_split_counts > 1]
        if len(leaking_families) > 0:
            logger.error(
                "INTERNAL CONSISTENCY CHECK FAILED: %d cycle famil(y/ies) span "
                "more than one split after propagation — this should be "
                "impossible and indicates a bug. Aborting WITHOUT writing "
                "metadata.parquet. Example family id(s): %s",
                len(leaking_families), leaking_families.index[:5].tolist(),
            )
            sys.exit(1)
        logger.info("Consistency check passed: no cycle family spans multiple splits.")

    if args.dry_run:
        logger.info("--dry-run set: metadata.parquet was NOT modified.")
        return

    df["split"] = updated_split
    df.to_parquet(meta_file, index=False)
    logger.info("metadata.parquet updated at %s", meta_file)


if __name__ == "__main__":
    main()