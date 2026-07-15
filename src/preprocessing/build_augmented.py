"""
build_balanced_augmented.py
----------------------------
CLI entry point for generating an OFFLINE, class-balanced augmented dataset.

Unlike build_augmented.py (which bakes a fixed grid of pitch shifts for every
real cycle) or dataset.py's on-the-fly augmentation (recomputed every epoch,
never cached), this script lets you say "I want N samples of class X" and it
will:

  1. Read the EXISTING metadata.parquet (built by build_features.py) and
     count how many REAL cycles of each class currently exist.
  2. For each class short of its target, repeatedly sample a base real cycle
     of that class (with replacement) and apply a randomly chosen chain of
     one or more waveform augmentation techniques (see
     augmentation/waveform.py's WAVEFORM_AUGMENTERS registry) with randomised
     parameters drawn from the acoustically-valid ranges documented there.
  3. Runs each generated waveform through the SAME filter/normalise/extract/
     resize steps as build_features.py (see "pipeline consistency" note
     below), and writes .npy files + metadata rows with
     source_type="augmented", generator="<prefix>__<technique(s)>".

The original real rows are never modified. Downstream training code selects
this data the same way it already does for build_augmented.py's output: via
data.source_types / data.generators in the experiment config (see
config/experiments/*.yaml and train.py).

Pipeline consistency
---------------------
build_features.py currently does four things to each waveform before
extraction: (1) an optional bandpass filter, (2) peak normalisation
(waveform / max(|waveform|)), (3) optional length normalisation controlled
by features.yaml's length_mode, (4) cv2.resize(..., (224, 224)) of the
extracted 2-D feature array. This script reproduces steps 1, 2 and 4 exactly
so that generated features are shape- and scale-compatible with the real
cache (mixing differently-shaped .npy arrays in one training batch would
break torch.stack in CachedFeatureDataset).

NOTE on the bandpass filter: build_features.py reads
feat_cfg.get("filter", {}), but features.yaml names the block
"butterworth_filter". That mismatch means the filter is currently a no-op in
build_features.py regardless of "enabled: true". This script intentionally
looks up the SAME ("filter") key so its output stays consistent with
whatever build_features.py actually does today. If you fix that key
mismatch in build_features.py, mirror the fix here (--filter-key lets you
override without editing code) and re-run both.

Step 3 (length normalisation) is handled a little differently here: some
augmentations (time_stretch, time_shift with wrap=False) change the
effective duration or content window, and one of them (time_stretch) is NOT
internally guarded against librosa's default n_fft=2048 on short cycles the
way pitch_shift is. So this script always pads to features.yaml's
target_duration_s BEFORE augmenting (same rationale build_augmented.py uses
for pitch-shifting) and re-normalises back to that same length AFTER
augmenting, regardless of the global length_mode setting — purely as a safe,
fixed-length staging step before feature extraction. This does not change
build_features.py's own behaviour for real data.

Usage
-----
    # See what would be generated without writing anything
    python build_balanced_augmented.py --target 2000 --dry-run

    # Top every class up to 2000 samples using all "safe" techniques
    python build_balanced_augmented.py --target 2000 --workers 4

    # Per-class targets, restricting to specific techniques
    python build_balanced_augmented.py \\
        --target-map crackle=1800 wheeze=1800 both=1800 \\
        --techniques additive_gaussian_noise additive_pink_noise random_gain time_shift pitch_shift \\
        --compose-min 1 --compose-max 2 \\
        --workers 4

Options
-------
    --datasets       Which datasets' real cycles to draw from. Default: icbhi sprsound
    --features       Which feature types to (re)extract. Default: all
    --features-cfg   Path to features config file. Default: config/features.yaml
    --label-col      Which metadata column defines "class" for balancing.
                     Default: label_4class (use label_coarse or label_fine to
                     balance a different label scheme — see dataset.py docstring)
    --classes        Restrict balancing to these class values. Default: all
                     classes found in the real data for --label-col.
    --target         Uniform target sample count applied to every class
                     (existing real + already-generated augmented count
                     against target for that class, both). Overridden per-class by --target-map.
    --target-map     One or more "class=count" overrides, e.g. crackle=2000 wheeze=2000
    --techniques     Which augmentation techniques are eligible. Default: all
                     of WAVEFORM_AUGMENTERS (pitch_shift, time_stretch,
                     additive_gaussian_noise, additive_pink_noise, random_gain,
                     time_shift).
    --compose-min    Minimum number of techniques chained per generated
                     sample (sampled without replacement from --techniques). Default: 1
    --compose-max    Maximum number of techniques chained per generated
                     sample. Default: 1 (set >1 for compound augmentations)
    --split          Which metadata "split" value to draw base real cycles
                     from. Default: train (never draw from val/test)
    --generator-prefix  Prefix used to build the generator column and to
                     detect already-generated rows on re-runs. Default: offline_aug
    --skip-existing / --no-skip-existing   Whether pre-existing augmented rows
                     from a previous run of THIS script (matched by
                     generator-prefix, class, dataset) count toward the
                     target, and whether individual .npy files are
                     regenerated if they already exist. Default: skip-existing on.
    --seed           Base random seed for technique/parameter/base-cycle selection. Default: 42
    --workers        Number of parallel worker processes. Default: 1
    --dry-run        Print the generation plan (per class: existing real,
                     existing augmented, target, to-generate) and exit
                     without writing anything.
    --log-level      Logging verbosity. Default: INFO
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
import yaml
from scipy.signal import butter, sosfilt

try:
    import cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False

sys.path.insert(0, str(Path(__file__).parent))

from restructure.resample import load_segment, seconds_to_samples
from restructure.pad import normalise_length
from restructure.features import extract, param_tag, EXTRACTORS
from augmentation.waveform import WAVEFORM_AUGMENTERS

logger = logging.getLogger(__name__)


# ── Config / path helpers (mirrors build_features.py / build_augmented.py) ──

def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def feature_npy_path(base_dir: Path, dataset: str, feature_type: str, ptag: str, cycle_id: str) -> Path:
    return base_dir / "features" / dataset / feature_type / ptag / f"{cycle_id}.npy"


def apply_bandpass_filter(waveform, sr, lowcut, highcut, order):
    """Duplicated from build_features.py to keep the offline-augmented cache
    on the exact same waveform-preprocessing path as the real cache."""
    sos = butter(order, [lowcut, highcut], btype="bandpass", fs=sr, output="sos")
    return sosfilt(sos, waveform)


# ── Randomised technique parameter sampling ─────────────────────────────────
# Ranges follow the "recommended" ranges documented in augmentation/waveform.py.

def _sample_pitch_shift(rng: np.random.Generator) -> dict:
    return {"n_steps": float(rng.choice([-2.0, -1.0, 1.0, 2.0]))}


def _sample_time_stretch(rng: np.random.Generator) -> dict:
    return {"rate": float(rng.uniform(0.9, 1.1))}


def _sample_gaussian_noise(rng: np.random.Generator) -> dict:
    return {"snr_db": float(rng.uniform(10.0, 25.0))}


def _sample_pink_noise(rng: np.random.Generator) -> dict:
    return {"snr_db": float(rng.uniform(10.0, 25.0))}


def _sample_random_gain(rng: np.random.Generator) -> dict:
    # random_gain(waveform, min_db, max_db) samples internally via np.random.
    # Passing min_db == max_db forces a specific, loggable gain value while
    # still going through the shared registry function unmodified.
    gain_db = float(rng.uniform(-6.0, 6.0))
    return {"min_db": gain_db, "max_db": gain_db}


def _sample_time_shift(rng: np.random.Generator) -> dict:
    return {
        "max_shift_fraction": float(rng.uniform(0.05, 0.15)),
        "wrap": bool(rng.integers(0, 2)),
    }


PARAM_SAMPLERS = {
    "pitch_shift": _sample_pitch_shift,
    "time_stretch": _sample_time_stretch,
    "additive_gaussian_noise": _sample_gaussian_noise,
    "additive_pink_noise": _sample_pink_noise,
    "random_gain": _sample_random_gain,
    "time_shift": _sample_time_shift,
}

# pitch_shift is the only registry function that takes `sr` as a kwarg.
_NEEDS_SR = {"pitch_shift"}


def apply_technique_chain(
    waveform: np.ndarray,
    sr: int,
    techniques: list[str],
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict]]:
    """Apply each technique in `techniques` in order, returning the final
    waveform and a list of {"technique": name, "params": {...}} dicts for
    the metadata row (full reproducibility of what was applied)."""
    applied = []
    for name in techniques:
        params = PARAM_SAMPLERS[name](rng)
        fn = WAVEFORM_AUGMENTERS[name]
        if name in _NEEDS_SR:
            waveform = fn(waveform, sr=sr, **params)
        else:
            waveform = fn(waveform, **params)
        applied.append({"technique": name, "params": params})
    return waveform, applied


def generator_tag(prefix: str, techniques: list[str]) -> str:
    return f"{prefix}__{'+'.join(techniques)}"


# ── Per-sample worker ────────────────────────────────────────────────────────

def process_generated_cycle(
    task: dict,
    feature_type: str,
    feat_cfg: dict,
    filter_key: str,
    ptag: str,
    base_dir: Path,
    generator_prefix: str,
    skip_existing: bool,
    resize_to: tuple[int, int] | None,
) -> dict | None:
    """
    task keys: base_record (dict, one real metadata row deduped by cycle_id),
    techniques (list[str]), new_cycle_id (str), task_seed (int)
    """
    record = task["base_record"]
    techniques = task["techniques"]
    new_cycle_id = task["new_cycle_id"]
    rng = np.random.default_rng(task["task_seed"])

    target_sr = feat_cfg.get("target_sr", 16000)
    target_dur = float(feat_cfg.get("target_duration_s", 8.0))
    pad_mode = feat_cfg.get("pad_mode", "reflect")
    pad_alignment = feat_cfg.get("pad_alignment", "center")

    dataset = record["source_dataset"]
    original_cycle_id = record["cycle_id"]
    generator_name = generator_tag(generator_prefix, techniques)

    out_path = feature_npy_path(base_dir, dataset, feature_type, ptag, new_cycle_id)

    if skip_existing and out_path.exists():
        return _build_meta_row(record, new_cycle_id, original_cycle_id, generator_name, [], feature_type, out_path)

    try:
        # 1. Load raw waveform segment
        waveform, sr = load_segment(
            audio_path=record["audio_path"],
            start_time=record["start_time"],
            end_time=record["end_time"],
            target_sr=target_sr,
            mono=True,
        )

        # 2. Bandpass filter — mirrors build_features.py's current (buggy
        #    key) behaviour so the augmented cache stays consistent with the
        #    real cache as it actually behaves today. See module docstring.
        filter_cfg = feat_cfg.get(filter_key, {})
        if filter_cfg.get("enabled", False):
            waveform = apply_bandpass_filter(
                waveform, sr=sr,
                order=filter_cfg.get("order", 10),
                lowcut=filter_cfg.get("lowcut", 25.0),
                highcut=filter_cfg.get("highcut", 2500.0),
            )

        # 3. Peak normalise — mirrors build_features.py
        peak = np.max(np.abs(waveform))
        if peak > 0:
            waveform = waveform / peak

        # 4. Pad to target_duration_s BEFORE augmenting. Needed because
        #    time_stretch (unlike pitch_shift) has no internal guard against
        #    librosa's default n_fft=2048 on short cycles — see module
        #    docstring. This is a staging step for safe augmentation/
        #    extraction; it does not affect build_features.py's own output.
        target_samples = seconds_to_samples(target_dur, sr)
        waveform = normalise_length(
            waveform, target_samples=target_samples,
            mode="pad_truncate", pad_mode=pad_mode, pad_alignment=pad_alignment,
        )

        # 5. Apply the randomly-parameterised technique chain
        if len(waveform) > 0:
            waveform, applied = apply_technique_chain(waveform, sr, techniques, rng)
        else:
            applied = []

        # 6. Re-normalise back to the same fixed length in case a technique
        #    (time_stretch) changed the duration.
        if len(waveform) != target_samples:
            waveform = normalise_length(
                waveform, target_samples=target_samples,
                mode="pad_truncate", pad_mode=pad_mode, pad_alignment=pad_alignment,
            )

        # 7. Extract feature
        feature = extract(feature_type, waveform, sr, feat_cfg)

        # 8. Resize — mirrors build_features.py's cv2.resize(..., (224, 224))
        #    so real and augmented .npy arrays are shape-compatible in the
        #    same training batch.
        if resize_to is not None:
            if _CV2_OK:
                feature = cv2.resize(feature, resize_to, interpolation=cv2.INTER_LANCZOS4)
            else:
                logger.warning("cv2 not available — skipping resize; shapes may not "
                                "match build_features.py's cached real features.")

        # 9. Save
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), feature)

    except Exception as e:
        logger.error(
            "Failed to generate cycle from base=%s (techniques=%s, %s): %s",
            original_cycle_id, techniques, feature_type, e,
        )
        return None

    return _build_meta_row(record, new_cycle_id, original_cycle_id, generator_name, applied, feature_type, out_path)


def _build_meta_row(record, new_cycle_id, original_cycle_id, generator_name, applied, feature_type, out_path) -> dict:
    new_row = dict(record)  # copy all original fields (label, split, device, etc.)
    new_row["cycle_id"] = new_cycle_id
    new_row["original_cycle_id"] = original_cycle_id
    new_row["source_type"] = "augmented"
    new_row["generator"] = generator_name
    new_row["aug_params"] = json.dumps(applied)
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


# ── Planning: figure out how many samples each class needs ─────────────────

def build_plan(
    df: pd.DataFrame,
    datasets: list[str],
    label_col: str,
    classes: list[str] | None,
    target: int | None,
    target_map: dict[str, int],
    split: str,
    generator_prefix: str,
    seed: int,
    techniques: list[str],
    compose_min: int,
    compose_max: int,
) -> tuple[list[dict], pd.DataFrame]:
    """Returns (tasks, plan_summary_df)."""
    df_real = df[(df["source_type"] == "real")
                 & (df["source_dataset"].isin(datasets))
                 & (df["split"] == split)]
    df_real = df_real.drop_duplicates(subset=["cycle_id"])

    if df_real.empty:
        raise ValueError(f"No real rows found for datasets={datasets}, split={split!r}.")

    df_prev_aug = df[(df["source_type"] == "augmented")
                      & (df["source_dataset"].isin(datasets))
                      & (df["generator"].astype(str).str.startswith(generator_prefix))]
    df_prev_aug = df_prev_aug.drop_duplicates(subset=["cycle_id"])

    all_classes = classes or sorted(df_real[label_col].dropna().unique().tolist())

    rng = np.random.default_rng(seed)
    tasks: list[dict] = []
    summary_rows = []
    task_counter = 0

    for cls in all_classes:
        cls_real = df_real[df_real[label_col] == cls]
        n_real = len(cls_real)
        if n_real == 0:
            logger.warning("No real cycles found for class %r — skipping (cannot sample a base cycle).", cls)
            continue

        cls_prev_aug = df_prev_aug[df_prev_aug[label_col] == cls]
        n_prev_aug = len(cls_prev_aug)

        cls_target = target_map.get(cls, target)
        if cls_target is None:
            logger.warning("No target specified for class %r (use --target or --target-map) — skipping.", cls)
            continue

        n_needed = max(0, cls_target - n_real - n_prev_aug)
        summary_rows.append({
            "class": cls, "n_real": n_real, "n_prev_augmented": n_prev_aug,
            "target": cls_target, "n_to_generate": n_needed,
        })

        base_records = cls_real.drop(columns=["feature_type", "feature_path"], errors="ignore").to_dict("records")

        for i in range(n_needed):
            base_record = base_records[rng.integers(0, len(base_records))]
            k = int(rng.integers(compose_min, compose_max + 1))
            k = min(k, len(techniques))
            chosen = list(rng.choice(techniques, size=k, replace=False))
            # The trailing hex suffix guarantees uniqueness across separate
            # runs of this script (e.g. a later top-up run) even though the
            # per-class counter `i` restarts at 0 each run — without it, a
            # resumed run could regenerate the same (base_cycle, i) pair as
            # an earlier run, silently overwriting that row via
            # drop_duplicates(keep="last") and undercounting the target.
            suffix = uuid.uuid4().hex[:8]
            new_cycle_id = f"{base_record['cycle_id']}__offaug_{cls}_{i:05d}_{suffix}"
            task_counter += 1
            tasks.append({
                "base_record": base_record,
                "techniques": chosen,
                "new_cycle_id": new_cycle_id,
                "task_seed": seed + task_counter,
            })

    plan_df = pd.DataFrame(summary_rows)
    return tasks, plan_df


# ── CLI ───────────────────────────────────────────────────────────────────

def _parse_target_map(pairs: list[str] | None) -> dict[str, int]:
    out = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise argparse.ArgumentTypeError(f"--target-map entries must be class=count, got {pair!r}")
        cls, count = pair.split("=", 1)
        out[cls.strip()] = int(count)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Generate an offline, class-balanced augmented dataset from cached real features."
    )
    parser.add_argument("--datasets", nargs="+", default=["icbhi", "sprsound"], choices=["icbhi", "sprsound"])
    parser.add_argument("--features", nargs="+", default=list(EXTRACTORS.keys()), choices=list(EXTRACTORS.keys()))
    parser.add_argument("--features-cfg", default="config/features.yaml")
    parser.add_argument("--label-col", default="label_4class")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Restrict balancing to these classes (default: all found)")
    parser.add_argument("--target", type=int, default=None,
                        help="Uniform target sample count per class")
    parser.add_argument("--target-map", nargs="+", default=None,
                        help='Per-class overrides, e.g. crackle=2000 wheeze=2000')
    parser.add_argument("--techniques", nargs="+", default=list(WAVEFORM_AUGMENTERS.keys()),
                        choices=list(WAVEFORM_AUGMENTERS.keys()))
    parser.add_argument("--compose-min", type=int, default=1)
    parser.add_argument("--compose-max", type=int, default=1)
    parser.add_argument("--split", default="train")
    parser.add_argument("--generator-prefix", default="offline_aug")
    parser.add_argument("--filter-key", default="butterworth_filter",
                        help="Config key read for the bandpass filter block. Default now matches "
                             "build_features.py's fixed lookup (see module docstring history) — "
                             "override if you're deliberately reproducing the old buggy behavior.")
    parser.add_argument("--resize-h", type=int, default=224,
                        help="Target height for the final feature array resize, matching "
                             "build_features.py's hardcoded cv2.resize(..., (224, 224)).")
    parser.add_argument("--resize-w", type=int, default=224,
                        help="Target width for the final feature array resize.")
    parser.add_argument("--no-resize", action="store_true",
                        help="Skip the resize step entirely. Only safe if build_features.py's "
                             "real cache also has no resize step and every real cycle already "
                             "produces identically-shaped features (e.g. length_mode is set and "
                             "fixed STFT params are used).")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.compose_min < 1 or args.compose_max < args.compose_min:
        logger.error("Require 1 <= --compose-min <= --compose-max, got %d, %d", args.compose_min, args.compose_max)
        sys.exit(1)

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
        generator_prefix=args.generator_prefix, seed=args.seed,
        techniques=args.techniques, compose_min=args.compose_min, compose_max=args.compose_max,
    )

    logger.info("Generation plan (split=%s, label_col=%s):\n%s", args.split, args.label_col,
                plan_df.to_string(index=False) if not plan_df.empty else "(empty)")

    if args.dry_run:
        logger.info("--dry-run set: no files written.")
        return

    if not tasks:
        logger.info("Nothing to generate — every class already meets its target.")
        return

    logger.info("Generating %d new cycles × %d feature types…", len(tasks), len(args.features))

    all_new_rows: list[dict] = []
    resize_to = None if args.no_resize else (args.resize_w, args.resize_h)
    for feature_type in args.features:
        ptag = param_tag(feature_type, feat_cfg, target_sr)
        worker_fn = partial(
            process_generated_cycle,
            feature_type=feature_type, feat_cfg=feat_cfg, filter_key=args.filter_key,
            ptag=ptag, base_dir=base_dir, generator_prefix=args.generator_prefix,
            skip_existing=args.skip_existing, resize_to=resize_to,
        )
        if args.workers > 1:
            results = _parallel_map(worker_fn, tasks, args.workers)
        else:
            results = [worker_fn(t) for t in tasks]

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
        logger.info(
            "Metadata updated at %s — added %d rows (%d total rows now).",
            meta_file, len(df_new), len(df_combined),
        )
    else:
        logger.warning("No rows produced — check errors above.")


if __name__ == "__main__":
    main()