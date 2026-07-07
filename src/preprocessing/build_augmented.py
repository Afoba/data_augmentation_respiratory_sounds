"""
build_augmented.py
-------------------
CLI entry point for pre-caching PITCH-SHIFTED augmented features.

Pitch shifting is the one augmentation expensive enough (phase-vocoder STFT
resynthesis, ~20-100ms/sample on CPU) that it's worth baking into the feature
cache once rather than recomputing on-the-fly every epoch. All other
augmentations (SpecAugment, noise, gain, time shift) stay on-the-fly — see
src/augmentation/waveform_augment.py and spectrogram_augment.py.

This script:
  1. Reads the EXISTING metadata.parquet (must have been built by
     build_features.py first) and selects rows with source_type == "real".
  2. For each requested semitone shift, length-normalises (pads/truncates)
     the waveform FIRST, then pitch-shifts the padded result, then extracts
     features — see note below on why padding comes before shifting.
  3. Writes new .npy files under the SAME feature directories (distinguished
     by a different cycle_id suffix), and appends new rows to metadata.parquet
     with source_type="augmented", generator="pitch_shift_<±N>",
     original_cycle_id=<the real cycle this was derived from>.

The original real rows in metadata.parquet are untouched. Downstream training
code controls inclusion purely via a `source_type` / `generator` filter — see
the project README for example queries.

Note on pad-before-shift ordering
----------------------------------
librosa.effects.pitch_shift() runs its own internal STFT with a fixed
default n_fft=2048, independent of anything in features.yaml. ICBHI has
some cycles shorter than 2048 samples at 16kHz (~0.13s), which would
otherwise trigger a "n_fft is too large for input signal" warning and an
uncontrolled internal zero-pad fallback inside librosa. Padding to
target_duration_s BEFORE pitch-shifting avoids that entirely. The tradeoff:
the reflect-padded edges of short cycles get pitch-shifted along with the
real signal, rather than only the real signal being shifted. Since the
padding is a reflection of the real signal (not silence), this is a minor
and arguably benign distortion compared to librosa's alternative internal
zero-pad fallback — but it's a real tradeoff, not a free fix.

Usage
-----
    python build_augmented.py --shifts -2 -1 1 2 [OPTIONS]

Options
-------
    --datasets      Which datasets to augment. Default: icbhi sprsound
    --features      Which feature types to (re)extract. Default: all
    --shifts        Semitone shifts to generate. Default: -2 -1 1 2
                    (Recommended to stay within ±2-3; see acoustic-validity
                    note in waveform_augment.py)
    --features-cfg  Path to features config file. Default: config/features.yaml
    --workers       Number of parallel worker processes. Default: 1
    --log-level     Logging verbosity. Default: INFO

Example
-------
    # Generate ±1 and ±2 semitone pitch-shifted variants for all cycles,
    # all three feature types
    python build_augmented.py --shifts -2 -1 1 2 --workers 4
"""

import argparse
import logging
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent / "src"))

from restructure.resample  import load_segment, seconds_to_samples
from restructure.pad       import normalise_length
from restructure.features  import extract, param_tag, EXTRACTORS
from augmentation.waveform import pitch_shift

logger = logging.getLogger(__name__)


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def feature_npy_path(base_dir: Path, dataset: str, feature_type: str, ptag: str, cycle_id: str) -> Path:
    return base_dir / "features" / dataset / feature_type / ptag / f"{cycle_id}.npy"


def shift_tag(n_steps: float) -> str:
    """e.g. +1 -> 'plus1', -2 -> 'minus2' — safe for filenames."""
    sign = "plus" if n_steps >= 0 else "minus"
    return f"{sign}{abs(n_steps):g}"


def process_pitch_shifted_cycle(
    record: dict,
    n_steps: float,
    feature_type: str,
    feat_cfg: dict,
    ptag: str,
    base_dir: Path,
    skip_existing: bool,
) -> dict | None:
    """
    Load the original waveform, pitch-shift it, then run the standard
    length-normalise + feature-extraction path. Returns a new metadata
    dict with source_type="augmented", or None on failure.
    """
    target_sr     = feat_cfg.get("target_sr", 16000)
    target_dur    = float(feat_cfg.get("target_duration_s", 8.0))
    length_mode   = feat_cfg.get("length_mode", "pad_truncate")
    pad_mode      = feat_cfg.get("pad_mode", "reflect")
    pad_alignment = feat_cfg.get("pad_alignment", "center")

    original_cycle_id = record["cycle_id"]
    dataset = record["source_dataset"]
    generator_name = f"pitch_shift_{'+' if n_steps >= 0 else ''}{n_steps:g}"
    new_cycle_id = f"{original_cycle_id}__aug_{shift_tag(n_steps)}"

    out_path = feature_npy_path(base_dir, dataset, feature_type, ptag, new_cycle_id)

    if skip_existing and out_path.exists():
        return _build_meta_row(record, new_cycle_id, original_cycle_id, generator_name, feature_type, out_path)

    try:
        # 1. Load original waveform segment (pre-shift, pre-pad)
        waveform, sr = load_segment(
            audio_path=record["audio_path"],
            start_time=record["start_time"],
            end_time=record["end_time"],
            target_sr=target_sr,
            mono=True,
        )

        # 2. Length normalise FIRST, before pitch shifting. This matters because
        #    librosa.effects.pitch_shift() runs its own internal STFT with a
        #    fixed default n_fft=2048 (independent of features.yaml) — if a raw
        #    cycle is shorter than that (ICBHI has cycles under 0.13s at 16kHz),
        #    librosa emits a "n_fft is too large for input signal" warning and
        #    effectively zero-pads internally in an uncontrolled way. Padding to
        #    the full target_duration_s up front guarantees the signal handed to
        #    pitch_shift() is always far longer than any reasonable n_fft, and
        #    matches the same reflect/centered padding used everywhere else in
        #    the pipeline rather than librosa's internal zero-padding fallback.
        target_samples = seconds_to_samples(target_dur, sr)
        waveform = normalise_length(
            waveform,
            target_samples=target_samples,
            mode=length_mode,
            pad_mode=pad_mode,
            pad_alignment=pad_alignment,
        )

        # 3. Pitch shift the already-padded, fixed-length waveform.
        if len(waveform) > 0:
            waveform = pitch_shift(waveform, sr=sr, n_steps=n_steps)

        # 4. Extract feature
        feature = extract(feature_type, waveform, sr, feat_cfg)

        # 5. Save
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), feature)

    except Exception as e:
        logger.error(
            "Failed to process pitch-shifted cycle %s (shift=%.1f, %s): %s",
            original_cycle_id, n_steps, feature_type, e,
        )
        return None

    return _build_meta_row(record, new_cycle_id, original_cycle_id, generator_name, feature_type, out_path)


def _build_meta_row(record, new_cycle_id, original_cycle_id, generator_name, feature_type, out_path) -> dict:
    new_row = dict(record)  # copy all original fields (label, split, device, etc.)
    new_row["cycle_id"] = new_cycle_id
    new_row["original_cycle_id"] = original_cycle_id
    new_row["source_type"] = "augmented"
    new_row["generator"] = generator_name
    new_row["feature_type"] = feature_type
    new_row["feature_path"] = str(out_path)
    return new_row


def main():
    parser = argparse.ArgumentParser(description="Pre-cache pitch-shifted augmented features.")
    parser.add_argument("--datasets", nargs="+", default=["icbhi", "sprsound"],
                        choices=["icbhi", "sprsound"])
    parser.add_argument("--features", nargs="+", default=list(EXTRACTORS.keys()),
                        choices=list(EXTRACTORS.keys()))
    parser.add_argument("--shifts", nargs="+", type=float, default=[-2, -1, 1, 2],
                        help="Semitone shifts to generate (default: -2 -1 1 2)")
    parser.add_argument("--features-cfg", default="config/features.yaml")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    for s in args.shifts:
        if abs(s) > 3:
            logger.warning(
                "Requested shift %.1f semitones exceeds the recommended ±3 range "
                "for respiratory sounds — wheeze/crackle frequency content may "
                "become acoustically unrealistic.", s,
            )

    feat_cfg  = load_yaml(args.features_cfg)
    base_dir  = Path(feat_cfg["output"]["base_dir"]).expanduser().resolve()
    meta_file = Path(feat_cfg["output"]["metadata_file"]).expanduser().resolve()
    skip_existing = feat_cfg["output"].get("skip_existing", True)
    target_sr = feat_cfg.get("target_sr", 16000)

    if not meta_file.exists():
        logger.error(
            "metadata.parquet not found at %s. Run build_features.py first — "
            "build_augmented.py derives augmented data from existing real records.",
            meta_file,
        )
        sys.exit(1)

    df = pd.read_parquet(meta_file)
    df_real = df[(df["source_type"] == "real") & (df["source_dataset"].isin(args.datasets))]

    if df_real.empty:
        logger.error("No real records found for datasets %s in %s", args.datasets, meta_file)
        sys.exit(1)

    # We only need ONE row per cycle_id (not per feature_type) to get the
    # source audio/timing info — feature_type-specific extraction happens below.
    base_records = (
        df_real.drop_duplicates(subset=["cycle_id"])
        .drop(columns=["feature_type", "feature_path"], errors="ignore")
        .to_dict("records")
    )

    logger.info(
        "Generating pitch-shift augmentations for %d base cycles × %d shifts × %d feature types",
        len(base_records), len(args.shifts), len(args.features),
    )

    all_new_rows: list[dict] = []

    for feature_type in args.features:
        ptag = param_tag(feature_type, feat_cfg, target_sr)

        for n_steps in args.shifts:
            logger.info(
                "Pitch shift %+.1f semitones — feature=%s [tag: %s] — %d cycles…",
                n_steps, feature_type, ptag, len(base_records),
            )

            worker_fn = partial(
                process_pitch_shifted_cycle,
                n_steps=n_steps,
                feature_type=feature_type,
                feat_cfg=feat_cfg,
                ptag=ptag,
                base_dir=base_dir,
                skip_existing=skip_existing,
            )

            if args.workers > 1:
                results = _parallel_map(worker_fn, base_records, args.workers)
            else:
                results = [worker_fn(r) for r in base_records]

            n_ok = sum(1 for r in results if r is not None)
            logger.info(
                "Shift %+.1f / %s: %d/%d succeeded.",
                n_steps, feature_type, n_ok, len(base_records),
            )
            all_new_rows.extend(r for r in results if r is not None)

    if all_new_rows:
        df_new = pd.DataFrame(all_new_rows)
        df_combined = (
            pd.concat([df, df_new], ignore_index=True)
            .drop_duplicates(subset=["cycle_id", "feature_type"], keep="last")
        )
        df_combined.to_parquet(meta_file, index=False)
        logger.info(
            "Metadata updated at %s — added %d augmented rows (%d total rows now).",
            meta_file, len(df_new), len(df_combined),
        )
    else:
        logger.warning("No augmented rows produced — check errors above.")


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


if __name__ == "__main__":
    main()