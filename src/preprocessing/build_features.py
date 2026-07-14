"""
build_features.py
-----------------
CLI entry point for the feature extraction pipeline.

For each requested dataset and feature type, this script:
  1. Parses cycle/event annotations into unified records.
  2. Loads each waveform segment from disk.
  3. Resamples to target_sr.
  4. Normalises length (pad/truncate or time-stretch).
  5. Extracts the requested spectral feature(s).
  6. Saves each feature as a .npy file under data/processed/features/.
  7. Writes/appends a metadata Parquet table.

Usage
-----
    python build_features.py [OPTIONS]

Options
-------
    --datasets      Which datasets to process. One or more of: icbhi sprsound
                    Default: all (icbhi sprsound)
    --features      Which feature types to extract. One or more of: logmel mfcc stft
                    Default: all (logmel mfcc stft)
    --icbhi-cfg     Path to ICBHI config file.       Default: config/icbhi.yaml
    --sprsound-cfg  Path to SPRSound config file.    Default: config/sprsound.yaml
    --features-cfg  Path to features config file.    Default: config/features.yaml
    --workers       Number of parallel worker processes. Default: 1
    --log-level     Logging verbosity. Default: INFO

Examples
--------
    # Extract all features for both datasets (single-threaded)
    python build_features.py

    # Extract only log-mel for ICBHI, 4 workers
    python build_features.py --datasets icbhi --features logmel --workers 4

    # Re-run without overwriting existing .npy files (controlled by skip_existing in features.yaml)
    python build_features.py
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from scipy.signal import butter, sosfilt

import numpy as np
import pandas as pd
import yaml
import cv2

# This script lives at src/preprocessing/build_features.py — its sibling
# packages (datasets/, restructure/, augmentation/) are in the SAME directory,
# not nested under a separate src/ folder, so we add this file's own
# directory to sys.path (not parent/"src", which doesn't exist here).
sys.path.insert(0, str(Path(__file__).parent))

from datasets.icbhi_loader    import load_icbhi
from datasets.sprsound_loader import load_sprsound
from restructure.resample     import load_segment, seconds_to_samples
from restructure.pad          import normalise_length
from restructure.features     import extract, param_tag, EXTRACTORS

logger = logging.getLogger(__name__)


# ── Config loading ─────────────────────────────────────────────────────────────

def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Output path helpers ────────────────────────────────────────────────────────

def feature_npy_path(
    base_dir: Path,
    dataset: str,
    feature_type: str,
    ptag: str,
    cycle_id: str,
) -> Path:
    return base_dir / "features" / dataset / feature_type / ptag / f"{cycle_id}.npy"


# ── Bandpass filter ─────────────────────────────────────────────

def apply_bandpass_filter(waveform, sr, lowcut, highcut, order):
    # Generate the filter coefficients
    # NOTE: Always use output='sos' (Second-Order Sections) for high-order filters like 10. 
    # Standard 'ba' output will cause severe numerical instability and ruin your audio!
    sos = butter(order, [lowcut, highcut], btype='bandpass', fs=sr, output='sos')
    
    # Apply the filter to your audio array
    filtered_waveform = sosfilt(sos, waveform)
    
    return filtered_waveform


# ── Per-cycle worker ───────────────────────────────────────────────────────────

def process_cycle(
    record: dict,
    feature_type: str,
    feat_cfg: dict,
    ptag: str,
    base_dir: Path,
    skip_existing: bool,
) -> dict | None:
    """
    Process one cycle: load waveform → resample → pad → extract feature → save.

    Returns a metadata dict (for the parquet row) or None on failure.
    """
    target_sr       = feat_cfg.get("target_sr", 16000)
    target_dur      = float(feat_cfg.get("target_duration_s", 8.0))
    length_mode     = feat_cfg.get("length_mode", None)
    pad_mode        = feat_cfg.get("pad_mode", "reflect")
    pad_alignment   = feat_cfg.get("pad_alignment", "center")

    cycle_id   = record["cycle_id"]
    dataset    = record["source_dataset"]

    out_path = feature_npy_path(base_dir, dataset, feature_type, ptag, cycle_id)

    if skip_existing and out_path.exists():
        # Return a lightweight record without the feature path recomputation
        return {**record, "feature_type": feature_type, "feature_path": str(out_path)}

    try:
        # 1. Load waveform segment
        waveform, sr = load_segment(
            audio_path=record["audio_path"],
            start_time=record["start_time"],
            end_time=record["end_time"],
            target_sr=target_sr,
            mono=True,
        )

        # Check if the filter config exists and is enabled
        filter_cfg = feat_cfg.get("butterworth_filter", {})
        if filter_cfg.get("enabled", False):
            waveform = apply_bandpass_filter(
                waveform, 
                sr=sr, 
                order=filter_cfg.get("order", 10),
                lowcut=filter_cfg.get("lowcut", 25.0),
                highcut=filter_cfg.get("highcut", 2500.0)
            )

        waveform = waveform / np.max(np.abs(waveform))

        # 2. Length normalise
        if length_mode is not None:
            target_samples = seconds_to_samples(target_dur, sr)
            waveform = normalise_length(
                waveform,
                target_samples=target_samples,
                mode=length_mode,
                pad_mode=pad_mode,
                pad_alignment=pad_alignment,
            )

        # 3. Extract feature
        feature = extract(feature_type, waveform, sr, feat_cfg)

        resized_spec = cv2.resize(feature, (224, 224), interpolation=cv2.INTER_LANCZOS4)

        # 4. Save
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), resized_spec)

    except Exception as e:
        logger.error("Failed to process cycle %s (%s): %s", cycle_id, feature_type, e)
        return None

    return {**record, "feature_type": feature_type, "feature_path": str(out_path)}


# ── Dataset loading ────────────────────────────────────────────────────────────

def load_records(datasets: list[str], icbhi_cfg: str, sprsound_cfg: str) -> list[dict]:
    records = []
    if "icbhi" in datasets:
        logger.info("Loading ICBHI annotations…")
        records.extend(load_icbhi(icbhi_cfg))
    if "sprsound" in datasets:
        logger.info("Loading SPRSound annotations…")
        records.extend(load_sprsound(sprsound_cfg))
    logger.info("Total cycles to process: %d", len(records))
    return records


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract and cache respiratory sound features.")
    parser.add_argument(
        "--datasets", nargs="+", default=["icbhi", "sprsound"],
        choices=["icbhi", "sprsound"],
        help="Datasets to process (default: both)",
    )
    parser.add_argument(
        "--features", nargs="+", default=list(EXTRACTORS.keys()),
        choices=list(EXTRACTORS.keys()),
        help="Feature types to extract (default: all)",
    )
    parser.add_argument("--icbhi-cfg",    default="config/icbhi.yaml")
    parser.add_argument("--sprsound-cfg", default="config/sprsound.yaml")
    parser.add_argument("--features-cfg", default="config/features.yaml")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel worker processes (default: 1)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    feat_cfg   = load_yaml(args.features_cfg)
    base_dir   = Path(feat_cfg["output"]["base_dir"]).expanduser().resolve()
    meta_file  = Path(feat_cfg["output"]["metadata_file"]).expanduser().resolve()
    skip_existing = feat_cfg["output"].get("skip_existing", True)
    target_sr  = feat_cfg.get("target_sr", 16000)

    records = load_records(args.datasets, args.icbhi_cfg, args.sprsound_cfg)
    if not records:
        logger.error("No records loaded. Check dataset config paths.")
        sys.exit(1)

    all_meta_rows: list[dict] = []

    for feature_type in args.features:
        ptag = param_tag(feature_type, feat_cfg, target_sr)
        logger.info("Extracting %s  [tag: %s]  for %d cycles…", feature_type, ptag, len(records))

        worker_fn = partial(
            process_cycle,
            feature_type=feature_type,
            feat_cfg=feat_cfg,
            ptag=ptag,
            base_dir=base_dir,
            skip_existing=skip_existing,
        )

        if args.workers > 1:
            results = _parallel_map(worker_fn, records, args.workers)
        else:
            results = [worker_fn(r) for r in records]

        n_ok  = sum(1 for r in results if r is not None)
        n_err = len(results) - n_ok
        logger.info(
            "%s: %d/%d cycles processed successfully (%d errors).",
            feature_type, n_ok, len(records), n_err,
        )
        all_meta_rows.extend(r for r in results if r is not None)

    # ── Write metadata Parquet ─────────────────────────────────────────────────
    if all_meta_rows:
        df_new = pd.DataFrame(all_meta_rows)
        meta_file.parent.mkdir(parents=True, exist_ok=True)

        if meta_file.exists():
            df_existing = pd.read_parquet(meta_file)
            # Merge: new rows overwrite existing rows for the same (cycle_id, feature_type)
            df_combined = (
                pd.concat([df_existing, df_new], ignore_index=True)
                .drop_duplicates(subset=["cycle_id", "feature_type"], keep="last")
            )
        else:
            df_combined = df_new

        df_combined.to_parquet(meta_file, index=False)
        logger.info(
            "Metadata written to %s  (%d total rows)", meta_file, len(df_combined)
        )
    else:
        logger.warning("No metadata rows to write — all cycles may have failed.")


def _parallel_map(fn, items: list, n_workers: int) -> list:
    """Run fn over items using a ProcessPoolExecutor, preserving order."""
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