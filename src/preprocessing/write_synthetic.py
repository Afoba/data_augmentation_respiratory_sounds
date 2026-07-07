"""
write_synthetic.py
-------------------
Registers synthetic respiratory sound waveforms (produced by an external
generative model — GAN, VAE, diffusion, etc.) into the SAME feature cache
and metadata schema used by real and augmented data.

This script does NOT contain any generative modelling code. It's a thin
adapter: you generate waveforms yourself (in whatever framework you like),
then call `register_synthetic_cycle()` (or run this script standalone with
a directory of .wav files + a labels file) to run them through the identical
length-normalisation + feature-extraction path as real data, and append rows
to metadata.parquet with source_type="synthetic".

Two ways to use this
--------------------
1. Programmatically, from your generative model's training/sampling script:

    from write_synthetic import register_synthetic_cycle

    register_synthetic_cycle(
        waveform=my_generated_waveform,     # 1-D numpy array
        sr=16000,                            # sample rate of the generated waveform
        label_4class="wheeze",
        generator="diffusion_v1",
        source_dataset="icbhi",              # which label space / device family it mimics
        feature_types=["logmel", "mfcc", "stft"],
        features_cfg_path="config/features.yaml",
    )

2. As a CLI, pointing at a directory of generated .wav files + a labels CSV:

    python write_synthetic.py \\
        --audio-dir generated_audio/diffusion_v1/ \\
        --labels-csv generated_audio/diffusion_v1/labels.csv \\
        --generator diffusion_v1 \\
        --source-dataset icbhi \\
        --features logmel mfcc stft

    labels.csv must have columns: filename,label_4class[,label_fine,split]
"""

import argparse
import csv
import logging
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent / "src"))

from restructure.pad      import normalise_length
from restructure.resample import seconds_to_samples
from restructure.features import extract, param_tag, EXTRACTORS

logger = logging.getLogger(__name__)

_VALID_4CLASS = {"normal", "crackle", "wheeze", "both"}
_BINARY_FROM_4CLASS = {
    "normal":  (0, 0),
    "crackle": (1, 0),
    "wheeze":  (0, 1),
    "both":    (1, 1),
}


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def feature_npy_path(base_dir: Path, dataset: str, feature_type: str, ptag: str, cycle_id: str) -> Path:
    return base_dir / "features" / dataset / feature_type / ptag / f"{cycle_id}.npy"


def register_synthetic_cycle(
    waveform: np.ndarray,
    sr: int,
    label_4class: str,
    generator: str,
    source_dataset: str = "icbhi",
    label_fine: str | None = None,
    split: str = "train",
    feature_types: list[str] | None = None,
    features_cfg_path: str | Path = "config/features.yaml",
    cycle_id: str | None = None,
    extra_metadata: dict | None = None,
) -> dict:
    """
    Run a single generated waveform through feature extraction and append
    a row to metadata.parquet for each requested feature type.

    Parameters
    ----------
    waveform        : 1-D numpy array, the raw generated audio
    sr              : sample rate of `waveform` as produced by the generator
    label_4class    : one of "normal" | "crackle" | "wheeze" | "both"
    generator       : free-text identifier for the generative model/run,
                      e.g. "diffusion_v1", "gan_run3"
    source_dataset  : which dataset's label/feature space this mimics —
                      determines output directory and which device/SR
                      conventions to assume. Must be "icbhi" or "sprsound".
    label_fine      : optional fine-grained label (SPRSound 7-class style).
                      Defaults to label_4class if not given.
    split           : "train" | "test" | "synthetic" — synthetic data is
                      commonly kept out of the test split entirely; defaults
                      to "train" but you should set this deliberately.
    feature_types   : which features to extract. Default: all (logmel, mfcc, stft)
    features_cfg_path : path to features.yaml
    cycle_id        : optional explicit cycle_id; auto-generated (uuid-based)
                      if not given
    extra_metadata  : optional dict of additional columns to attach to the
                      metadata row (e.g. {"seed": 42, "model_checkpoint": "ep100"})

    Returns
    -------
    dict summary: {"cycle_id": ..., "n_features_written": int, "errors": [...]}
    """
    if label_4class not in _VALID_4CLASS:
        raise ValueError(f"label_4class must be one of {_VALID_4CLASS}, got {label_4class!r}")
    if source_dataset not in ("icbhi", "sprsound"):
        raise ValueError(f"source_dataset must be 'icbhi' or 'sprsound', got {source_dataset!r}")

    feat_cfg = load_yaml(features_cfg_path)
    base_dir = Path(feat_cfg["output"]["base_dir"]).expanduser().resolve()
    meta_file = Path(feat_cfg["output"]["metadata_file"]).expanduser().resolve()
    target_sr = feat_cfg.get("target_sr", 16000)
    target_dur = float(feat_cfg.get("target_duration_s", 8.0))
    length_mode = feat_cfg.get("length_mode", "pad_truncate")
    pad_mode = feat_cfg.get("pad_mode", "reflect")
    pad_alignment = feat_cfg.get("pad_alignment", "center")

    feature_types = feature_types or list(EXTRACTORS.keys())
    cycle_id = cycle_id or f"synthetic__{generator}__{uuid.uuid4().hex[:12]}"
    label_fine = label_fine or label_4class
    crackle, wheeze = _BINARY_FROM_4CLASS[label_4class]

    # Resample to target_sr if the generator output is at a different rate
    if target_sr is not None and sr != target_sr:
        try:
            import librosa
            waveform = librosa.resample(waveform.astype(np.float32), orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        except ImportError:
            raise ImportError("librosa is required to resample synthetic audio: pip install librosa")

    # Length normalise — identical path to real/augmented data
    target_samples = seconds_to_samples(target_dur, sr)
    waveform = normalise_length(
        waveform.astype(np.float32),
        target_samples=target_samples,
        mode=length_mode,
        pad_mode=pad_mode,
        pad_alignment=pad_alignment,
    )

    new_rows = []
    errors = []

    for feature_type in feature_types:
        try:
            ptag = param_tag(feature_type, feat_cfg, target_sr)
            feature = extract(feature_type, waveform, sr, feat_cfg)

            out_path = feature_npy_path(base_dir, source_dataset, feature_type, ptag, cycle_id)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(out_path), feature)

            row = {
                "cycle_id": cycle_id,
                "source_dataset": source_dataset,
                "audio_path": None,            # no real audio file backs this row
                "start_time": None,
                "end_time": None,
                "duration": target_dur,
                "label_4class": label_4class,
                "label_fine": label_fine,
                "label_coarse": label_4class,
                "crackle": crackle,
                "wheeze": wheeze,
                "split": split,
                # dataset-specific fields left empty/None for schema consistency
                "patient_id": None, "session_id": None, "location": None,
                "mode": None, "device": None, "diagnosis": None,
                "record_label": None, "quality": None,
                # provenance
                "source_type": "synthetic",
                "generator": generator,
                "original_cycle_id": None,
                # bookkeeping
                "feature_type": feature_type,
                "feature_path": str(out_path),
            }
            if extra_metadata:
                row.update(extra_metadata)

            new_rows.append(row)

        except Exception as e:
            logger.error("Failed to extract %s for synthetic cycle %s: %s", feature_type, cycle_id, e)
            errors.append((feature_type, str(e)))

    # Append to metadata.parquet
    if new_rows:
        df_new = pd.DataFrame(new_rows)
        meta_file.parent.mkdir(parents=True, exist_ok=True)
        if meta_file.exists():
            df_existing = pd.read_parquet(meta_file)
            df_combined = (
                pd.concat([df_existing, df_new], ignore_index=True)
                .drop_duplicates(subset=["cycle_id", "feature_type"], keep="last")
            )
        else:
            df_combined = df_new
        df_combined.to_parquet(meta_file, index=False)

    return {
        "cycle_id": cycle_id,
        "n_features_written": len(new_rows),
        "errors": errors,
    }


# ── CLI: batch-register from a directory of .wav files + labels CSV ────────────

def _cli_main():
    parser = argparse.ArgumentParser(
        description="Register a directory of synthetic .wav files into the feature cache."
    )
    parser.add_argument("--audio-dir", required=True, help="Directory containing generated .wav files")
    parser.add_argument("--labels-csv", required=True,
                        help="CSV with columns: filename,label_4class[,label_fine,split]")
    parser.add_argument("--generator", required=True, help="Identifier for this generative model/run")
    parser.add_argument("--source-dataset", default="icbhi", choices=["icbhi", "sprsound"])
    parser.add_argument("--features", nargs="+", default=list(EXTRACTORS.keys()),
                        choices=list(EXTRACTORS.keys()))
    parser.add_argument("--features-cfg", default="config/features.yaml")
    parser.add_argument("--split", default="train", help="Default split if not specified per-row in CSV")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        import soundfile as sf
    except ImportError:
        logger.error("soundfile is required for the CLI mode: pip install soundfile")
        sys.exit(1)

    audio_dir = Path(args.audio_dir)
    n_ok, n_err = 0, 0

    with open(args.labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    logger.info("Registering %d synthetic samples from %s", len(rows), args.audio_dir)

    for row in rows:
        wav_path = audio_dir / row["filename"]
        if not wav_path.exists():
            logger.warning("Audio file not found: %s — skipping", wav_path)
            n_err += 1
            continue

        try:
            waveform, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)  # mixdown to mono

            result = register_synthetic_cycle(
                waveform=waveform,
                sr=sr,
                label_4class=row["label_4class"],
                generator=args.generator,
                source_dataset=args.source_dataset,
                label_fine=row.get("label_fine") or None,
                split=row.get("split") or args.split,
                feature_types=args.features,
                features_cfg_path=args.features_cfg,
            )
            if result["errors"]:
                n_err += 1
            else:
                n_ok += 1

        except Exception as e:
            logger.error("Failed to register %s: %s", wav_path, e)
            n_err += 1

    logger.info("Done. %d succeeded, %d failed.", n_ok, n_err)


if __name__ == "__main__":
    _cli_main()