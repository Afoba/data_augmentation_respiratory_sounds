"""
dataset.py
----------
PyTorch Dataset wrapping the cached metadata.parquet + .npy feature files,
with on-the-fly augmentation applied at load time.

This is the bridge between the offline preprocessing/caching pipeline
(build_features.py, build_augmented.py, write_synthetic.py) and a training
loop. It does NOT know about model architecture — it just returns
(feature_tensor, label) pairs, with augmentation controlled by config.

Data selection (which rows of metadata.parquet to use) is expressed as a
pandas query/filter BEFORE constructing this Dataset — see the README
examples. This Dataset only handles loading + augmenting whatever rows
it's given.

On-the-fly augmentation layers
-------------------------------
1. Waveform-level (applied to the cached real-data waveform if you want
   noise/gain/time-shift on top of an existing log-mel cache, you would
   normally instead re-extract from raw audio — see note below). In THIS
   Dataset, on-the-fly augmentation operates on the CACHED SPECTROGRAM
   (logmel/mfcc/stft), not the raw waveform, since that's what's cached.
   Waveform-level on-the-fly augmentation (noise/gain/time-shift) is
   provided separately in waveform_augment.py for use in a RAW-AUDIO
   dataset variant if you'd rather augment before feature extraction.
2. Spectrogram-level (SpecAugment, optional flip) via spectrogram_augment.py,
   applied per-sample in __getitem__.
3. Mixup is NOT applied here (it needs a batch of paired samples) — apply
   SpectrogramMixup as a collate_fn step or inside your training loop after
   batching. See the README for an example collate_fn.

Two Dataset variants are provided:
    CachedFeatureDataset — loads pre-extracted .npy spectrograms (FAST,
                           recommended for normal training/iteration).
    RawAudioDataset       — loads raw audio and extracts features on-the-fly,
                           enabling waveform-level augmentation
                           (noise/gain/pitch/time-shift/time-stretch) for
                           cycles NOT already pre-cached via build_augmented.py.
                           Slower; use selectively.
"""

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from preprocessing.restructure.resample import load_segment, seconds_to_samples
from preprocessing.restructure.pad import normalise_length
from preprocessing.restructure.features import extract as extract_feature
from preprocessing.augmentation import waveform as wa

logger = logging.getLogger(__name__)


_LABEL_COLUMNS = {
    "4class": "label_4class",
    "fine":   "label_fine",
    "coarse": "label_coarse",
}


class CachedFeatureDataset(Dataset):
    """
    Loads pre-extracted feature arrays (.npy) referenced by a filtered
    metadata DataFrame, with optional on-the-fly spectrogram augmentation.

    Parameters
    ----------
    metadata        : pre-filtered DataFrame (one row per cycle/feature_type
                       combination — already subset by source_type, split,
                       feature_type, etc. by the caller)
    feature_type    : which feature_type rows to use ("logmel" | "mfcc" | "stft").
                       Rows not matching this are ignored (filtered internally
                       too, in case `metadata` contains multiple feature types).
    label_scheme    : "4class" (default) | "fine" | "coarse" — see note below.
                       "4class" and "coarse" are aliases for the SAME column
                       (label_coarse): SPRSound's 7-class taxonomy is collapsed
                       via coarse_label_map in sprsound.yaml (e.g. stridor and
                       rhonchi both become "wheeze") so it lines up with
                       ICBHI's native 4 classes (normal/crackle/wheeze/both).
                       "fine" uses label_fine instead — SPRSound's native
                       classes with NO merging at all (stridor stays
                       "stridor", rhonchi stays "rhonchi"; ICBHI cycles are
                       unaffected either way since label_fine == label_4class
                       for ICBHI). Use "fine" if you don't want SPRSound's
                       coarse mapping applied; use "4class"/"coarse" if you
                       want ICBHI-comparable labels. See
                       preprocessing/datasets/sprsound_loader.py for the
                       exact mapping table.
    class_list      : ordered list of class names defining the label index
                       mapping (e.g. ["normal","crackle","wheeze","both"]).
                       Required so label indices are consistent across runs.
    transform       : optional callable (e.g. spectrogram_augment.ComposeAugment
                       instance) applied to the loaded spectrogram tensor.
                       Pass None to disable (e.g. for validation/test sets).
    return_metadata : if True, __getitem__ also returns the row's cycle_id
                       and source_type (useful for debugging/error analysis)
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        feature_type: str,
        label_scheme: str = "4class",
        class_list: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        return_metadata: bool = False,
    ):
        if label_scheme not in _LABEL_COLUMNS:
            raise ValueError(f"label_scheme must be one of {list(_LABEL_COLUMNS)}, got {label_scheme!r}")

        self.df = metadata[metadata["feature_type"] == feature_type].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(
                f"No rows with feature_type={feature_type!r} found in the provided metadata. "
                f"Available feature_types: {metadata['feature_type'].unique().tolist() if 'feature_type' in metadata else 'N/A'}"
            )

        self.label_col = _LABEL_COLUMNS[label_scheme]
        self.class_list = class_list or sorted(self.df[self.label_col].dropna().unique().tolist())
        self.class_to_idx = {c: i for i, c in enumerate(self.class_list)}
        self.transform = transform
        self.return_metadata = return_metadata

        unknown = set(self.df[self.label_col].dropna().unique()) - set(self.class_list)
        if unknown:
            logger.warning(
                "Found labels not in class_list and will raise on access: %s. "
                "Pass a complete class_list to avoid this.", unknown,
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        feature = np.load(row["feature_path"]).astype(np.float32)
        spec = torch.from_numpy(feature)

        if self.transform is not None:
            spec = self.transform(spec)

        label_str = row[self.label_col]
        if label_str not in self.class_to_idx:
            raise KeyError(
                f"Label {label_str!r} (row cycle_id={row['cycle_id']}) not found in "
                f"class_to_idx={self.class_to_idx}. Pass a complete class_list."
            )
        label_idx = self.class_to_idx[label_str]

        if self.return_metadata:
            return spec, label_idx, {"cycle_id": row["cycle_id"], "source_type": row["source_type"]}
        return spec, label_idx


class RawAudioDataset(Dataset):
    """
    Loads RAW audio segments and performs length-normalisation + feature
    extraction on-the-fly, with optional waveform-level augmentation
    applied BEFORE feature extraction. Use this when you want stochastic
    waveform augmentations (noise, gain, time-shift, time-stretch) that
    were not pre-cached by build_augmented.py.

    This is slower than CachedFeatureDataset since it recomputes the
    spectrogram every access — use a reasonable number of DataLoader
    workers to hide the cost.

    Parameters
    ----------
    metadata        : DataFrame with at least one row per cycle (audio_path,
                       start_time, end_time, label columns) — typically the
                       `source_type == "real"` subset, deduplicated by cycle_id
    feature_type    : "logmel" | "mfcc" | "stft"
    features_cfg    : the parsed features.yaml dict (for target_sr, padding,
                       and feature-extraction parameters)
    label_scheme    : "4class" | "fine" | "coarse" — see CachedFeatureDataset
                       docstring above for the full explanation; in short,
                       "fine" avoids SPRSound's coarse mapping (stridor/rhonchi
                       stay separate) while "4class"/"coarse" merges them into
                       ICBHI-compatible classes.
    class_list      : ordered class names (see CachedFeatureDataset)
    waveform_augment_fn : optional callable(waveform, sr) -> waveform applied
                       BEFORE length normalisation. Compose your own from
                       waveform_augment.py functions, e.g.:

                           def my_aug(wav, sr):
                               if random.random() < 0.5:
                                   wav = wa.random_gain(wav)
                               if random.random() < 0.3:
                                   wav = wa.additive_gaussian_noise(wav, snr_db=15)
                               if random.random() < 0.3:
                                   wav = wa.time_shift(wav)
                               return wav

    spectrogram_transform : optional callable applied to the resulting
                       spectrogram AFTER extraction (e.g. SpecAugment)
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        feature_type: str,
        features_cfg: dict,
        label_scheme: str = "4class",
        class_list: Optional[list[str]] = None,
        waveform_augment_fn: Optional[Callable] = None,
        spectrogram_transform: Optional[Callable] = None,
    ):
        if label_scheme not in _LABEL_COLUMNS:
            raise ValueError(f"label_scheme must be one of {list(_LABEL_COLUMNS)}, got {label_scheme!r}")

        self.df = metadata.drop_duplicates(subset=["cycle_id"]).reset_index(drop=True)
        self.feature_type = feature_type
        self.features_cfg = features_cfg
        self.label_col = _LABEL_COLUMNS[label_scheme]
        self.class_list = class_list or sorted(self.df[self.label_col].dropna().unique().tolist())
        self.class_to_idx = {c: i for i, c in enumerate(self.class_list)}
        self.waveform_augment_fn = waveform_augment_fn
        self.spectrogram_transform = spectrogram_transform

        self.target_sr = features_cfg.get("target_sr", 16000)
        self.target_dur = float(features_cfg.get("target_duration_s", 8.0))
        self.length_mode = features_cfg.get("length_mode", "pad_truncate")
        self.pad_mode = features_cfg.get("pad_mode", "reflect")
        self.pad_alignment = features_cfg.get("pad_alignment", "center")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        waveform, sr = load_segment(
            audio_path=row["audio_path"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            target_sr=self.target_sr,
            mono=True,
        )

        if self.waveform_augment_fn is not None:
            waveform = self.waveform_augment_fn(waveform, sr)

        target_samples = seconds_to_samples(self.target_dur, sr)
        waveform = normalise_length(
            waveform,
            target_samples=target_samples,
            mode=self.length_mode,
            pad_mode=self.pad_mode,
            pad_alignment=self.pad_alignment,
        )

        feature = extract_feature(self.feature_type, waveform, sr, self.features_cfg)
        spec = torch.from_numpy(feature.astype(np.float32))

        if self.spectrogram_transform is not None:
            spec = self.spectrogram_transform(spec)

        label_str = row[self.label_col]
        label_idx = self.class_to_idx[label_str]

        return spec, label_idx


# ── Example collate_fn for applying Mixup after batching ───────────────────────

def make_mixup_collate_fn(mixup_module, n_classes: int):
    """
    Returns a collate_fn that stacks a batch and applies SpectrogramMixup.

    Usage:
        from augmentation.spectrogram_augment import SpectrogramMixup
        mixer = SpectrogramMixup(alpha=0.4, label_mode="soft")
        loader = DataLoader(dataset, batch_size=32,
                            collate_fn=make_mixup_collate_fn(mixer, n_classes=4))
    """
    def collate_fn(batch):
        specs = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        labels_onehot = torch.nn.functional.one_hot(labels, num_classes=n_classes).float()
        mixed_specs, mixed_labels = mixup_module(specs, labels_onehot)
        return mixed_specs, mixed_labels

    return collate_fn