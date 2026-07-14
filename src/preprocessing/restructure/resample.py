"""
resample.py
-----------
Audio loading and resampling utilities.

Loads a single-channel waveform from a given audio file, slices out a
[start_time, end_time] segment, and resamples to the target sample rate.

Depends on librosa for audio I/O and resampling. soundfile is used as the
backend where possible (faster and more format-flexible than audioread).
"""
from __future__ import annotations
import logging
from pathlib import Path

import numpy as np

try:
    import librosa
    import soundfile as sf
    _LIBROSA_AVAILABLE = True
except ImportError as e:
    _LIBROSA_AVAILABLE = False
    _IMPORT_ERROR = e

logger = logging.getLogger(__name__)


def load_segment(
    audio_path: str | Path,
    start_time: float,
    end_time: float,
    target_sr: int | None = None,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Load a time segment from an audio file and optionally resample.

    Parameters
    ----------
    audio_path : path to the audio file
    start_time : segment start in seconds (inclusive)
    end_time   : segment end in seconds (exclusive)
    target_sr  : desired sample rate after resampling.
                 Pass None to keep the file's native sample rate.
    mono       : if True, mix down to mono after loading

    Returns
    -------
    (waveform, sample_rate)
        waveform    — 1-D float32 numpy array
        sample_rate — integer sample rate of the returned waveform
    """
    if not _LIBROSA_AVAILABLE:
        raise ImportError(
            "librosa and soundfile are required. "
            "Install with: pip install librosa soundfile"
        )

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    duration = end_time - start_time
    if duration <= 0:
        raise ValueError(
            f"end_time ({end_time}) must be greater than start_time ({start_time})"
        )

    # librosa.load handles resampling natively when res_type is specified.
    # Using 'offset'/'duration' avoids loading the entire file into memory.
    waveform, sr = librosa.load(
        str(audio_path),
        sr=target_sr,       # None = native SR; int = resample on load
        mono=mono,
        offset=start_time,
        duration=duration,
        res_type="kaiser_best",   # high-quality resampler; use "kaiser_fast" to trade quality for speed
    )

    # librosa always returns float32, but assert to be safe
    waveform = waveform.astype(np.float32)

    actual_sr = target_sr if target_sr is not None else sr

    if len(waveform) == 0:
        logger.warning(
            "Empty waveform loaded from %s [%.3f–%.3f s] — "
            "start/end times may exceed file duration.",
            audio_path,
            start_time,
            end_time,
        )

    return waveform, actual_sr


def seconds_to_samples(seconds: float, sr: int) -> int:
    """Convert a duration in seconds to the number of samples at a given SR."""
    return int(round(seconds * sr))