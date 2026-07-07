"""
pad.py
------
Waveform-level preprocessing: length normalisation (pad/truncate or time-stretch)
and smart padding strategies.

All functions operate on 1-D numpy arrays (single-channel waveforms).
Multi-channel input is rejected early — callers are expected to already have
mixed down to mono before calling these utilities.
"""

import numpy as np
import warnings

try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False


# ── Public API ─────────────────────────────────────────────────────────────────

def normalise_length(
    waveform: np.ndarray,
    target_samples: int,
    mode: str = "pad_truncate",
    pad_mode: str = "reflect",
    pad_alignment: str = "center",
) -> np.ndarray:
    """
    Normalise a 1-D waveform to exactly `target_samples` samples.

    Parameters
    ----------
    waveform       : 1-D float32 array
    target_samples : desired output length in samples
    mode           : "pad_truncate" | "time_stretch"
                     pad_truncate — pad short / truncate long (no pitch distortion)
                     time_stretch — phase-vocoder stretch to target length
    pad_mode       : "reflect" | "zero" | "replicate"
                     How to fill padding region (only used in pad_truncate mode).
    pad_alignment  : "center" | "start" | "end"
                     Where to place the real signal within the padded window.

    Returns
    -------
    1-D float32 array of length exactly target_samples.
    """
    if waveform.ndim != 1:
        raise ValueError(f"Expected 1-D waveform, got shape {waveform.shape}")

    n = len(waveform)

    if n == target_samples:
        return waveform.astype(np.float32)

    if mode == "pad_truncate":
        return _pad_truncate(waveform, target_samples, pad_mode, pad_alignment)
    elif mode == "time_stretch":
        return _time_stretch(waveform, target_samples)
    else:
        raise ValueError(f"Unknown length normalisation mode: {mode!r}. "
                         f"Choose 'pad_truncate' or 'time_stretch'.")


# ── Internal helpers ───────────────────────────────────────────────────────────

def _pad_truncate(
    waveform: np.ndarray,
    target_samples: int,
    pad_mode: str,
    pad_alignment: str,
) -> np.ndarray:
    n = len(waveform)
    waveform = waveform.astype(np.float32)

    # ── Truncate if too long ───────────────────────────────────────────────────
    if n > target_samples:
        if pad_alignment == "center":
            # Take the centre crop
            start = (n - target_samples) // 2
            return waveform[start : start + target_samples]
        elif pad_alignment == "start":
            return waveform[:target_samples]
        else:  # "end"
            return waveform[n - target_samples :]

    # ── Pad if too short ───────────────────────────────────────────────────────
    total_pad = target_samples - n

    if pad_alignment == "center":
        pad_left  = total_pad // 2
        pad_right = total_pad - pad_left
    elif pad_alignment == "start":
        pad_left, pad_right = 0, total_pad
    else:  # "end"
        pad_left, pad_right = total_pad, 0

    return _apply_padding(waveform, pad_left, pad_right, pad_mode)


def _apply_padding(
    waveform: np.ndarray,
    pad_left: int,
    pad_right: int,
    pad_mode: str,
) -> np.ndarray:
    n = len(waveform)

    if pad_mode == "zero":
        return np.pad(waveform, (pad_left, pad_right), mode="constant", constant_values=0.0)

    elif pad_mode == "replicate":
        return np.pad(waveform, (pad_left, pad_right), mode="edge")

    elif pad_mode == "reflect":
        # numpy reflect mode requires pad width ≤ n-1 on each side.
        # If the waveform is very short relative to the required padding,
        # we tile the signal enough times first, then reflect-pad normally.
        max_single_pass = max(n - 1, 1)

        if pad_left <= max_single_pass and pad_right <= max_single_pass:
            return np.pad(waveform, (pad_left, pad_right), mode="reflect")

        # Tile until a single reflect pass can cover the required padding
        repeats_needed = (max(pad_left, pad_right) // max_single_pass) + 2
        tiled = np.tile(waveform, repeats_needed)
        tiled_n = len(tiled)
        # Now reflect-pad the tiled array — pad widths are now well within bounds
        result = np.pad(tiled, (pad_left, pad_right), mode="reflect")
        # Extract the centre window of exactly target length
        total_target = pad_left + n + pad_right
        start = (len(result) - total_target) // 2
        return result[start : start + total_target].astype(np.float32)

    else:
        raise ValueError(
            f"Unknown pad_mode: {pad_mode!r}. Choose 'reflect', 'zero', or 'replicate'."
        )


def _safe_stretch_n_fft(signal_len: int, default: int = 2048) -> int:
    """
    Pick an n_fft for librosa.effects.time_stretch() that never exceeds the
    input signal length. librosa.effects.time_stretch() defaults to
    n_fft=2048 internally regardless of input length, which triggers a
    "n_fft is too large for input signal" warning (and an uncontrolled
    internal zero-pad fallback) on short cycles — ICBHI has some cycles
    under 2048 samples at 16kHz (~0.13s). Returns the largest power of 2
    that is <= signal_len, capped at `default`, floored at 32 to avoid a
    degenerately small analysis window on extremely short signals.
    """
    if signal_len >= default:
        return default
    n = 1
    while n * 2 <= signal_len:
        n *= 2
    return max(n, 32)


def _time_stretch(waveform: np.ndarray, target_samples: int) -> np.ndarray:
    """
    Phase-vocoder time-stretch to reach exactly target_samples.
    Requires librosa. Pitch is preserved (only duration changes).
    """
    if not _LIBROSA_AVAILABLE:
        raise ImportError(
            "librosa is required for time_stretch mode. "
            "Install with: pip install librosa"
        )

    n = len(waveform)
    if n == 0:
        return np.zeros(target_samples, dtype=np.float32)

    rate = n / target_samples   # <1 slows down (stretches), >1 speeds up (squashes)

    # n_fft is sized to the INPUT signal (n), not the target — librosa's STFT
    # runs on the original unstretched waveform internally regardless of the
    # requested stretch rate, so what matters is whether n_fft fits within n.
    stretch_n_fft = _safe_stretch_n_fft(n)

    stretched = librosa.effects.time_stretch(
        waveform.astype(np.float32), rate=rate, n_fft=stretch_n_fft
    )

    # Phase vocoder output length is approximate; enforce exact length
    if len(stretched) < target_samples:
        stretched = np.pad(stretched, (0, target_samples - len(stretched)), mode="constant")
    elif len(stretched) > target_samples:
        stretched = stretched[:target_samples]

    return stretched.astype(np.float32)