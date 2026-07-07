"""
waveform_augment.py
--------------------
Waveform-level augmentation functions for respiratory sound data.

These operate on 1-D float32 numpy arrays (already resampled / before length
normalisation, or after — see notes per function). They are framework-agnostic
(pure numpy/librosa) so they can be used either:

  (a) Offline, baked into the feature cache (pitch shifting — see
      build_augmented.py), because it is too slow to run per-sample at train time.

  (b) On-the-fly inside a PyTorch Dataset.__getitem__ before feature extraction
      (noise, gain, time shift, time stretch — all cheap enough for this).

Each function takes a waveform + parameters and returns an augmented waveform
of the SAME shape as the input (callers handle any necessary re-padding).

── Acoustic validity notes ──────────────────────────────────────────────────
- pitch_shift: kept to a small semitone range by convention (±1, ±2). Larger
  shifts move wheeze frequency content outside clinically realistic ranges.
- time_stretch: changes the apparent rate of crackle events per cycle, which
  is itself a clinically meaningful feature. Use small stretch factors and
  treat this as a higher-risk augmentation; included here because it was
  explicitly requested, but consider ablating with/without it.
- additive_noise / gain / time_shift: considered safe, do not alter the
  underlying spectral/temporal characteristics that define crackle/wheeze.
"""

import logging
import numpy as np

try:
    import librosa
    _LIBROSA_OK = True
except ImportError:
    _LIBROSA_OK = False

logger = logging.getLogger(__name__)


def _require_librosa():
    if not _LIBROSA_OK:
        raise ImportError("librosa is required for this augmentation: pip install librosa")


# ── Pitch shifting (expensive — intended for offline pre-caching) ──────────────

def _safe_n_fft(signal_len: int, default: int = 2048) -> int:
    """
    Pick an n_fft that never exceeds the input signal length, avoiding
    librosa's "n_fft is too large for input signal" warning (and the
    uncontrolled internal zero-pad fallback that comes with it) on short
    waveforms. Returns the largest power of 2 <= signal_len, capped at
    `default`, floored at 32.
    """
    if signal_len >= default:
        return default
    n = 1
    while n * 2 <= signal_len:
        n *= 2
    return max(n, 32)


def pitch_shift(waveform: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """
    Shift pitch by n_steps semitones, preserving duration.

    Parameters
    ----------
    waveform : 1-D float32 array
    sr       : sample rate
    n_steps  : semitones to shift (positive = up, negative = down).
               Recommended range: [-2, +2] for respiratory sounds.

    Returns
    -------
    1-D float32 array, same length as input.
    """
    _require_librosa()
    if abs(n_steps) > 3:
        logger.warning(
            "pitch_shift: |n_steps|=%.1f exceeds the recommended ±3 semitone range "
            "for respiratory sounds — wheeze/crackle frequency content may become unrealistic.",
            n_steps,
        )
    # n_fft sized to the input — librosa.effects.pitch_shift() defaults its
    # internal STFT to n_fft=2048 regardless of input length. Callers that
    # invoke this directly on a short, unpadded waveform (e.g. an on-the-fly
    # RawAudioDataset augmentation hook running BEFORE length normalisation)
    # would otherwise hit the same warning/fallback that build_augmented.py
    # avoids by padding first — this guards the function itself, independent
    # of call order.
    n_fft = _safe_n_fft(len(waveform))
    shifted = librosa.effects.pitch_shift(
        waveform.astype(np.float32), sr=sr, n_steps=n_steps, n_fft=n_fft
    )
    # pitch_shift preserves length, but guard against off-by-one from the STFT
    if len(shifted) != len(waveform):
        if len(shifted) > len(waveform):
            shifted = shifted[: len(waveform)]
        else:
            shifted = np.pad(shifted, (0, len(waveform) - len(shifted)))
    return shifted.astype(np.float32)


# ── Time stretching (waveform-level; distinct from preprocessing's length-fit version) ──

def time_stretch(waveform: np.ndarray, rate: float) -> np.ndarray:
    """
    Stretch/compress waveform duration by `rate` while preserving pitch.

    Parameters
    ----------
    waveform : 1-D float32 array
    rate     : >1.0 speeds up (shortens), <1.0 slows down (lengthens).
               Recommended range: [0.9, 1.1] to limit distortion of crackle
               event rate / breath cycle timing.

    Returns
    -------
    1-D float32 array. NOTE: length will differ from input — caller must
    re-pad/truncate to the model's expected fixed duration (use
    preprocessing.pad.normalise_length).
    """
    _require_librosa()
    if rate <= 0:
        raise ValueError(f"rate must be positive, got {rate}")
    if not (0.8 <= rate <= 1.25):
        logger.warning(
            "time_stretch: rate=%.3f is outside the recommended [0.8, 1.25] range — "
            "crackle event rate / breath timing may become unrealistic.",
            rate,
        )
    stretched = librosa.effects.time_stretch(waveform.astype(np.float32), rate=rate)
    return stretched.astype(np.float32)


# ── Additive noise ───────────────────────────────────────────────────────────

def additive_gaussian_noise(waveform: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """
    Add white Gaussian noise at a target signal-to-noise ratio.

    Parameters
    ----------
    waveform : 1-D float32 array
    snr_db   : desired SNR in dB. Lower = noisier. Recommended range: 10-30 dB.

    Returns
    -------
    1-D float32 array, same shape as input.
    """
    signal_power = np.mean(waveform ** 2) + 1e-12
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0.0, np.sqrt(noise_power), size=waveform.shape)
    return (waveform + noise).astype(np.float32)


def additive_pink_noise(waveform: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """
    Add pink (1/f) noise at a target SNR. Pink noise more closely resembles
    real ambient/clinical-environment noise than white noise.

    Parameters
    ----------
    waveform : 1-D float32 array
    snr_db   : desired SNR in dB.

    Returns
    -------
    1-D float32 array, same shape as input.
    """
    n = len(waveform)
    # Generate pink noise via FFT filtering of white noise (1/sqrt(f) magnitude)
    white = np.random.randn(n)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1] if n > 1 else 1.0  # avoid divide-by-zero at DC
    scale = 1.0 / np.sqrt(freqs)
    spectrum = np.fft.rfft(white) * scale
    pink = np.fft.irfft(spectrum, n=n)
    pink = pink / (np.std(pink) + 1e-12)  # normalise to unit std before scaling to SNR

    signal_power = np.mean(waveform ** 2) + 1e-12
    noise_power = signal_power / (10 ** (snr_db / 10))
    pink = pink * np.sqrt(noise_power)

    return (waveform + pink).astype(np.float32)


# ── Gain / amplitude scaling ────────────────────────────────────────────────

def random_gain(
    waveform: np.ndarray,
    min_db: float = -6.0,
    max_db: float = 6.0,
) -> np.ndarray:
    """
    Apply a random gain (amplitude scale) sampled uniformly in dB.

    Parameters
    ----------
    waveform : 1-D float32 array
    min_db   : minimum gain in dB (can be negative)
    max_db   : maximum gain in dB

    Returns
    -------
    1-D float32 array, same shape as input.
    """
    gain_db = np.random.uniform(min_db, max_db)
    gain_linear = 10 ** (gain_db / 20)
    return (waveform * gain_linear).astype(np.float32)


# ── Time shifting (waveform roll within window) ─────────────────────────────

def time_shift(
    waveform: np.ndarray,
    max_shift_fraction: float = 0.1,
    wrap: bool = True,
) -> np.ndarray:
    """
    Roll the waveform circularly (or zero-fill) by a random offset.

    Parameters
    ----------
    waveform           : 1-D float32 array
    max_shift_fraction : maximum shift as a fraction of total length (0-1)
    wrap                : if True, wrapped samples re-enter at the other end
                          (circular shift). If False, vacated region is
                          zero-filled (linear shift).

    Returns
    -------
    1-D float32 array, same shape as input.
    """
    n = len(waveform)
    max_shift = int(n * max_shift_fraction)
    if max_shift == 0:
        return waveform.astype(np.float32)

    shift = np.random.randint(-max_shift, max_shift + 1)

    if wrap:
        return np.roll(waveform, shift).astype(np.float32)

    shifted = np.zeros_like(waveform)
    if shift > 0:
        shifted[shift:] = waveform[: n - shift]
    elif shift < 0:
        shifted[: n + shift] = waveform[-shift:]
    else:
        shifted[:] = waveform
    return shifted.astype(np.float32)


# ── Registry for config-driven dispatch ─────────────────────────────────────

WAVEFORM_AUGMENTERS = {
    "pitch_shift": pitch_shift,
    "time_stretch": time_stretch,
    "additive_gaussian_noise": additive_gaussian_noise,
    "additive_pink_noise": additive_pink_noise,
    "random_gain": random_gain,
    "time_shift": time_shift,
}