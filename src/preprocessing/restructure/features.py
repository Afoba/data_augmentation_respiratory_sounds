"""
features.py
-----------
Spectral feature extraction from preprocessed (padded, resampled) waveforms.

Three representations are supported:
    logmel  — log-compressed mel spectrogram  (n_mels × T)
    mfcc    — MFCCs + optional delta/delta2   (n_mfcc[×1,2,3] × T)
    stft    — STFT magnitude/power/dB          (freq_bins × T)

All functions take a 1-D float32 waveform and a config dict (parsed from
features.yaml) and return a 2-D float32 numpy array.

The caller (build_features.py) is responsible for:
  - loading and resampling the waveform (resample.py)
  - length normalisation (pad.py)
  - writing the returned array to disk
"""

import numpy as np

try:
    import librosa
    _LIBROSA_OK = True
except ImportError:
    _LIBROSA_OK = False


# ── Helpers ────────────────────────────────────────────────────────────────────

def _require_librosa():
    if not _LIBROSA_OK:
        raise ImportError("librosa is required: pip install librosa")


def _shared_stft_kwargs(cfg: dict) -> dict:
    """Extract n_fft / hop_length / win_length from a sub-config dict."""
    return {
        "n_fft":      int(cfg.get("n_fft",      1024)),
        "hop_length": int(cfg.get("hop_length",  256)),
        "win_length": int(cfg.get("win_length", 1024)),
    }


def param_tag(feature_type: str, cfg: dict, target_sr: int) -> str:
    """
    Build a short human-readable tag encoding the key parameters for a
    given feature type. Used as the subdirectory name under features/.

    Example:  "logmel__nm128_h256_w1024_sr16000"
    """
    fc = cfg.get(feature_type, {})
    sr_tag = f"sr{target_sr}"

    if feature_type == "logmel":
        return (
            f"logmel__nm{fc.get('n_mels', 128)}"
            f"_h{fc.get('hop_length', 256)}"
            f"_w{fc.get('win_length', 1024)}"
            f"_{sr_tag}"
        )
    elif feature_type == "mel":
        return (
            f"mel__nm{fc.get('n_mels', 128)}"
            f"_h{fc.get('hop_length', 256)}"
            f"_w{fc.get('win_length', 1024)}"
            f"_{sr_tag}"
        )
    elif feature_type == "mfcc":
        d_tag = ""
        if fc.get("include_delta", True):
            d_tag += "d1"
        if fc.get("include_delta2", True):
            d_tag += "d2"
        return (
            f"mfcc__nc{fc.get('n_mfcc', 40)}"
            f"_h{fc.get('hop_length', 256)}"
            f"_{d_tag}"
            f"_{sr_tag}"
        )
    elif feature_type == "stft":
        return (
            f"stft__{fc.get('output', 'magnitude')}"
            f"_h{fc.get('hop_length', 256)}"
            f"_w{fc.get('win_length', 1024)}"
            f"_{sr_tag}"
        )
    else:
        raise ValueError(f"Unknown feature type: {feature_type!r}")


def extract_mel(waveform: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    """
    Compute a log-compressed mel spectrogram.

    Parameters
    ----------
    waveform : 1-D float32 array (already padded/truncated to target length)
    sr       : sample rate of the waveform
    cfg      : full features.yaml config dict (reads cfg["logmel"])

    Returns
    -------
    2-D float32 array of shape (n_mels, T)
    """
    _require_librosa()
    fc = cfg.get("logmel", {})
    stft_kw = _shared_stft_kwargs(fc)

    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_mels=int(fc.get("n_mels", 128)),
        fmin=float(fc.get("fmin", 50.0)),
        fmax=float(fc.get("fmax", 2000.0)),
        power=float(fc.get("power", 2.0)),
        **stft_kw,
    )

    return mel_spec.astype(np.float32)


# ── Log-mel spectrogram ────────────────────────────────────────────────────────

def extract_logmel(waveform: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    """
    Compute a log-compressed mel spectrogram.

    Parameters
    ----------
    waveform : 1-D float32 array (already padded/truncated to target length)
    sr       : sample rate of the waveform
    cfg      : full features.yaml config dict (reads cfg["logmel"])

    Returns
    -------
    2-D float32 array of shape (n_mels, T)
    """
    _require_librosa()
    fc = cfg.get("logmel", {})
    stft_kw = _shared_stft_kwargs(fc)

    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_mels=int(fc.get("n_mels", 128)),
        fmin=float(fc.get("fmin", 50.0)),
        fmax=float(fc.get("fmax", 2000.0)),
        power=float(fc.get("power", 2.0)),
        **stft_kw,
    )

    log_mel = librosa.power_to_db(mel_spec, top_db=float(fc.get("top_db", 80.0)))
    return log_mel.astype(np.float32)


# ── MFCC ──────────────────────────────────────────────────────────────────────

def extract_mfcc(waveform: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    """
    Compute MFCCs, optionally appending delta and delta-delta coefficients.

    Parameters
    ----------
    waveform : 1-D float32 array
    sr       : sample rate
    cfg      : full features.yaml config dict (reads cfg["mfcc"])

    Returns
    -------
    2-D float32 array of shape (n_mfcc * n_stacks, T)
    where n_stacks = 1 (mfcc only), 2 (+ delta), or 3 (+ delta2)
    """
    _require_librosa()
    fc = cfg.get("mfcc", {})
    stft_kw = _shared_stft_kwargs(fc)
    n_mfcc = int(fc.get("n_mfcc", 40))

    coeffs = librosa.feature.mfcc(
        y=waveform,
        sr=sr,
        n_mfcc=n_mfcc,
        n_mels=int(fc.get("n_mels", 128)),
        fmin=float(fc.get("fmin", 50.0)),
        fmax=float(fc.get("fmax", 2000.0)),
        **stft_kw,
    )

    stacks = [coeffs]

    if fc.get("include_delta", True):
        stacks.append(librosa.feature.delta(coeffs, order=1))

    if fc.get("include_delta2", True):
        stacks.append(librosa.feature.delta(coeffs, order=2))

    return np.vstack(stacks).astype(np.float32)


# ── STFT ──────────────────────────────────────────────────────────────────────

def extract_stft(waveform: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    """
    Compute an STFT representation (magnitude, power, or dB).

    Parameters
    ----------
    waveform : 1-D float32 array
    sr       : sample rate
    cfg      : full features.yaml config dict (reads cfg["stft"])

    Returns
    -------
    2-D float32 array of shape (freq_bins, T)
    where freq_bins is determined by n_fft and optional max_freq_hz truncation.
    """
    _require_librosa()
    fc = cfg.get("stft", {})
    stft_kw = _shared_stft_kwargs(fc)
    output_type = fc.get("output", "magnitude")

    D = librosa.stft(waveform, **stft_kw)       # complex (freq_bins, T)
    magnitude = np.abs(D).astype(np.float32)    # (freq_bins, T)

    if output_type == "magnitude":
        result = magnitude
    elif output_type == "power":
        result = magnitude ** 2
    elif output_type == "db":
        result = librosa.amplitude_to_db(magnitude)
    else:
        raise ValueError(
            f"Unknown stft output type: {output_type!r}. "
            "Choose 'magnitude', 'power', or 'db'."
        )

    # Optional frequency truncation
    max_freq_hz = fc.get("max_freq_hz", None)
    if max_freq_hz is not None:
        n_fft = stft_kw["n_fft"]
        max_bin = int(np.floor(float(max_freq_hz) / sr * n_fft)) + 1
        max_bin = min(max_bin, result.shape[0])
        result = result[:max_bin, :]

    return result.astype(np.float32)


# ── Dispatcher ─────────────────────────────────────────────────────────────────

EXTRACTORS = {
    "mel":   extract_mel,
    "logmel": extract_logmel,
    "mfcc":   extract_mfcc,
    "stft":   extract_stft,
}


def extract(
    feature_type: str,
    waveform: np.ndarray,
    sr: int,
    cfg: dict,
) -> np.ndarray:
    """
    Dispatch to the appropriate extractor by name.

    Parameters
    ----------
    feature_type : "logmel" | "mfcc" | "stft"
    waveform     : 1-D float32 array
    sr           : sample rate
    cfg          : full features.yaml config dict

    Returns
    -------
    2-D float32 feature array
    """
    if feature_type not in EXTRACTORS:
        raise ValueError(
            f"Unknown feature type: {feature_type!r}. "
            f"Available: {list(EXTRACTORS.keys())}"
        )
    return EXTRACTORS[feature_type](waveform, sr, cfg)