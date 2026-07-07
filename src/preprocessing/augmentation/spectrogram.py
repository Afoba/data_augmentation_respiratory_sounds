"""
spectrogram_augment.py
-----------------------
On-the-fly spectrogram-level augmentation as PyTorch nn.Module transforms.

These operate on 2-D (freq, time) or batched 3-D/4-D (batch, [channel,] freq, time)
torch.Tensor spectrograms — i.e. whatever was loaded from the cached .npy feature
files (log-mel, MFCC, or STFT all share this shape convention).

Modules included
-----------------
    SpecAugment        — frequency + time masking (Park et al. 2019)
    SpectrogramFlip     — time-reverse / frequency-reverse. DISABLED BY DEFAULT,
                          see acoustic-validity warning below.
    SpectrogramMixup    — linear interpolation of two spectrograms + labels,
                          supports both soft (interpolated) and hard
                          (majority/same-class-only) label modes.

── Acoustic validity warning: SpectrogramFlip ──────────────────────────────
Time-reversal destroys the characteristic rapid-onset/decay shape of crackles,
and frequency-reversal inverts harmonic structure into a physically unrealistic
pattern no real respiratory sound produces. This transform is provided for
ablation/completeness only. It is OFF by default (`p=0.0`) and logs a warning
the first time it is constructed with a nonzero probability. Enable deliberately
and validate its effect on validation metrics before trusting it.

Usage
-----
    transform = ComposeAugment([
        SpecAugment(freq_mask_param=15, time_mask_param=25, n_freq_masks=2, n_time_masks=2, p=0.8),
        SpectrogramFlip(time_flip_p=0.0, freq_flip_p=0.0),   # explicitly off
    ])
    augmented_spec = transform(spec)   # spec: torch.Tensor (freq, time)

    # Mixup is applied at the batch level (needs pairs + labels), used separately:
    mixer = SpectrogramMixup(alpha=0.4, label_mode="soft")
    mixed_specs, mixed_labels = mixer(batch_specs, batch_labels_onehot)
"""

import logging
import warnings

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ── SpecAugment ──────────────────────────────────────────────────────────────

class SpecAugment(nn.Module):
    """
    Frequency and time masking augmentation (Park et al., 2019).

    Applies `n_freq_masks` frequency-axis masks and `n_time_masks` time-axis
    masks, each of random width up to the given parameter, with masked
    regions set to a fill value (default: the spectrogram's own mean, which
    is gentler than zero for log-scaled features where 0 may not represent
    "silence").

    Parameters
    ----------
    freq_mask_param : max width of each frequency mask (in bins)
    time_mask_param : max width of each time mask (in frames)
    n_freq_masks    : number of frequency masks to apply
    n_time_masks    : number of time masks to apply
    fill_value      : "mean" to use the spectrogram's mean, or a float constant
    p               : probability of applying SpecAugment at all to a given sample
    """

    def __init__(
        self,
        freq_mask_param: int = 15,
        time_mask_param: int = 25,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
        fill_value: str | float = "mean",
        p: float = 1.0,
    ):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.fill_value = fill_value
        self.p = p

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        spec : torch.Tensor of shape (freq, time) or (channel, freq, time)
        Returns a tensor of the same shape with masks applied.
        """
        if torch.rand(1).item() > self.p:
            return spec

        spec = spec.clone()
        *_, n_freq, n_time = spec.shape

        fill = spec.mean().item() if self.fill_value == "mean" else float(self.fill_value)

        for _ in range(self.n_freq_masks):
            width = int(torch.randint(0, self.freq_mask_param + 1, (1,)).item())
            if width == 0 or width >= n_freq:
                continue
            start = int(torch.randint(0, n_freq - width + 1, (1,)).item())
            spec[..., start : start + width, :] = fill

        for _ in range(self.n_time_masks):
            width = int(torch.randint(0, self.time_mask_param + 1, (1,)).item())
            if width == 0 or width >= n_time:
                continue
            start = int(torch.randint(0, n_time - width + 1, (1,)).item())
            spec[..., :, start : start + width] = fill

        return spec


# ── Spectrogram flip (disabled-by-default) ───────────────────────────────────

class SpectrogramFlip(nn.Module):
    """
    Time-reverse and/or frequency-reverse a spectrogram.

    *** ACOUSTIC VALIDITY WARNING ***
    Flipping destroys real diagnostic structure:
      - time_flip reverses the rapid-onset/decay envelope that characterises
        crackles, producing an event shape that does not occur physiologically.
      - freq_flip inverts harmonic/formant structure, producing a frequency
        pattern no real respiratory sound exhibits.

    Both probabilities default to 0.0 (fully disabled). A warning is logged
    once at construction time if either probability is set above 0.

    Parameters
    ----------
    time_flip_p : probability of applying a time-axis (horizontal) flip
    freq_flip_p : probability of applying a frequency-axis (vertical) flip
    """

    def __init__(self, time_flip_p: float = 0.0, freq_flip_p: float = 0.0):
        super().__init__()
        self.time_flip_p = time_flip_p
        self.freq_flip_p = freq_flip_p

        if time_flip_p > 0 or freq_flip_p > 0:
            warnings.warn(
                "SpectrogramFlip enabled with time_flip_p=%.2f, freq_flip_p=%.2f. "
                "Flipping can produce acoustically unrealistic crackle/wheeze patterns "
                "(reversed onset/decay shape, inverted harmonic structure). "
                "Recommended: validate against held-out metrics before trusting this "
                "augmentation; consider keeping it as an ablation rather than a default."
                % (time_flip_p, freq_flip_p),
                UserWarning,
                stacklevel=2,
            )
            logger.warning(
                "SpectrogramFlip constructed with nonzero flip probability "
                "(time=%.2f, freq=%.2f) — see acoustic-validity warning in docstring.",
                time_flip_p, freq_flip_p,
            )

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if self.time_flip_p > 0 and torch.rand(1).item() < self.time_flip_p:
            spec = torch.flip(spec, dims=[-1])
        if self.freq_flip_p > 0 and torch.rand(1).item() < self.freq_flip_p:
            spec = torch.flip(spec, dims=[-2])
        return spec


# ── Spectrogram mixup ──────────────────────────────────────────────────────

class SpectrogramMixup(nn.Module):
    """
    Mixup augmentation applied at the spectrogram level (Zhang et al., 2018,
    adapted to spectrograms as used in respiratory-sound classification).

    Linearly interpolates pairs of spectrograms within a batch:
        mixed_x = lam * x_i + (1 - lam) * x_j
    and handles labels according to `label_mode`:

        "soft" — labels are also interpolated:
                     mixed_y = lam * y_i + (1 - lam) * y_j
                 Appropriate when classes can genuinely co-occur in your label
                 space (e.g. mixing a "crackle" and "wheeze" example produces
                 a soft target that leans toward the existing "both" class
                 region of label space).

        "same_class_only" — only mixes pairs that share the same hard label;
                 the resulting label stays the hard original label (no
                 interpolation). Use this if you want mixup purely as a
                 regulariser/data-density augmentation without implying any
                 partial-class semantics. Pairs with different labels in the
                 batch are left unmixed (passed through unchanged).

    Parameters
    ----------
    alpha      : Beta(alpha, alpha) distribution parameter controlling the
                 mixing ratio. Higher alpha -> mixing ratios closer to 0.5;
                 lower alpha -> ratios closer to 0 or 1 (mild mixing).
                 Typical range: 0.2-0.4.
    label_mode : "soft" | "same_class_only"
    p          : probability of applying mixup to a given batch at all

    Forward signature
    -----------------
    forward(specs, labels) -> (mixed_specs, mixed_labels)
        specs  : torch.Tensor (batch, freq, time) or (batch, channel, freq, time)
        labels : torch.Tensor (batch, n_classes) one-hot or soft labels.
                 For "same_class_only" mode, hard integer labels of shape
                 (batch,) are also accepted and returned unchanged in shape.
    """

    def __init__(self, alpha: float = 0.4, label_mode: str = "soft", p: float = 0.5):
        super().__init__()
        if label_mode not in ("soft", "same_class_only"):
            raise ValueError(
                f"label_mode must be 'soft' or 'same_class_only', got {label_mode!r}"
            )
        self.alpha = alpha
        self.label_mode = label_mode
        self.p = p

    def forward(
        self, specs: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() > self.p:
            return specs, labels

        batch_size = specs.shape[0]
        if batch_size < 2:
            return specs, labels  # nothing to mix

        perm = torch.randperm(batch_size, device=specs.device)

        # Sample one lambda per batch (standard mixup) from Beta(alpha, alpha)
        lam = float(torch.distributions.Beta(self.alpha, self.alpha).sample())

        if self.label_mode == "soft":
            mixed_specs = lam * specs + (1 - lam) * specs[perm]
            mixed_labels = lam * labels.float() + (1 - lam) * labels[perm].float()
            return mixed_specs, mixed_labels

        # same_class_only: only mix within pairs sharing the same hard label;
        # leave other samples untouched, label unchanged.
        if labels.dim() > 1:
            hard_labels = labels.argmax(dim=-1)
        else:
            hard_labels = labels

        same_class_mask = hard_labels == hard_labels[perm]

        mixed_specs = specs.clone()
        mix_idx = same_class_mask.nonzero(as_tuple=True)[0]
        if len(mix_idx) > 0:
            mixed_specs[mix_idx] = (
                lam * specs[mix_idx] + (1 - lam) * specs[perm][mix_idx]
            )
        # Labels are unchanged in same_class_only mode — same hard class either way
        return mixed_specs, labels


# ── Compose helper ───────────────────────────────────────────────────────────

class ComposeAugment(nn.Module):
    """
    Apply a sequence of single-spectrogram transforms in order
    (e.g. SpecAugment then SpectrogramFlip). Does NOT include SpectrogramMixup,
    since mixup operates on batches of (spec, label) pairs rather than a
    single spectrogram — apply it separately after batching/collation.
    """

    def __init__(self, transforms: list[nn.Module]):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            spec = t(spec)
        return spec