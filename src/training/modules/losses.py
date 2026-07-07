"""
losses.py
---------
Loss function construction for the classification pipeline. Off by default —
plain (unweighted) cross-entropy unless explicitly configured otherwise.

── Class weighting strategies ───────────────────────────────────────────────
All weighting strategies are computed from the ACTUAL training-pool class
counts you pass in (i.e. whatever source_type/split/etc. filters you already
applied before building the DataLoader) — NOT from some assumed real-only
distribution. This matters when combining with augmentation: if you're
training on real+augmented data, weights computed here reflect what the
model is actually seeing per epoch, avoiding the double-counting problem
described in conversation (weighting against a stale real-only ratio while
ALSO oversampling via augmentation compounds the correction twice).

    inverse_frequency      : weight_c = total / (n_classes * count_c)
                              Aggressive — can heavily dominate gradients
                              early in training if imbalance is severe.
    inverse_sqrt_frequency : weight_c = sqrt(total / count_c), then renormalised
                              Gentler than inverse_frequency; usually better
                              behaved for severe imbalance.
    effective_number        : weight_c = (1 - beta) / (1 - beta^count_c)
                              Cui et al. 2019 "Class-Balanced Loss" — accounts
                              for diminishing returns of additional samples
                              within a class (marginal samples in a large
                              class are more redundant than in a small one).
                              beta is a config knob, default 0.999.
    none                     : plain unweighted cross-entropy (default).
"""

import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def compute_class_weights(
    class_counts: dict[str, int],
    class_list: list[str],
    strategy: str = "none",
    beta: float = 0.999,
) -> torch.Tensor | None:
    """
    Compute per-class loss weights from observed training-pool class counts.

    Parameters
    ----------
    class_counts : {class_name: count} — typically
                    df_train_pool[label_col].value_counts().to_dict()
                    computed AFTER all data-selection filters (source_type,
                    split, etc.) are applied, so it reflects the actual
                    training distribution, not just the real-data distribution.
    class_list    : ordered class names — determines output tensor index order,
                    must match whatever order your Dataset uses for label indices.
    strategy      : "none" | "inverse_frequency" | "inverse_sqrt_frequency" | "effective_number"
    beta          : only used for "effective_number" strategy.

    Returns
    -------
    torch.Tensor of shape (n_classes,), or None if strategy == "none"
    (None is the correct value to pass as nn.CrossEntropyLoss(weight=None),
    i.e. unweighted).
    """
    if strategy == "none":
        return None

    counts = np.array([class_counts.get(c, 0) for c in class_list], dtype=np.float64)

    if (counts == 0).any():
        missing = [c for c, n in zip(class_list, counts) if n == 0]
        logger.warning(
            "compute_class_weights: class(es) %s have ZERO samples in the "
            "training pool — their weight will be set to 0 contribution-wise "
            "is undefined; using a small epsilon count instead to avoid "
            "division by zero. Double check this is intended (e.g. you may "
            "have filtered out a class entirely).",
            missing,
        )
        counts = np.maximum(counts, 1.0)  # avoid divide-by-zero; weight becomes large but finite

    total = counts.sum()
    n_classes = len(class_list)

    if strategy == "inverse_frequency":
        weights = total / (n_classes * counts)

    elif strategy == "inverse_sqrt_frequency":
        weights = np.sqrt(total / counts)
        weights = weights / weights.sum() * n_classes  # renormalise so mean weight ≈ 1

    elif strategy == "effective_number":
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * n_classes  # renormalise so mean weight ≈ 1

    else:
        raise ValueError(
            f"Unknown class weighting strategy: {strategy!r}. "
            "Choose 'none', 'inverse_frequency', 'inverse_sqrt_frequency', or 'effective_number'."
        )

    logger.info(
        "Class weights (%s): %s",
        strategy, {c: round(float(w), 3) for c, w in zip(class_list, weights)},
    )

    return torch.tensor(weights, dtype=torch.float32)


def build_loss(cfg: dict, class_counts: dict[str, int], class_list: list[str]) -> nn.Module:
    """
    Build the loss function from experiment config.

    Config keys (under cfg["loss"] in the experiment YAML):
        class_weighting : "none" (default) | "inverse_frequency" |
                           "inverse_sqrt_frequency" | "effective_number"
        beta             : float, default 0.999 — only used for "effective_number"
        label_smoothing  : float, default 0.0 — passed through to CrossEntropyLoss

    Parameters
    ----------
    cfg          : full experiment config dict
    class_counts : {class_name: count} from the actual training pool (see
                   compute_class_weights docstring)
    class_list   : ordered class names

    Returns
    -------
    nn.CrossEntropyLoss, configured with weight= and label_smoothing= as specified.
    """
    loss_cfg = cfg.get("loss", {})
    strategy = loss_cfg.get("class_weighting", "none")
    beta = loss_cfg.get("beta", 0.999)
    label_smoothing = loss_cfg.get("label_smoothing", 0.0)

    weights = compute_class_weights(class_counts, class_list, strategy=strategy, beta=beta)

    return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)