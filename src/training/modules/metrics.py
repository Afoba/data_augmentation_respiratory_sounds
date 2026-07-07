"""
metrics.py
----------
Classification metrics for the respiratory sound pipeline, including the
official ICBHI 2017 challenge score.

── The ICBHI score, precisely ───────────────────────────────────────────────
This is NOT the same as macro-averaged recall over the 4 classes. The
official definition collapses the 4-class problem (normal/crackle/wheeze/both)
into a BINARY normal-vs-abnormal problem first:

    Sensitivity (Se) = correctly classified ABNORMAL cycles / total abnormal cycles
                        where ABNORMAL = crackle + wheeze + both
    Specificity (Sp) = correctly classified NORMAL cycles / total normal cycles
    ICBHI Score       = (Se + Sp) / 2

A cycle is "correctly classified abnormal" if its true label is abnormal AND
the predicted label is ALSO abnormal (any of crackle/wheeze/both — the
specific abnormal subtype predicted does not need to match for this metric).
Likewise a normal cycle is "correctly classified normal" only if predicted
normal; predicting it as any abnormal subtype counts as a miss for Sp.

This means a model can have mediocre per-class accuracy on crackle vs wheeze
vs both specifically, but still score well on the ICBHI metric, AS LONG AS it
reliably distinguishes normal from (any kind of) abnormal. Computing this as
plain macro recall over 4 classes would NOT reproduce the official number —
that's a common mistake worth avoiding when comparing against published
ICBHI results.

Requires class_list to be exactly ["normal", "crackle", "wheeze", "both"]
(any order is fine, but all four must be present) — this metric is only
defined for the 4-class / coarse label scheme, not the SPRSound fine-grained
7-class scheme.
"""

import logging
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix as sk_confusion_matrix,
)

logger = logging.getLogger(__name__)

_ABNORMAL_CLASSES = {"crackle", "wheeze", "both"}


def icbhi_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_list: list[str],
) -> dict:
    """
    Compute the official ICBHI 2017 score: (Sensitivity + Specificity) / 2,
    where Se/Sp are defined on the binary normal-vs-abnormal collapse of the
    4-class predictions (see module docstring).

    Parameters
    ----------
    y_true     : 1-D array of true class INDICES (not strings)
    y_pred     : 1-D array of predicted class INDICES, same length as y_true
    class_list : ordered list of class name strings corresponding to the
                 indices used in y_true/y_pred (e.g. ["normal","crackle","wheeze","both"])

    Returns
    -------
    dict with keys: "se" (sensitivity), "sp" (specificity), "icbhi_score",
    plus the raw counts used to compute them (for sanity-checking / logging).
    """
    if "normal" not in class_list:
        raise ValueError(
            f"icbhi_score requires 'normal' in class_list, got {class_list}. "
            "This metric is only defined for the 4-class/coarse label scheme."
        )
    missing_abnormal = _ABNORMAL_CLASSES - set(class_list)
    if missing_abnormal:
        logger.warning(
            "icbhi_score: class_list %s is missing some standard abnormal "
            "classes %s — Se will only be computed over whichever abnormal "
            "classes ARE present. This is expected if you've filtered out "
            "a class (e.g. excluded 'both'), but double check this is intended.",
            class_list, missing_abnormal,
        )

    normal_idx = class_list.index("normal")
    abnormal_indices = {class_list.index(c) for c in _ABNORMAL_CLASSES if c in class_list}

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_is_abnormal = np.isin(y_true, list(abnormal_indices))
    true_is_normal = y_true == normal_idx
    pred_is_abnormal = np.isin(y_pred, list(abnormal_indices))
    pred_is_normal = y_pred == normal_idx

    n_abnormal = true_is_abnormal.sum()
    n_normal = true_is_normal.sum()

    if n_abnormal == 0:
        logger.warning("icbhi_score: no abnormal cycles in y_true — Se is undefined, returning 0.0")
        se = 0.0
    else:
        se = (true_is_abnormal & pred_is_abnormal).sum() / n_abnormal

    if n_normal == 0:
        logger.warning("icbhi_score: no normal cycles in y_true — Sp is undefined, returning 0.0")
        sp = 0.0
    else:
        sp = (true_is_normal & pred_is_normal).sum() / n_normal

    return {
        "se": float(se),
        "sp": float(sp),
        "icbhi_score": float((se + sp) / 2),
        "n_normal": int(n_normal),
        "n_abnormal": int(n_abnormal),
    }


def standard_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_list: list[str],
) -> dict:
    """
    Standard classification metrics: accuracy, macro/weighted F1, and the
    confusion matrix.

    Parameters
    ----------
    y_true, y_pred : 1-D arrays of class indices
    class_list     : ordered class names (for confusion matrix labelling)

    Returns
    -------
    dict with "accuracy", "f1_macro", "f1_weighted", "confusion_matrix"
    (the confusion matrix is a nested list, JSON-serialisable, rows=true,
    cols=predicted, in class_list order).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(len(class_list)))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "confusion_matrix": sk_confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "class_list": class_list,
    }


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_list: list[str],
    test_subset: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute the full metric suite: standard metrics + ICBHI score (if the
    class_list supports it), optionally broken down by SPRSound's
    inter/intra test_subset.

    Parameters
    ----------
    y_true, y_pred : 1-D arrays of class indices
    class_list     : ordered class names
    test_subset    : optional 1-D array, same length as y_true, with values
                      "inter" | "intra" | None per sample. If provided,
                      metrics are computed separately for each subset in
                      addition to the overall (merged) numbers.

    Returns
    -------
    dict:
        "overall": {standard metrics + icbhi_score if applicable}
        "inter":   same, computed only on test_subset == "inter" rows (if test_subset given)
        "intra":   same, computed only on test_subset == "intra" rows (if test_subset given)
    """
    result = {"overall": _compute_one(y_true, y_pred, class_list)}

    if test_subset is not None:
        test_subset = np.asarray(test_subset)
        for subset_name in ("inter", "intra"):
            mask = test_subset == subset_name
            if mask.sum() == 0:
                continue
            result[subset_name] = _compute_one(y_true[mask], y_pred[mask], class_list)

    return result


def _compute_one(y_true: np.ndarray, y_pred: np.ndarray, class_list: list[str]) -> dict:
    metrics = standard_metrics(y_true, y_pred, class_list)
    if "normal" in class_list and _ABNORMAL_CLASSES & set(class_list):
        metrics.update(icbhi_score(y_true, y_pred, class_list))
    return metrics


# ── Convenience: run inference over a DataLoader and collect predictions ───────

@torch.no_grad()
def predict_dataset(model: torch.nn.Module, dataloader, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """
    Run model in eval mode over a DataLoader, returning (y_true, y_pred) as
    numpy arrays of class indices. Assumes the dataloader yields (spec, label)
    pairs where label is an integer class index (NOT one-hot — i.e. do not
    use this with a mixup-collated loader, which produces soft labels not
    suited to this metric computation).
    """
    model.eval()
    all_true, all_pred = [], []
    for specs, labels in dataloader:
        specs = specs.to(device)
        logits = model(specs)
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_pred.append(preds)
        all_true.append(np.asarray(labels))
    return np.concatenate(all_true), np.concatenate(all_pred)