"""
train.py
--------
Main training orchestrator. Reads an experiment YAML config (see
config/experiments/baseline_icbhi_resnet18.yaml for the full schema), builds
the dataset/dataloader/model/loss/optimizer from it, runs the training loop
with per-epoch validation, saves checkpoints, and writes out a metrics.json
with the full training history.

Usage
-----
    python src/training/train.py --config config/experiments/baseline_icbhi_resnet18.yaml

This script imports from preprocessing.restructure / preprocessing.augmentation
(sibling package under src/preprocessing/), so it needs `src/` on sys.path —
same convention as src/training/dataset.py. Run from the project root.

Output layout
-------------
    <run.output_dir>/<run.name>/
        checkpoints/
            best.pt       # best val checkpoint_metric so far
            last.pt       # most recent epoch (always overwritten)
        metrics.json      # full per-epoch history + final test metrics
        config.yaml       # copy of the config used, for reproducibility
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).parent.parent))  # adds src/ to path

from modules.dataset import CachedFeatureDataset, make_mixup_collate_fn, _LABEL_COLUMNS
from modules.models import build_model
from modules.losses import build_loss
from modules.metrics import compute_all_metrics, predict_dataset
from preprocessing.augmentation.spectrogram import SpecAugment, ComposeAugment, SpectrogramMixup

logger = logging.getLogger(__name__)


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("Requested device='cuda' but CUDA is not available — falling back to cpu.")
        return "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        logger.warning("Requested device='mps' but MPS is not available — falling back to cpu.")
        return "cpu"
    return requested


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Data selection ──────────────────────────────────────────────────────────

def build_metadata_filters(meta: pd.DataFrame, data_cfg: dict, seed: int = 42) -> dict[str, pd.DataFrame]:
    """
    Apply the experiment config's data-selection rules to the full metadata
    DataFrame, returning {"train": df, "val": df, "test": df}.

    val/test ALWAYS restrict to source_type == "real" regardless of
    data_cfg["source_types"], since augmented/synthetic data should never be
    used to estimate generalisation.

    data_cfg["max_per_class"], if set, is a {label_value: max_count} dict
    that DOWNSAMPLES the TRAIN pool only (never val/test) for classes whose
    row count exceeds the given cap — e.g. {"normal": 2000} to reproduce a
    majority class being capped rather than augmented. Classes not present
    as keys are left untouched. Sampling is row-level (this function is
    always called after the pool has already been filtered to a single
    feature_type, so each cycle_id appears at most once here) and seeded by
    `seed` for reproducibility.
    """
    feature_type = data_cfg["feature_type"]
    datasets = data_cfg["datasets"]
    source_types = data_cfg.get("source_types", ["real"])
    generators = data_cfg.get("generators")
    exclude_poor_quality = data_cfg.get("exclude_poor_quality", False)
    max_per_class = data_cfg.get("max_per_class")

    df = meta[
        (meta["feature_type"] == feature_type)
        & (meta["source_dataset"].isin(datasets))
    ].copy()

    if exclude_poor_quality and "quality" in df.columns:
        before = len(df)
        df = df[df["quality"] != "poor_quality"]
        logger.info("Excluded %d poor_quality rows (quality filter)", before - len(df))

    def _filter_pool(d: pd.DataFrame, allowed_source_types: list[str]) -> pd.DataFrame:
        out = d[d["source_type"].isin(allowed_source_types)]
        has_derivative_types = "augmented" in allowed_source_types or "synthetic" in allowed_source_types
        if generators is not None and has_derivative_types:
            # Only constrain rows that actually have a generator value; real
            # rows have generator=None and are unaffected by this filter.
            mask = out["generator"].isna() | out["generator"].isin(generators)
            out = out[mask]
        return out

    train_pool = _filter_pool(df[df["split"] == "train"], source_types)

    if max_per_class:
        label_col = _LABEL_COLUMNS[data_cfg.get("label_scheme", "4class")]
        pieces = []
        for label, group in train_pool.groupby(label_col, dropna=False):
            cap = max_per_class.get(label)
            if cap is not None and len(group) > cap:
                before = len(group)
                group = group.sample(n=cap, random_state=seed)
                logger.info(
                    "max_per_class: downsampled %r from %d to %d rows "
                    "(label_col=%s, seed=%d)", label, before, cap, label_col, seed,
                )
            pieces.append(group)
        train_pool = pd.concat(pieces).sort_index()

    val_pool = df[(df["split"] == "val") & (df["source_type"] == "real")]
    test_pool = df[(df["split"] == "test") & (df["source_type"] == "real")]

    if val_pool.empty:
        logger.warning(
            "No split=='val' rows found — did you run build_val_split.py? "
            "Training will proceed but validation metrics will be empty."
        )
    if test_pool.empty:
        logger.warning("No split=='test' rows found for the selected dataset(s).")

    logger.info(
        "Data pool sizes — train: %d, val: %d, test: %d (feature_type=%s, datasets=%s, source_types=%s)",
        len(train_pool), len(val_pool), len(test_pool), feature_type, datasets, source_types,
    )

    return {"train": train_pool, "val": val_pool, "test": test_pool}


def build_train_transform(aug_cfg: dict):
    spec_cfg = aug_cfg.get("spec_augment", {})
    if not spec_cfg.get("enabled", False):
        return None
    return ComposeAugment([
        SpecAugment(
            freq_mask_param=spec_cfg.get("freq_mask_param", 15),
            time_mask_param=spec_cfg.get("time_mask_param", 25),
            n_freq_masks=spec_cfg.get("n_freq_masks", 2),
            n_time_masks=spec_cfg.get("n_time_masks", 2),
            p=spec_cfg.get("p", 0.8),
        )
    ])


def build_collate_fn(aug_cfg: dict, n_classes: int):
    mixup_cfg = aug_cfg.get("mixup", {})
    if not mixup_cfg.get("enabled", False):
        return None
    mixer = SpectrogramMixup(
        alpha=mixup_cfg.get("alpha", 0.4),
        label_mode=mixup_cfg.get("label_mode", "soft"),
        p=mixup_cfg.get("p", 0.5),
    )
    return make_mixup_collate_fn(mixer, n_classes=n_classes)


# ── Weighted sampler ────────────────────────────────────────────────────────

def build_weighted_sampler(
    dataset: "CachedFeatureDataset",
    class_counts: dict[str, int],
    class_list: list[str],
    strategy: str = "inverse_sqrt_frequency",
) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that draws batches with oversampled minority
    classes, without requiring any new cached files. Each sample in the
    dataset is assigned a weight proportional to the inverse frequency (or
    inverse-sqrt-frequency, etc.) of its class, then PyTorch's sampler draws
    `len(dataset)` indices per epoch with replacement according to those
    weights — minority class cycles appear more often in each epoch than
    their raw count would suggest.

    Parameters
    ----------
    dataset      : the training CachedFeatureDataset (used to read per-sample labels)
    class_counts : {class_name: count} from the training pool
    class_list   : ordered class names, must match dataset.class_list
    strategy     : same options as losses.compute_class_weights:
                   "inverse_frequency" | "inverse_sqrt_frequency" |
                   "effective_number" | "none" (none -> uniform, no oversampling)

    Returns
    -------
    WeightedRandomSampler configured to draw len(dataset) samples per epoch
    with replacement.

    ── Interaction with loss.class_weighting ─────────────────────────────────
    Using BOTH weighted_sampler and loss.class_weighting simultaneously
    applies two independent corrections for the same imbalance problem and
    can over-correct: the sampler changes which samples appear per epoch
    (frequency-level correction) while class weighting changes their gradient
    contribution (loss-level correction). Unless you have a specific reason to
    combine them, pick one or the other. train.py logs a warning when both
    are active together.
    """
    from modules.losses import compute_class_weights

    # Reuse the same weight-computation logic from losses.py — the formulas
    # are identical; the difference is where those weights get applied (sampler
    # vs. loss function).
    class_weights_tensor = compute_class_weights(class_counts, class_list, strategy=strategy)
    if class_weights_tensor is None:
        # strategy == "none" — return a uniform sampler (no oversampling)
        sample_weights = [1.0] * len(dataset)
    else:
        weight_map = {cls: float(class_weights_tensor[i]) for i, cls in enumerate(class_list)}
        # Map each sample in the dataset to its class weight
        label_col = dataset.label_col
        sample_weights = [
            weight_map.get(dataset.df.iloc[i][label_col], 1.0)
            for i in range(len(dataset))
        ]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
    )


# ── Optimizer / scheduler ───────────────────────────────────────────────────

def build_optimizer(model: nn.Module, opt_cfg: dict) -> torch.optim.Optimizer:
    name = opt_cfg.get("name", "adamw")
    lr = opt_cfg.get("lr", 3e-4)
    weight_decay = opt_cfg.get("weight_decay", 1e-4)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if name == "adamw":
        return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        momentum = opt_cfg.get("momentum", 0.9)
        return torch.optim.SGD(trainable_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {name!r}. Choose 'adamw' or 'sgd'.")


def build_scheduler(optimizer: torch.optim.Optimizer, sched_cfg: dict, epochs: int):
    name = sched_cfg.get("name", "none")
    if name == "none":
        return None
    elif name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=sched_cfg.get("step_size", 10), gamma=sched_cfg.get("gamma", 0.1)
        )
    elif name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',      # 'min' if monitoring validation loss, 'max' for accuracy
            factor=sched_cfg.get("factor", 0.5),      # The "gradient factor"
            patience=sched_cfg.get("patience", 10), 
            verbose=True
        )
    else:
        raise ValueError(f"Unknown scheduler: {name!r}. Choose 'none', 'cosine', or 'step'.")


# ── Training / eval loops ───────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip_norm=None, mixup_active=False):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        specs, labels = batch
        specs = specs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(specs)

        if mixup_active:
            # labels are soft (B, n_classes) float tensors from make_mixup_collate_fn
            # (one_hot is always applied there, even on batches where SpectrogramMixup's
            # own `p` didn't fire, so this branch handles both mixed and unmixed batches).
            # nn.CrossEntropyLoss accepts (B, n_classes) float targets directly since
            # PyTorch 1.10, with class_weighting and label_smoothing both applying
            # correctly — previously this used a manual log_softmax implementation that
            # silently bypassed both, making those config settings ineffective for any
            # batch processed through the mixup path.
            loss = criterion(logits, labels)
        else:
            loss = criterion(logits, labels)

        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def compute_val_loss(model, loader, criterion, device) -> float | None:
    """
    Compute average loss over a validation/test loader, using the same
    criterion as training. Returns None if the loader is empty (e.g. no val
    split configured) rather than 0.0, so callers can distinguish "no data"
    from "zero loss".

    Val data is never mixup-collated (mixup is train-only — see
    build_collate_fn, only ever applied to train_loader), so this always
    uses plain integer labels with criterion(logits, labels) directly; no
    soft-label branch needed here unlike train_one_epoch.
    """
    if loader is None or len(loader.dataset) == 0:
        return None
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for specs, labels in loader:
        specs = specs.to(device)
        labels = labels.to(device)
        logits = model(specs)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, class_list, device, test_subset_col=None) -> dict:
    if loader is None or len(loader.dataset) == 0:
        return {}
    y_true, y_pred = predict_dataset(model, loader, device=device)
    subset_arr = None
    if test_subset_col is not None:
        subset_arr = np.asarray(test_subset_col)
    return compute_all_metrics(y_true, y_pred, class_list, test_subset=subset_arr)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train a respiratory sound classifier from an experiment config.")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML config")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_yaml(args.config)
    run_cfg = cfg["run"]
    data_cfg = cfg["data"]

    set_seed(run_cfg.get("seed", 42))
    device = resolve_device(run_cfg.get("device", "cpu"))
    logger.info("Using device: %s", device)

    out_dir = Path(run_cfg.get("output_dir", "results")) / run_cfg["name"]
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out_dir / "config.yaml")

    # ── Load metadata, apply data-selection filters ─────────────────────────
    feat_cfg = load_yaml(data_cfg["features_cfg"])
    meta_file = Path(feat_cfg["output"]["metadata_file"]).expanduser().resolve()
    if not meta_file.exists():
        logger.error("metadata.parquet not found at %s. Run build_features.py first.", meta_file)
        sys.exit(1)
    meta = pd.read_parquet(meta_file)

    pools = build_metadata_filters(meta, data_cfg, seed=run_cfg.get("seed", 42))
    class_list = data_cfg["class_list"]
    feature_type = data_cfg["feature_type"]

    # ── Datasets / loaders ───────────────────────────────────────────────────
    train_transform = build_train_transform(cfg.get("augmentation", {}))
    train_ds = CachedFeatureDataset(
        pools["train"], feature_type=feature_type, label_scheme=data_cfg["label_scheme"],
        class_list=class_list, transform=train_transform,
    )
    val_ds = (
        CachedFeatureDataset(pools["val"], feature_type=feature_type, label_scheme=data_cfg["label_scheme"], class_list=class_list)
        if not pools["val"].empty else None
    )
    test_ds = (
        CachedFeatureDataset(pools["test"], feature_type=feature_type, label_scheme=data_cfg["label_scheme"], class_list=class_list)
        if not pools["test"].empty else None
    )

    dl_cfg = cfg.get("data_loader", {})
    collate_fn = build_collate_fn(cfg.get("augmentation", {}), n_classes=len(class_list))
    mixup_active = collate_fn is not None

    # ── Optional WeightedRandomSampler (opt-in, off by default) ──────────────
    use_weighted_sampler = dl_cfg.get("weighted_sampler", False)
    sampler = None
    if use_weighted_sampler:
        sampler_strategy = dl_cfg.get("weighted_sampler_strategy", "inverse_sqrt_frequency")
        loss_strategy = cfg.get("loss", {}).get("class_weighting", "none")
        if loss_strategy != "none":
            logger.warning(
                "Both weighted_sampler (strategy=%r) and loss.class_weighting=%r are enabled. "
                "This applies two independent corrections for the same class imbalance — "
                "the sampler changes which samples appear per epoch (frequency-level) while "
                "class_weighting changes their gradient contribution (loss-level). This can "
                "over-correct. Consider using one or the other unless you have a specific "
                "reason to combine them.",
                sampler_strategy, loss_strategy,
            )
        # class_counts is available below, so sampler is built after that line —
        # build it after pools/class_counts are established (see below).

    train_loader = DataLoader(
        train_ds, batch_size=dl_cfg.get("batch_size", 32),
        # shuffle and sampler are mutually exclusive in PyTorch — when a sampler
        # is active, it controls the draw order, so shuffle must be False.
        shuffle=dl_cfg.get("shuffle_train", True) if not use_weighted_sampler else False,
        num_workers=dl_cfg.get("num_workers", 0),
        collate_fn=collate_fn,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=dl_cfg.get("batch_size", 32), shuffle=False,
                  num_workers=dl_cfg.get("num_workers", 0))
        if val_ds is not None else None
    )
    test_loader = (
        DataLoader(test_ds, batch_size=dl_cfg.get("batch_size", 32), shuffle=False,
                  num_workers=dl_cfg.get("num_workers", 0))
        if test_ds is not None else None
    )

    # ── Model / loss / optimizer / scheduler ─────────────────────────────────
    model = build_model(cfg, n_classes=len(class_list)).to(device)

    train_label_col = {"4class": "label_4class", "coarse": "label_coarse", "fine": "label_fine"}[data_cfg["label_scheme"]]
    class_counts = pools["train"].drop_duplicates(subset=["cycle_id"])[train_label_col].value_counts().to_dict()
    criterion = build_loss(cfg, class_counts, class_list)
    if hasattr(criterion, "weight") and criterion.weight is not None:
        criterion.weight = criterion.weight.to(device)

    # ── Build sampler now that class_counts is available ──────────────────────
    if use_weighted_sampler:
        sampler = build_weighted_sampler(
            train_ds, class_counts, class_list,
            strategy=dl_cfg.get("weighted_sampler_strategy", "inverse_sqrt_frequency"),
        )
        # Rebuild train_loader with the sampler (sampler and shuffle are mutually
        # exclusive in PyTorch — shuffle was already set to False above for this case)
        train_loader = DataLoader(
            train_ds, batch_size=dl_cfg.get("batch_size", 32),
            sampler=sampler,
            num_workers=dl_cfg.get("num_workers", 0),
            collate_fn=collate_fn,
        )
        logger.info(
            "WeightedRandomSampler active (strategy=%r) — minority classes will be "
            "oversampled per epoch. Effective samples per epoch: %d.",
            dl_cfg.get("weighted_sampler_strategy", "inverse_sqrt_frequency"),
            len(train_ds),
        )

    optimizer = build_optimizer(model, cfg.get("optimizer", {}))
    train_cfg = cfg.get("training", {})
    epochs = train_cfg.get("epochs", 30)
    scheduler = build_scheduler(optimizer, cfg.get("scheduler", {}), epochs)

    checkpoint_metric = train_cfg.get("checkpoint_metric", "icbhi_score")
    patience = train_cfg.get("early_stopping_patience")
    grad_clip_norm = train_cfg.get("grad_clip_norm")

    # ── Training loop ────────────────────────────────────────────────────────
    history = []
    best_metric_value = -float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            grad_clip_norm=grad_clip_norm, mixup_active=mixup_active,
        )

        val_metrics = evaluate(model, val_loader, class_list, device) if val_loader else {}
        val_overall = val_metrics.get("overall", {})
        val_loss = compute_val_loss(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step(val_loss if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else None)

        current_metric_value = val_overall.get(checkpoint_metric, -float("inf"))
        logger.info(
            "Epoch %d/%d — train_loss=%.4f  val_loss=%s  val_%s=%.4f  val_accuracy=%.4f",
            epoch, epochs, train_loss,
            f"{val_loss:.4f}" if val_loss is not None else "n/a",
            checkpoint_metric,
            current_metric_value if current_metric_value != -float("inf") else float("nan"),
            val_overall.get("accuracy", float("nan")),
        )

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_metrics": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        })

        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "class_list": class_list},
                  ckpt_dir / "last.pt")

        if current_metric_value > best_metric_value:
            best_metric_value = current_metric_value
            epochs_without_improvement = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "class_list": class_list},
                      ckpt_dir / "best.pt")
            logger.info("New best %s=%.4f — checkpoint saved.", checkpoint_metric, current_metric_value)
        else:
            epochs_without_improvement += 1

        if patience is not None and epochs_without_improvement >= patience:
            logger.info("Early stopping: no improvement in %d epochs.", patience)
            break

    # ── Final test evaluation (using best checkpoint) ────────────────────────
    test_metrics = {}
    if test_loader is not None and (ckpt_dir / "last.pt").exists():
        best_ckpt = torch.load(ckpt_dir / "last.pt", map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        test_subset_col = (
            pools["test"].drop_duplicates(subset=["cycle_id"])["test_subset"].values
            if "test_subset" in pools["test"].columns else None
        )
        test_metrics = evaluate(model, test_loader, class_list, device, test_subset_col=test_subset_col)
        logger.info("Final test metrics (best checkpoint): %s", test_metrics.get("overall", {}))

    # ── Write results ────────────────────────────────────────────────────────
    results = {
        "run_name": run_cfg["name"],
        "config_path": str(args.config),
        "history": history,
        "best_val_metric": {"name": checkpoint_metric, "value": best_metric_value},
        "test_metrics": test_metrics,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Training complete. Results written to %s", out_dir / "metrics.json")


if __name__ == "__main__":
    main()