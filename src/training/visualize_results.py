"""
visualize_results.py
---------------------
Generates archival PNG plots from a training run's metrics.json (produced by
train.py). Saves into <run_dir>/plots/, alongside checkpoints/ and config.yaml.

Usage
-----
    python src/training/visualize_results.py --run-name baseline_icbhi_resnet18

    # or point directly at a metrics.json (e.g. a custom output_dir)
    python src/training/visualize_results.py --metrics-json results/my_run/metrics.json

Produces
--------
    plots/training_curves.png      train_loss, val icbhi_score/accuracy/
                                    sensitivity/specificity, val F1
                                    (macro/weighted), learning rate — all vs
                                    epoch. If the run's val data has
                                    inter/intra subsets (SPRSound), those are
                                    plotted alongside "overall" with a
                                    consistent colour-per-subset scheme.
    plots/test_metrics_bars.png    grouped bar chart of final test metrics
                                    (accuracy, f1_macro, f1_weighted,
                                    icbhi_score, sensitivity, specificity —
                                    where present), one group of bars per
                                    metric, one bar per subset (overall /
                                    inter / intra) if applicable.
    plots/confusion_matrix.png     heatmap of the final test confusion matrix
                                    (overall), annotated with counts and
                                    row-normalised percentages.

Each plot is generated independently and skipped (with a warning, not a
crash) if the data it needs isn't present — e.g. icbhi_score is absent for
label_scheme: "fine" runs (SPRSound's native classes don't define a
normal-vs-abnormal binary collapse the same way), confusion_matrix needs at
least a test split, inter/intra lines only appear if that data exists.
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe backend, no display needed
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Consistent styling across all plots
_SUBSET_COLORS = {"overall": "#2C7FB8", "inter": "#D95F02", "intra": "#1B9E77"}
_SUBSET_ORDER = ["overall", "inter", "intra"]


def load_results(metrics_json_path: Path) -> dict:
    with open(metrics_json_path) as f:
        return json.load(f)


def _extract_series(history: list[dict], subset: str, metric: str) -> tuple[list[int], list[float]]:
    """
    Pull (epochs, values) for a given (subset, metric) pair out of the
    per-epoch history, skipping epochs where that subset/metric isn't present
    (e.g. val_loader was empty, or this run has no inter/intra data).
    """
    epochs, values = [], []
    for entry in history:
        val_metrics = entry.get("val_metrics", {})
        subset_metrics = val_metrics.get(subset)
        if subset_metrics is None or metric not in subset_metrics:
            continue
        epochs.append(entry["epoch"])
        values.append(subset_metrics[metric])
    return epochs, values


def _find_best_epoch(history: list[dict], best_val_metric: dict) -> int | None:
    """Locate the epoch whose val 'overall' metric matches best_val_metric's value (first match)."""
    metric_name = best_val_metric.get("name")
    target_value = best_val_metric.get("value")
    if metric_name is None or target_value is None:
        return None
    for entry in history:
        overall = entry.get("val_metrics", {}).get("overall", {})
        if metric_name in overall and abs(overall[metric_name] - target_value) < 1e-9:
            return entry["epoch"]
    return None


# ── Plot 1: training curves ─────────────────────────────────────────────────

def plot_training_curves(results: dict, out_path: Path):
    history = results.get("history", [])
    if not history:
        logger.warning("No 'history' found in results — skipping training_curves.png")
        return

    available_subsets = [
        s for s in _SUBSET_ORDER
        if any(s in entry.get("val_metrics", {}) for entry in history)
    ]
    if not available_subsets:
        logger.warning("No val_metrics found in any epoch — skipping training_curves.png")
        return

    best_epoch = _find_best_epoch(history, results.get("best_val_metric", {}))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Training curves — {results.get('run_name', 'run')}", fontsize=13, fontweight="bold")

    # ── Top-left: train_loss & val_loss ──────────────────────────────────
    ax = axes[0, 0]
    epochs = [e["epoch"] for e in history]
    train_loss = [e["train_loss"] for e in history]
    ax.plot(epochs, train_loss, color="#555555", linewidth=1.8, marker="o", markersize=3, label="train")

    # val_loss may be entirely absent (older metrics.json from before this
    # was tracked) or None per-epoch (no val split configured) — handle both
    # without crashing or plotting a broken/empty line.
    val_loss_epochs, val_loss_vals = [], []
    for e in history:
        vl = e.get("val_loss")
        if vl is not None:
            val_loss_epochs.append(e["epoch"])
            val_loss_vals.append(vl)
    if val_loss_epochs:
        ax.plot(val_loss_epochs, val_loss_vals, color="#D95F02", linewidth=1.8,
                marker="s", markersize=3, label="val")
        ax.legend(fontsize=8)
    else:
        logger.info("No val_loss data found in history — plotting train_loss only "
                    "(older metrics.json, or no val split configured for this run)")

    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)

    # ── Top-right: ICBHI score, accuracy, sensitivity, specificity ──────────
    ax = axes[0, 1]
    any_plotted = False
    for subset in available_subsets:
        color = _SUBSET_COLORS[subset]
        ep, vals = _extract_series(history, subset, "icbhi_score")
        if ep:
            ax.plot(ep, vals, color=color, linestyle="-", marker="o", markersize=3,
                    linewidth=1.8, label=f"{subset} icbhi_score")
            any_plotted = True
        ep, vals = _extract_series(history, subset, "accuracy")
        if ep:
            ax.plot(ep, vals, color=color, linestyle="--", marker="^", markersize=3,
                    linewidth=1.8, label=f"{subset} accuracy")
            any_plotted = True
        # Se/Sp plotted thinner, no markers, and partly transparent so they
        # recede behind icbhi_score/accuracy rather than competing for
        # attention — they're supporting detail (icbhi_score IS (se+sp)/2),
        # not the primary signal this panel is for.
        ep, vals = _extract_series(history, subset, "se")
        if ep:
            ax.plot(ep, vals, color=color, linestyle=":", linewidth=1.1, alpha=0.55,
                    label=f"{subset} sensitivity")
            any_plotted = True
        ep, vals = _extract_series(history, subset, "sp")
        if ep:
            ax.plot(ep, vals, color=color, linestyle="-.", linewidth=1.1, alpha=0.55,
                    label=f"{subset} specificity")
            any_plotted = True
    if not any_plotted:
        ax.text(0.5, 0.5, "No icbhi_score/accuracy/se/sp data", ha="center", va="center", transform=ax.transAxes)
        logger.info("No icbhi_score/accuracy/se/sp series found for any subset (e.g. label_scheme='fine' has none of these)")
    ax.set_title("Validation: ICBHI score, accuracy, Se & Sp")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=6.5, loc="lower right", ncol=2 if len(available_subsets) > 1 else 1)
    ax.grid(alpha=0.3)

    # ── Bottom-left: F1 macro/weighted ─────────────────────────────────────
    ax = axes[1, 0]
    any_plotted = False
    for subset in available_subsets:
        color = _SUBSET_COLORS[subset]
        ep, vals = _extract_series(history, subset, "f1_macro")
        if ep:
            ax.plot(ep, vals, color=color, linestyle="-", marker="o", markersize=3,
                    label=f"{subset} f1_macro")
            any_plotted = True
        ep, vals = _extract_series(history, subset, "f1_weighted")
        if ep:
            ax.plot(ep, vals, color=color, linestyle="--", marker="^", markersize=3,
                    label=f"{subset} f1_weighted")
            any_plotted = True
    if not any_plotted:
        ax.text(0.5, 0.5, "No F1 data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Validation: F1 (macro & weighted)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    # ── Bottom-right: learning rate ─────────────────────────────────────────
    ax = axes[1, 1]
    lrs = [e.get("lr") for e in history]
    if any(lr is not None for lr in lrs):
        ax.plot(epochs, lrs, color="#888888", linewidth=1.8, marker="o", markersize=3)
        ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "No LR data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Learning rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR (log scale)")
    ax.grid(alpha=0.3)

    # Mark best-checkpoint epoch on all four subplots
    if best_epoch is not None:
        for ax in axes.flat:
            ax.axvline(best_epoch, color="red", linestyle=":", linewidth=1.3, alpha=0.7)
        axes[0, 0].text(
            best_epoch, axes[0, 0].get_ylim()[1], "  best ckpt", color="red",
            fontsize=8, va="top", ha="left",
        )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


# ── Plot 2: final test metrics bar chart ────────────────────────────────────

def plot_test_metrics_bars(results: dict, out_path: Path):
    test_metrics = results.get("test_metrics", {})
    if not test_metrics:
        logger.warning("No 'test_metrics' found in results — skipping test_metrics_bars.png")
        return

    available_subsets = [s for s in _SUBSET_ORDER if s in test_metrics]
    if not available_subsets:
        logger.warning("test_metrics present but empty — skipping test_metrics_bars.png")
        return

    metric_names = ["accuracy", "f1_macro", "f1_weighted", "icbhi_score", "se", "sp"]
    _DISPLAY_LABEL = {"se": "sensitivity", "sp": "specificity"}
    # Only plot metrics that exist in at least one subset
    metric_names = [m for m in metric_names if any(m in test_metrics[s] for s in available_subsets)]
    if not metric_names:
        logger.warning("No plottable metrics found in test_metrics — skipping test_metrics_bars.png")
        return

    n_metrics = len(metric_names)
    n_subsets = len(available_subsets)
    x = np.arange(n_metrics)
    width = 0.8 / n_subsets

    fig, ax = plt.subplots(figsize=(max(6, n_metrics * 1.6), 5))
    for i, subset in enumerate(available_subsets):
        values = [test_metrics[subset].get(m, np.nan) for m in metric_names]
        offset = (i - (n_subsets - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=subset, color=_SUBSET_COLORS[subset])
        for bar, v in zip(bars, values):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([_DISPLAY_LABEL.get(m, m) for m in metric_names])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(f"Final test metrics — {results.get('run_name', 'run')}")
    if n_subsets > 1:
        ax.legend(title="Subset", loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


# ── Plot 3: confusion matrix heatmap ────────────────────────────────────────

def plot_confusion_matrix(results: dict, out_path: Path):
    test_overall = results.get("test_metrics", {}).get("overall", {})
    cm = test_overall.get("confusion_matrix")
    class_list = test_overall.get("class_list")

    if cm is None or class_list is None:
        logger.warning("No confusion_matrix/class_list found in test_metrics['overall'] — skipping confusion_matrix.png")
        return

    cm = np.array(cm)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid divide-by-zero for classes absent from test data
    cm_pct = cm / row_sums * 100

    fig, ax = plt.subplots(figsize=(max(5, len(class_list) * 1.3), max(4.5, len(class_list) * 1.1)))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)

    ax.set_xticks(range(len(class_list)))
    ax.set_yticks(range(len(class_list)))
    ax.set_xticklabels(class_list, rotation=45, ha="right")
    ax.set_yticklabels(class_list)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (test, overall) — {results.get('run_name', 'run')}")

    for i in range(len(class_list)):
        for j in range(len(class_list)):
            text_color = "white" if cm_pct[i, j] > 50 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                    ha="center", va="center", color=text_color, fontsize=9)

    fig.colorbar(im, ax=ax, label="Row-normalised %")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate archival plots from a training run's metrics.json.")
    parser.add_argument("--run-name", default=None,
                        help="Run name — looks for results/<run-name>/metrics.json (or <output_dir>/<run-name>/metrics.json if --output-dir given)")
    parser.add_argument("--output-dir", default="results",
                        help="Base output dir used by train.py (default: results). Only used with --run-name.")
    parser.add_argument("--metrics-json", default=None,
                        help="Direct path to a metrics.json file (overrides --run-name/--output-dir)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if args.metrics_json:
        metrics_json_path = Path(args.metrics_json)
    elif args.run_name:
        metrics_json_path = Path(args.output_dir) / args.run_name / "metrics.json"
    else:
        parser.error("Provide either --run-name or --metrics-json")
        return

    if not metrics_json_path.exists():
        logger.error("metrics.json not found at %s", metrics_json_path)
        raise SystemExit(1)

    results = load_results(metrics_json_path)
    run_dir = metrics_json_path.parent
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_training_curves(results, plots_dir / "training_curves.png")
    plot_test_metrics_bars(results, plots_dir / "test_metrics_bars.png")
    plot_confusion_matrix(results, plots_dir / "confusion_matrix.png")

    logger.info("Done. Plots saved under %s", plots_dir)


if __name__ == "__main__":
    main()