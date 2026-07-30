"""
models.py
---------
Model registry for the classification pipeline. Each entry is a builder
function that takes a config dict and returns a torch.nn.Module ready for
training — input: a batch of spectrograms (B, freq, time) or (B, 1, freq,
time); output: raw logits (B, n_classes).

Adding a new architecture later: write a build_<name>(cfg, n_classes) -> nn.Module
function and add it to MODEL_REGISTRY. Nothing else in the training pipeline
needs to change — train.py only ever calls build_model(cfg, n_classes).

Currently registered: "resnet18" and "vgg11" (pretrained ImageNet backbones
+ the shared SpectrogramToRGB adapter) and "simple_cnn" (small CNN built
natively for spectrograms, no adapter). See each builder's docstring for
config options and design rationale.

── ResNet-18 + spectrogram adapter ──────────────────────────────────────────
torchvision's resnet18 expects (B, 3, H, W) ImageNet-style input. Our cached
features are single-channel (B, freq, time) with arbitrary aspect ratio and
log-power/dB value ranges, not 0-255 pixel intensities. SpectrogramToRGB
bridges this gap:

  1. Add a channel dim if missing: (B, freq, time) -> (B, 1, freq, time)
  2. Per-sample min-max normalise to roughly [0, 1] (spectrograms have very
     different absolute scales per feature type/config; this puts them in a
     comparable range to natural images before ImageNet normalisation)
  3. Replicate 1 -> 3 channels (rather than replacing the first conv layer —
     this keeps ALL pretrained weights intact, including the most broadly
     transferable low-level filters, at the cost of the 3 channels being
     identical rather than carrying distinct color information, which
     spectrograms don't have anyway)
  4. Resize (bilinear) to the backbone's expected input size (224x224 for
     standard torchvision ResNet-18 pretrained weights)
  5. Apply ImageNet channel-wise normalisation (mean/std), since the
     pretrained weights expect inputs in that distribution

This adapter is a separate nn.Module (not baked into preprocessing) so it
runs on-GPU as part of the forward pass and so swapping it out for a
different adaptation strategy later is a one-line change in build_resnet18,
not a pipeline-wide change.
"""
from __future__ import annotations
import logging
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Standard ImageNet normalisation stats — what torchvision's pretrained
# ResNet-18 weights were trained with.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class SpectrogramToRGB(nn.Module):
    """
    Adapts a single-channel spectrogram batch to 3-channel, ImageNet-normalised,
    backbone-resolution input. See module docstring for the full rationale.

    Parameters
    ----------
    target_size : (H, W) the backbone expects. Default (224, 224) for
                  standard torchvision ResNet-18.
    """

    def __init__(self, target_size: tuple[int, int] = (224, 224)):
        super().__init__()
        self.target_size = target_size
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, freq, time) -> (B, 1, freq, time)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() != 4:
            raise ValueError(f"Expected 3D (B,F,T) or 4D (B,1,F,T) input, got shape {tuple(x.shape)}")

        # Per-sample min-max normalise to [0, 1]. Computed per-sample (not
        # per-batch) since different cycles/feature configs can have very
        # different absolute dB/power ranges.
        b = x.shape[0]
        flat = x.view(b, -1)
        x_min = flat.min(dim=1).values.view(b, 1, 1, 1)
        x_max = flat.max(dim=1).values.view(b, 1, 1, 1)
        denom = (x_max - x_min).clamp(min=1e-6)
        x = (x - x_min) / denom

        # Replicate to 3 channels
        x = x.repeat(1, 3, 1, 1)

        # Resize to backbone input resolution
        x = F.interpolate(x, size=self.target_size, mode="bilinear", align_corners=False)

        # ImageNet normalisation
        x = (x - self.mean) / self.std

        return x


def build_resnet18(cfg: dict, n_classes: int) -> nn.Module:
    """
    Build a ResNet-18 classifier with a spectrogram input adapter.

    Config keys (all under cfg["model"] in the experiment YAML):
        pretrained    : bool, default True. Load ImageNet-pretrained weights.
                        Requires internet access on first run (torchvision
                        downloads and caches weights under ~/.cache/torch).
        freeze_backbone : bool, default False.
                        True  -> linear probe: backbone weights frozen, only
                                 the final FC layer (and the adapter, which
                                 has no learnable params) trains. Fast,
                                 lower overfitting risk on small datasets.
                        False -> full fine-tune: entire network trains.
        input_size    : [H, W], default [224, 224]. Resize target fed to the
                        backbone. 224 matches standard pretrained weights;
                        changing this is allowed but moves off-distribution
                        from what the weights were trained on.
        dropout       : float, default 0.0 (disabled). Standard torchvision
                        ResNet-18 has NO dropout anywhere — unlike VGG-11,
                        there's nothing pre-existing to override. Setting
                        this > 0 inserts a new nn.Dropout right before the
                        final Linear(512, n_classes) layer (applied to the
                        512-d pooled feature vector), the standard place to
                        add it to a ResNet head. Always trains regardless of
                        freeze_backbone, same as the fc layer itself.

    Returns
    -------
    nn.Module — calling model(batch) where batch is (B, freq, time) or
    (B, 1, freq, time) returns (B, n_classes) logits.
    """
    import torchvision
    from torchvision.models import resnet18, ResNet18_Weights

    model_cfg = cfg.get("model", {})
    pretrained = model_cfg.get("pretrained", True)
    freeze_backbone = model_cfg.get("freeze_backbone", False)
    input_size = tuple(model_cfg.get("input_size", [224, 224]))
    dropout = model_cfg.get("dropout", 0.0)

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    try:
        backbone = resnet18(weights=weights)
    except Exception as e:
        if pretrained:
            logger.warning(
                "Failed to download/load pretrained ResNet-18 weights (%s). "
                "Falling back to random initialisation. Check internet access "
                "if this is unexpected — torchvision needs to reach "
                "download.pytorch.org on first use.", e,
            )
            backbone = resnet18(weights=None)
        else:
            raise

    # Replace the final FC layer to match our class count. This layer is
    # ALWAYS trained from scratch (random init), regardless of freeze_backbone,
    # since the pretrained 1000-way ImageNet head has no meaning for our task.
    # If dropout > 0, insert it right before this layer — ResNet-18 has no
    # dropout of its own to override, so this is purely additive.
    in_features = backbone.fc.in_features
    if dropout > 0:
        backbone.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, n_classes))
    else:
        backbone.fc = nn.Linear(in_features, n_classes)

    if freeze_backbone:
        for name, param in backbone.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False
        logger.info(
            "ResNet-18: backbone frozen (linear probe mode) — only fc layer "
            "(%d params, dropout=%.2f) will train.",
            sum(p.numel() for p in backbone.fc.parameters()), dropout,
        )
    else:
        logger.info(
            "ResNet-18: full fine-tune mode — all %d params will train (dropout=%.2f).",
            sum(p.numel() for p in backbone.parameters()), dropout,
        )

    adapter = SpectrogramToRGB(target_size=input_size)

    return nn.Sequential(adapter, backbone)


def build_resnet34(cfg: dict, n_classes: int) -> nn.Module:
    """
    Build a ResNet-34 classifier with a spectrogram input adapter.

    Config keys (all under cfg["model"] in the experiment YAML):
        pretrained    : bool, default True. Load ImageNet-pretrained weights.
                        Requires internet access on first run (torchvision
                        downloads and caches weights under ~/.cache/torch).
        freeze_backbone : bool, default False.
                        True  -> linear probe: backbone weights frozen, only
                                 the final FC layer (and the adapter, which
                                 has no learnable params) trains. Fast,
                                 lower overfitting risk on small datasets.
                        False -> full fine-tune: entire network trains.
        input_size    : [H, W], default [224, 224]. Resize target fed to the
                        backbone. 224 matches standard pretrained weights;
                        changing this is allowed but moves off-distribution
                        from what the weights were trained on.

    Returns
    -------
    nn.Module — calling model(batch) where batch is (B, freq, time) or
    (B, 1, freq, time) returns (B, n_classes) logits.
    """
    import torchvision
    from torchvision.models import resnet34, ResNet34_Weights

    model_cfg = cfg.get("model", {})
    pretrained = model_cfg.get("pretrained", True)
    freeze_backbone = model_cfg.get("freeze_backbone", False)
    input_size = tuple(model_cfg.get("input_size", [224, 224]))

    weights = ResNet34_Weights.DEFAULT if pretrained else None
    try:
        backbone = resnet34(weights=weights)
    except Exception as e:
        if pretrained:
            logger.warning(
                "Failed to download/load pretrained ResNet-34 weights (%s). "
                "Falling back to random initialisation. Check internet access "
                "if this is unexpected — torchvision needs to reach "
                "download.pytorch.org on first use.", e,
            )
            backbone = resnet34(weights=None)
        else:
            raise

    # Replace the final FC layer to match our class count. This layer is
    # ALWAYS trained from scratch (random init), regardless of freeze_backbone,
    # since the pretrained 1000-way ImageNet head has no meaning for our task.
    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, n_classes)

    if freeze_backbone:
        for name, param in backbone.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False
        logger.info(
            "ResNet-34: backbone frozen (linear probe mode) — only fc layer "
            "(%d params) will train.", sum(p.numel() for p in backbone.fc.parameters()),
        )
    else:
        logger.info(
            "ResNet-34: full fine-tune mode — all %d params will train.",
            sum(p.numel() for p in backbone.parameters()),
        )

    adapter = SpectrogramToRGB(target_size=input_size)

    return nn.Sequential(adapter, backbone)


def build_vgg11(cfg: dict, n_classes: int) -> nn.Module:
    """
    Build a VGG-11 classifier with the same spectrogram-to-RGB adapter used
    by ResNet-18 (VGG-11 has the identical (B, 3, H, W) ImageNet input
    contract, so no new adapter is needed).

    Config keys (all under cfg["model"] in the experiment YAML):
        pretrained      : bool, default True. Load ImageNet-pretrained
                          weights. Requires internet access on first run.
        freeze_backbone : bool, default False.
                          True  -> linear probe: the convolutional feature
                                   extractor (backbone.features +
                                   backbone.avgpool) is frozen; the entire
                                   classifier head (backbone.classifier,
                                   three Linear layers) trains.
                          False -> full fine-tune: entire network trains.
        input_size      : [H, W], default [224, 224]. Resize target fed to
                          the backbone — same role as in ResNet-18.
        dropout         : float or null, default null (leave as-is).
                          Unlike ResNet-18, torchvision's VGG-11 ALREADY has
                          two nn.Dropout(p=0.5) layers built into its
                          classifier (classifier[2] and classifier[5], each
                          between the 4096-wide Linear layers) — this is
                          standard VGG architecture, not something this repo
                          added, and it's what the paper's own VGG-11
                          inherits too ("the only difference from the
                          original model lies on the softmax layer"). The
                          default (null) leaves both at their pretrained
                          p=0.5, matching the paper. Setting dropout to a
                          float overrides BOTH layers' p to that value —
                          e.g. dropout: 0.3 to reduce regularisation
                          strength, or dropout: 0.0 to disable dropout
                          entirely and compare against ResNet-18's default.

    Architectural note: VGG-11's classifier is a 3-layer Sequential
    (4096 -> 4096 -> 1000 for ImageNet), unlike ResNet-18's single fc layer.
    Only the FINAL layer (classifier[6], 4096 -> 1000) is ImageNet-specific
    and gets replaced; the two preceding 4096-wide layers are general-purpose
    feature transformations and are kept (with their pretrained weights, if
    pretrained=True) — replacing only the head, not the whole classifier,
    preserves more of what pretraining learned. This also means VGG-11 has
    far more parameters in its head than ResNet-18 (the two retained 4096-wide
    layers are ~16.8M params alone), which is worth knowing when comparing
    architectures: a like-for-like "linear probe" here trains a much larger
    head than ResNet-18's linear probe does.

    Returns
    -------
    nn.Module — calling model(batch) where batch is (B, freq, time) or
    (B, 1, freq, time) returns (B, n_classes) logits.
    """
    from torchvision.models import vgg11, VGG11_Weights

    model_cfg = cfg.get("model", {})
    pretrained = model_cfg.get("pretrained", True)
    freeze_backbone = model_cfg.get("freeze_backbone", False)
    input_size = tuple(model_cfg.get("input_size", [224, 224]))
    dropout = model_cfg.get("dropout")  # None = leave torchvision's default p=0.5 as-is

    weights = VGG11_Weights.DEFAULT if pretrained else None
    try:
        backbone = vgg11(weights=weights)
    except Exception as e:
        if pretrained:
            logger.warning(
                "Failed to download/load pretrained VGG-11 weights (%s). "
                "Falling back to random initialisation. Check internet access "
                "if this is unexpected — torchvision needs to reach "
                "download.pytorch.org on first use.", e,
            )
            backbone = vgg11(weights=None)
        else:
            raise

    # Replace only the FINAL classifier layer (classifier[6]: 4096 -> 1000)
    # to match our class count. The two preceding 4096-wide Linear layers
    # are kept as-is (including their pretrained weights) since they're
    # general-purpose feature transformations, not ImageNet-class-specific.
    in_features = backbone.classifier[6].in_features
    backbone.classifier[6] = nn.Linear(in_features, n_classes)

    # Override the two existing Dropout layers' p if requested. indices 2
    # and 5 are architectural constants of torchvision's VGG — see the
    # printed Sequential in this function's docstring history / torchvision
    # source, not something that shifts with n_classes or pretrained.
    if dropout is not None:
        backbone.classifier[2] = nn.Dropout(p=dropout)
        backbone.classifier[5] = nn.Dropout(p=dropout)

    if freeze_backbone:
        # "Backbone" = convolutional feature extractor only (features +
        # avgpool). The entire classifier Sequential (all three Linear
        # layers, not just the replaced final one) stays trainable — VGG's
        # classifier head is large enough that "linear probe" here still
        # means training a substantial 3-layer MLP, not a single layer.
        for name, param in backbone.named_parameters():
            if not name.startswith("classifier."):
                param.requires_grad = False
        trainable = sum(p.numel() for p in backbone.classifier.parameters())
        logger.info(
            "VGG-11: backbone (features+avgpool) frozen (linear probe mode) — "
            "classifier head (%d params across 3 Linear layers, dropout=%s) will train.",
            trainable, "0.5 (default)" if dropout is None else f"{dropout:.2f}",
        )
    else:
        logger.info(
            "VGG-11: full fine-tune mode — all %d params will train (dropout=%s).",
            sum(p.numel() for p in backbone.parameters()),
            "0.5 (default)" if dropout is None else f"{dropout:.2f}",
        )

    adapter = SpectrogramToRGB(target_size=input_size)

    return nn.Sequential(adapter, backbone)


class SimpleSpectrogramCNN(nn.Module):
    """
    A small CNN built natively for spectrogram input — no resizing, no
    3-channel replication, no ImageNet normalisation. Operates directly on
    whatever (freq, time) shape your features.yaml config produces.

    Architecture: a stack of conv blocks (Conv2d -> BatchNorm -> ReLU ->
    MaxPool), doubling channel width each block, followed by adaptive average
    pooling to collapse the spatial dimensions to a fixed size REGARDLESS of
    input shape (so the same model works whether your logmel cache is
    128x501 or 64x250 or anything else — no hardcoded spatial assumptions),
    then a small classifier head.

    This is meaningfully different from the ResNet-18 path: it has far fewer
    parameters (faster to train, less prone to overfitting on a dataset this
    size), no pretrained-weight dependency (no internet access needed, no
    ImageNet-domain-mismatch question), and it sees the spectrogram at its
    native aspect ratio rather than squashed to a square — comparing the two
    tells you whether the pretrained-and-adapted approach is actually buying
    you anything over a model designed for the real input shape.

    Parameters (read from cfg["model"] in the experiment YAML)
    ----------
    n_blocks       : int, default 4. Number of conv blocks. Each block halves
                     both spatial dimensions (via MaxPool2d(2)) and doubles
                     channel width starting from base_channels.
    base_channels  : int, default 32. Channel width of the first conv block;
                     block i has base_channels * 2**i channels.
    dropout        : float, default 0.3. Applied before the final linear layer.
    """

    def __init__(self, n_classes: int, n_blocks: int = 4, base_channels: int = 32, dropout: float = 0.3):
        super().__init__()

        blocks = []
        in_channels = 1
        out_channels = base_channels
        for _ in range(n_blocks):
            blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ))
            in_channels = out_channels
            out_channels *= 2

        self.conv_blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # collapses (C, H, W) -> (C, 1, 1) regardless of input H, W
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_channels, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, freq, time) -> (B, 1, freq, time)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() != 4:
            raise ValueError(f"Expected 3D (B,F,T) or 4D (B,1,F,T) input, got shape {tuple(x.shape)}")

        x = self.conv_blocks(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def build_simple_cnn(cfg: dict, n_classes: int) -> nn.Module:
    """
    Build a SimpleSpectrogramCNN from config.

    Config keys (under cfg["model"] in the experiment YAML):
        n_blocks       : int, default 4
        base_channels  : int, default 32
        dropout        : float, default 0.3

    No pretrained weights, no freeze_backbone option (the whole thing always
    trains — there's no pretrained portion to optionally freeze) and no
    internet dependency.
    """
    model_cfg = cfg.get("model", {})
    n_blocks = model_cfg.get("n_blocks", 4)
    base_channels = model_cfg.get("base_channels", 32)
    dropout = model_cfg.get("dropout", 0.3)

    model = SimpleSpectrogramCNN(
        n_classes=n_classes, n_blocks=n_blocks, base_channels=base_channels, dropout=dropout,
    )
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("SimpleSpectrogramCNN: built with %d blocks, base_channels=%d — %d total params (all trainable).",
                n_blocks, base_channels, total_params)
    return model


# ── Registry ─────────────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, Callable[[dict, int], nn.Module]] = {
    "resnet18": build_resnet18,
    "resnet34": build_resnet34,
    "vgg11": build_vgg11,
    "simple_cnn": build_simple_cnn,
}


def build_model(cfg: dict, n_classes: int) -> nn.Module:
    """
    Dispatch to the appropriate model builder by name.

    Parameters
    ----------
    cfg       : full experiment config dict; reads cfg["model"]["name"]
    n_classes : number of output classes (determines final layer size)

    Returns
    -------
    nn.Module, ready for training.
    """
    model_name = cfg.get("model", {}).get("name")
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model name {model_name!r}. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name](cfg, n_classes)