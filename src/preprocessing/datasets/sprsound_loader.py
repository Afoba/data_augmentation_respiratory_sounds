"""
sprsound_loader.py
------------------
Parses SPRSound (BioCAS2022 release) annotation JSON files into unified
CycleRecord dicts. Each CycleRecord matches the same schema as
icbhi_loader.py so that downstream preprocessing and feature extraction
code is fully dataset-agnostic.

Expected raw directory layout (see config/sprsound.yaml for the authoritative
list, since directory names are configurable). TWO physical layouts are
supported, auto-detected at load time:

    Layout (a) — combined test set with separate subset-tag folders:
        raw_dir/
            train2022_wav/            <recording_id>.wav
            train2022_json/           <recording_id>.json
            test2022_wav/             <recording_id>.wav
            test2022_json/            <recording_id>.json   (ALL test recordings)
            inter_test_json/          <recording_id>.json   (SUBSET: unseen patients)
            intra_test_json/          <recording_id>.json   (SUBSET: patients also in train)

    Layout (b) — inter/intra nested inside test2022_json/, no combined set:
        raw_dir/
            train2022_wav/            <recording_id>.wav
            train2022_json/           <recording_id>.json
            test2022_wav/             <recording_id>.wav
            test2022_json/
                inter_test_json/      <recording_id>.json   (SUBSET: unseen patients)
                intra_test_json/      <recording_id>.json   (SUBSET: patients also in train)
            (test2022_json/ has NO loose .json files directly inside it —
             inter_test_json/ + intra_test_json/ together ARE the test set)

The loader detects which layout is present by checking whether test_json_dir
contains loose .json files directly inside it (layout a) or only the two
subfolders (layout b — confirmed as the actual on-disk structure for some
dataset mirrors/downloads). In layout (a), inter/intra are used only to TAG
recordings already found in the combined test_json_dir. In layout (b), each
recording's subset is known directly from which folder it was read from —
no membership lookup needed, and no recording can fall into "unknown".

inter_test_json/ and intra_test_json/ contain ANNOTATION FILES ONLY in both
layouts — the corresponding audio always lives in test2022_wav/ and is
matched by filename.

Real SPRSound JSON schema (verified against the official documentation):
    {
        "recording_annotation": "Normal",   # record-level: Normal | CAS | DAS |
                                             # "CAS & DAS" | "Poor Quality"
        "event_annotation": [
            {"start": 342, "end": 2515, "type": "Normal"},   # start/end in MILLISECONDS
            ...
        ]
    }
There is NO separate "quality" field at the event level — poor-quality
recordings are signalled entirely through `recording_annotation == "Poor
Quality"`. The loader derives a normalised `quality` column from this
(`"poor_quality"` vs `""`), while also keeping the raw record-level category
in `record_label`.

Filename convention (from the official README):
    <patient_number>_<age>_<gender>_<location>_<recording_number>
e.g. 65101170_0.4_0_p1_3246  — these are parsed loosely; only the leading
patient_number token before the first underscore is required, and is used
as `patient_id` (useful context metadata; NOT used for grouping since
SPRSound's own intra/inter test split already encodes patient overlap —
see note in build_val_split.py guidance below).
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# Default coarse label mapping (overridden by config if provided)
_DEFAULT_COARSE_MAP = {
    "normal": "normal",
    "rhonchi": "wheeze",
    "wheeze": "wheeze",
    "stridor": "wheeze",
    "coarse_crackle": "crackle",
    "fine_crackle": "crackle",
    "wheeze_and_crackle": "both",
}

# Derived binary flags from the coarse label
_COARSE_TO_BINARY = {
    "normal":  (0, 0),
    "crackle": (1, 0),
    "wheeze":  (0, 1),
    "both":    (1, 1),
}


def load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _normalise_label(raw: str) -> str:
    """
    Lowercase and replace spaces/hyphens/ampersands/plus-signs with underscores.

    Real SPRSound labels use different separators at different levels:
      - record level uses " & " (e.g. "CAS & DAS")
      - event level uses "+" (e.g. "Wheeze+Crackle")
    Both must normalise to the same underscore convention used in
    coarse_label_map / fine_label_set, or label mapping silently fails
    (falls through to "unknown").
    """
    return (
        raw.strip().lower()
        .replace(" & ", "_and_")
        .replace("+", "_and_")
        .replace(" ", "_")
        .replace("-", "_")
    )


def _find_audio_file(wav_dir: Path, recording_id: str) -> Optional[Path]:
    for ext in (".wav", ".flac", ".mp3"):
        candidate = wav_dir / f"{recording_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def _list_recording_ids(json_dir: Path) -> set[str]:
    """Return the set of recording_ids (filename stems) present in a json dir."""
    if not json_dir.exists():
        return set()
    return {p.stem for p in json_dir.glob("*.json")}


def load_sprsound(config_path: str | Path) -> list[dict]:
    """
    Parse all SPRSound events into unified CycleRecord dicts.

    Parameters
    ----------
    config_path : path to config/sprsound.yaml

    Returns
    -------
    List of CycleRecord dicts (one per respiratory event/cycle). Test-split
    records additionally carry a `test_subset` field: "inter" | "intra" |
    "unknown" (the last only for test recordings absent from BOTH official
    subset directories, which would indicate a non-standard dataset copy).
    Train-split records have `test_subset = None`.
    """
    cfg = load_config(config_path)
    raw_dir = Path(cfg["raw_dir"]).expanduser().resolve()

    jf = cfg.get("json_fields", {})
    field_label  = jf.get("event_label",  "type")
    field_start  = jf.get("event_start",  "start")
    field_end    = jf.get("event_end",    "end")
    field_reclab = jf.get("record_label", "recording_annotation")
    time_unit    = jf.get("event_time_unit", "ms")
    time_scale   = 0.001 if time_unit == "ms" else 1.0

    poor_quality_values = {
        _normalise_label(v) for v in cfg.get("poor_quality_values", ["poor_quality"])
    }
    coarse_map = cfg.get("coarse_label_map", _DEFAULT_COARSE_MAP)

    # Resolve directory layout.
    #
    # The official BioCAS2022 release has been observed in TWO different
    # physical layouts across download mirrors/versions:
    #   (a) inter_test_json/ and intra_test_json/ as SIBLINGS of test2022_json/
    #       under raw_dir, with test2022_json/ itself also containing the
    #       full combined set of loose per-recording .json files.
    #   (b) inter_test_json/ and intra_test_json/ NESTED INSIDE test2022_json/,
    #       with test2022_json/ containing ONLY those two subfolders and no
    #       loose files directly inside it.
    # Layout (b) has no separate "combined" file set at all — inter+intra
    # together ARE the complete test set. We detect which layout is present
    # and resolve accordingly, rather than assuming (a) and silently loading
    # zero test events when (b) is what's actually on disk.
    train_wav_dir  = raw_dir / cfg.get("train_wav_dir",  "train2022_wav")
    train_json_dir = raw_dir / cfg.get("train_json_dir", "train2022_json")
    test_wav_dir   = raw_dir / cfg.get("test_wav_dir",   "test2022_wav")
    test_json_dir  = raw_dir / cfg.get("test_json_dir",  "test2022_json")

    inter_dirname = cfg.get("inter_test_json_dir", "inter_test_json")
    intra_dirname = cfg.get("intra_test_json_dir", "intra_test_json")

    # Try sibling-of-raw_dir first (layout a / as originally configured),
    # then nested-under-test_json_dir (layout b — matches the structure
    # confirmed on disk for this project).
    inter_json_dir = raw_dir / inter_dirname
    intra_json_dir = raw_dir / intra_dirname
    if not inter_json_dir.exists() and (test_json_dir / inter_dirname).exists():
        inter_json_dir = test_json_dir / inter_dirname
    if not intra_json_dir.exists() and (test_json_dir / intra_dirname).exists():
        intra_json_dir = test_json_dir / intra_dirname

    inter_ids = _list_recording_ids(inter_json_dir)
    intra_ids = _list_recording_ids(intra_json_dir)
    overlap = inter_ids & intra_ids
    if overlap:
        logger.warning(
            "SPRSound: %d recording(s) appear in BOTH inter_test_json and "
            "intra_test_json — this should not happen; tagging as 'intra' "
            "(arbitrary tie-break). Recordings: %s",
            len(overlap), sorted(overlap)[:5],
        )

    # Does test_json_dir itself contain loose .json files (layout a), or only
    # the two subfolders (layout b)? Glob is non-recursive, so this correctly
    # distinguishes the two cases.
    test_json_dir_has_loose_files = bool(list(test_json_dir.glob("*.json"))) if test_json_dir.exists() else False

    if test_json_dir_has_loose_files:
        # Layout (a): test_json_dir is the combined source of truth; inter/intra
        # are used only to TAG recordings already found there (existing behaviour).
        effective_test_json_dir = test_json_dir
        logger.info(
            "SPRSound: using %s as the combined test-event source "
            "(loose .json files found directly inside it).", test_json_dir,
        )
    else:
        # Layout (b): no combined file set exists — inter_test_json/ and
        # intra_test_json/ together ARE the test set. We merge them into a
        # single synthetic listing below rather than reading test_json_dir.
        effective_test_json_dir = None
        logger.info(
            "SPRSound: %s contains no loose .json files directly inside it — "
            "treating inter_test_json/ (%d files) + intra_test_json/ (%d files) "
            "as the complete test set instead.",
            test_json_dir, len(inter_ids), len(intra_ids),
        )

    records: list[dict] = []
    missing_audio: list[str] = []
    unknown_labels: set[str] = set()

    if effective_test_json_dir is not None:
        split_dirs = [
            ("train", train_wav_dir, train_json_dir, None),
            ("test",  test_wav_dir,  effective_test_json_dir, None),
        ]
    else:
        # Layout (b): two separate test directories, each with a KNOWN subset
        # tag from the start (no membership-matching needed/possible).
        split_dirs = [
            ("train", train_wav_dir, train_json_dir, None),
            ("test",  test_wav_dir,  inter_json_dir,  "inter"),
            ("test",  test_wav_dir,  intra_json_dir,  "intra"),
        ]

    for split, wav_dir, json_dir, forced_subset in split_dirs:
        if not json_dir.exists():
            logger.warning("SPRSound %s annotation directory not found: %s — skipping", split, json_dir)
            continue

        json_files = sorted(json_dir.glob("*.json"))
        for json_path in json_files:
            recording_id = json_path.stem

            audio_path = _find_audio_file(wav_dir, recording_id)
            if audio_path is None:
                missing_audio.append(str(json_path))
                continue

            try:
                with open(json_path) as f:
                    annotation = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.error("Failed to parse %s: %s — skipping", json_path, e)
                continue

            record_label_raw = annotation.get(field_reclab, "")
            record_label = _normalise_label(record_label_raw) if record_label_raw else ""
            quality = record_label if record_label in poor_quality_values else ""

            events = annotation.get("event_annotation", [])

            # Determine intra/inter tag for test-split recordings. If the
            # caller already knows the subset (forced_subset, set when
            # reading directly from inter_test_json/ or intra_test_json/ —
            # layout (b)), use that directly rather than re-deriving it via
            # membership lookup, which is both redundant and, for layout (b),
            # impossible (recording_ids in inter/intra wouldn't be found in
            # a test_json_dir that has no loose files of its own).
            if split == "test":
                if forced_subset is not None:
                    test_subset = forced_subset
                elif recording_id in intra_ids:
                    test_subset = "intra"
                elif recording_id in inter_ids:
                    test_subset = "inter"
                else:
                    test_subset = "unknown"
            else:
                test_subset = None

            # Loosely parse patient_id as the leading underscore-delimited
            # token of the filename, per the documented naming convention.
            patient_id = recording_id.split("_")[0] if "_" in recording_id else recording_id

            for idx, event in enumerate(events):
                raw_label = event.get(field_label, "")
                label_fine = _normalise_label(raw_label)
                start_time = float(event.get(field_start, 0.0)) * time_scale
                end_time   = float(event.get(field_end,   0.0)) * time_scale

                if label_fine not in coarse_map:
                    unknown_labels.add(label_fine)
                    label_coarse = "unknown"
                else:
                    label_coarse = coarse_map[label_fine]

                crackle, wheeze = _COARSE_TO_BINARY.get(label_coarse, (0, 0))
                cycle_id = f"sprsound__{split}__{recording_id}__{idx:04d}"

                records.append(
                    {
                        # ── Universal fields ──────────────────────────────────
                        "cycle_id":       cycle_id,
                        "source_dataset": "sprsound",
                        "audio_path":     str(audio_path),
                        "start_time":     start_time,
                        "end_time":       end_time,
                        "duration":       end_time - start_time,
                        "label_fine":     label_fine,    # 7-class SPRSound label
                        "label_coarse":   label_coarse,  # 4-class ICBHI-compatible label
                        "label_4class":   label_coarse,  # alias for uniform access
                        "crackle":        crackle,
                        "wheeze":         wheeze,
                        "split":          split,
                        # ── SPRSound-specific metadata ────────────────────────
                        "record_label":   record_label,
                        "quality":        quality,
                        "recording_id":   recording_id,
                        "test_subset":    test_subset,   # "inter" | "intra" | "unknown" | None
                        # ICBHI-only fields left as None for schema consistency
                        "patient_id":     patient_id,
                        "session_id":     None,
                        "location":       None,
                        "mode":           None,
                        "device":         None,
                        "diagnosis":      None,
                        # ── Provenance fields (shared schema across real/augmented/synthetic) ──
                        "source_type": "real",
                        "generator": None,
                        "original_cycle_id": None,
                    }
                )

    if missing_audio:
        logger.warning(
            "SPRSound: %d JSON files had no matching audio file and were skipped.",
            len(missing_audio),
        )
    if unknown_labels:
        logger.warning(
            "SPRSound: encountered unrecognised event labels (coarse mapped to 'unknown'): %s",
            unknown_labels,
        )

    n_test = sum(1 for r in records if r["split"] == "test")
    n_inter = sum(1 for r in records if r.get("test_subset") == "inter")
    n_intra = sum(1 for r in records if r.get("test_subset") == "intra")
    n_unknown_subset = sum(1 for r in records if r.get("test_subset") == "unknown")
    logger.info(
        "SPRSound: loaded %d events total (test split: %d events — %d inter, %d intra, %d unknown-subset)",
        len(records), n_test, n_inter, n_intra, n_unknown_subset,
    )
    return records