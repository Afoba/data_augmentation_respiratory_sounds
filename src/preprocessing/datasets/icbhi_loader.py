"""
icbhi_loader.py
---------------
Parses ICBHI 2017 annotation files into a list of unified CycleRecord dicts.

Each CycleRecord has the following keys (all downstream code depends on these,
not on any ICBHI-specific field):

    cycle_id        str   unique identifier: "icbhi__<filename_stem>__<idx:04d>"
    source_dataset  str   always "icbhi"
    audio_path      str   absolute path to the recording wav/flac
    start_time      float cycle start time within the recording (seconds)
    end_time        float cycle end time within the recording (seconds)
    label_4class    str   "normal" | "crackle" | "wheeze" | "both"
    crackle         int   1 if crackle present, else 0  (binary)
    wheeze          int   1 if wheeze present, else 0   (binary)
    split           str   "train" | "test"
    # ICBHI-specific metadata (stored but not required by downstream code)
    patient_id      str
    session_id      str
    location        str   auscultation location code
    mode            str   "sc" (single channel) | "mc" (multichannel)
    device          str   recording device name
    diagnosis       str   patient diagnosis (from ICBHI_Challenge_diagnosis.txt)
"""
from __future__ import annotations
import os
import re
import logging
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# Filename pattern: <pid>_<session>_<location>_<mode>_<device>
# Real ICBHI session tokens are alphanumeric (e.g. "1b1", "2p2"), not purely
# numeric, so session uses \w+ rather than \d+.
_FILENAME_RE = re.compile(
    r"^(?P<pid>\d+)_(?P<session>\w+)_(?P<location>[A-Za-z]+)_"
    r"(?P<mode>sc|mc)_(?P<device>.+)$"
)

_LABEL_MAP = {
    (0, 0): "normal",
    (1, 0): "crackle",
    (0, 1): "wheeze",
    (1, 1): "both",
}


def load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _find_audio_file(raw_dir: Path, stem: str, extensions: list[str]) -> Optional[Path]:
    for ext in extensions:
        candidate = raw_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _parse_annotation_file(txt_path: Path) -> list[dict]:
    """Parse one ICBHI .txt annotation file into a list of raw cycle dicts."""
    cycles = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                logger.warning("Skipping malformed annotation line in %s: %r", txt_path, line)
                continue
            start, end, crackle, wheeze = (
                float(parts[0]),
                float(parts[1]),
                int(parts[2]),
                int(parts[3]),
            )
            cycles.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "crackle": crackle,
                    "wheeze": wheeze,
                    "label_4class": _LABEL_MAP[(crackle, wheeze)],
                }
            )
    return cycles


def _resolve_case_insensitive(directory: Path, filename: str) -> Optional[Path]:
    """
    Resolve `filename` inside `directory`, tolerating case differences.
    Tries an exact match first (fast path), then falls back to a
    case-insensitive scan of the directory. Returns None if no match is found.
    """
    exact = directory / filename
    if exact.exists():
        return exact

    target_lower = filename.lower()
    if not directory.exists():
        return None
    for candidate in directory.iterdir():
        if candidate.is_file() and candidate.name.lower() == target_lower:
            return candidate
    return None


def _load_split_map(split_file: Path) -> dict[str, str]:
    """Return {filename_stem: 'train'|'test'} from the ICBHI split file."""
    split_map = {}
    with open(split_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                stem = Path(parts[0]).stem  # strip extension if present
                split_map[stem] = parts[1].lower()
    return split_map


def _load_diagnosis_map(diag_file: Path) -> dict[str, str]:
    """Return {patient_id: diagnosis} from ICBHI_Challenge_diagnosis.txt."""
    diag_map = {}
    with open(diag_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                diag_map[parts[0].strip()] = parts[1].strip()
    return diag_map


def load_icbhi(config_path: str | Path) -> list[dict]:
    """
    Parse all ICBHI cycles into unified CycleRecord dicts.

    Parameters
    ----------
    config_path : path to config/icbhi.yaml

    Returns
    -------
    List of CycleRecord dicts (one per respiratory cycle).
    """
    cfg = load_config(config_path)
    raw_dir = Path(cfg["raw_dir"]).expanduser().resolve()
    extensions = cfg.get("audio_extensions", [".wav", ".flac"])

    # Load auxiliary maps. File lookup is case-insensitive: different ICBHI
    # release copies use inconsistent casing for these filenames (e.g.
    # "ICBHI_Challenge_train_test.txt" vs "ICBHI_challenge_train_test.txt"),
    # and relying on an exact case match in cfg would silently produce
    # split="unknown" for every cycle if the casing happens to differ.
    split_file = _resolve_case_insensitive(raw_dir, cfg["split_file"])
    diag_file = _resolve_case_insensitive(raw_dir, cfg["diagnosis_file"])

    split_map = _load_split_map(split_file) if split_file and split_file.exists() else {}
    diag_map = _load_diagnosis_map(diag_file) if diag_file and diag_file.exists() else {}

    if not split_map:
        logger.warning("Split file not found at %s — all cycles will have split=unknown", raw_dir / cfg["split_file"])
    if not diag_map:
        logger.warning("Diagnosis file not found at %s — diagnosis field will be empty", raw_dir / cfg["diagnosis_file"])

    records: list[dict] = []

    # Iterate over all annotation txt files
    txt_files = sorted(raw_dir.glob("*.txt"))
    # Exclude known non-annotation files (the resolved split/diagnosis files,
    # using whatever casing they actually have on disk) — comparison is
    # case-insensitive as a second layer of safety.
    skip_stems = {
        (split_file.stem.lower() if split_file else Path(cfg["split_file"]).stem.lower()),
        (diag_file.stem.lower() if diag_file else Path(cfg["diagnosis_file"]).stem.lower()),
    }
    txt_files = [p for p in txt_files if p.stem.lower() not in skip_stems]

    for txt_path in txt_files:
        stem = txt_path.stem

        # Parse filename metadata
        m = _FILENAME_RE.match(stem)
        if m is None:
            logger.info(
                "Filename %r does not match expected ICBHI cycle-annotation pattern — "
                "skipping (expected for auxiliary files like filename_differences.txt, "
                "filename_format.txt that ship with the official ICBHI release)",
                stem,
            )
            continue

        pid = m.group("pid")
        session = m.group("session")
        location = m.group("location")
        mode = m.group("mode")
        device = m.group("device")

        # Find corresponding audio
        audio_path = _find_audio_file(raw_dir, stem, extensions)
        if audio_path is None:
            logger.warning("No audio file found for annotation %s — skipping", txt_path)
            continue

        split = split_map.get(stem, "unknown")
        diagnosis = diag_map.get(pid, "")

        raw_cycles = _parse_annotation_file(txt_path)

        for idx, cycle in enumerate(raw_cycles):
            cycle_id = f"icbhi__{stem}__{idx:04d}"
            records.append(
                {
                    # ── Universal fields ──────────────────────────────────────
                    "cycle_id": cycle_id,
                    "source_dataset": "icbhi",
                    "audio_path": str(audio_path),
                    "start_time": cycle["start_time"],
                    "end_time": cycle["end_time"],
                    "duration": cycle["end_time"] - cycle["start_time"],
                    "label_4class": cycle["label_4class"],
                    "label_fine": cycle["label_4class"],   # same as 4class for ICBHI
                    "label_coarse": cycle["label_4class"],
                    "crackle": cycle["crackle"],
                    "wheeze": cycle["wheeze"],
                    "split": split,
                    # ── ICBHI-specific metadata ───────────────────────────────
                    "patient_id": pid,
                    "session_id": session,
                    "location": location,
                    "mode": mode,
                    "device": device,
                    "diagnosis": diagnosis,
                    # SPRSound-only fields left as None for schema consistency
                    "record_label": None,
                    "quality": None,
                    # ── Provenance fields (shared schema across real/augmented/synthetic) ──
                    "source_type": "real",        # "real" | "augmented" | "synthetic"
                    "generator": None,             # e.g. "pitch_shift_+1", "diffusion_v1"
                    "original_cycle_id": None,     # set for augmented/synthetic derivatives
                }
            )

    logger.info("ICBHI: loaded %d cycles from %d annotation files", len(records), len(txt_files))
    return records