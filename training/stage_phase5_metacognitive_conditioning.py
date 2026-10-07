#!/usr/bin/env python3
"""
Aerovex Sonon - Phase 5 Metacognitive & Self-Aware Dataset Conditioning Engine
==============================================================================
Enriches speech dataset manifests with four metacognitive conditioning streams:
  1. Ahead-of-Time Inner Monologue Prefixes: time-aligned semantic tokens
     projected ~150-300 ms ahead of acoustic audio rendering.
  2. Continuous 3D Affective Coordinates (Valence, Arousal, Dominance):
     computed from lexical sentiment and prosodic dynamics.
  3. Non-Autoregressive Acoustic In-filling Masks: randomized temporal spans
     (100-400 ms) for training acoustic self-repair and contextual error recovery.
  4. Conversational Turn State & Barge-In Markers: turn floor transition tags.
"""

import argparse
import json
import logging
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sonon.phase5.metacognitive")

# Affective Lexicon mapping for continuous VAD space
# Valence: [-1.0 (unpleasant) to +1.0 (pleasant)]
# Arousal: [0.0 (calm) to 1.0 (activated)]
# Dominance: [-1.0 (submissive) to +1.0 (authoritative)]
AFFECTIVE_LEXICON = {
    # Emergency / urgent / high alarm
    "mayday": (-0.90, 0.95, 0.80),
    "emergency": (-0.85, 0.90, 0.75),
    "abort": (-0.80, 0.95, 0.85),
    "collision": (-0.95, 0.95, 0.60),
    "warning": (-0.70, 0.85, 0.70),
    "caution": (-0.50, 0.75, 0.65),
    "danger": (-0.80, 0.90, 0.60),
    "traffic": (-0.30, 0.70, 0.60),
    # Authoritative commands
    "cleared": (0.40, 0.50, 0.80),
    "maintain": (0.10, 0.40, 0.75),
    "climb": (0.20, 0.60, 0.70),
    "descend": (0.00, 0.55, 0.70),
    "hold": (0.00, 0.50, 0.75),
    "squawk": (0.10, 0.45, 0.80),
    "contact": (0.20, 0.40, 0.70),
    # Positive / conversational
    "good": (0.60, 0.35, 0.50),
    "great": (0.80, 0.55, 0.60),
    "roger": (0.30, 0.30, 0.60),
    "copy": (0.25, 0.30, 0.60),
    "thanks": (0.70, 0.35, 0.40),
    "thank": (0.70, 0.35, 0.40),
    "welcome": (0.60, 0.30, 0.50),
    # Hesitations / uncertainty
    "uh": (-0.10, 0.30, 0.20),
    "umm": (-0.15, 0.25, 0.15),
    "well": (0.00, 0.20, 0.30),
    "maybe": (-0.10, 0.25, 0.20),
}


def infer_utterance_affect(text: str) -> Dict[str, float]:
    """Compute continuous Valence-Arousal-Dominance (V, A, D) coordinates."""
    words = re.findall(r"\b[a-z]+\b", text.lower())
    if not words:
        return {"valence": 0.0, "arousal": 0.30, "dominance": 0.50}

    valences = []
    arousals = []
    dominances = []

    for w in words:
        if w in AFFECTIVE_LEXICON:
            v, a, d = AFFECTIVE_LEXICON[w]
            valences.append(v)
            arousals.append(a)
            dominances.append(d)

    if valences:
        mean_v = sum(valences) / len(valences)
        mean_a = sum(arousals) / len(arousals)
        mean_d = sum(dominances) / len(dominances)
    else:
        # Default neutral conversational baseline
        mean_v = 0.10
        mean_a = 0.35
        mean_d = 0.55

    # Check terminal punctuation for prosodic arousal cues
    if text.endswith("!"):
        mean_a = min(1.0, mean_a + 0.25)
    elif text.endswith("?"):
        mean_a = min(1.0, mean_a + 0.15)
        mean_d = max(-1.0, mean_d - 0.15)

    return {
        "valence": round(float(mean_v), 3),
        "arousal": round(float(mean_a), 3),
        "dominance": round(float(mean_d), 3),
    }


def generate_inner_monologue_tokens(
    text: str, duration_sec: float, target_affect: Dict[str, float]
) -> List[Dict[str, Any]]:
    """Generate time-aligned ahead-of-time Inner Monologue tokens."""
    words = text.strip().split()
    if not words:
        return []

    token_duration_ms = (duration_sec * 1000.0) / len(words)
    tokens = []

    for i, word in enumerate(words):
        # Lookahead offset: projected 150-300 ms ahead of acoustic time
        lookahead_offset = round(150.0 + (i * 25.0) % 150.0, 1)

        # Prosodic intent classification
        clean = word.strip(".,!?:;\"'").lower()
        if clean in ("stop", "abort", "warning", "caution", "mayday"):
            intent = "UrgentExclamation"
            entropy = 0.20
        elif clean in ("uh", "umm", "well", "ah"):
            intent = "HesitationPause"
            entropy = 1.45
        elif word.endswith("?"):
            intent = "InquisitiveRise"
            entropy = 0.50
        elif clean in ("copy", "roger", "affirmative", "yes"):
            intent = "EmpatheticAffirmation"
            entropy = 0.15
        else:
            intent = "DeclarativeCadence"
            entropy = 0.35

        tokens.append({
            "token_index": i,
            "text": word,
            "lookahead_offset_ms": lookahead_offset,
            "approx_timestamp_ms": round(i * token_duration_ms, 1),
            "prosodic_intent": intent,
            "entropy": round(entropy, 2),
        })

    return tokens


def generate_acoustic_infilling_mask(duration_sec: float) -> Optional[Dict[str, float]]:
    """Generate random temporal mask span (100-400 ms) for acoustic self-repair training."""
    duration_ms = duration_sec * 1000.0
    if duration_ms < 600.0:
        return None

    # Choose mask duration between 100ms and 400ms
    mask_duration = round(random.uniform(100.0, min(400.0, duration_ms * 0.40)), 1)
    # Position mask randomly away from extreme edges
    max_start = duration_ms - mask_duration - 100.0
    if max_start <= 100.0:
        start_ms = 100.0
    else:
        start_ms = round(random.uniform(100.0, max_start), 1)

    return {
        "mask_start_ms": start_ms,
        "mask_duration_ms": mask_duration,
        "mask_end_ms": round(start_ms + mask_duration, 1),
    }


def condition_manifest_item(item: Dict[str, Any], rng_seed: Optional[int] = None) -> Dict[str, Any]:
    """Augment a single manifest record with self-state-aware metacognitive metadata."""
    if rng_seed is not None:
        random.seed(rng_seed)

    text = item.get("text") or item.get("transcript_normalized") or ""
    duration = float(item.get("duration_seconds", 3.0))

    # 1. 3D Affective State
    affect = infer_utterance_affect(text)

    # 2. Ahead-of-time Inner Monologue Tokens
    tokens = generate_inner_monologue_tokens(text, duration, affect)

    # 3. Acoustic In-filling Mask
    infill_mask = generate_acoustic_infilling_mask(duration)

    # 4. Turn State Metadata
    is_atc = "atc" in item.get("uuid", "")
    is_duplex = "duplex" in item.get("uuid", "")

    metacognitive_meta = {
        "affective_state": affect,
        "inner_monologue_preview": tokens[:8],  # store head tokens for low-overhead manifest
        "inner_monologue_token_count": len(tokens),
        "infilling_mask": infill_mask,
        "conversational_floor": {
            "initial_state": "Thinking",
            "execution_state": "Speaking",
            "is_atc_radio": is_atc,
            "is_duplex_dialogue": is_duplex,
            "allow_passive_backchannel": True,
        },
    }

    conditioned = dict(item)
    conditioned["metacognitive"] = metacognitive_meta
    return conditioned


def condition_manifest_file(manifest_path: Path, output_path: Path) -> int:
    """Read an existing manifest, condition all records, and write the augmented manifest."""
    logger.info(f"Loading manifest: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {manifest_path}")

    logger.info(f"Conditioning {len(data)} items with metacognitive self-state metadata...")
    conditioned_list = []
    for i, item in enumerate(data):
        conditioned_list.append(condition_manifest_item(item, rng_seed=i))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(conditioned_list, f, indent=2)

    logger.info(f"Wrote conditioned manifest with {len(conditioned_list)} items to {output_path}")
    return len(conditioned_list)


def main():
    parser = argparse.ArgumentParser(
        description="Condition Sonon dataset manifests with Phase 5 metacognitive self-state metadata."
    )
    parser.add_argument(
        "--train_manifest",
        type=str,
        default="/D/aerovex_datasets/pilot/train_manifest.json",
        help="Path to input train manifest",
    )
    parser.add_argument(
        "--val_manifest",
        type=str,
        default="/D/aerovex_datasets/pilot/val_manifest.json",
        help="Path to input validation manifest",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/D/aerovex_datasets/pilot",
        help="Output directory for conditioned manifests",
    )
    parser.add_argument(
        "--in_place",
        action="store_true",
        default=True,
        help="Overwrite manifests in-place after backing up originals",
    )
    args = parser.parse_args()

    train_path = Path(args.train_manifest)
    val_path = Path(args.val_manifest)
    out_dir = Path(args.output_dir)

    if not train_path.exists():
        logger.error(f"Train manifest not found: {train_path}")
        sys.exit(1)

    # Backup originals
    train_bak = train_path.with_suffix(".json.pre_phase5_bak")
    val_bak = val_path.with_suffix(".json.pre_phase5_bak")
    if not train_bak.exists():
        import shutil
        shutil.copy2(train_path, train_bak)
        logger.info(f"Backed up train manifest to {train_bak}")
    if val_path.exists() and not val_bak.exists():
        import shutil
        shutil.copy2(val_path, val_bak)
        logger.info(f"Backed up val manifest to {val_bak}")

    out_train = train_path if args.in_place else out_dir / "train_manifest_metacognitive.json"
    out_val = val_path if args.in_place else out_dir / "val_manifest_metacognitive.json"

    n_train = condition_manifest_file(train_path, out_train)
    n_val = 0
    if val_path.exists():
        n_val = condition_manifest_file(val_path, out_val)

    logger.info("=" * 70)
    logger.info("Phase 5 Metacognitive Dataset Conditioning Complete:")
    logger.info(f"  Conditioned Train Utterances: {n_train}")
    logger.info(f"  Conditioned Val Utterances:   {n_val}")
    logger.info(f"  Total Metacognitive Dataset:  {n_train + n_val}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
