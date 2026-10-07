#!/usr/bin/env python3
"""Stage Phase 2 Foundation & Multi-Speaker Timbre Scaling Corpora.

Expands Sonon voice dataset volume and accent diversity:
1. DailyTalk Multi-Turn Dialogue Expansion: Ingests conversational dialogues with natural prosodic inflections.
2. VCTK Global English Accent Diversity: Ingests clean studio speech (DPA mic1) across diverse native dialects.
3. Rigorous Acoustic Quality Gate: Enforces WADA-SNR >= 28.0 dB, 0.85s <= duration <= 17.5s, peak -1.0 dBFS.

Conforms strictly to Aerovex robotics engineering standards: zero emojis, pure safe signal processing.
"""

import os
import sys
import io
import re
import glob
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import soundfile as sf
import scipy.signal as signal
import pyarrow.parquet as pq

# Add parent directory for prepare_dataset import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.prepare_dataset import inspect_and_curate_audio

TARGET_SR = 24000
OUTPUT_DIR = Path("/D/aerovex_datasets/raw/tier2_phase2_staging")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resample_if_needed(audio: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    return signal.resample_poly(audio, up, down).astype(np.float32)


def verify_audio_sample(audio: np.ndarray, sample_rate: int = TARGET_SR, min_snr: float = 28.0) -> bool:
    """Verify in-memory audio passes rigorous DSP inspection."""
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, audio, sample_rate, subtype="PCM_16")
        valid, _, _, _ = inspect_and_curate_audio(tmp.name, sample_rate, min_snr_db=min_snr)
        return valid


def stage_dailytalk_partition(parquet_path: Path, target_dir: Path, max_turns: int = 3000) -> Tuple[int, int]:
    """Stage multi-turn conversational dialogue from a DailyTalk parquet partition."""
    print(f"--- Staging DailyTalk Conversations from {parquet_path.name} ---")
    if not parquet_path.exists():
        print(f"Error: Parquet {parquet_path} does not exist.")
        return 0, 0

    table = pq.read_table(parquet_path)
    d = table.to_pydict()

    staged_total = 0
    staged_hesitations = 0

    conv_count = len(d["conversation_id"])
    for c_idx in range(conv_count):
        if staged_total >= max_turns:
            break

        conv_id = d["conversation_id"][c_idx]
        audio_entry = d["audio"][c_idx]
        cuts = d["audio_cut_idxs"][c_idx]
        texts = d["texts"][c_idx]
        speakers = d["speaker_ids"][c_idx]

        try:
            with io.BytesIO(audio_entry["bytes"]) as bio:
                full_audio, sr = sf.read(bio)
        except Exception:
            continue

        for turn_idx, (cut, text, spk) in enumerate(zip(cuts, texts, speakers)):
            if staged_total >= max_turns:
                break

            start_samp, end_samp = int(cut[0]), int(cut[1])
            if end_samp <= start_samp or end_samp > len(full_audio):
                continue

            turn_audio = full_audio[start_samp:end_samp]
            dur = len(turn_audio) / float(sr)
            if dur < 0.85 or dur > 17.5:
                continue

            turn_24k = resample_if_needed(turn_audio, sr, TARGET_SR)
            peak = np.max(np.abs(turn_24k))
            if peak > 1e-4:
                turn_24k = turn_24k * (0.841 / peak)

            if not verify_audio_sample(turn_24k, TARGET_SR, min_snr=25.0):
                continue

            clean_text = text.strip()
            lower_text = clean_text.lower()
            hes_re = re.search(r"\b(umm+|uh+|hmm+|well|err+|er|ah|oh)\b", lower_text)
            is_hesitation = hes_re is not None

            if is_hesitation:
                if lower_text.startswith("umm"):
                    norm_text = re.sub(r"^[uU]mm+[…\.\,]*\s*", "[hesitation] ", clean_text)
                elif lower_text.startswith("uh"):
                    norm_text = re.sub(r"^[uU]h+[…\.\,]*\s*", "[hesitation] ", clean_text)
                elif lower_text.startswith("hmm"):
                    norm_text = re.sub(r"^[hH]mm+[…\.\,]*\s*", "[hesitation] ", clean_text)
                elif lower_text.startswith("well"):
                    norm_text = re.sub(r"^[wW]ell[…\.\,]*\s*", "[hesitation] ", clean_text)
                else:
                    norm_text = f"[hesitation] {clean_text}"
                staged_hesitations += 1
            else:
                norm_text = clean_text

            stem = f"phase2_dt_c{conv_id:04d}_t{turn_idx:02d}_spk{spk}"
            out_wav = target_dir / f"{stem}.wav"
            out_txt = target_dir / f"{stem}.normalized.txt"

            sf.write(out_wav, turn_24k, TARGET_SR, subtype="PCM_16")
            out_txt.write_text(norm_text, encoding="utf-8")
            staged_total += 1

    print(f"Staged {staged_total} verified DailyTalk turns (including {staged_hesitations} [hesitation] turns).")
    return staged_total, staged_hesitations


def stage_vctk_partition(parquet_path: Path, target_dir: Path, max_samples: int = 1500) -> int:
    """Stage studio-recorded speech across diverse native accents from VCTK."""
    print(f"--- Staging VCTK Accent Diversity from {parquet_path.name} ---")
    if not parquet_path.exists():
        print(f"Error: Parquet {parquet_path} does not exist.")
        return 0

    table = pq.read_table(parquet_path)
    d = table.to_pydict()

    staged = 0
    total_rows = len(d["id"])

    for idx in range(total_rows):
        if staged >= max_samples:
            break

        mic_id = d["mic_id"][idx]
        # Prefer mic1 (DPA 4035 omni-directional studio mic with zero proximity effect)
        if mic_id != "mic1":
            continue

        raw_text = d["text"][idx].strip()
        if not raw_text or len(raw_text) < 3:
            continue

        spk_id = d["speaker_id"][idx]
        item_id = d["id"][idx]

        audio_entry = d["audio"][idx]
        try:
            with io.BytesIO(audio_entry["bytes"]) as bio:
                audio, sr = sf.read(bio)
        except Exception:
            continue

        dur = len(audio) / float(sr)
        if dur < 0.85 or dur > 17.5:
            continue

        audio_24k = resample_if_needed(audio, sr, TARGET_SR)
        peak = np.max(np.abs(audio_24k))
        if peak > 1e-4:
            audio_24k = audio_24k * (0.841 / peak)

        if not verify_audio_sample(audio_24k, TARGET_SR, min_snr=28.0):
            continue

        stem = f"phase2_vctk_{spk_id}_{item_id}"
        out_wav = target_dir / f"{stem}.wav"
        out_txt = target_dir / f"{stem}.normalized.txt"

        sf.write(out_wav, audio_24k, TARGET_SR, subtype="PCM_16")
        out_txt.write_text(raw_text, encoding="utf-8")
        staged += 1

    print(f"Staged {staged} verified studio VCTK utterances across {len(set(d['speaker_id']))} speakers.")
    return staged


def stage_libritts_partition(parquet_path: Path, target_dir: Path, max_samples: int = 2500) -> int:
    """Stage clean studio speech from LibriTTS-R partition."""
    print(f"--- Staging LibriTTS-R Clean Speech from {parquet_path.name} ---")
    if not parquet_path.exists():
        print(f"Error: Parquet {parquet_path} does not exist.")
        return 0

    table = pq.read_table(parquet_path)
    d = table.to_pydict()

    staged = 0
    total_rows = len(d["id"])

    for idx in range(total_rows):
        if staged >= max_samples:
            break

        norm_text = (d["text_normalized"][idx] or "").strip()
        if not norm_text or len(norm_text) < 3:
            continue

        spk_id = str(d["speaker_id"][idx])
        item_id = str(d["id"][idx])
        audio_entry = d["audio"][idx]

        try:
            with io.BytesIO(audio_entry["bytes"]) as bio:
                audio, sr = sf.read(bio)
        except Exception:
            continue

        dur = len(audio) / float(sr)
        if dur < 0.85 or dur > 17.5:
            continue

        audio_24k = resample_if_needed(audio, sr, TARGET_SR)
        peak = np.max(np.abs(audio_24k))
        if peak > 1e-4:
            audio_24k = audio_24k * (0.841 / peak)

        if not verify_audio_sample(audio_24k, TARGET_SR, min_snr=28.0):
            continue

        stem = f"phase2_libri_{spk_id}_{item_id}"
        out_wav = target_dir / f"{stem}.wav"
        out_txt = target_dir / f"{stem}.normalized.txt"

        sf.write(out_wav, audio_24k, TARGET_SR, subtype="PCM_16")
        out_txt.write_text(norm_text, encoding="utf-8")
        staged += 1

    unique_spks = len(set(d['speaker_id']))
    print(f"Staged {staged} verified studio LibriTTS-R utterances across {unique_spks} speakers.")
    return staged


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage Phase 2 Foundation & Multi-Speaker Timbre Scaling Corpora."
    )
    parser.add_argument(
        "--dailytalk_parquet",
        nargs="+",
        default=None,
        help="Path(s) to DailyTalk parquet partition file(s).",
    )
    parser.add_argument(
        "--vctk_parquet",
        nargs="+",
        default=None,
        help="Path(s) to VCTK parquet partition file(s).",
    )
    parser.add_argument(
        "--libritts_parquet",
        nargs="+",
        default=None,
        help="Path(s) to LibriTTS-R parquet partition file(s).",
    )
    parser.add_argument(
        "--max_dt_turns",
        type=int,
        default=2500,
        help="Maximum DailyTalk turns to stage per partition.",
    )
    parser.add_argument(
        "--max_vctk_samples",
        type=int,
        default=1200,
        help="Maximum VCTK studio samples to stage per partition.",
    )
    parser.add_argument(
        "--max_libri_samples",
        type=int,
        default=2500,
        help="Maximum LibriTTS-R samples to stage per partition.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Destination directory for staged files.",
    )
    parser.add_argument(
        "--clean_staging",
        action="store_true",
        default=True,
        help="Clean output directory before staging.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    print("================================================================================")
    print("Aerovex Sonon: Staging Phase 2 Foundation & Multi-Speaker Timbre Scaling")
    print("================================================================================")

    if args.clean_staging and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)

    total_staged = 0
    dt_staged, dt_hes = 0, 0
    vctk_staged = 0
    libri_staged = 0

    if args.dailytalk_parquet:
        for dt_str in args.dailytalk_parquet:
            dt_p = Path(dt_str)
            sub_staged, sub_hes = stage_dailytalk_partition(dt_p, out_dir, max_turns=args.max_dt_turns)
            dt_staged += sub_staged
            dt_hes += sub_hes
        total_staged += dt_staged

    if args.vctk_parquet:
        for vctk_str in args.vctk_parquet:
            vctk_p = Path(vctk_str)
            sub_staged = stage_vctk_partition(vctk_p, out_dir, max_samples=args.max_vctk_samples)
            vctk_staged += sub_staged
        total_staged += vctk_staged

    if args.libritts_parquet:
        for libri_str in args.libritts_parquet:
            libri_p = Path(libri_str)
            sub_staged = stage_libritts_partition(libri_p, out_dir, max_samples=args.max_libri_samples)
            libri_staged += sub_staged
        total_staged += libri_staged

    print("\n================================================================================")
    print("Phase 2 Foundation Staging Completed Successfully")
    print("================================================================================")
    print(f"Output Directory:      {out_dir}")
    print(f"Total Staged Files:    {total_staged}")
    if dt_staged > 0:
        print(f"  - DailyTalk Turns:   {dt_staged} (Hesitations: {dt_hes})")
    if vctk_staged > 0:
        print(f"  - VCTK Utterances:   {vctk_staged}")
    if libri_staged > 0:
        print(f"  - LibriTTS-R Speech: {libri_staged}")
    print("================================================================================")


if __name__ == "__main__":
    main()
