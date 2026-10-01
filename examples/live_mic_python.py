#!/usr/bin/env python3
"""
Sonon Live Microphone Listening Example (Python).

Captures streaming audio from your microphone at 16 kHz and spots
enrolled wake-words in real-time.

Features:
- Energy-based Voice Activity Detection (VAD) silence trimming
- Immediate audio playback of enrolled wake-word via `aplay` / `sounddevice`
- Real-time visual audio level and DTW distance indicator
- Zero-dependency ALSA fallback (`arecord` + `aplay`)

Usage:
    python live_mic_python.py
"""

import math
import os
import struct
import subprocess
import sys
import time
import wave
from typing import List

try:
    import sonon
except ImportError:
    # Try local package path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_dir = os.path.abspath(os.path.join(script_dir, "..", "bindings", "python"))
    if os.path.exists(pkg_dir):
        sys.path.insert(0, pkg_dir)
        import sonon
    else:
        print("Please install sonon first: pip install sonon")
        sys.exit(1)


def save_wav(filepath: str, samples: List[float], sample_rate: int = 16000) -> None:
    """Saves float32 audio samples as a 16-bit PCM WAV file."""
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        raw_bytes = bytearray()
        for s in samples:
            clamped = max(-1.0, min(1.0, s))
            val = int(clamped * 32767.0)
            raw_bytes.extend(struct.pack("<h", val))
        wf.writeframes(raw_bytes)


def play_audio_file(filepath: str) -> None:
    """Plays a WAV file using system audio player (aplay / afplay / ffplay)."""
    if subprocess.run(["which", "aplay"], capture_output=True).returncode == 0:
        subprocess.run(["aplay", "-q", filepath], check=False)
    elif subprocess.run(["which", "afplay"], capture_output=True).returncode == 0:
        subprocess.run(["afplay", filepath], check=False)
    elif subprocess.run(["which", "ffplay"], capture_output=True).returncode == 0:
        subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", filepath], check=False)


def listen_with_system_mic():
    """Zero-dependency ALSA microphone capture and playback using `arecord` and `aplay`."""
    sample_rate = 16000.0
    engine = sonon.SononEngine(sample_rate=sample_rate)

    has_arecord = subprocess.run(["which", "arecord"], capture_output=True).returncode == 0
    has_aplay = subprocess.run(["which", "aplay"], capture_output=True).returncode == 0

    if not has_arecord:
        print("Error: `arecord` was not found on your system.")
        print("Install ALSA utilities or use `pip install sounddevice`.")
        return

    wav_path = "/tmp/sonon_wake_word.wav"

    print("==================================================")
    print("Sonon Live Microphone Wake-Word Spotter (ALSA)")
    print("==================================================")
    print("Step 1: Wake-Word Enrollment")
    print("Press Enter and say your wake word (e.g. 'Take Off' or 'Plank')...")
    input()

    print("Recording for 2.0 seconds... speak now!")
    rec_cmd = ["arecord", "-q", "-r", "16000", "-c", "1", "-f", "S16_LE", "-d", "2"]
    rec_proc = subprocess.Popen(rec_cmd, stdout=subprocess.PIPE)
    raw_audio, _ = rec_proc.communicate()

    num_samples = len(raw_audio) // 2
    if num_samples == 0:
        print("Error: No audio samples captured.")
        return

    samples = [
        struct.unpack("<h", raw_audio[i * 2 : (i + 1) * 2])[0] / 32768.0
        for i in range(num_samples)
    ]

    # Auto-trim silence to isolate active speech
    trimmed = sonon.SononEngine.trim_silence(samples, sample_rate=sample_rate)
    trimmed_duration = len(trimmed) / sample_rate
    print(f"Captured {len(samples)/sample_rate:.2f}s -> Trimmed to {trimmed_duration:.2f}s of active speech.")

    # Save to disk
    save_wav(wav_path, trimmed, int(sample_rate))

    # Play back to user immediately
    if has_aplay:
        print(f"\n[PLAYBACK] Playing back your enrolled wake-word via `aplay`...")
        subprocess.run(["aplay", "-q", wav_path], check=False)
        time.sleep(0.3)
    else:
        print(f"\n[PLAYBACK] Saved wake word to {wav_path}.")

    features = engine.extract_features(trimmed)
    if not features:
        print("Error: Feature extraction yielded 0 frames. Please try again.")
        return

    keyword_name = "wake_word"
    threshold = 0.58
    engine.enroll_keyword(keyword_name, features, threshold=threshold)
    print(f"Enrolled '{keyword_name}' ({len(features)} feature frames, threshold: {threshold}).")

    print("\nStep 2: Continuous Real-Time Listening")
    print("Speak naturally! When you say your wake word, it will trigger below.")
    print("Press Ctrl+C to stop.\n")
    print("--------------------------------------------------")

    stream_cmd = ["arecord", "-q", "-r", "16000", "-c", "1", "-f", "S16_LE"]
    stream_proc = subprocess.Popen(stream_cmd, stdout=subprocess.PIPE)
    chunk_samples_count = 320  # 20 ms
    chunk_bytes = chunk_samples_count * 2

    last_trigger_time = 0.0

    try:
        while True:
            data = stream_proc.stdout.read(chunk_bytes)
            if not data or len(data) < chunk_bytes:
                break

            n = len(data) // 2
            chunk = [
                struct.unpack("<h", data[i * 2 : (i + 1) * 2])[0] / 32768.0
                for i in range(n)
            ]

            # Ingest and detect
            events = engine.ingest_samples(chunk)
            dist = engine.get_last_distance(keyword_name)
            now = time.time()

            # Visual volume meter (DC-free AC energy)
            mean_c = sum(chunk) / max(1, len(chunk))
            ac_rms = math.sqrt(sum((s - mean_c) ** 2 for s in chunk) / max(1, len(chunk)))
            bars = min(15, int(ac_rms * 40))
            meter = "#" * bars + " " * (15 - bars)
            dist_str = f"{dist:.3f}" if dist is not None and math.isfinite(dist) else "..."

            sys.stdout.write(f"\r[LISTENING] Level: [{meter}] (RMS: {ac_rms:.3f}) | Best Dist: {dist_str} (Thresh: {threshold:.2f})  ")
            sys.stdout.flush()

            for ev in events:
                if now - last_trigger_time > 1.0:  # 1-second debounce
                    last_trigger_time = now
                    sys.stdout.write(
                        f"\n\n==================================================\n"
                        f"*** [TRIGGER] WAKE-WORD DETECTED: '{ev.keyword}'! ***\n"
                        f"Confidence: {ev.confidence*100:.1f}% | Time: {ev.timestamp_sec:.2f}s\n"
                        f"==================================================\n\n"
                    )
                    sys.stdout.flush()

    except KeyboardInterrupt:
        print("\nStopping audio stream...")
    finally:
        stream_proc.terminate()
        try:
            stream_proc.wait(timeout=1.0)
        except Exception:
            stream_proc.kill()


def listen_with_sounddevice():
    """Microphone capture and playback using `sounddevice`."""
    import sounddevice as sd
    import numpy as np

    sample_rate = 16000.0
    engine = sonon.SononEngine(sample_rate=sample_rate)

    print("==================================================")
    print("Sonon Live Microphone Wake-Word Spotter (SoundDevice)")
    print("==================================================")
    print("Step 1: Wake-Word Enrollment")
    print("Press Enter and say your wake word (e.g. 'Take Off' or 'Plank')...")
    input()
    time.sleep(0.2)

    enroll_sec = 2.0
    print("Recording for 2.0 seconds... speak now!")
    rec = sd.rec(
        int(enroll_sec * sample_rate),
        samplerate=int(sample_rate),
        channels=1,
        dtype="float32",
    )
    sd.wait()
    samples = rec.flatten().tolist()

    trimmed = sonon.SononEngine.trim_silence(samples, sample_rate=sample_rate)
    trimmed_duration = len(trimmed) / sample_rate
    print(f"Captured {enroll_sec:.2f}s -> Trimmed to {trimmed_duration:.2f}s of active speech.")

    wav_path = "/tmp/sonon_wake_word.wav"
    save_wav(wav_path, trimmed, int(sample_rate))

    print(f"\n[PLAYBACK] Playing back your enrolled wake-word...")
    sd.play(np.array(trimmed, dtype=np.float32), int(sample_rate))
    sd.wait()
    time.sleep(0.3)

    features = engine.extract_features(trimmed)
    if not features:
        print("Error: Feature extraction yielded 0 frames. Please try again.")
        return

    keyword_name = "wake_word"
    threshold = 0.58
    engine.enroll_keyword(keyword_name, features, threshold=threshold)
    print(f"Enrolled '{keyword_name}' ({len(features)} feature frames, threshold: {threshold}).")

    print("\nStep 2: Continuous Real-Time Listening")
    print("Speak naturally! When you say your wake word, it will trigger below.")
    print("Press Ctrl+C to stop.\n")
    print("--------------------------------------------------")

    block_size = 320
    last_trigger_time = 0.0

    def audio_callback(indata, frames, time_info, status):
        nonlocal last_trigger_time
        chunk = indata[:, 0].tolist()
        events = engine.ingest_samples(chunk)
        dist = engine.get_last_distance(keyword_name)
        now = time.time()

        mean_c = sum(chunk) / max(1, len(chunk))
        ac_rms = math.sqrt(sum((s - mean_c) ** 2 for s in chunk) / max(1, len(chunk)))
        bars = min(15, int(ac_rms * 40))
        meter = "#" * bars + " " * (15 - bars)
        dist_str = f"{dist:.3f}" if dist is not None and math.isfinite(dist) else "..."

        sys.stdout.write(f"\r[LISTENING] Level: [{meter}] (RMS: {ac_rms:.3f}) | Best Dist: {dist_str} (Thresh: {threshold:.2f})  ")
        sys.stdout.flush()

        for ev in events:
            if now - last_trigger_time > 1.0:
                last_trigger_time = now
                sys.stdout.write(
                    f"\n\n==================================================\n"
                    f"*** [TRIGGER] WAKE-WORD DETECTED: '{ev.keyword}'! ***\n"
                    f"Confidence: {ev.confidence*100:.1f}% | Time: {ev.timestamp_sec:.2f}s\n"
                    f"==================================================\n\n"
                )
                sys.stdout.flush()

    try:
        with sd.InputStream(
            samplerate=int(sample_rate),
            channels=1,
            blocksize=block_size,
            dtype="float32",
            callback=audio_callback,
        ):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping audio stream...")


if __name__ == "__main__":
    # If arecord is available on Linux, prioritize ALSA for zero-latency direct access
    if sys.platform.startswith("linux") and subprocess.run(["which", "arecord"], capture_output=True).returncode == 0:
        listen_with_system_mic()
    else:
        try:
            import sounddevice

            listen_with_sounddevice()
        except ImportError:
            listen_with_system_mic()

