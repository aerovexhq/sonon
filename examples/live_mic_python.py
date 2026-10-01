#!/usr/bin/env python3
"""
Sonon Live Microphone Listening Example (Python).

Captures streaming audio from your microphone at 16 kHz and spots
enrolled wake-words in real-time.

Requirements:
    pip install sonon sounddevice
    (Or zero-dependency fallback via system `arecord` / `ffmpeg`)
"""

import sys
import time

try:
    import sonon
except ImportError:
    print("Please install sonon first: pip install sonon")
    sys.exit(1)


def listen_with_sounddevice():
    import sounddevice as sd
    import numpy as np

    sample_rate = 16000.0
    engine = sonon.SononEngine(sample_rate=sample_rate)

    print("==================================================")
    print("Sonon Live Microphone Wake-Word Spotter")
    print("==================================================")
    print("Step 1: Calibration & Enrollment")
    print("Press Enter and immediately say your keyword (e.g. 'Take Off' or 'Plank')...")
    input()

    enroll_duration_sec = 1.0
    print("Recording keyword for 1.0 second... speak now!")
    rec = sd.rec(
        int(enroll_duration_sec * sample_rate),
        samplerate=int(sample_rate),
        channels=1,
        dtype="float32",
    )
    sd.wait()
    samples = rec.flatten().tolist()

    features = engine.extract_features(samples)
    if not features:
        print("Error: No speech detected in enrollment. Try again.")
        return

    keyword_name = "wake_word"
    engine.enroll_keyword(keyword_name, features, threshold=3.5)
    print(f"Enrolled keyword '{keyword_name}' ({len(features)} feature frames).")
    print()
    print("Step 2: Continuous Real-Time Listening")
    print("Speak naturally. When you say the keyword, it will trigger below.")
    print("Press Ctrl+C to stop.")
    print("--------------------------------------------------")

    block_size = 320  # 20 ms blocks

    def audio_callback(indata, frames, time_info, status):
        chunk = indata[:, 0].tolist()
        detections = engine.ingest_samples(chunk)
        for ev in detections:
            print(
                f"[TRIGGER] Wake-Word Detected: '{ev.keyword}' | "
                f"Confidence: {ev.confidence:.2f} | Time: {ev.timestamp_sec:.2f}s"
            )

    with sd.InputStream(
        samplerate=int(sample_rate),
        channels=1,
        blocksize=block_size,
        dtype="float32",
        callback=audio_callback,
    ):
        while True:
            time.sleep(0.1)


def listen_with_system_mic():
    """Zero-dependency fallback piping directly from system audio tools (arecord / ffmpeg)."""
    import subprocess
    import struct

    sample_rate = 16000.0
    engine = sonon.SononEngine(sample_rate=sample_rate)

    # Detect available recording utility
    cmd = None
    if subprocess.run(["which", "arecord"], capture_output=True).returncode == 0:
        cmd = ["arecord", "-q", "-r", "16000", "-c", "1", "-f", "S16_LE"]
    elif subprocess.run(["which", "ffmpeg"], capture_output=True).returncode == 0:
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "quiet",
            "-f",
            "pulse",
            "-i",
            "default",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-f",
            "s16le",
            "-",
        ]

    if not cmd:
        print("Please install sounddevice (`pip install sounddevice`) or `arecord`.")
        return

    print("Running zero-dependency microphone capture via:", cmd[0])
    print("Enrollment: Record 1.5 seconds of your wake-word...")
    print("Press Enter to begin recording...")
    input()

    rec_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    raw_enroll = rec_proc.stdout.read(int(16000 * 2 * 1.5))
    rec_proc.terminate()

    num_samples = len(raw_enroll) // 2
    enroll_samples = [
        struct.unpack("<h", raw_enroll[i * 2 : (i + 1) * 2])[0] / 32768.0
        for i in range(num_samples)
    ]
    features = engine.extract_features(enroll_samples)
    engine.enroll_keyword("wake_word", features, threshold=3.5)
    print("Enrolled! Listening continuously...")

    stream_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    chunk_bytes = 320 * 2  # 20 ms chunk (16-bit)

    try:
        while True:
            data = stream_proc.stdout.read(chunk_bytes)
            if not data:
                break
            n = len(data) // 2
            chunk_samples = [
                struct.unpack("<h", data[i * 2 : (i + 1) * 2])[0] / 32768.0
                for i in range(n)
            ]
            events = engine.ingest_samples(chunk_samples)
            for ev in events:
                print(
                    f"[TRIGGER] Wake-Word Detected: '{ev.keyword}' | "
                    f"Confidence: {ev.confidence:.2f} | Time: {ev.timestamp_sec:.2f}s"
                )
    except KeyboardInterrupt:
        stream_proc.terminate()


if __name__ == "__main__":
    try:
        import sounddevice

        listen_with_sounddevice()
    except ImportError:
        listen_with_system_mic()
