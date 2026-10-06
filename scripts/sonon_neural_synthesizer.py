#!/usr/bin/env python3
"""Sonon Neural Speech Synthesizer

Industrial-grade neural speech synthesis engine utilizing deep learning foundation
weights with Sonon multi-speaker voiceprint blending and acoustic signal mastering.

Zero emojis. Safe local execution.
"""

import os
import sys
import time
import argparse
import numpy as np
import scipy.signal as signal
import soundfile as sf
import kokoro_onnx

MODEL_DIR = "/home/usr/.cache/sonon_models/kokoro"
MODEL_PATH = os.path.join(MODEL_DIR, "kokoro-v1.0.onnx")
VOICES_PATH = os.path.join(MODEL_DIR, "voices-v1.0.bin")

OUTPUT_DIR = "/home/usr/Projects/aerovex/modules/sonon/output/speech_synthesis"
DOCS_AUDIO_DIR = "/home/usr/Projects/aerovex/modules/sonon/docs/public/audio"


class SononAcousticMaster:
    """Acoustic signal mastering chain for flight-deck and studio playback."""

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        # 4-pole highpass Butterworth filter at 45 Hz (subsonic rumble suppression)
        self.sos_hp = signal.butter(4, 45, btype='highpass', fs=sample_rate, output='sos')
        # Sibilance notch filter at 7200 Hz with Q=3.0 to tame harsh fricatives
        b_notch, a_notch = signal.iirnotch(7200, 3.0, fs=sample_rate)
        self.b_notch = b_notch
        self.a_notch = a_notch

    def process(self, audio: np.ndarray, apply_warmth: bool = True) -> np.ndarray:
        if len(audio) == 0:
            return audio

        # 1. DC offset removal
        out = audio - np.mean(audio)

        # 2. Subsonic highpass filtering
        out = signal.sosfilt(self.sos_hp, out)

        # 3. Sibilance control
        out = signal.lfilter(self.b_notch, self.a_notch, out)

        # 4. Subtle vocal warmth harmonic saturation
        if apply_warmth:
            # Soft-knee polynomial saturation for natural warmth
            out = np.clip(out, -1.0, 1.0)
            out = out + 0.04 * (out ** 2) * np.sign(out) - 0.02 * (out ** 3)

        # 5. Broadcast peak normalization to -1.0 dBFS (amplitude 0.891)
        peak = np.max(np.abs(out))
        if peak > 1e-4:
            target_peak = 0.891  # -1.0 dBFS
            out = out * (target_peak / peak)

        return out.astype(np.float32)


class SononNeuralVoiceEngine:
    """Sonon Neural Speech Synthesis Engine with multi-speaker voiceprint blending."""

    def __init__(self, model_path: str = MODEL_PATH, voices_path: str = VOICES_PATH):
        if not os.path.exists(model_path) or not os.path.exists(voices_path):
            raise FileNotFoundError(
                f"Model weights not found. Please ensure {model_path} and {voices_path} exist."
            )
        print(f"Loading neural foundation weights from {model_path}...")
        t0 = time.time()
        self.kokoro = kokoro_onnx.Kokoro(model_path, voices_path)
        self.master = SononAcousticMaster(sample_rate=24000)
        self.voices = self.kokoro.get_voices()
class AerospaceTextNormalizer:
    """Python bridge for Sonon safe Rust aerospace text normalization."""

    ACRONYMS = {
        "UAV": "U A V",
        "UAS": "U A S",
        "VTOL": "V-TOL",
        "EO/IR": "E O I R",
        "TACAN": "tack-an",
        "METAR": "mee-tar",
        "NOTAM": "no-tam",
        "VFR": "V F R",
        "IFR": "I F R",
        "ATC": "A T C",
        "TCAS": "tee-cas",
        "ILS": "I L S",
        "AGL": "A G L",
        "MSL": "M S L",
        "ETA": "E T A",
        "RPM": "R P M",
        "VHF": "V H F",
        "UHF": "U H F",
        "GNSS": "G N S S",
        "GPS": "G P S",
        "ADS-B": "ads bee",
    }

    DIGITS = {
        '0': "zero", '1': "one", '2': "two", '3': "three", '4': "four",
        '5': "five", '6': "six", '7': "seven", '8': "eight", '9': "niner",
    }

    @classmethod
    def expand_digits(cls, s: str) -> str:
        return " ".join(cls.DIGITS.get(c, c) for c in s if c.isdigit())

    @classmethod
    def normalize(cls, text: str) -> str:
        words = []
        for token in text.split():
            clean = token.strip(".,;:!?()[]{}'\"")
            punct = token[len(token) - len(token.lstrip(".,;:!?()[]{}'\"")):]
            suffix = token[len(token.rstrip(".,;:!?()[]{}'\"")):]

            if clean in cls.ACRONYMS:
                words.append(f"{cls.ACRONYMS[clean]}{suffix}")
                continue

            # Flight level (e.g. FL350)
            if clean.startswith("FL") and len(clean) >= 4 and clean[2:].isdigit():
                words.append(f"flight level {cls.expand_digits(clean[2:])}{suffix}")
                continue

            # Heading (e.g. HDG090)
            if clean.startswith("HDG") and len(clean) >= 5 and clean[3:].isdigit():
                words.append(f"heading {cls.expand_digits(clean[3:])}{suffix}")
                continue

            # Runway (e.g. RWY28R)
            if clean.startswith("RWY") and len(clean) >= 5:
                sub = clean[3:]
                digits = "".join(c for c in sub if c.isdigit())
                side = ""
                if sub.endswith(("R", "r")):
                    side = " right"
                elif sub.endswith(("L", "l")):
                    side = " left"
                elif sub.endswith(("C", "c")):
                    side = " center"
                if digits:
                    words.append(f"runway {cls.expand_digits(digits)}{side}{suffix}")
                    continue

            words.append(token)
        return " ".join(words)


class SononNeuralVoiceEngine:
    """Sonon Neural Speech Synthesis Engine with multi-speaker voiceprint blending and urgency control."""

    def __init__(self, model_path: str = MODEL_PATH, voices_path: str = VOICES_PATH):
        if not os.path.exists(model_path) or not os.path.exists(voices_path):
            raise FileNotFoundError(
                f"Model weights not found. Please ensure {model_path} and {voices_path} exist."
            )
        print(f"Loading neural foundation weights from {model_path}...")
        t0 = time.time()
        self.kokoro = kokoro_onnx.Kokoro(model_path, voices_path)
        self.master = SononAcousticMaster(sample_rate=24000)
        self.normalizer = AerospaceTextNormalizer()
        self.voices = self.kokoro.get_voices()
        print(f"Loaded neural model in {time.time() - t0:.2f}s. Available voice vectors: {len(self.voices)}")

        # Precompute Sonon domain voice profiles
        self.voice_profiles = self._build_voice_profiles()

    def _build_voice_profiles(self) -> dict:
        profiles = {}

        # 1. Flight Specialist Female (Warm, clear, natural flight-deck cadence)
        v_heart = self.kokoro.get_voice_style("af_heart")
        v_bella = self.kokoro.get_voice_style("af_bella")
        profiles["flight_specialist_female"] = 0.75 * v_heart + 0.25 * v_bella

        # 2. Flight Commander Male (Deep, authoritative, crisp articulation)
        v_adam = self.kokoro.get_voice_style("am_adam")
        v_michael = self.kokoro.get_voice_style("am_michael")
        profiles["flight_commander_male"] = 0.65 * v_adam + 0.35 * v_michael

        # 3. Conversational AI Natural (Studio-grade expressive human speech)
        profiles["conversational_natural"] = v_heart

        # 4. Tactical Operations Male (Crisp, fast cadence)
        v_onyx = self.kokoro.get_voice_style("am_onyx") if "am_onyx" in self.voices else v_adam
        profiles["tactical_male"] = 0.5 * v_adam + 0.5 * v_onyx

        # 5. Mission Dispatch British (International aviation standard)
        v_george = self.kokoro.get_voice_style("bm_george")
        profiles["mission_dispatch"] = v_george

        return profiles

    def synthesize(
        self,
        text: str,
        profile_name: str = "conversational_natural",
        urgency: str = "calm",
        speed: float = 1.0,
        apply_mastering: bool = True,
        apply_aerospace_norm: bool = True,
    ) -> tuple[np.ndarray, int]:
        """Synthesize high-fidelity speech from text using specified voice profile and situational urgency."""
        if apply_aerospace_norm:
            norm_text = self.normalizer.normalize(text)
        else:
            norm_text = text

        if profile_name in self.voice_profiles:
            voice_style = np.copy(self.voice_profiles[profile_name])
        elif profile_name in self.voices:
            voice_style = np.copy(self.kokoro.get_voice_style(profile_name))
        else:
            voice_style = np.copy(self.voice_profiles["conversational_natural"])

        # Urgency prosody modulation: modify speed and style latent energy
        effective_speed = speed
        if urgency == "caution":
            effective_speed *= 1.08
            # Modulate style vector tension
            voice_style *= 1.05
        elif urgency == "emergency":
            effective_speed *= 1.18
            voice_style *= 1.12

        t0 = time.time()
        raw_samples, sample_rate = self.kokoro.create(
            text=norm_text,
            voice=voice_style,
            speed=effective_speed,
            lang="en-us",
        )
        synth_time = time.time() - t0
        audio_dur = len(raw_samples) / sample_rate
        rtf = synth_time / max(audio_dur, 0.01)

        print(
            f"Synthesized '{norm_text[:35]}...' [{profile_name}|{urgency}] in {synth_time:.2f}s "
            f"(Duration: {audio_dur:.2f}s, RTF: {rtf:.3f})"
        )

        if apply_mastering:
            mastered_samples = self.master.process(raw_samples)
            return mastered_samples, sample_rate
        else:
            return raw_samples, sample_rate


def run_test_synthesis():
    """Run baseline test synthesis suite and write audio files."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DOCS_AUDIO_DIR, exist_ok=True)

    engine = SononNeuralVoiceEngine()

    test_cases = [
        {
            "filename": "sonon_neural_waypoint_reached.wav",
            "profile": "flight_specialist_female",
            "speed": 1.0,
            "text": "Waypoint reached. Autopilot online. All flight parameters nominal.",
            "description": "Direct benchmark against former robotic formant synthesis",
        },
        {
            "filename": "sonon_neural_commander_takeoff.wav",
            "profile": "flight_commander_male",
            "speed": 0.98,
            "text": "Tower, this is flight lead. Pre-flight checks are complete, engines stabilized, cleared for immediate departure.",
            "description": "Authoritative male commander voice for flight maneuvers",
        },
        {
            "filename": "sonon_neural_conversational_intro.wav",
            "profile": "conversational_natural",
            "speed": 1.0,
            "text": "Hello, I am Sonon, Aerovex's neural speech intelligence system. This is our foundation acoustic model running deep learning inference.",
            "description": "Natural human conversational cadence and warmth",
        },
        {
            "filename": "sonon_neural_tactical_status.wav",
            "profile": "tactical_male",
            "speed": 1.05,
            "text": "System health normal. Radar contact acquired at three zero miles. Echolocation and acoustic beamforming synchronized.",
            "description": "Fast-paced tactical operational status",
        },
        {
            "filename": "sonon_neural_whisper_grade_clarity.wav",
            "profile": "flight_specialist_female",
            "urgency": "calm",
            "speed": 0.95,
            "text": "Our objective is to deliver speech synthesis and recognition as accurate and natural as human speech, across both cloud servers and embedded aerospace hardware.",
            "description": "High-fidelity industrial speech AI mission statement",
        },
        {
            "filename": "sonon_neural_urgency_calm.wav",
            "profile": "conversational_natural",
            "urgency": "calm",
            "speed": 1.0,
            "text": "Traffic advisory. Inbound aircraft five miles south, level at five thousand feet.",
            "description": "Situational urgency: calm traffic advisory",
        },
        {
            "filename": "sonon_neural_urgency_caution.wav",
            "profile": "flight_specialist_female",
            "urgency": "caution",
            "speed": 1.0,
            "text": "Caution. Terrain ahead. Pull up. Terrain ahead.",
            "description": "Situational urgency: terrain caution alert",
        },
        {
            "filename": "sonon_neural_urgency_emergency.wav",
            "profile": "flight_commander_male",
            "urgency": "emergency",
            "speed": 1.0,
            "text": "Warning. Engine flameout detected on left turbine. Initiate restart sequence immediately.",
            "description": "Situational urgency: emergency tactical alert",
        },
        {
            "filename": "sonon_neural_aerospace_expansion.wav",
            "profile": "flight_specialist_female",
            "urgency": "calm",
            "speed": 1.0,
            "text": "FL350 heading HDG090 cleared ILS approach RWY28R. UAV monitoring TCAS.",
            "description": "Aerospace domain acronym and flight level phonetic expansion",
        },
    ]

    generated_paths = []
    print("\n--- Generating Neural Speech Audio Samples ---")
    for tc in test_cases:
        audio, sr = engine.synthesize(
            text=tc["text"],
            profile_name=tc["profile"],
            urgency=tc.get("urgency", "calm"),
            speed=tc.get("speed", 1.0),
            apply_mastering=True,
            apply_aerospace_norm=True,
        )

        out_path = os.path.join(OUTPUT_DIR, tc["filename"])
        docs_path = os.path.join(DOCS_AUDIO_DIR, tc["filename"])

        sf.write(out_path, audio, sr, subtype='PCM_16')
        sf.write(docs_path, audio, sr, subtype='PCM_16')

        file_size_kb = os.path.getsize(out_path) / 1024.0
        duration_s = len(audio) / sr
        print(f"Generated: {out_path} ({duration_s:.2f}s, {file_size_kb:.1f} KB)")
        generated_paths.append((tc["filename"], out_path, duration_s, tc["description"]))

    print("\nAll neural speech files generated successfully.")
    return generated_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sonon Neural Speech Synthesizer")
    parser.add_argument("--test", action="store_true", help="Run full test synthesis suite")
    parser.add_argument("--text", type=str, help="Text to synthesize")
    parser.add_argument("--voice", type=str, default="conversational_natural", help="Voice profile")
    parser.add_argument("--urgency", type=str, default="calm", choices=["calm", "caution", "emergency"], help="Situational urgency level")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech rate multiplier")
    parser.add_argument("--out", type=str, default="/tmp/sonon_output.wav", help="Output file path")
    args = parser.parse_args()

    if args.test or not args.text:
        run_test_synthesis()
    else:
        engine = SononNeuralVoiceEngine()
        audio, sr = engine.synthesize(
            args.text,
            profile_name=args.voice,
            urgency=args.urgency,
            speed=args.speed,
        )
        sf.write(args.out, audio, sr, subtype='PCM_16')
        print(f"Wrote output to {args.out}")
