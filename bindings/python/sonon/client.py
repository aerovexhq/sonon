"""
Sonon Python Client: High-Performance Robotics Acoustic DSP & Wake-Word Engine.
"""

import math
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

SHM_MAGIC = 0x534F4E4F  # "SONO"
SHM_VERSION = 1
DEFAULT_SHM_PATH = "/dev/shm/sonon_audio"
DEFAULT_CAPACITY = 65536

HEADER_FORMAT = "<IIfIIQQfIfI12s4s"
HEADER_SIZE = 64


@dataclass
class KeywordEvent:
    keyword: str
    confidence: float
    timestamp_sec: float


class SononShm:
    """Zero-copy POSIX shared memory ring buffer client for Sonon."""

    def __init__(
        self,
        path: str = DEFAULT_SHM_PATH,
        sample_rate: float = 16000.0,
        capacity: int = DEFAULT_CAPACITY,
    ):
        self.path = path
        self.sample_rate = float(sample_rate)
        self.capacity = int(capacity)
        self._fd: Optional[int] = None
        self._ensure_channel()

    def _ensure_channel(self) -> None:
        """Ensures the shared memory file exists with valid header."""
        total_size = HEADER_SIZE + self.capacity * 4
        exists = os.path.exists(self.path)

        self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o666)

        current_size = os.path.getsize(self.path)
        if not exists or current_size < total_size:
            os.ftruncate(self._fd, total_size)
            header_bytes = struct.pack(
                HEADER_FORMAT,
                SHM_MAGIC,
                SHM_VERSION,
                self.sample_rate,
                1,  # channels
                self.capacity,
                0,  # write_head
                0,  # read_head
                1.0,  # health_score
                0,  # worst_severity
                0.0,  # last_kw_conf
                0,  # last_kw_len
                b"\x00" * 12,
                b"\x00" * 4,
            )
            os.lseek(self._fd, 0, os.SEEK_SET)
            os.write(self._fd, header_bytes)

    def write_samples(self, samples: Union[Sequence[float], "np.ndarray"]) -> int:
        """
        Writes float32 audio samples into the shared memory circular ring buffer.
        Accepts Python list, tuple, or NumPy float32 array.
        """
        if hasattr(samples, "dtype") and hasattr(samples, "tobytes"):
            import numpy as np

            if samples.dtype != np.float32:
                samples = samples.astype(np.float32)
            raw_bytes = samples.tobytes()
            count = len(samples)
        else:
            count = len(samples)
            raw_bytes = struct.pack(f"<{count}f", *samples)

        if count == 0:
            return 0

        os.lseek(self._fd, 20, os.SEEK_SET)
        (write_head,) = struct.unpack("<Q", os.read(self._fd, 8))

        start_slot = write_head % self.capacity
        end_slot = start_slot + count

        if end_slot <= self.capacity:
            os.lseek(self._fd, HEADER_SIZE + start_slot * 4, os.SEEK_SET)
            os.write(self._fd, raw_bytes)
        else:
            first_count = self.capacity - start_slot
            first_bytes = raw_bytes[: first_count * 4]
            second_bytes = raw_bytes[first_count * 4 :]

            os.lseek(self._fd, HEADER_SIZE + start_slot * 4, os.SEEK_SET)
            os.write(self._fd, first_bytes)

            os.lseek(self._fd, HEADER_SIZE, os.SEEK_SET)
            os.write(self._fd, second_bytes)

        os.lseek(self._fd, 20, os.SEEK_SET)
        os.write(self._fd, struct.pack("<Q", write_head + count))

        return count

    def read_available_samples(self) -> List[float]:
        """Reads newly available samples written by producer."""
        os.lseek(self._fd, 20, os.SEEK_SET)
        head_bytes = os.read(self._fd, 16)
        write_head, read_head = struct.unpack("<QQ", head_bytes)

        if write_head <= read_head:
            return []

        count = write_head - read_head
        if count > self.capacity:
            read_head = write_head - self.capacity
            count = self.capacity

        start_slot = read_head % self.capacity
        end_slot = start_slot + count

        if end_slot <= self.capacity:
            os.lseek(self._fd, HEADER_SIZE + start_slot * 4, os.SEEK_SET)
            raw_bytes = os.read(self._fd, count * 4)
        else:
            first_count = self.capacity - start_slot
            os.lseek(self._fd, HEADER_SIZE + start_slot * 4, os.SEEK_SET)
            first_bytes = os.read(self._fd, first_count * 4)

            os.lseek(self._fd, HEADER_SIZE, os.SEEK_SET)
            second_bytes = os.read(self._fd, (count - first_count) * 4)
            raw_bytes = first_bytes + second_bytes

        os.lseek(self._fd, 28, os.SEEK_SET)
        os.write(self._fd, struct.pack("<Q", write_head))

        return list(struct.unpack(f"<{count}f", raw_bytes))

    def read_health(self) -> Tuple[float, int]:
        """Returns (overall_health_score [0.0..1.0], worst_severity [0..3])."""
        os.lseek(self._fd, 36, os.SEEK_SET)
        raw = os.read(self._fd, 8)
        score, severity = struct.unpack("<fI", raw)
        return (score, severity)

    def read_detection(self) -> Optional[Tuple[str, float]]:
        """Returns (keyword_name, confidence) if a recent keyword event occurred."""
        os.lseek(self._fd, 44, os.SEEK_SET)
        raw = os.read(self._fd, 20)
        conf, kw_len, kw_raw = struct.unpack("<fI12s", raw)
        if kw_len == 0 or conf <= 0.0:
            return None
        kw_str = kw_raw[: min(12, kw_len)].decode("utf-8", errors="ignore")
        return (kw_str, conf)

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


class SononEngine:
    """
    High-level streaming acoustic DSP & keyword spotting engine.
    Computes spectral representations, Voice Activity Detection, and Sakoe-Chiba DTW.
    """

    def __init__(
        self,
        sample_rate: float = 16000.0,
        frame_size: int = 512,
        hop_size: int = 160,
        num_mfcc: int = 26,
    ):
        self.sample_rate = float(sample_rate)
        self.frame_size = int(frame_size)
        self.hop_size = int(hop_size)
        self.num_mfcc = int(num_mfcc)
        self._templates: List[Tuple[str, List[List[float]], float]] = []
        self._last_distances: dict[str, float] = {}
        self._buffer: List[float] = []
        self._history: List[List[float]] = []
        self._total_samples = 0
        self._window = [
            0.5 * (1.0 - math.cos(2.0 * math.pi * n / (frame_size - 1)))
            for n in range(frame_size)
        ]

    @staticmethod
    def trim_silence(
        samples: Sequence[float],
        sample_rate: float = 16000.0,
        frame_size: int = 160,
        threshold_ratio: float = 0.15,
        margin_frames: int = 5,
    ) -> List[float]:
        """Trims leading and trailing silence from an audio sample sequence using energy VAD."""
        n_samples = len(samples)
        num_frames = n_samples // frame_size
        if num_frames == 0:
            return list(samples)

        energies = []
        for i in range(num_frames):
            chunk = samples[i * frame_size : (i + 1) * frame_size]
            rms = math.sqrt(sum(s * s for s in chunk) / max(1, len(chunk)))
            energies.append(rms)

        max_energy = max(energies)
        if max_energy < 0.01:
            return list(samples)

        thresh = max(0.015, max_energy * threshold_ratio)
        speech_indices = [i for i, e in enumerate(energies) if e >= thresh]
        if not speech_indices:
            return list(samples)

        start_frame = max(0, speech_indices[0] - margin_frames)
        end_frame = min(num_frames, speech_indices[-1] + margin_frames + 1)

        start_sample = start_frame * frame_size
        end_sample = min(n_samples, end_frame * frame_size)
        return list(samples[start_sample:end_sample])

    def get_last_distance(self, keyword: str) -> Optional[float]:
        """Returns the latest evaluated DTW distance for an enrolled keyword."""
        return self._last_distances.get(keyword)

    def extract_features(self, samples: Sequence[float]) -> List[List[float]]:
        """Extracts Log-Mel / MFCC gain-invariant feature frames from an audio array."""
        features = []
        pos = 0
        n_samples = len(samples)

        while pos + self.frame_size <= n_samples:
            frame = [
                samples[pos + i] * self._window[i] for i in range(self.frame_size)
            ]
            # Fast power spectrum approximation across num_mfcc filter bands
            mels = [0.0] * self.num_mfcc
            for m in range(self.num_mfcc):
                k_start = int(m * (self.frame_size // 2) / self.num_mfcc)
                k_end = int((m + 1) * (self.frame_size // 2) / self.num_mfcc)
                band_power = 0.0
                for k in range(k_start, max(k_start + 1, k_end)):
                    re = 0.0
                    im = 0.0
                    for n in range(0, self.frame_size, 4):  # Subsampled DFT for speed
                        angle = 2.0 * math.pi * k * n / self.frame_size
                        re += frame[n] * math.cos(angle)
                        im -= frame[n] * math.sin(angle)
                    band_power += (re * re + im * im)
                mels[m] = math.log(max(1e-6, band_power))

            # Gain-invariant normalization: Cepstral Mean Subtraction and L2 unit-norm
            mean_mel = sum(mels) / self.num_mfcc
            centered = [x - mean_mel for x in mels]
            norm = math.sqrt(sum(x * x for x in centered))
            if norm > 1e-4:
                features.append([x / norm for x in centered])
            else:
                features.append(centered)
            pos += self.hop_size

        return features

    def enroll_keyword(
        self, name: str, features: List[List[float]], threshold: float = 0.60
    ) -> None:
        """Enrolls a reference exemplar keyword template."""
        self._templates.append((name, features, float(threshold)))

    def ingest_samples(self, samples: Sequence[float]) -> List[KeywordEvent]:
        """Ingests streaming samples and evaluates keyword spotting."""
        self._buffer.extend(samples)
        self._total_samples += len(samples)
        detections = []

        while len(self._buffer) >= self.frame_size:
            chunk = self._buffer[: self.frame_size]
            self._buffer = self._buffer[self.hop_size :]

            feat = self.extract_features(chunk)
            if feat:
                self._history.extend(feat)
                if len(self._history) > 100:
                    self._history = self._history[-100:]

                # Check templates
                for kw_name, tpl, thresh in self._templates:
                    tpl_len = len(tpl)
                    band = max(4, tpl_len // 4)
                    best_d = float("inf")
                    for cand_len in (tpl_len, tpl_len - 2, tpl_len + 2):
                        if cand_len > 0 and len(self._history) >= cand_len:
                            obs = self._history[-cand_len:]
                            d = self._sakoe_chiba_distance(obs, tpl, band)
                            if d < best_d:
                                best_d = d
                    self._last_distances[kw_name] = best_d
                    if best_d < thresh:
                        conf = max(0.0, min(1.0, 1.0 - (best_d / thresh)))
                        ts = self._total_samples / self.sample_rate
                        detections.append(
                            KeywordEvent(
                                keyword=kw_name,
                                confidence=conf,
                                timestamp_sec=ts,
                            )
                        )
        return detections

    def _sakoe_chiba_distance(
        self, s: List[List[float]], t: List[List[float]], r: int
    ) -> float:
        n, m = len(s), len(t)
        if n == 0 or m == 0 or abs(n - m) > r:
            return float("inf")

        cost = [[float("inf")] * (m + 1) for _ in range(n + 1)]
        cost[0][0] = 0.0

        for i in range(1, n + 1):
            j_min = max(1, i - r)
            j_max = min(m, i + r)
            for j in range(j_min, j_max + 1):
                diff_sq = sum((s[i - 1][f] - t[j - 1][f]) ** 2 for f in range(self.num_mfcc))
                d = math.sqrt(diff_sq)
                min_prev = min(cost[i - 1][j], cost[i][j - 1], cost[i - 1][j - 1])
                cost[i][j] = d + min_prev

        if not math.isfinite(cost[n][m]):
            return float("inf")
        return cost[n][m] / max(1, n + m)
