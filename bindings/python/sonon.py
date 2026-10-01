"""
Sonon Python Client: POSIX Shared Memory Interface.

Provides zero-copy, ultra-low-latency audio streaming and acoustic health monitoring
integration between Python autonomous agents / companion computers and the Sonon DSP engine.
"""

import os
import struct
import time
from typing import List, Optional, Sequence, Tuple, Union

SHM_MAGIC = 0x534F4E4F  # "SONO"
SHM_VERSION = 1
DEFAULT_SHM_PATH = "/dev/shm/sonon_audio"
DEFAULT_CAPACITY = 65536

# Header layout (64 bytes):
# magic (I), version (I), sample_rate (f), channels (I), capacity (I) -> 20 bytes
# write_head (Q), read_head (Q) -> 16 bytes
# health_score (f), worst_severity (I), last_kw_conf (f), last_kw_len (I) -> 16 bytes
# last_keyword (12s), reserved (4s) -> 16 bytes
HEADER_FORMAT = "<IIfIIQQfIfI12s4s"
HEADER_SIZE = 64


class SononShm:
    """Zero-copy POSIX shared memory ring buffer client for Sonon."""

    def __init__(self, path: str = DEFAULT_SHM_PATH, sample_rate: float = 16000.0, capacity: int = DEFAULT_CAPACITY):
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
            # NumPy array zero-copy serialization
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

        # Read current write_head
        os.lseek(self._fd, 20, os.SEEK_SET)
        (write_head,) = struct.unpack("<Q", os.read(self._fd, 8))

        start_slot = write_head % self.capacity
        end_slot = start_slot + count

        if end_slot <= self.capacity:
            # Single contiguous write
            os.lseek(self._fd, HEADER_SIZE + start_slot * 4, os.SEEK_SET)
            os.write(self._fd, raw_bytes)
        else:
            # Wrapped ring buffer write
            first_count = self.capacity - start_slot
            first_bytes = raw_bytes[: first_count * 4]
            second_bytes = raw_bytes[first_count * 4 :]

            os.lseek(self._fd, HEADER_SIZE + start_slot * 4, os.SEEK_SET)
            os.write(self._fd, first_bytes)

            os.lseek(self._fd, HEADER_SIZE, os.SEEK_SET)
            os.write(self._fd, second_bytes)

        # Update write_head in header
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

        # Advance read_head
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
