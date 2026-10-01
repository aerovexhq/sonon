//! Fixed-capacity circular audio sample buffer for streaming audio ingestion.

/// High-throughput contiguous audio ring buffer.
#[derive(Debug, Clone)]
pub struct AudioRingBuffer {
    buffer: Vec<f32>,
    head: usize,
    count: usize,
    capacity: usize,
}

impl AudioRingBuffer {
    /// Create a new ring buffer with the given sample capacity.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "Capacity must be non-zero");
        Self {
            buffer: vec![0.0; capacity],
            head: 0,
            count: 0,
            capacity,
        }
    }

    /// Push an audio slice into the buffer, overwriting oldest samples if capacity is exceeded.
    pub fn push_slice(&mut self, samples: &[f32]) {
        for &sample in samples {
            let write_pos = (self.head + self.count) % self.capacity;
            self.buffer[write_pos] = sample;
            if self.count < self.capacity {
                self.count += 1;
            } else {
                self.head = (self.head + 1) % self.capacity;
            }
        }
    }

    /// Read the latest `n` samples in chronological order into `out`.
    /// Returns true if at least `n` samples were available.
    pub fn read_latest(&self, n: usize, out: &mut [f32]) -> bool {
        if n > self.count || n > out.len() {
            return false;
        }

        let start_offset = self.count - n;
        for i in 0..n {
            let idx = (self.head + start_offset + i) % self.capacity;
            out[i] = self.buffer[idx];
        }
        true
    }

    /// Peek the oldest `n` samples in chronological order into `out` without advancing.
    /// Returns true if at least `n` samples were available.
    pub fn peek(&self, n: usize, out: &mut [f32]) -> bool {
        if n > self.count || n > out.len() {
            return false;
        }

        for i in 0..n {
            let idx = (self.head + i) % self.capacity;
            out[i] = self.buffer[idx];
        }
        true
    }

    /// Discard the oldest `n` samples from the ring buffer.
    /// Returns the number of samples actually discarded.
    pub fn pop_front(&mut self, n: usize) -> usize {
        let to_pop = n.min(self.count);
        self.head = (self.head + to_pop) % self.capacity;
        self.count -= to_pop;
        to_pop
    }

    /// Return current number of buffered samples.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Return total buffer capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Clear all buffered samples.
    pub fn clear(&mut self) {
        self.head = 0;
        self.count = 0;
    }
}
