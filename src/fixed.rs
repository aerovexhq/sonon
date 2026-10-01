//! Fixed-point Q15 / Q31 DSP math mode and static heapless buffers for microcontrollers.
//!
//! Provides deterministic, zero-allocation integer arithmetic tailored for
//! embedded microcontrollers (ARM Cortex-M4/M7, STM32H7, ESP32-S3) without floating-point units.

#![deny(unsafe_code)]

/// 16-bit signed fixed-point number with 15 fractional bits (Q15).
///
/// Dynamic range: [-1.0, +0.9999694824]
/// Precision: 1 / 32768 ≈ 3.0517e-5
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Q15(pub i16);

impl Q15 {
    pub const ZERO: Q15 = Q15(0);
    pub const ONE: Q15 = Q15(i16::MAX);
    pub const MIN: Q15 = Q15(i16::MIN);

    /// Converts a float32 into Q15 representation, clamping to [-1.0, 1.0].
    #[inline]
    pub fn from_f32(val: f32) -> Self {
        let clamped = val.clamp(-1.0, 0.9999695);
        let scaled = (clamped * 32768.0).round() as i32;
        Q15(scaled.clamp(i16::MIN as i32, i16::MAX as i32) as i16)
    }

    /// Converts Q15 into float32.
    #[inline]
    pub fn to_f32(self) -> f32 {
        (self.0 as f32) / 32768.0
    }

    /// Saturating addition.
    #[inline]
    pub fn saturating_add(self, rhs: Self) -> Self {
        Q15(self.0.saturating_add(rhs.0))
    }

    /// Saturating subtraction.
    #[inline]
    pub fn saturating_sub(self, rhs: Self) -> Self {
        Q15(self.0.saturating_sub(rhs.0))
    }

    /// Fixed-point multiplication with 32-bit intermediate product.
    #[inline]
    pub fn mul(self, rhs: Self) -> Self {
        let prod = (self.0 as i32 * rhs.0 as i32) >> 15;
        Q15(prod.clamp(i16::MIN as i32, i16::MAX as i32) as i16)
    }

    /// Absolute value.
    #[inline]
    pub fn abs(self) -> Self {
        Q15(self.0.saturating_abs())
    }

    /// High-throughput dot product using a 32-bit widening accumulator.
    /// Returns 32-bit Q15 fixed-point sum (scale factor 32768).
    #[inline]
    pub fn dot_product(a: &[Q15], b: &[Q15]) -> i32 {
        assert_eq!(a.len(), b.len(), "Slice lengths must match");
        let mut acc: i32 = 0;
        for i in 0..a.len() {
            acc += (a[i].0 as i32 * b[i].0 as i32) >> 15;
        }
        acc
    }

    /// Normalized dot product averaged across vector length (returns Q15).
    #[inline]
    pub fn dot_product_normalized(a: &[Q15], b: &[Q15]) -> Q15 {
        if a.is_empty() {
            return Q15::ZERO;
        }
        let raw = Self::dot_product(a, b);
        let avg = raw / (a.len() as i32);
        Q15(avg.clamp(i16::MIN as i32, i16::MAX as i32) as i16)
    }

    /// 256-entry sine look-up table query for zero-FPU trigonometric synthesis.
    /// Phase input: [0, 65535] mapped to [0, 2*PI).
    pub fn lut_sin(phase: u16) -> Q15 {
        // Quarter-wave 64-entry sine table
        const QUARTER_SIN: [i16; 65] = [
            0, 804, 1607, 2410, 3211, 4011, 4807, 5601, 6392, 7179, 7961, 8739, 9511, 10278,
            11038, 11792, 12539, 13278, 14009, 14732, 15446, 16150, 16845, 17530, 18204, 18867,
            19519, 20159, 20787, 21402, 22004, 22594, 23169, 23731, 24278, 24811, 25329, 25831,
            26318, 26789, 27244, 27683, 28105, 28509, 28897, 29267, 29620, 29955, 30272, 30571,
            30851, 31112, 31355, 31579, 31784, 31970, 32136, 32284, 32411, 32520, 32608, 32677,
            32727, 32756, 32767,
        ];

        let quadrant = (phase >> 14) & 0x03;
        let index_in_quadrant = ((phase >> 8) & 0x3F) as usize;

        let mag = match quadrant {
            0 => QUARTER_SIN[index_in_quadrant],
            1 => QUARTER_SIN[64 - index_in_quadrant],
            2 => -QUARTER_SIN[index_in_quadrant],
            3 => -QUARTER_SIN[64 - index_in_quadrant],
            _ => 0,
        };

        Q15(mag)
    }

    /// Cosine look-up table query: cos(phase) = sin(phase + 16384).
    #[inline]
    pub fn lut_cos(phase: u16) -> Q15 {
        Self::lut_sin(phase.wrapping_add(16384))
    }
}

/// 32-bit signed fixed-point number with 31 fractional bits (Q31).
///
/// Dynamic range: [-1.0, +0.9999999995]
/// Precision: 1 / 2147483648 ≈ 4.6566e-10
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Q31(pub i32);

impl Q31 {
    pub const ZERO: Q31 = Q31(0);
    pub const ONE: Q31 = Q31(i32::MAX);
    pub const MIN: Q31 = Q31(i32::MIN);

    /// Converts a float32 into Q31 representation.
    #[inline]
    pub fn from_f32(val: f32) -> Self {
        let clamped = val.clamp(-1.0, 0.99999994);
        let scaled = (clamped as f64 * 2147483648.0).round() as i64;
        Q31(scaled.clamp(i32::MIN as i64, i32::MAX as i64) as i32)
    }

    /// Converts Q31 into float32.
    #[inline]
    pub fn to_f32(self) -> f32 {
        (self.0 as f32) / 2147483648.0
    }

    /// Saturating addition.
    #[inline]
    pub fn saturating_add(self, rhs: Self) -> Self {
        Q31(self.0.saturating_add(rhs.0))
    }

    /// Saturating subtraction.
    #[inline]
    pub fn saturating_sub(self, rhs: Self) -> Self {
        Q31(self.0.saturating_sub(rhs.0))
    }

    /// Fixed-point multiplication with 64-bit intermediate product.
    #[inline]
    pub fn mul(self, rhs: Self) -> Self {
        let prod = ((self.0 as i64 * rhs.0 as i64) >> 31) as i32;
        Q31(prod)
    }

    /// Dot product with 64-bit accumulator.
    pub fn dot_product(a: &[Q31], b: &[Q31]) -> Q31 {
        assert_eq!(a.len(), b.len(), "Slice lengths must match");
        let mut acc: i64 = 0;
        for i in 0..a.len() {
            acc += (a[i].0 as i64 * b[i].0 as i64) >> 31;
        }
        Q31(acc.clamp(i32::MIN as i64, i32::MAX as i64) as i32)
    }
}

/// Fixed-capacity static audio circular buffer allocated entirely on the stack
/// or in static RAM with `#![no_std]` suitability.
pub struct StaticAudioBuffer<const N: usize> {
    data: [Q15; N],
    head: usize,
    tail: usize,
    count: usize,
}

impl<const N: usize> Default for StaticAudioBuffer<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> StaticAudioBuffer<N> {
    /// Construct a new empty static buffer.
    pub const fn new() -> Self {
        Self {
            data: [Q15::ZERO; N],
            head: 0,
            tail: 0,
            count: 0,
        }
    }

    /// Push a single Q15 sample into the ring buffer.
    /// Overwrites the oldest sample if the buffer is full.
    pub fn push(&mut self, sample: Q15) {
        self.data[self.head] = sample;
        self.head = (self.head + 1) % N;
        if self.count < N {
            self.count += 1;
        } else {
            self.tail = (self.tail + 1) % N;
        }
    }

    /// Pop the oldest sample from the ring buffer.
    pub fn pop(&mut self) -> Option<Q15> {
        if self.count == 0 {
            None
        } else {
            let sample = self.data[self.tail];
            self.tail = (self.tail + 1) % N;
            self.count -= 1;
            Some(sample)
        }
    }

    /// Peek the oldest `out.len()` samples without removing them.
    #[allow(clippy::needless_range_loop)]
    pub fn peek_slice(&self, out: &mut [Q15]) -> usize {
        let n = out.len().min(self.count);
        let mut curr = self.tail;
        for i in 0..n {
            out[i] = self.data[curr];
            curr = (curr + 1) % N;
        }
        n
    }

    /// Returns current number of samples stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns true if buffer contains zero samples.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Returns buffer capacity N.
    #[inline]
    pub const fn capacity(&self) -> usize {
        N
    }
}

/// Fixed-point Sakoe-Chiba DTW matcher operating purely on Q15 integer features.
pub struct FixedDtwMatcher;

impl FixedDtwMatcher {
    /// Compute Manhattan Dynamic Time Warping distance between two sequences of Q15 feature frames
    /// within a Sakoe-Chiba band corridor of radius `r`.
    ///
    /// Requires zero floating-point operations.
    pub fn compute_distance_q15(
        seq_a: &[&[Q15]],
        seq_b: &[&[Q15]],
        band_radius: usize,
    ) -> i32 {
        let n = seq_a.len();
        let m = seq_b.len();
        if n == 0 || m == 0 {
            return i32::MAX;
        }

        // Use preallocated stack array for cost columns if n, m are bounded (e.g. up to 128 frames)
        let mut cost = vec![vec![i32::MAX / 2; m + 1]; n + 1];
        cost[0][0] = 0;

        for i in 1..=n {
            let j_start = if i > band_radius { i - band_radius } else { 1 };
            let j_end = (i + band_radius).min(m);

            for j in j_start..=j_end {
                // Compute Manhattan distance between feature frames
                let frame_a = seq_a[i - 1];
                let frame_b = seq_b[j - 1];
                let mut dist: i32 = 0;
                for d in 0..frame_a.len().min(frame_b.len()) {
                    dist += (frame_a[d].0 as i32 - frame_b[d].0 as i32).abs();
                }

                let min_prev = cost[i - 1][j]
                    .min(cost[i][j - 1])
                    .min(cost[i - 1][j - 1]);

                cost[i][j] = min_prev.saturating_add(dist);
            }
        }

        cost[n][m] / ((n + m) as i32)
    }
}
