//! Ultra-Low-Power RISC-V Vector (RVV 1.0) and PULP-NN / XpulpNN micro-engine acceleration.
//!
//! Provides deterministic, zero-heap `#![no_std]` capable primitives, packed SIMD dot products,
//! scalable vector operations, static memory arenas, and battery-perched surveillance power modeling (< 1 mW).

#![deny(unsafe_code)]

use crate::health::MavlinkNamedValueFloat;
use serde::{Deserialize, Serialize};

/// Packed 4x 8-bit signed integer vector for XpulpNN SIMD dot products (`pv.dotsp.b`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PackedI8x4(pub [i8; 4]);

/// Packed 2x 16-bit signed integer vector for XpulpNN halfword SIMD (`pv.dotsp.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PackedI16x2(pub [i16; 2]);

/// PULP-NN / XpulpNN ISA Extension emulation kernel.
///
/// Implements custom packed SIMD instructions developed by ETH Zürich and University of Bologna
/// for the PULP platform (GAP8, GAP9, Siracusa, Vega):
/// - `pv.dotsp.b`: 4-element 8-bit signed vector dot product with 32-bit accumulation.
/// - `pv.dotsp.h`: 2-element 16-bit signed vector dot product with 32-bit accumulation.
/// - `pv.sad.h`: 2-element 16-bit sum of absolute differences for Dynamic Time Warping (DTW).
/// - `pv.clip`: single-cycle hardware saturation.
pub struct PulpVectorEngine;

impl PulpVectorEngine {
    /// 4-way 8-bit signed vector dot product (`pv.dotsp.b`).
    /// Computes sum(a[i] * b[i]) for i in 0..4 in a single cycle.
    #[inline]
    pub fn pv_dotsp_b(a: [i8; 4], b: [i8; 4]) -> i32 {
        (a[0] as i32 * b[0] as i32)
            + (a[1] as i32 * b[1] as i32)
            + (a[2] as i32 * b[2] as i32)
            + (a[3] as i32 * b[3] as i32)
    }

    /// 4-way 8-bit unsigned vector dot product (`pv.dotup.b`).
    #[inline]
    pub fn pv_dotup_b(a: [u8; 4], b: [u8; 4]) -> u32 {
        (a[0] as u32 * b[0] as u32)
            + (a[1] as u32 * b[1] as u32)
            + (a[2] as u32 * b[2] as u32)
            + (a[3] as u32 * b[3] as u32)
    }

    /// 2-way 16-bit signed halfword vector dot product (`pv.dotsp.h`).
    #[inline]
    pub fn pv_dotsp_h(a: [i16; 2], b: [i16; 2]) -> i32 {
        (a[0] as i32 * b[0] as i32) + (a[1] as i32 * b[1] as i32)
    }

    /// 2-way 16-bit sum of absolute differences (`pv.sad.h`).
    /// Computes |a[0] - b[0]| + |a[1] - b[1]| in a single cycle for rapid DTW matching.
    #[inline]
    pub fn pv_sad_h(a: [i16; 2], b: [i16; 2]) -> u32 {
        let diff0 = (a[0] as i32 - b[0] as i32).abs() as u32;
        let diff1 = (a[1] as i32 - b[1] as i32).abs() as u32;
        diff0 + diff1
    }

    /// 2-way 16-bit packed absolute value (`pv.sabs.h`).
    #[inline]
    pub fn pv_sabs_h(a: [i16; 2]) -> [i16; 2] {
        [a[0].saturating_abs(), a[1].saturating_abs()]
    }

    /// Single-cycle hardware clipping/saturation (`p.clip`).
    #[inline]
    pub fn pv_clip_q15(val: i32, min_val: i16, max_val: i16) -> i16 {
        val.clamp(min_val as i32, max_val as i32) as i16
    }

    /// 2-way 16-bit packed minimum (`pv.min.h`).
    #[inline]
    pub fn pv_min_h(a: [i16; 2], b: [i16; 2]) -> [i16; 2] {
        [a[0].min(b[0]), a[1].min(b[1])]
    }

    /// 2-way 16-bit packed maximum (`pv.max.h`).
    #[inline]
    pub fn pv_max_h(a: [i16; 2], b: [i16; 2]) -> [i16; 2] {
        [a[0].max(b[0]), a[1].max(b[1])]
    }

    /// High-throughput Q15 vector dot product using 4-way SIMD loop unrolling.
    /// Returns 64-bit accumulator value.
    pub fn vector_dot_product_q15(a: &[i16], b: &[i16]) -> i64 {
        let n = a.len().min(b.len());
        let mut acc: i64 = 0;
        let mut i = 0;

        // Process in chunks of 4 elements (two pv.dotsp.h pairs)
        while i + 4 <= n {
            let p0 = Self::pv_dotsp_h([a[i], a[i + 1]], [b[i], b[i + 1]]);
            let p1 = Self::pv_dotsp_h([a[i + 2], a[i + 3]], [b[i + 2], b[i + 3]]);
            acc += (p0 as i64) + (p1 as i64);
            i += 4;
        }

        // Remainder loop
        while i < n {
            acc += (a[i] as i64) * (b[i] as i64);
            i += 1;
        }

        acc
    }

    /// Accelerated Manhattan distance between two Q15 feature frames using `pv.sad.h`.
    pub fn vector_dtw_distance_q15(frame_a: &[i16], frame_b: &[i16]) -> u32 {
        let n = frame_a.len().min(frame_b.len());
        let mut sum_dist: u32 = 0;
        let mut i = 0;

        while i + 2 <= n {
            sum_dist += Self::pv_sad_h([frame_a[i], frame_a[i + 1]], [frame_b[i], frame_b[i + 1]]);
            i += 2;
        }

        if i < n {
            sum_dist += (frame_a[i] as i32 - frame_b[i] as i32).unsigned_abs();
        }

        sum_dist
    }
}

/// RISC-V Vector Extension (RVV 1.0) configuration parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RvvConfig {
    /// Vector register length in bits (e.g. 128, 256, 512).
    pub vlen_bits: usize,
    /// Selected element width in bits (8, 16, 32).
    pub sew_bits: usize,
    /// Vector register grouping multiplier (1, 2, 4, 8).
    pub lmul: usize,
}

impl Default for RvvConfig {
    fn default() -> Self {
        Self {
            vlen_bits: 128, // Standard RV32/RV64 embedded vector length
            sew_bits: 16,   // Q15 standard audio element width
            lmul: 1,
        }
    }
}

impl RvvConfig {
    /// Maximum number of vector elements processed per vector instruction (VLMAX).
    #[inline]
    pub fn vlmax(&self) -> usize {
        let sew_bytes = (self.sew_bits / 8).max(1);
        (self.vlen_bits / 8 / sew_bytes) * self.lmul
    }
}

/// RISC-V Vector (RVV 1.0) scalable vectorization engine.
pub struct RvvVectorEngine {
    config: RvvConfig,
}

impl RvvVectorEngine {
    /// Creates a new RVV vector engine with specified configuration.
    pub fn new(config: RvvConfig) -> Self {
        Self { config }
    }

    /// Access active configuration.
    pub fn config(&self) -> &RvvConfig {
        &self.config
    }

    /// Vector fused multiply-accumulate on floating-point data (`vfmacc.vv`).
    /// Performs `acc[i] += a[i] * b[i]` across dynamic vector striping.
    pub fn vfmacc_f32(&self, acc: &mut [f32], a: &[f32], b: &[f32]) {
        let vlmax = self.config.vlmax().max(1);
        let n = acc.len().min(a.len()).min(b.len());
        let mut offset = 0;

        while offset < n {
            let vl = (n - offset).min(vlmax);
            for i in 0..vl {
                let idx = offset + i;
                acc[idx] += a[idx] * b[idx];
            }
            offset += vl;
        }
    }

    /// Widening vector multiply-accumulate from Q15 to Q31 (`vwmacc.vv`).
    /// Performs `acc[i] += (a[i] as i32) * (b[i] as i32)` without intermediate overflow.
    pub fn vwmacc_q15_to_q31(&self, acc: &mut [i32], a: &[i16], b: &[i16]) {
        let vlmax = self.config.vlmax().max(1);
        let n = acc.len().min(a.len()).min(b.len());
        let mut offset = 0;

        while offset < n {
            let vl = (n - offset).min(vlmax);
            for i in 0..vl {
                let idx = offset + i;
                acc[idx] += (a[idx] as i32) * (b[idx] as i32);
            }
            offset += vl;
        }
    }

    /// Tree-structured vector reduction sum (`vfredusum.vs`).
    pub fn vfredsum_f32(&self, vec: &[f32]) -> f32 {
        let mut sum = 0.0f32;
        let vlmax = self.config.vlmax().max(1);
        let mut offset = 0;

        while offset < vec.len() {
            let vl = (vec.len() - offset).min(vlmax);
            let mut chunk_sum = 0.0f32;
            for i in 0..vl {
                chunk_sum += vec[offset + i];
            }
            sum += chunk_sum;
            offset += vl;
        }

        sum
    }

    /// Vectorized FIR filter kernel.
    /// Convolves input signal with filter taps using vector registers.
    pub fn vector_fir_filter_f32(&self, input: &[f32], taps: &[f32], output: &mut [f32]) {
        let n_out = output.len().min(input.len().saturating_sub(taps.len() - 1));
        let num_taps = taps.len();

        for i in 0..n_out {
            let slice = &input[i..i + num_taps];
            let mut acc = 0.0f32;
            for k in 0..num_taps {
                acc += slice[num_taps - 1 - k] * taps[k];
            }
            output[i] = acc;
        }
    }
}

/// Static zero-heap memory arena for deterministic embedded microcontroller execution.
/// Guarantees strictly zero heap allocations (malloc/free) and deterministic O(1) allocation.
pub struct PulpMemoryArena<const AUDIO_CAP: usize = 1024, const FEAT_CAP: usize = 512, const MATRIX_CAP: usize = 1024> {
    pub audio_buffer: [i16; AUDIO_CAP],
    pub audio_len: usize,
    pub feature_buffer: [i16; FEAT_CAP],
    pub feature_len: usize,
    pub scratch_matrix: [i32; MATRIX_CAP],
    pub scratch_len: usize,
    pub peak_bytes_used: usize,
}

impl<const AUDIO_CAP: usize, const FEAT_CAP: usize, const MATRIX_CAP: usize> Default
    for PulpMemoryArena<AUDIO_CAP, FEAT_CAP, MATRIX_CAP>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const AUDIO_CAP: usize, const FEAT_CAP: usize, const MATRIX_CAP: usize>
    PulpMemoryArena<AUDIO_CAP, FEAT_CAP, MATRIX_CAP>
{
    /// Creates a new zero-initialized static memory arena.
    pub fn new() -> Self {
        Self {
            audio_buffer: [0i16; AUDIO_CAP],
            audio_len: 0,
            feature_buffer: [0i16; FEAT_CAP],
            feature_len: 0,
            scratch_matrix: [0i32; MATRIX_CAP],
            scratch_len: 0,
            peak_bytes_used: 0,
        }
    }

    /// Resets all arena allocation lengths to zero for the next frame.
    pub fn reset(&mut self) {
        self.audio_len = 0;
        self.feature_len = 0;
        self.scratch_len = 0;
    }

    /// Pushes audio samples into the static audio arena slice.
    pub fn push_audio(&mut self, samples: &[i16]) -> usize {
        let available = AUDIO_CAP.saturating_sub(self.audio_len);
        let count = samples.len().min(available);
        self.audio_buffer[self.audio_len..self.audio_len + count].copy_from_slice(&samples[..count]);
        self.audio_len += count;
        self.update_peak();
        count
    }

    /// Pushes feature values into the static feature arena slice.
    pub fn push_features(&mut self, features: &[i16]) -> usize {
        let available = FEAT_CAP.saturating_sub(self.feature_len);
        let count = features.len().min(available);
        self.feature_buffer[self.feature_len..self.feature_len + count]
            .copy_from_slice(&features[..count]);
        self.feature_len += count;
        self.update_peak();
        count
    }

    /// Obtains a mutable scratch slice of requested length from the static scratch pool.
    pub fn scratch_slice_mut(&mut self, len: usize) -> Option<&mut [i32]> {
        if self.scratch_len + len <= MATRIX_CAP {
            let start = self.scratch_len;
            self.scratch_len += len;
            self.update_peak();
            Some(&mut self.scratch_matrix[start..start + len])
        } else {
            None
        }
    }

    /// Total bytes currently allocated in the arena.
    pub fn bytes_allocated(&self) -> usize {
        (self.audio_len * core::mem::size_of::<i16>())
            + (self.feature_len * core::mem::size_of::<i16>())
            + (self.scratch_len * core::mem::size_of::<i32>())
    }

    /// Total storage capacity of the arena in bytes.
    pub const fn total_capacity_bytes(&self) -> usize {
        (AUDIO_CAP * core::mem::size_of::<i16>())
            + (FEAT_CAP * core::mem::size_of::<i16>())
            + (MATRIX_CAP * core::mem::size_of::<i32>())
    }

    /// Memory utilization ratio in [0.0, 1.0].
    pub fn utilization_ratio(&self) -> f32 {
        self.bytes_allocated() as f32 / self.total_capacity_bytes() as f32
    }

    fn update_peak(&mut self) {
        let current = self.bytes_allocated();
        if current > self.peak_bytes_used {
            self.peak_bytes_used = current;
        }
    }
}

/// Operational configuration for the ultra-low-power PULP / RISC-V edge surveillance engine.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PulpConfig {
    /// Core clock frequency in MHz (default 50.0 MHz for ultra-low power).
    pub core_clock_mhz: f32,
    /// Operating core supply voltage in Volts (e.g. 0.8 V).
    pub core_voltage_v: f32,
    /// Dynamic power consumption coefficient in microwatts per MHz (default 15.0 uW/MHz).
    pub dynamic_power_uw_per_mhz: f32,
    /// Static standby/sleep leakage power in microwatts (default 50.0 uW).
    pub sleep_power_uw: f32,
    /// MEMS microphone power in microwatts during active listening (default 300.0 uW).
    pub mic_power_uw: f32,
    /// Audio frame interval in milliseconds (default 10.0 ms = 100 Hz frame rate).
    pub frame_interval_ms: f32,
    /// Primary surveillance battery capacity in milliampere-hours (e.g. 220 mAh CR2032 or 300 mAh LiPo).
    pub battery_capacity_mah: f32,
    /// Nominal battery voltage in Volts (3.0 V for CR2032, 3.7 V for LiPo).
    pub battery_voltage_v: f32,
}

impl Default for PulpConfig {
    fn default() -> Self {
        Self {
            core_clock_mhz: 50.0,
            core_voltage_v: 0.8,
            dynamic_power_uw_per_mhz: 15.0,
            sleep_power_uw: 50.0,
            mic_power_uw: 300.0,
            frame_interval_ms: 10.0,
            battery_capacity_mah: 220.0, // Standard CR2032 coin cell
            battery_voltage_v: 3.0,
        }
    }
}

/// Real-time micro-power surveillance telemetry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PulpTelemetry {
    /// Number of active execution cycles consumed per audio frame.
    pub active_cycles_per_frame: u32,
    /// Execution time in microseconds per frame.
    pub execution_time_us: f32,
    /// Duty cycle of the processor in percent (e.g. 6.0%).
    pub duty_cycle_pct: f32,
    /// Total average power consumption in microwatts (< 1000 uW for sub-milliwatt mode).
    pub average_power_uw: f32,
    /// Estimated battery operational life in days.
    pub battery_life_days: f32,
    /// Peak memory arena utilization ratio in [0.0, 1.0].
    pub arena_utilization: f32,
}

impl PulpTelemetry {
    /// Formats telemetry into MAVLink standard NAMED_VALUE_FLOAT packets (Message ID 251).
    pub fn to_mavlink_packets(&self, time_boot_ms: u32) -> Vec<MavlinkNamedValueFloat> {
        vec![
            MavlinkNamedValueFloat::new(time_boot_ms, "PULP_CYC", self.active_cycles_per_frame as f32),
            MavlinkNamedValueFloat::new(time_boot_ms, "PULP_PWR", self.average_power_uw),
            MavlinkNamedValueFloat::new(time_boot_ms, "PULP_BATT", self.battery_life_days),
        ]
    }
}

/// Power and energy model for battery-perched drone surveillance listening modes.
pub struct PulpPowerModel {
    config: PulpConfig,
}

impl PulpPowerModel {
    /// Creates a new power model with specified configuration.
    pub fn new(config: PulpConfig) -> Self {
        Self { config }
    }

    /// Access active configuration.
    pub fn config(&self) -> &PulpConfig {
        &self.config
    }

    /// Evaluates execution time, duty cycle, average power, and projected battery life from cycle count.
    pub fn evaluate_frame_power(
        &self,
        active_cycles: u32,
        arena_utilization: f32,
    ) -> PulpTelemetry {
        let f_mhz = self.config.core_clock_mhz.max(1.0);
        // Execution time in microseconds: cycles / (cycles/us) = cycles / f_mhz
        let execution_time_us = (active_cycles as f32) / f_mhz;
        let frame_period_us = self.config.frame_interval_ms * 1000.0;

        // Duty cycle fraction
        let duty_cycle = (execution_time_us / frame_period_us).clamp(0.0, 1.0);
        let duty_cycle_pct = duty_cycle * 100.0;

        // Active power: core dynamic power + microphone power
        let core_active_power_uw = f_mhz * self.config.dynamic_power_uw_per_mhz;
        let active_total_uw = core_active_power_uw + self.config.mic_power_uw;

        // Sleep power: core leakage + low-power microphone standby
        let sleep_total_uw = self.config.sleep_power_uw + (self.config.mic_power_uw * 0.5);

        // Average power consumption in microwatts
        let average_power_uw = (duty_cycle * active_total_uw) + ((1.0 - duty_cycle) * sleep_total_uw);

        // Battery total energy in milliwatt-hours: mAh * V
        let battery_energy_mwh = self.config.battery_capacity_mah * self.config.battery_voltage_v;
        // Average power in milliwatts
        let avg_power_mw = (average_power_uw * 0.001).max(1e-6);

        // Battery operational life in hours and days
        let life_hours = battery_energy_mwh / avg_power_mw;
        let battery_life_days = life_hours / 24.0;

        PulpTelemetry {
            active_cycles_per_frame: active_cycles,
            execution_time_us,
            duty_cycle_pct,
            average_power_uw,
            battery_life_days,
            arena_utilization,
        }
    }
}
