#![deny(unsafe_code)]

use std::f32::consts::PI;

/// Direct Form II Transposed Second-Order IIR Biquad Notch Filter.
///
/// Implements a symmetric notch filter with transfer function:
/// H(z) = (b0 + b1 * z^-1 + b2 * z^-2) / (1 + a1 * z^-1 + a2 * z^-2)
///
/// Features sub-microsecond coefficient recalculation for real-time
/// drone motor telemetry tracking and zero runtime memory allocations.
#[derive(Debug, Clone)]
pub struct BiquadNotchFilter {
    sample_rate: f32,
    center_freq: f32,
    q_factor: f32,
    // Normalized coefficients (divided by a0)
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    // Transposed Direct Form II state registers
    s1: f32,
    s2: f32,
    // Whether filter is active (disabled if center_freq >= Nyquist or <= 10 Hz)
    active: bool,
}

impl BiquadNotchFilter {
    /// Constructs a new Biquad Notch Filter.
    ///
    /// # Arguments
    /// * `sample_rate` - Audio sampling frequency in Hz (e.g., 16000.0).
    /// * `center_freq` - Notch center frequency in Hz.
    /// * `q_factor` - Quality factor Q = f0 / Bandwidth (typically 5.0 to 20.0 for narrow tone rejection).
    pub fn new(sample_rate: f32, center_freq: f32, q_factor: f32) -> Self {
        let mut filter = Self {
            sample_rate,
            center_freq,
            q_factor: q_factor.max(0.1),
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            s1: 0.0,
            s2: 0.0,
            active: false,
        };
        filter.recalculate_coefficients();
        filter
    }

    /// Dynamically updates the center frequency while preserving state register continuity.
    pub fn update_frequency(&mut self, center_freq: f32) {
        if (self.center_freq - center_freq).abs() > 0.01 {
            self.center_freq = center_freq;
            self.recalculate_coefficients();
        }
    }

    /// Returns the current center frequency in Hz.
    pub fn center_freq(&self) -> f32 {
        self.center_freq
    }

    /// Returns whether the filter is actively filtering.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Recalculates normalized biquad coefficients based on sample rate, center frequency, and Q.
    fn recalculate_coefficients(&mut self) {
        let nyquist = self.sample_rate * 0.5;
        if self.center_freq <= 10.0 || self.center_freq >= nyquist - 10.0 {
            self.active = false;
            self.b0 = 1.0;
            self.b1 = 0.0;
            self.b2 = 0.0;
            self.a1 = 0.0;
            self.a2 = 0.0;
            return;
        }

        self.active = true;
        let omega0 = 2.0 * PI * (self.center_freq / self.sample_rate);
        let cos_omega0 = omega0.cos();
        let sin_omega0 = omega0.sin();
        let alpha = sin_omega0 / (2.0 * self.q_factor);

        let a0 = 1.0 + alpha;
        let inv_a0 = 1.0 / a0;

        self.b0 = inv_a0;
        self.b1 = -2.0 * cos_omega0 * inv_a0;
        self.b2 = inv_a0;
        self.a1 = -2.0 * cos_omega0 * inv_a0;
        self.a2 = (1.0 - alpha) * inv_a0;
    }

    /// Processes a single audio sample through the Transposed Direct Form II structure.
    #[inline]
    pub fn process_sample(&mut self, x: f32) -> f32 {
        if !self.active {
            return x;
        }

        let y = self.b0 * x + self.s1;
        self.s1 = self.b1 * x - self.a1 * y + self.s2;
        self.s2 = self.b2 * x - self.a2 * y;

        y
    }

    /// Processes a slice of audio samples in place.
    pub fn process_block(&mut self, samples: &mut [f32]) {
        if !self.active {
            return;
        }
        for s in samples.iter_mut() {
            *s = self.process_sample(*s);
        }
    }

    /// Resets internal state registers to zero.
    pub fn reset(&mut self) {
        self.s1 = 0.0;
        self.s2 = 0.0;
    }
}

/// Dynamic Rotor Blade Pass Frequency (BPF) Harmonic Notch Filter Bank.
///
/// Tracks brushless motor RPM telemetry from the autopilot (e.g., Kestrel ESC telemetry,
/// DShot RPM feedback, or MAVLink `ESC_TELEMETRY` packets) and dynamically updates
/// a cascade of high-Q notch filters targeted at:
/// f_k = k * (N_blades * RPM) / 60, for k = 1..=num_harmonics.
#[derive(Debug, Clone)]
pub struct RotorHarmonicNotchBank {
    sample_rate: f32,
    num_blades: usize,
    num_harmonics: usize,
    q_factor: f32,
    filters: Vec<BiquadNotchFilter>,
    current_rpm: f32,
}

impl RotorHarmonicNotchBank {
    /// Constructs a new Rotor Harmonic Notch Filter Bank.
    ///
    /// # Arguments
    /// * `sample_rate` - Audio sampling rate in Hz (e.g., 16000.0).
    /// * `num_blades` - Number of blades per propeller (typically 2 or 3).
    /// * `num_harmonics` - Number of harmonics to track (typically 2 to 4).
    /// * `q_factor` - Notch filter quality factor (typically 8.0 to 15.0).
    pub fn new(sample_rate: f32, num_blades: usize, num_harmonics: usize, q_factor: f32) -> Self {
        let blades = num_blades.max(1);
        let harmonics = num_harmonics.max(1);

        let mut filters = Vec::with_capacity(harmonics);
        for _ in 0..harmonics {
            filters.push(BiquadNotchFilter::new(sample_rate, 0.0, q_factor));
        }

        Self {
            sample_rate,
            num_blades: blades,
            num_harmonics: harmonics,
            q_factor,
            filters,
            current_rpm: 0.0,
        }
    }

    /// Synchronizes the notch filter bank with incoming motor RPM telemetry.
    ///
    /// Calculates the fundamental Blade Pass Frequency:
    /// f_BPF = (N_blades * RPM) / 60
    /// and tunes each harmonic k * f_BPF.
    pub fn update_rpm(&mut self, rpm: f32) {
        self.current_rpm = rpm.max(0.0);
        let bpf = (self.num_blades as f32 * self.current_rpm) / 60.0;

        for k in 1..=self.num_harmonics {
            let harmonic_freq = bpf * (k as f32);
            self.filters[k - 1].update_frequency(harmonic_freq);
        }
    }

    /// Ingests multi-motor telemetry (e.g., 4 individual motor RPMs on a quadcopter)
    /// and tunes the harmonic notches to the mean rotor speed.
    pub fn update_multi_motor_rpm(&mut self, motor_rpms: &[f32]) {
        if motor_rpms.is_empty() {
            return;
        }
        let sum: f32 = motor_rpms.iter().sum();
        let avg_rpm = sum / motor_rpms.len() as f32;
        self.update_rpm(avg_rpm);
    }

    /// Processes a slice of audio samples in place through all active harmonic notches.
    pub fn process_block(&mut self, samples: &mut [f32]) {
        for filter in self.filters.iter_mut() {
            if filter.is_active() {
                filter.process_block(samples);
            }
        }
    }

    /// Processes a single audio sample through the cascaded notch bank.
    #[inline]
    pub fn process_sample(&mut self, mut x: f32) -> f32 {
        for filter in self.filters.iter_mut() {
            if filter.is_active() {
                x = filter.process_sample(x);
            }
        }
        x
    }

    /// Returns the active notch frequencies in Hz.
    pub fn active_frequencies(&self) -> Vec<f32> {
        self.filters
            .iter()
            .filter(|f| f.is_active())
            .map(|f| f.center_freq())
            .collect()
    }

    /// Returns the audio sample rate in Hz.
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Returns the notch filter quality factor Q.
    pub fn q_factor(&self) -> f32 {
        self.q_factor
    }

    /// Returns the current RPM.
    pub fn current_rpm(&self) -> f32 {
        self.current_rpm
    }

    /// Resets all internal filter states.
    pub fn reset(&mut self) {
        for f in self.filters.iter_mut() {
            f.reset();
        }
    }
}
