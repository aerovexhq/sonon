//! Bio-inspired micro-tympanum differential microphone emulation based on *Ormia ochracea* mechanics.
//!
//! Emulates the mechanically coupled inter-tympanic cuticular bridge of the parasitoid fly *Ormia ochracea*,
//! amplifying microscopic acoustic time differences (ITD) and intensity differences (IID) across sub-2mm dual MEMS
//! microphone layouts to deliver > 20 dB of effective spatial amplification and precise 3D direction-of-arrival (DoA).

#![deny(unsafe_code)]

use crate::health::MavlinkNamedValueFloat;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Configuration parameters for the bio-inspired Ormia ochracea micro-tympanum bridge.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OrmiaConfig {
    /// Inter-microphone spacing in meters (sub-2mm, e.g. 0.0012 m = 1.2 mm).
    pub mic_distance_m: f32,
    /// Speed of sound in ambient air in m/s (default 343.0 m/s).
    pub speed_of_sound: f32,
    /// Audio sampling rate in Hz (e.g. 16000.0, 48000.0).
    pub sample_rate: f32,
    /// Symmetric (bending) mode resonance frequency in Hz (typically ~2200.0 Hz).
    pub f_bending: f32,
    /// Symmetric (bending) mode quality factor Q (typically ~2.2).
    pub q_bending: f32,
    /// Anti-symmetric (rocking) mode resonance frequency in Hz (typically ~3100.0 Hz).
    pub f_rocking: f32,
    /// Anti-symmetric (rocking) mode quality factor Q (typically ~3.5).
    pub q_rocking: f32,
    /// Mechanical coupling stiffness ratio Kc / K (typically ~0.5 to ~1.0).
    pub coupling_ratio: f32,
    /// Maximum expected IID amplification in dB at 90-degree incident angle (typically ~24.0 dB).
    pub iid_calibration_db: f32,
    /// Exponential smoothing alpha for power and direction tracking (e.g. 0.05).
    pub smoothing_alpha: f32,
}

impl Default for OrmiaConfig {
    fn default() -> Self {
        Self {
            mic_distance_m: 0.0012, // 1.2 mm anatomical Ormia ochracea tympanal separation
            speed_of_sound: 343.0,
            sample_rate: 16000.0,
            f_bending: 2200.0,
            q_bending: 2.2,
            f_rocking: 3100.0,
            q_rocking: 3.5,
            coupling_ratio: 1.0,
            iid_calibration_db: 24.0,
            smoothing_alpha: 0.05,
        }
    }
}

/// Second-order resonant biquad filter in Direct Form II Transposed structure.
///
/// Implements continuous-to-discrete bilinear transform with pre-warping:
/// H(s) = gain * w0^2 / (s^2 + (w0 / Q) * s + w0^2).
#[derive(Debug, Clone, PartialEq)]
pub struct OrmiaBiquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    s1: f32,
    s2: f32,
}

impl OrmiaBiquad {
    /// Constructs a stable Direct Form II Transposed biquad resonator.
    pub fn new(f0: f32, q: f32, sample_rate: f32, gain: f32) -> Self {
        assert!(f0 > 0.0 && f0 < sample_rate * 0.5, "f0 must be within (0, Nyquist)");
        assert!(q > 0.0, "Q factor must be positive");
        assert!(sample_rate > 0.0, "Sample rate must be positive");

        let omega = 2.0 * PI * f0 / sample_rate;
        let k = (omega * 0.5).tan();
        let k2 = k * k;
        let norm = 1.0 + k / q + k2;

        let b0 = gain * k2 / norm;
        let b1 = 2.0 * b0;
        let b2 = b0;
        let a1 = 2.0 * (k2 - 1.0) / norm;
        let a2 = (1.0 - k / q + k2) / norm;

        Self {
            b0,
            b1,
            b2,
            a1,
            a2,
            s1: 0.0,
            s2: 0.0,
        }
    }

    /// Evaluates one audio sample in-place using Direct Form II Transposed state recurrence.
    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.s1;
        self.s1 = self.b1 * x - self.a1 * y + self.s2;
        self.s2 = self.b2 * x - self.a2 * y;
        y
    }

    /// Clears internal delay state registers.
    pub fn reset(&mut self) {
        self.s1 = 0.0;
        self.s2 = 0.0;
    }

    /// Return feedforward coefficients [b0, b1, b2].
    pub fn b_coeffs(&self) -> [f32; 3] {
        [self.b0, self.b1, self.b2]
    }

    /// Return feedback coefficients [a1, a2].
    pub fn a_coeffs(&self) -> [f32; 2] {
        [self.a1, self.a2]
    }

    /// Computes analytical complex frequency response (magnitude, phase_rad) at specified frequency.
    pub fn frequency_response(&self, freq_hz: f32, sample_rate: f32) -> (f32, f32) {
        let omega = 2.0 * PI * freq_hz / sample_rate;
        let cos1 = omega.cos();
        let sin1 = -omega.sin();
        let cos2 = (2.0 * omega).cos();
        let sin2 = -(2.0 * omega).sin();

        let num_re = self.b0 + self.b1 * cos1 + self.b2 * cos2;
        let num_im = self.b1 * sin1 + self.b2 * sin2;

        let den_re = 1.0 + self.a1 * cos1 + self.a2 * cos2;
        let den_im = self.a1 * sin1 + self.a2 * sin2;

        let den_mag2 = den_re * den_re + den_im * den_im;
        if den_mag2 < 1e-12 {
            return (0.0, 0.0);
        }

        let re = (num_re * den_re + num_im * den_im) / den_mag2;
        let im = (num_im * den_re - num_re * den_im) / den_mag2;

        let mag = (re * re + im * im).sqrt();
        let phase = im.atan2(re);
        (mag, phase)
    }
}

/// Mechanically coupled inter-tympanic bridge simulation filter.
///
/// Converts dual microscopic microphone pressures s1(t) and s2(t) into symmetric bending
/// and anti-symmetric rocking modes, filters each through modal biquads, and resynthesizes
/// amplified tympanal displacements x1(t) and x2(t).
#[derive(Debug, Clone, PartialEq)]
pub struct OrmiaBridgeFilter {
    config: OrmiaConfig,
    biquad_bending: OrmiaBiquad,
    biquad_rocking: OrmiaBiquad,
}

impl OrmiaBridgeFilter {
    /// Creates a new Ormia bridge filter configured with biological or MEMS parameters.
    pub fn new(config: OrmiaConfig) -> Self {
        let f_center = (config.f_bending * config.f_rocking).sqrt();
        let omega_center = 2.0 * PI * f_center;

        // Characteristic acoustic difference ratio at 52-degree reference angle
        let theta_ref = 52.0f32.to_radians();
        let tau_ref = (config.mic_distance_m * theta_ref.sin()) / config.speed_of_sound;
        let delta_u_ref = 0.5 * omega_center * tau_ref;

        // Unit biquads to evaluate analytical frequency response magnitudes at geometric center
        let unit_s = OrmiaBiquad::new(config.f_bending, config.q_bending, config.sample_rate, 1.0);
        let unit_a = OrmiaBiquad::new(config.f_rocking, config.q_rocking, config.sample_rate, 1.0);
        let (mag_s, _) = unit_s.frequency_response(f_center, config.sample_rate);
        let (mag_a, _) = unit_a.frequency_response(f_center, config.sample_rate);

        // Bridge rocking mode leverage factor compensating for microscopic spatial aperture
        let nominal_leverage = mag_s / (mag_a * delta_u_ref.max(1e-6));
        let gain_rocking = nominal_leverage * config.coupling_ratio;

        let biquad_bending = OrmiaBiquad::new(
            config.f_bending,
            config.q_bending,
            config.sample_rate,
            1.0,
        );
        let biquad_rocking = OrmiaBiquad::new(
            config.f_rocking,
            config.q_rocking,
            config.sample_rate,
            gain_rocking,
        );

        Self {
            config,
            biquad_bending,
            biquad_rocking,
        }
    }

    /// Access active configuration.
    pub fn config(&self) -> &OrmiaConfig {
        &self.config
    }

    /// Access symmetric bending mode biquad.
    pub fn biquad_bending(&self) -> &OrmiaBiquad {
        &self.biquad_bending
    }

    /// Access anti-symmetric rocking mode biquad.
    pub fn biquad_rocking(&self) -> &OrmiaBiquad {
        &self.biquad_rocking
    }

    /// Processes a single pair of raw microphone samples and yields amplified tympanal outputs (x1, x2).
    #[inline]
    pub fn process_sample(&mut self, s1: f32, s2: f32) -> (f32, f32) {
        let u_s = 0.5 * (s1 + s2);
        let u_a = 0.5 * (s1 - s2);

        let z_s = self.biquad_bending.process(u_s);
        let z_a = self.biquad_rocking.process(u_a);

        // Ormia inter-tympanic destructive/constructive modal interference:
        // When sound originates on mic 1 side (theta > 0), u_a has +90 deg phase lead over u_s.
        // H_a has ~88 deg phase lead over H_s at operating band, giving z_a ~178 deg relative phase.
        // Thus (z_s - z_a) adds in-phase on ipsilateral side, while (z_s + z_a) cancels on contralateral side.
        let x1 = z_s - z_a;
        let x2 = z_s + z_a;

        (x1, x2)
    }

    /// Processes block of dual-channel microphone samples.
    pub fn process_block(
        &mut self,
        ch1: &[f32],
        ch2: &[f32],
        out1: &mut [f32],
        out2: &mut [f32],
    ) {
        let n = ch1.len().min(ch2.len()).min(out1.len()).min(out2.len());
        for i in 0..n {
            let (x1, x2) = self.process_sample(ch1[i], ch2[i]);
            out1[i] = x1;
            out2[i] = x2;
        }
    }

    /// Resets internal modal filter states.
    pub fn reset(&mut self) {
        self.biquad_bending.reset();
        self.biquad_rocking.reset();
    }
}

/// Real-time diagnostic and spatial telemetry emitted by the Ormia bridge.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OrmiaTelemetry {
    /// Estimated sound source azimuth angle in degrees (-90.0 to +90.0).
    pub azimuth_deg: f32,
    /// Mechanically amplified Interaural Intensity Difference in dB.
    pub iid_db: f32,
    /// Effective Interaural Time Difference lag in samples.
    pub itd_samples: f32,
    /// Spatial amplification gain in dB relative to acoustic free-field.
    pub amplification_gain_db: f32,
    /// Confidence score in [0.0, 1.0] based on acoustic SNR and signal validity.
    pub confidence: f32,
    /// RMS power level of ipsilateral mechanical displacement x1.
    pub x1_rms: f32,
    /// RMS power level of contralateral mechanical displacement x2.
    pub x2_rms: f32,
}

impl OrmiaTelemetry {
    /// Converts telemetry metrics to standard MAVLink NAMED_VALUE_FLOAT packets (Message ID 251).
    pub fn to_mavlink_packets(&self, time_boot_ms: u32) -> Vec<MavlinkNamedValueFloat> {
        vec![
            MavlinkNamedValueFloat::new(time_boot_ms, "ORM_AZIM", self.azimuth_deg),
            MavlinkNamedValueFloat::new(time_boot_ms, "ORM_IID", self.iid_db),
            MavlinkNamedValueFloat::new(time_boot_ms, "ORM_GAIN", self.amplification_gain_db),
        ]
    }
}

/// Continuous Direction-of-Arrival (DoA) estimator based on Ormia mechanical amplification.
pub struct OrmiaDirectionEstimator {
    config: OrmiaConfig,
    bridge: OrmiaBridgeFilter,
    p1_smooth: f32,
    p2_smooth: f32,
    raw_p1_smooth: f32,
    raw_p2_smooth: f32,
    x1_history: Vec<f32>,
    x2_history: Vec<f32>,
    history_idx: usize,
    history_len: usize,
    sample_count: u64,
    cached_itd: f32,
    latest_telemetry: OrmiaTelemetry,
}

const CORR_WINDOW: usize = 64;
const MAX_LAG: isize = 8;

impl OrmiaDirectionEstimator {
    /// Creates a new Ormia direction estimator with specified configuration.
    pub fn new(config: OrmiaConfig) -> Self {
        let bridge = OrmiaBridgeFilter::new(config.clone());
        let latest_telemetry = OrmiaTelemetry {
            azimuth_deg: 0.0,
            iid_db: 0.0,
            itd_samples: 0.0,
            amplification_gain_db: 0.0,
            confidence: 0.0,
            x1_rms: 0.0,
            x2_rms: 0.0,
        };

        Self {
            config,
            bridge,
            p1_smooth: 0.0,
            p2_smooth: 0.0,
            raw_p1_smooth: 0.0,
            raw_p2_smooth: 0.0,
            x1_history: vec![0.0f32; CORR_WINDOW],
            x2_history: vec![0.0f32; CORR_WINDOW],
            history_idx: 0,
            history_len: 0,
            sample_count: 0,
            cached_itd: 0.0,
            latest_telemetry,
        }
    }

    /// Access active configuration.
    pub fn config(&self) -> &OrmiaConfig {
        &self.config
    }

    /// Access inner mechanical bridge filter.
    pub fn bridge(&self) -> &OrmiaBridgeFilter {
        &self.bridge
    }

    /// Access mutable inner mechanical bridge filter.
    pub fn bridge_mut(&mut self) -> &mut OrmiaBridgeFilter {
        &mut self.bridge
    }

    /// Access latest evaluated telemetry.
    pub fn latest_telemetry(&self) -> &OrmiaTelemetry {
        &self.latest_telemetry
    }

    /// Ingests a single sample pair, updates mechanical state, and returns amplified displacements and telemetry.
    pub fn process_sample(&mut self, s1: f32, s2: f32) -> (f32, f32, OrmiaTelemetry) {
        let (x1, x2) = self.bridge.process_sample(s1, s2);

        // Store into circular cross-correlation history
        self.x1_history[self.history_idx] = x1;
        self.x2_history[self.history_idx] = x2;
        self.history_idx = (self.history_idx + 1) % CORR_WINDOW;
        if self.history_len < CORR_WINDOW {
            self.history_len += 1;
        }

        // Exponential power smoothing
        let alpha = self.config.smoothing_alpha;
        self.p1_smooth = (1.0 - alpha) * self.p1_smooth + alpha * (x1 * x1);
        self.p2_smooth = (1.0 - alpha) * self.p2_smooth + alpha * (x2 * x2);
        self.raw_p1_smooth = (1.0 - alpha) * self.raw_p1_smooth + alpha * (s1 * s1);
        self.raw_p2_smooth = (1.0 - alpha) * self.raw_p2_smooth + alpha * (s2 * s2);

        let eps = 1e-12f32;
        let iid_db = 10.0 * ((self.p1_smooth + eps) / (self.p2_smooth + eps)).log10();
        let iid_clamped = iid_db.clamp(-30.0, 30.0);

        let raw_iid_db = 10.0 * ((self.raw_p1_smooth + eps) / (self.raw_p2_smooth + eps)).log10();
        let amplification_gain_db = (iid_clamped.abs() - raw_iid_db.abs()).max(0.0);

        self.sample_count += 1;
        if self.sample_count % 32 == 0 {
            self.cached_itd = self.estimate_itd_samples();
        }
        let itd_samples = self.cached_itd;

        // Direction-of-arrival mapping
        let sin_theta = (iid_clamped / self.config.iid_calibration_db).clamp(-1.0, 1.0);
        let azimuth_deg = sin_theta.asin() * (180.0 / PI);

        let total_power = self.p1_smooth + self.p2_smooth;
        let confidence = (total_power / (total_power + 1e-5)).clamp(0.0, 1.0);

        self.latest_telemetry = OrmiaTelemetry {
            azimuth_deg,
            iid_db: iid_clamped,
            itd_samples,
            amplification_gain_db,
            confidence,
            x1_rms: self.p1_smooth.sqrt(),
            x2_rms: self.p2_smooth.sqrt(),
        };

        (x1, x2, self.latest_telemetry.clone())
    }

    /// Ingests blocks of dual-channel audio and returns latest telemetry.
    pub fn process_block(
        &mut self,
        mic1: &[f32],
        mic2: &[f32],
        out1: &mut [f32],
        out2: &mut [f32],
    ) -> OrmiaTelemetry {
        let n = mic1.len().min(mic2.len()).min(out1.len()).min(out2.len());
        let alpha = self.config.smoothing_alpha;
        let one_minus_alpha = 1.0 - alpha;

        for i in 0..n {
            let (x1, x2) = self.bridge.process_sample(mic1[i], mic2[i]);
            out1[i] = x1;
            out2[i] = x2;

            self.x1_history[self.history_idx] = x1;
            self.x2_history[self.history_idx] = x2;
            self.history_idx = (self.history_idx + 1) % CORR_WINDOW;

            self.p1_smooth = one_minus_alpha * self.p1_smooth + alpha * (x1 * x1);
            self.p2_smooth = one_minus_alpha * self.p2_smooth + alpha * (x2 * x2);
            self.raw_p1_smooth = one_minus_alpha * self.raw_p1_smooth + alpha * (mic1[i] * mic1[i]);
            self.raw_p2_smooth = one_minus_alpha * self.raw_p2_smooth + alpha * (mic2[i] * mic2[i]);
        }
        self.history_len = (self.history_len + n).min(CORR_WINDOW);
        self.sample_count += n as u64;

        let eps = 1e-12f32;
        let iid_db = 10.0 * ((self.p1_smooth + eps) / (self.p2_smooth + eps)).log10();
        let iid_clamped = iid_db.clamp(-30.0, 30.0);

        let raw_iid_db = 10.0 * ((self.raw_p1_smooth + eps) / (self.raw_p2_smooth + eps)).log10();
        let amplification_gain_db = (iid_clamped.abs() - raw_iid_db.abs()).max(0.0);

        self.cached_itd = self.estimate_itd_samples();
        let itd_samples = self.cached_itd;

        let sin_theta = (iid_clamped / self.config.iid_calibration_db).clamp(-1.0, 1.0);
        let azimuth_deg = sin_theta.asin() * (180.0 / PI);

        let total_power = self.p1_smooth + self.p2_smooth;
        let confidence = (total_power / (total_power + 1e-5)).clamp(0.0, 1.0);

        self.latest_telemetry = OrmiaTelemetry {
            azimuth_deg,
            iid_db: iid_clamped,
            itd_samples,
            amplification_gain_db,
            confidence,
            x1_rms: self.p1_smooth.sqrt(),
            x2_rms: self.p2_smooth.sqrt(),
        };

        self.latest_telemetry.clone()
    }

    /// Internal sub-sample ITD estimation using parabolic interpolation on cross-correlation.
    fn estimate_itd_samples(&self) -> f32 {
        if self.history_len < CORR_WINDOW {
            return 0.0;
        }

        let mut r = [0.0f32; (2 * MAX_LAG + 1) as usize];
        let n_corr = CORR_WINDOW - 2 * (MAX_LAG as usize);

        for (lag_idx, lag) in (-MAX_LAG..=MAX_LAG).enumerate() {
            let mut sum = 0.0f32;
            for k in 0..n_corr {
                let idx1 = (self.history_idx + k) % CORR_WINDOW;
                let idx2 = (self.history_idx as isize + k as isize + lag + CORR_WINDOW as isize) as usize % CORR_WINDOW;
                sum += self.x1_history[idx1] * self.x2_history[idx2];
            }
            r[lag_idx] = sum;
        }

        // Find peak index
        let mut best_idx = MAX_LAG as usize;
        let mut max_val = r[best_idx];
        for (i, &val) in r.iter().enumerate() {
            if val > max_val {
                max_val = val;
                best_idx = i;
            }
        }

        let best_lag = best_idx as isize - MAX_LAG;
        if best_idx > 0 && best_idx + 1 < r.len() {
            let y_prev = r[best_idx - 1];
            let y_curr = r[best_idx];
            let y_next = r[best_idx + 1];
            let denom = 2.0 * (2.0 * y_curr - y_prev - y_next);
            if denom > 1e-12 {
                let delta = ((y_next - y_prev) / denom).clamp(-0.5, 0.5);
                return best_lag as f32 + delta;
            }
        }

        best_lag as f32
    }

    /// Resets all internal bridge filters, smoothing powers, and history buffers.
    pub fn reset(&mut self) {
        self.bridge.reset();
        self.p1_smooth = 0.0;
        self.p2_smooth = 0.0;
        self.raw_p1_smooth = 0.0;
        self.raw_p2_smooth = 0.0;
        self.x1_history.fill(0.0);
        self.x2_history.fill(0.0);
        self.history_idx = 0;
        self.history_len = 0;
        self.sample_count = 0;
        self.cached_itd = 0.0;
        self.latest_telemetry = OrmiaTelemetry {
            azimuth_deg: 0.0,
            iid_db: 0.0,
            itd_samples: 0.0,
            amplification_gain_db: 0.0,
            confidence: 0.0,
            x1_rms: 0.0,
            x2_rms: 0.0,
        };
    }
}
