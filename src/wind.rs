#![deny(unsafe_code)]

//! Aerodynamic Wind Buffeting Incoherent Noise Separation & Turbulent Boundary Layer (TBL) Suppression.
//!
//! Features:
//! - Multi-channel convective turbulence phase-decorrelation filter separating acoustic sound waves
//!   from aerodynamic pressure fluctuations (pseudosound).
//! - Physical Corcos (1964) turbulent boundary layer wall-pressure cross-spectral density modeling.
//! - Magnitude-Squared Coherence (MSC) tracking with adaptive exponential forgetting.
//! - Convective phase-slowness discriminator rejecting non-propagating hydrodynamic eddies ($U_c \ll c$).
//! - Adaptive aerodynamic rumble high-pass filter dynamically tuned to flight airspeed ($f_c(V_\infty)$).
//! - Restoration of positive voice SNR under severe $15\text{ m/s}$ ($54\text{ km/h}$) forward laminar flight airflow.
//! - Constant Overlap-Add (COLA) time-frequency resynthesis with zero amplitude ripple.
//! - Standard MAVLink v2 `NAMED_VALUE_FLOAT` telemetry packets (`WIND_COH`, `WIND_SUPP`, `WIND_SPD`).
//! - Rigorous synthetic wind tunnel and turbulent eddy generator (`AeroacousticWindSimulator`).

use crate::beamforming::SPEED_OF_SOUND;
use crate::health::MavlinkNamedValueFloat;
use crate::stft::FftProcessor;
use crate::tse::Complex32;
use crate::window::{Window, WindowType};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Configuration for Aerodynamic Wind Buffeting and Turbulent Boundary Layer Suppression.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WindTurbulenceConfig {
    /// Audio sampling rate in Hertz (e.g. 16000.0).
    pub sample_rate: f32,
    /// STFT FFT size (must be power of 2, e.g. 512).
    pub fft_size: usize,
    /// STFT hop size (e.g. 256 for 50% overlap).
    pub hop_size: usize,
    /// Physical distance between primary and reference microphones in meters (e.g. 0.02 = 20 mm).
    pub mic_distance_m: f32,
    /// Forward flight or ambient airflow airspeed in m/s (e.g. 15.0 m/s).
    pub forward_airspeed_mps: f32,
    /// Minimum noise coherence floor threshold below which energy is classified as turbulent pseudosound.
    pub min_coherence_threshold: f32,
    /// Maximum allowable wind suppression attenuation in decibels (e.g. 24.0 dB).
    pub max_attenuation_db: f32,
    /// Exponential temporal smoothing factor for Cross-Power Spectral Densities [0.5, 0.95].
    pub coherence_smoothing: f32,
    /// Enable convective phase-velocity / apparent slowness discriminator.
    pub enable_convective_slowness_gate: bool,
    /// Safety margin factor over acoustic speed of sound for slowness gating (e.g. 1.35).
    pub convective_slowness_margin: f32,
    /// Enable adaptive aerodynamic low-frequency rumble high-pass filter.
    pub enable_adaptive_rumble_filter: bool,
    /// Minimum rumble high-pass cutoff frequency in Hz under calm/stationary conditions.
    pub min_rumble_cutoff_hz: f32,
    /// Maximum rumble high-pass cutoff frequency in Hz under maximum slipstream airspeed.
    pub max_rumble_cutoff_hz: f32,
    /// Reference airspeed in m/s at which maximum rumble cutoff is reached.
    pub reference_airspeed_mps: f32,
}

impl Default for WindTurbulenceConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000.0,
            fft_size: 512,
            hop_size: 256,
            mic_distance_m: 0.02,
            forward_airspeed_mps: 15.0,
            min_coherence_threshold: 0.12,
            max_attenuation_db: 24.0,
            coherence_smoothing: 0.85,
            enable_convective_slowness_gate: true,
            convective_slowness_margin: 1.35,
            enable_adaptive_rumble_filter: true,
            min_rumble_cutoff_hz: 60.0,
            max_rumble_cutoff_hz: 220.0,
            reference_airspeed_mps: 15.0,
        }
    }
}

/// Real-time diagnostic telemetry for aerodynamic wind buffeting and boundary layer suppression.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WindNoiseTelemetry {
    /// Mean Magnitude-Squared Coherence (MSC) across human speech formants (300 Hz - 3400 Hz) [0.0, 1.0].
    pub mean_speech_coherence: f32,
    /// Instantaneous wind buffeting noise suppression in decibels (dB).
    pub suppression_db: f32,
    /// Acoustically estimated convective airflow airspeed in meters per second (m/s).
    pub estimated_airspeed_mps: f32,
    /// Active aerodynamic rumble high-pass filter cutoff frequency in Hertz (Hz).
    pub rumble_cutoff_hz: f32,
    /// Flag indicating active severe turbulent wind buffeting detection.
    pub is_wind_buffeting: bool,
    /// Flight timestamp in seconds.
    pub timestamp_sec: f64,
}

impl WindNoiseTelemetry {
    /// Format into standard MAVLink v2 `NAMED_VALUE_FLOAT` telemetry packets.
    pub fn to_mavlink_telemetry(&self) -> Vec<MavlinkNamedValueFloat> {
        let time_boot_ms = (self.timestamp_sec * 1000.0) as u32;
        vec![
            MavlinkNamedValueFloat::new(time_boot_ms, "WIND_COH", self.mean_speech_coherence),
            MavlinkNamedValueFloat::new(time_boot_ms, "WIND_SUPP", self.suppression_db),
            MavlinkNamedValueFloat::new(time_boot_ms, "WIND_SPD", self.estimated_airspeed_mps),
        ]
    }
}

/// 2nd-order Direct Form II Transposed adaptive Butterworth high-pass filter
/// suppressing aerodynamic turbulence rumble in drone audio.
#[derive(Debug, Clone)]
pub struct AdaptiveRumbleFilter {
    sample_rate: f32,
    current_cutoff_hz: f32,
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    d1: f32,
    d2: f32,
}

impl AdaptiveRumbleFilter {
    /// Construct a new adaptive rumble filter with initial cutoff frequency.
    pub fn new(sample_rate: f32, initial_cutoff_hz: f32) -> Self {
        let mut filter = Self {
            sample_rate,
            current_cutoff_hz: initial_cutoff_hz.max(10.0),
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            d1: 0.0,
            d2: 0.0,
        };
        filter.update_cutoff(initial_cutoff_hz);
        filter
    }

    /// Update cutoff frequency and recompute 2nd-order Butterworth coefficients via Bilinear Transform.
    pub fn update_cutoff(&mut self, cutoff_hz: f32) {
        let nyquist = self.sample_rate * 0.5;
        let fc = cutoff_hz.clamp(10.0, nyquist * 0.95);
        self.current_cutoff_hz = fc;

        // Bilinear Transform with frequency prewarping:
        let omega = 2.0 * PI * fc / self.sample_rate;
        let k = (omega * 0.5).tan();
        let q = std::f32::consts::FRAC_1_SQRT_2; // Butterworth Q = 1/sqrt(2) = 0.7071
        let k_sq = k * k;
        let norm = 1.0 + k / q + k_sq;

        self.b0 = 1.0 / norm;
        self.b1 = -2.0 / norm;
        self.b2 = 1.0 / norm;
        self.a1 = 2.0 * (k_sq - 1.0) / norm;
        self.a2 = (1.0 - k / q + k_sq) / norm;
    }

    /// Process a single sample through Direct Form II Transposed structure.
    #[inline]
    pub fn process_sample(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.d1;
        self.d1 = self.b1 * x - self.a1 * y + self.d2;
        self.d2 = self.b2 * x - self.a2 * y;
        y
    }

    /// Process a block of samples in-place.
    pub fn process_block(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            *s = self.process_sample(*s);
        }
    }

    /// Return active cutoff frequency in Hz.
    pub fn current_cutoff_hz(&self) -> f32 {
        self.current_cutoff_hz
    }

    /// Reset filter internal delay state.
    pub fn reset(&mut self) {
        self.d1 = 0.0;
        self.d2 = 0.0;
    }
}

/// Turbulent Boundary Layer (TBL) and Wind Buffeting Noise Suppressor.
///
/// Implements dual-channel STFT cross-spectral coherence tracking, Corcos aerodynamic
/// decorrelation gating, convective phase-slowness discrimination, and COLA resynthesis.
#[derive(Debug, Clone)]
pub struct TurbulentBoundaryLayerSuppressor {
    config: WindTurbulenceConfig,
    fft: FftProcessor,
    window: Window,
    rumble_filter1: AdaptiveRumbleFilter,
    rumble_filter2: AdaptiveRumbleFilter,

    // Spectral power tracking buffers [num_bins]
    p11: Vec<f32>,
    p22: Vec<f32>,
    p12: Vec<Complex32>,

    // Time-domain ring and overlap buffers
    input_buffer1: Vec<f32>,
    input_buffer2: Vec<f32>,
    overlap_buffer: Vec<f32>,

    // Scratch buffers for zero-allocation streaming execution
    scratch_re1: Vec<f32>,
    scratch_im1: Vec<f32>,
    scratch_re2: Vec<f32>,
    scratch_im2: Vec<f32>,
    scratch_out_re: Vec<f32>,
    scratch_out_im: Vec<f32>,

    // Telemetry tracking
    smoothed_airspeed_mps: f32,
    latest_telemetry: WindNoiseTelemetry,
}

impl TurbulentBoundaryLayerSuppressor {
    /// Construct a new turbulent boundary layer wind suppressor.
    pub fn new(config: WindTurbulenceConfig) -> Self {
        assert!(
            config.fft_size > 0 && (config.fft_size & (config.fft_size - 1)) == 0,
            "FFT size must be a power of 2"
        );
        assert!(
            config.hop_size > 0 && config.hop_size <= config.fft_size,
            "Hop size must be <= FFT size"
        );
        assert!(
            config.sample_rate > 0.0,
            "Sample rate must be positive"
        );

        let n = config.fft_size;
        let num_bins = n / 2 + 1;
        let fft = FftProcessor::new(n);
        let window = Window::new(WindowType::Hann, n);

        let rumble_filter1 = AdaptiveRumbleFilter::new(config.sample_rate, config.min_rumble_cutoff_hz);
        let rumble_filter2 = AdaptiveRumbleFilter::new(config.sample_rate, config.min_rumble_cutoff_hz);

        let p11 = vec![1e-6; num_bins];
        let p22 = vec![1e-6; num_bins];
        let p12 = vec![Complex32::zero(); num_bins];

        let input_buffer1 = Vec::with_capacity(n * 2);
        let input_buffer2 = Vec::with_capacity(n * 2);
        let overlap_buffer = vec![0.0; n];

        let scratch_re1 = vec![0.0; n];
        let scratch_im1 = vec![0.0; n];
        let scratch_re2 = vec![0.0; n];
        let scratch_im2 = vec![0.0; n];
        let scratch_out_re = vec![0.0; n];
        let scratch_out_im = vec![0.0; n];

        let latest_telemetry = WindNoiseTelemetry {
            mean_speech_coherence: 1.0,
            suppression_db: 0.0,
            estimated_airspeed_mps: 0.0,
            rumble_cutoff_hz: config.min_rumble_cutoff_hz,
            is_wind_buffeting: false,
            timestamp_sec: 0.0,
        };

        let initial_airspeed = config.forward_airspeed_mps;

        let mut suppressor = Self {
            config,
            fft,
            window,
            rumble_filter1,
            rumble_filter2,
            p11,
            p22,
            p12,
            input_buffer1,
            input_buffer2,
            overlap_buffer,
            scratch_re1,
            scratch_im1,
            scratch_re2,
            scratch_im2,
            scratch_out_re,
            scratch_out_im,
            smoothed_airspeed_mps: initial_airspeed,
            latest_telemetry,
        };

        suppressor.update_airspeed(suppressor.config.forward_airspeed_mps);
        suppressor
    }

    /// Access active configuration.
    pub fn config(&self) -> &WindTurbulenceConfig {
        &self.config
    }

    /// Update flight vehicle airspeed and adapt aerodynamic rumble filters accordingly.
    pub fn update_airspeed(&mut self, airspeed_mps: f32) {
        self.config.forward_airspeed_mps = airspeed_mps.max(0.0);
        if self.config.enable_adaptive_rumble_filter {
            let ratio = (airspeed_mps / self.config.reference_airspeed_mps.max(1.0)).clamp(0.0, 1.0);
            let cutoff = self.config.min_rumble_cutoff_hz
                + ratio * (self.config.max_rumble_cutoff_hz - self.config.min_rumble_cutoff_hz);
            self.rumble_filter1.update_cutoff(cutoff);
            self.rumble_filter2.update_cutoff(cutoff);
        }
    }

    /// Process a block of dual-channel audio of length `hop_size`.
    ///
    /// Outputs clean, wind-suppressed audio to `output` (length must equal `hop_size`).
    /// Returns the evaluated `WindNoiseTelemetry`.
    /// Executes with zero dynamic heap allocations in steady-state streaming.
    pub fn process_block(
        &mut self,
        mic1: &[f32],
        mic2: &[f32],
        output: &mut [f32],
        timestamp_sec: f64,
    ) -> WindNoiseTelemetry {
        let hop = self.config.hop_size;
        let n = self.config.fft_size;
        let num_bins = n / 2 + 1;
        assert_eq!(output.len(), hop, "Output length must match hop size");
        assert!(mic1.len() >= hop, "Mic 1 slice must be at least hop size");
        assert!(mic2.len() >= hop, "Mic 2 slice must be at least hop size");

        // Optional: Run pre-filtering through adaptive rumble filters
        for i in 0..hop {
            let s1 = if self.config.enable_adaptive_rumble_filter {
                self.rumble_filter1.process_sample(mic1[i])
            } else {
                mic1[i]
            };
            let s2 = if self.config.enable_adaptive_rumble_filter {
                self.rumble_filter2.process_sample(mic2[i])
            } else {
                mic2[i]
            };
            self.input_buffer1.push(s1);
            self.input_buffer2.push(s2);
        }

        // Check if sufficient samples are buffered for a complete STFT analysis window
        if self.input_buffer1.len() < n {
            output.fill(0.0);
            return self.latest_telemetry.clone();
        }

        // 1. Copy buffered time slices into scratch buffers and window them
        self.scratch_re1[..n].copy_from_slice(&self.input_buffer1[..n]);
        self.scratch_im1.fill(0.0);
        self.scratch_re2[..n].copy_from_slice(&self.input_buffer2[..n]);
        self.scratch_im2.fill(0.0);

        self.window.apply(&mut self.scratch_re1[..n]);
        self.window.apply(&mut self.scratch_re2[..n]);

        // 2. Compute in-place forward Radix-2 FFT
        self.fft.fft_in_place(&mut self.scratch_re1, &mut self.scratch_im1);
        self.fft.fft_in_place(&mut self.scratch_re2, &mut self.scratch_im2);

        // 3. Spectral Coherence Estimation and Convective Slowness Discrimination
        let alpha = self.config.coherence_smoothing;
        let one_minus_alpha = 1.0 - alpha;
        let freq_bin_hz = self.config.sample_rate / (n as f32);

        let min_coh = self.config.min_coherence_threshold;
        let min_gain = 10.0f32.powf(-self.config.max_attenuation_db / 20.0);
        let max_acoustic_slowness = (1.0 / SPEED_OF_SOUND) * self.config.convective_slowness_margin;

        let mut input_power_sum = 0.0f32;
        let mut output_power_sum = 0.0f32;

        let mut speech_band_coh_sum = 0.0f32;
        let mut speech_band_bin_count = 0;

        let mut low_freq_phase_sum = 0.0f32;
        let mut low_freq_bin_count = 0;

        for k in 0..num_bins {
            let freq_hz = (k as f32) * freq_bin_hz;
            let x1 = Complex32::new(self.scratch_re1[k], self.scratch_im1[k]);
            let x2 = Complex32::new(self.scratch_re2[k], self.scratch_im2[k]);

            let p1 = x1.norm_sqr();
            let p2 = x2.norm_sqr();
            let p12_inst = x1.mul(x2.conj());

            input_power_sum += p1;

            // Recursive exponential power averaging:
            let p11_smooth = alpha * self.p11[k] + one_minus_alpha * p1;
            let p22_smooth = alpha * self.p22[k] + one_minus_alpha * p2;
            let p12_smooth = self.p12[k].scale(alpha).add(p12_inst.scale(one_minus_alpha));

            self.p11[k] = p11_smooth;
            self.p22[k] = p22_smooth;
            self.p12[k] = p12_smooth;

            // Compute Magnitude-Squared Coherence (MSC):
            // MSC(k) = |P12(k)|^2 / (P11(k) * P22(k))
            let p12_mag_sq = p12_smooth.norm_sqr();
            let denom = (p11_smooth * p22_smooth).max(1e-12);
            let msc = (p12_mag_sq / denom).clamp(0.0, 1.0);

            // Accumulate speech band coherence (300 Hz - 3400 Hz)
            if (300.0..=3400.0).contains(&freq_hz) {
                speech_band_coh_sum += msc;
                speech_band_bin_count += 1;
            }

            // Estimate low-frequency convective phase lag for airspeed inversion (60 Hz - 300 Hz)
            if (60.0..=300.0).contains(&freq_hz) && p12_smooth.norm() > 1e-6 {
                let phase_diff = p12_smooth.im.atan2(p12_smooth.re);
                let tau = phase_diff / (2.0 * PI * freq_hz);
                low_freq_phase_sum += tau;
                low_freq_bin_count += 1;
            }

            // A. Base Wiener-like Coherence Gain
            let mut gain = if msc > min_coh {
                ((msc - min_coh) / (1.0 - min_coh)).sqrt()
            } else {
                0.0
            };

            // B. Convective Phase-Slowness Discrimination Gate
            if self.config.enable_convective_slowness_gate && freq_hz >= 100.0 {
                let phase_diff = p12_smooth.im.atan2(p12_smooth.re).abs();
                let omega = 2.0 * PI * freq_hz;
                let apparent_slowness = phase_diff / (omega * self.config.mic_distance_m.max(1e-3));

                // If apparent slowness exceeds physical acoustic propagation bounds in air,
                // the component is strictly aerodynamic convective turbulence / pseudosound!
                if apparent_slowness > max_acoustic_slowness {
                    // Suppress convective pseudosound with heavy attenuation factor
                    gain *= 0.15;
                }
            }

            // Enforce minimum attenuation floor to prevent musical noise
            let final_gain = gain.max(min_gain);

            // Apply suppression mask to primary channel STFT bin
            let y_k = x1.scale(final_gain);
            output_power_sum += y_k.norm_sqr();

            self.scratch_out_re[k] = y_k.re;
            self.scratch_out_im[k] = y_k.im;
        }

        // 4. Inverse STFT Reconstruction via Conjugate FFT (Hermitian Symmetry)
        self.scratch_re1[0] = self.scratch_out_re[0];
        self.scratch_im1[0] = 0.0;

        for k in 1..(n / 2) {
            self.scratch_re1[k] = self.scratch_out_re[k];
            self.scratch_im1[k] = -self.scratch_out_im[k]; // Conjugate for IFFT trick

            self.scratch_re1[n - k] = self.scratch_out_re[k];
            self.scratch_im1[n - k] = self.scratch_out_im[k];
        }

        self.scratch_re1[n / 2] = self.scratch_out_re[n / 2];
        self.scratch_im1[n / 2] = 0.0;

        self.fft.fft_in_place(&mut self.scratch_re1, &mut self.scratch_im1);

        let inv_n = 1.0 / (n as f32);
        for re in &mut self.scratch_re1 {
            *re *= inv_n;
        }

        // 5. Overlap-Add Resynthesis (50% Overlap Hann Analysis yields exact unity COLA)
        for i in 0..n {
            self.overlap_buffer[i] += self.scratch_re1[i];
        }

        // Drain hop_size samples to output
        output.copy_from_slice(&self.overlap_buffer[..hop]);

        // Shift overlap buffer forward by hop_size
        self.overlap_buffer.copy_within(hop..n, 0);
        self.overlap_buffer[(n - hop)..n].fill(0.0);

        // Slide input history buffers forward by hop_size
        self.input_buffer1.drain(..hop);
        self.input_buffer2.drain(..hop);

        // 6. Compute Telemetry Snapshot
        let mean_speech_coherence = if speech_band_bin_count > 0 {
            speech_band_coh_sum / (speech_band_bin_count as f32)
        } else {
            1.0
        };

        let suppression_db = if input_power_sum > 1e-12 && output_power_sum > 1e-12 {
            (10.0 * (input_power_sum / output_power_sum).log10()).clamp(0.0, 40.0)
        } else {
            0.0
        };

        let inst_airspeed = if low_freq_bin_count > 0 {
            let mean_tau = (low_freq_phase_sum / (low_freq_bin_count as f32)).abs();
            if mean_tau > 1e-4 && mean_tau < 0.05 {
                let u_c = (self.config.mic_distance_m / mean_tau).clamp(1.0, 30.0);
                (u_c / 0.7).clamp(1.0, 35.0)
            } else {
                self.smoothed_airspeed_mps
            }
        } else {
            self.smoothed_airspeed_mps
        };

        // Smooth airspeed estimate (alpha = 0.85)
        self.smoothed_airspeed_mps = 0.85 * self.smoothed_airspeed_mps + 0.15 * inst_airspeed;
        let estimated_airspeed_mps = self.smoothed_airspeed_mps;

        let is_wind_buffeting = mean_speech_coherence < 0.40 || suppression_db > 6.0;

        self.latest_telemetry = WindNoiseTelemetry {
            mean_speech_coherence,
            suppression_db,
            estimated_airspeed_mps,
            rumble_cutoff_hz: self.rumble_filter1.current_cutoff_hz(),
            is_wind_buffeting,
            timestamp_sec,
        };

        self.latest_telemetry.clone()
    }

    /// Access latest evaluated wind telemetry.
    pub fn latest_telemetry(&self) -> &WindNoiseTelemetry {
        &self.latest_telemetry
    }

    /// Reset internal state buffers and filters.
    pub fn reset(&mut self) {
        self.rumble_filter1.reset();
        self.rumble_filter2.reset();
        self.p11.fill(1e-6);
        self.p22.fill(1e-6);
        self.p12.fill(Complex32::zero());
        self.input_buffer1.clear();
        self.input_buffer2.clear();
        self.overlap_buffer.fill(0.0);
        self.smoothed_airspeed_mps = self.config.forward_airspeed_mps;
        self.latest_telemetry = WindNoiseTelemetry {
            mean_speech_coherence: 1.0,
            suppression_db: 0.0,
            estimated_airspeed_mps: self.config.forward_airspeed_mps,
            rumble_cutoff_hz: self.rumble_filter1.current_cutoff_hz(),
            is_wind_buffeting: false,
            timestamp_sec: 0.0,
        };
    }
}

/// Aeroacoustic Wind Simulator.
///
/// Synthesizes realistic 2-channel aerodynamic wind buffeting and Turbulent Boundary Layer (TBL)
/// wall-pressure fluctuations using the Corcos (1964) cross-spectral density formulation and
/// Kolmogorov $-5/3$ turbulence energy spectrum.
pub struct AeroacousticWindSimulator;

impl AeroacousticWindSimulator {
    /// Generate 2-channel Corcos Turbulent Boundary Layer wall-pressure noise.
    ///
    /// - `num_samples`: Total audio samples to generate.
    /// - `sample_rate`: Audio sample rate in Hz (e.g. 16000.0).
    /// - `airspeed_mps`: Flow velocity in m/s (e.g. 15.0 m/s).
    /// - `mic_distance_m`: Microphone separation along streamwise flow in meters (e.g. 0.02 m).
    pub fn generate_corcos_wind(
        num_samples: usize,
        sample_rate: f32,
        airspeed_mps: f32,
        mic_distance_m: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let n = num_samples.next_power_of_two().max(1024);
        let num_bins = n / 2 + 1;
        let freq_bin_hz = sample_rate / (n as f32);

        // Corcos parameters:
        // Convective velocity Uc ≈ 0.7 * U_inf
        let u_c = (0.7 * airspeed_mps).max(1.0);
        let alpha_x = 0.12f32; // Streamwise decay parameter
        let alpha_y = 0.75f32; // Spanwise decay parameter (assuming slight 5mm spanwise offset)
        let delta_x = mic_distance_m;
        let delta_y = 0.005f32;

        let mut spec1_re = vec![0.0f32; n];
        let mut spec1_im = vec![0.0f32; n];
        let mut spec2_re = vec![0.0f32; n];
        let mut spec2_im = vec![0.0f32; n];

        // LCG deterministic pseudorandom generator for zero dependencies and repeatability
        let mut rng_state = 0x12345678_u64;
        let mut uniform = || -> f32 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = ((rng_state >> 32) as u32) as f32 / 4294967296.0;
            val
        };
        let mut box_muller = || -> (f32, f32) {
            let u1 = uniform().max(1e-7);
            let u2 = uniform();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * PI * u2;
            (r * theta.cos(), r * theta.sin())
        };

        for k in 1..num_bins {
            let freq_hz = (k as f32) * freq_bin_hz;
            let omega = 2.0 * PI * freq_hz;

            // Kolmogorov -5/3 spectral density with low-frequency turnover at 80 Hz
            let f0 = 80.0f32;
            let phi_0 = 1.0 / (1.0 + (freq_hz / f0).powi(2)).powf(5.0 / 6.0);

            // Corcos coherence decay:
            let decay = (-(alpha_x * omega * delta_x + alpha_y * omega * delta_y) / u_c).exp();
            let phase = -omega * delta_x / u_c;

            // 2x2 Cholesky decomposition of Cross-Spectral Matrix:
            // S11 = phi_0, S22 = phi_0
            // S12 = phi_0 * decay * e^(j*phase)
            // L11 = sqrt(phi_0)
            // L21 = sqrt(phi_0) * decay * e^(-j*phase)
            // L22 = sqrt(phi_0 * (1 - decay^2))
            let l11 = phi_0.sqrt();
            let l21_mag = l11 * decay;
            let l21_re = l21_mag * (-phase).cos();
            let l21_im = l21_mag * (-phase).sin();
            let l22 = (phi_0 * (1.0 - decay * decay).max(0.0)).sqrt();

            // Independent standard complex Gaussians W1, W2
            let (w1_re, w1_im) = box_muller();
            let (w2_re, w2_im) = box_muller();

            // Channel 1 spectrum: X1 = L11 * W1
            let x1_re = l11 * w1_re;
            let x1_im = l11 * w1_im;

            // Channel 2 spectrum: X2 = L21 * W1 + L22 * W2
            let x2_re = (l21_re * w1_re - l21_im * w1_im) + l22 * w2_re;
            let x2_im = (l21_re * w1_im + l21_im * w1_re) + l22 * w2_im;

            spec1_re[k] = x1_re;
            spec1_im[k] = x1_im;
            spec2_re[k] = x2_re;
            spec2_im[k] = x2_im;

            if k < n / 2 {
                spec1_re[n - k] = x1_re;
                spec1_im[n - k] = -x1_im;
                spec2_re[n - k] = x2_re;
                spec2_im[n - k] = -x2_im;
            }
        }

        let fft = FftProcessor::new(n);
        fft.fft_in_place(&mut spec1_re, &mut spec1_im);
        fft.fft_in_place(&mut spec2_re, &mut spec2_im);

        let inv_n = 1.0 / (n as f32).sqrt();
        let mut ch1 = Vec::with_capacity(num_samples);
        let mut ch2 = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            ch1.push(spec1_re[i] * inv_n);
            ch2.push(spec2_re[i] * inv_n);
        }

        (ch1, ch2)
    }

    /// Simulate in-flight audio mixture of clean acoustic speech and severe turbulent wind noise.
    ///
    /// - `clean_speech`: Monophonic acoustic speech samples.
    /// - `sample_rate`: Sample rate in Hz.
    /// - `airspeed_mps`: Forward airspeed (e.g. 15.0 m/s).
    /// - `snr_db`: Target Signal-to-Noise Ratio in dB (e.g. -10.0 dB for wind-dominated speech).
    /// - `mic_distance_m`: Distance between microphones in meters.
    /// - `source_azimuth_rad`: Azimuth angle of speaker relative to array axis (e.g. 0.0 rad for broadside).
    pub fn simulate_flight_mixture(
        clean_speech: &[f32],
        sample_rate: f32,
        airspeed_mps: f32,
        snr_db: f32,
        mic_distance_m: f32,
        source_azimuth_rad: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let n = clean_speech.len();
        let (wind1, wind2) = Self::generate_corcos_wind(n, sample_rate, airspeed_mps, mic_distance_m);

        // Acoustic delay between microphones for plane sound wave:
        // tau_s = d * sin(azimuth) / c
        let acoustic_delay_sec = mic_distance_m * source_azimuth_rad.sin() / SPEED_OF_SOUND;
        let delay_samples = (acoustic_delay_sec * sample_rate).round() as isize;

        // Compute speech power
        let speech_pwr = clean_speech.iter().map(|&s| s * s).sum::<f32>() / (n as f32).max(1.0);
        let wind_pwr = wind1.iter().map(|&w| w * w).sum::<f32>() / (n as f32).max(1.0);

        // Scale wind to achieve specified SNR:
        // SNR_dB = 10 * log10(P_speech / P_wind) => P_wind_target = P_speech / 10^(SNR/10)
        let target_wind_pwr = speech_pwr / (10.0f32.powf(snr_db / 10.0)).max(1e-6);
        let wind_scale = (target_wind_pwr / wind_pwr.max(1e-12)).sqrt();

        let mut mix1 = Vec::with_capacity(n);
        let mut mix2 = Vec::with_capacity(n);

        for i in 0..n {
            let s1 = clean_speech[i];
            let s2_idx = (i as isize) - delay_samples;
            let s2 = if s2_idx >= 0 && (s2_idx as usize) < n {
                clean_speech[s2_idx as usize]
            } else {
                clean_speech[i]
            };

            let m1 = s1 + wind1[i] * wind_scale;
            let m2 = s2 + wind2[i] * wind_scale;

            mix1.push(m1);
            mix2.push(m2);
        }

        (mix1, mix2)
    }
}
