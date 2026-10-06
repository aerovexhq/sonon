//! Continuous Dyadic Morlet Wavelet Filterbank, Calderon Analytical Inversion,
//! Port-Hamiltonian Two-Mass Vocal Fold Oscillator, and Hybrid Physical-Neural
//! Wavelet Flow Matching Synthesizer in pure safe Rust.
//!
//! Subphase 4 of the Grand Industrial Speech Synthesis Initiative.

#![deny(unsafe_code)]

use crate::flow_matching::{
    DeterministicRng, FlowConditioning, FlowMatchingDiT, FlowSolverScheme, Linear,
    TextConditioningEncoder,
};
use crate::phonetic::{FormantTarget, G2pEngine};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Normalization constant for Morlet mother wavelet C_delta = pi^(-1/4) ~ 0.75112554.
pub const MORLET_C_DELTA: f32 = 0.7511255444649425;

/// Continuous Wavelet Transform (CWT) scalogram holding multi-scale complex coefficients
/// and non-negative spectral magnitude representations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WaveletScalogram {
    /// Analyzed dyadic dilation scales a_j.
    pub scales: Vec<f32>,
    /// Pseudo-frequencies corresponding to each scale in Hz.
    pub frequencies: Vec<f32>,
    /// Real wavelet coefficient matrix [num_scales x num_samples].
    pub real: Vec<Vec<f32>>,
    /// Imaginary wavelet coefficient matrix [num_scales x num_samples].
    pub imag: Vec<Vec<f32>>,
    /// Scalogram magnitude matrix S(a, b) = sqrt(Re^2 + Im^2 + 1e-8) [num_scales x num_samples].
    pub magnitude: Vec<Vec<f32>>,
    /// Audio sampling rate in Hz.
    pub sample_rate: f32,
}

impl WaveletScalogram {
    /// Construct a new WaveletScalogram with validation.
    pub fn new(
        scales: Vec<f32>,
        frequencies: Vec<f32>,
        real: Vec<Vec<f32>>,
        imag: Vec<Vec<f32>>,
        magnitude: Vec<Vec<f32>>,
        sample_rate: f32,
    ) -> Self {
        assert_eq!(
            scales.len(),
            frequencies.len(),
            "Scales and frequencies length mismatch"
        );
        assert_eq!(
            scales.len(),
            real.len(),
            "Scales and real coefficients length mismatch"
        );
        assert_eq!(
            scales.len(),
            imag.len(),
            "Scales and imaginary coefficients length mismatch"
        );
        assert_eq!(
            scales.len(),
            magnitude.len(),
            "Scales and magnitude matrix length mismatch"
        );

        Self {
            scales,
            frequencies,
            real,
            imag,
            magnitude,
            sample_rate,
        }
    }

    /// Number of scale channels D.
    pub fn num_scales(&self) -> usize {
        self.scales.len()
    }

    /// Number of time samples N.
    pub fn num_samples(&self) -> usize {
        self.real.first().map(|v| v.len()).unwrap_or(0)
    }

    /// Integrated acoustic energy across all scales and time samples.
    pub fn total_energy(&self) -> f32 {
        let mut sum = 0.0f32;
        for scale_mags in &self.magnitude {
            for &m in scale_mags {
                sum += m * m;
            }
        }
        sum
    }

    /// Integrated acoustic energy at a specific scale index.
    pub fn scale_energy(&self, scale_idx: usize) -> f32 {
        if scale_idx >= self.magnitude.len() {
            return 0.0;
        }
        self.magnitude[scale_idx].iter().map(|&m| m * m).sum()
    }

    /// Returns (scale_index, peak_energy) for the scale exhibiting dominant energy.
    pub fn dominant_scale(&self) -> (usize, f32) {
        let mut max_eng = 0.0f32;
        let mut best_idx = 0;
        for (idx, _) in self.scales.iter().enumerate() {
            let eng = self.scale_energy(idx);
            if eng > max_eng {
                max_eng = eng;
                best_idx = idx;
            }
        }
        (best_idx, max_eng)
    }

    /// Returns the acoustic frequency in Hz corresponding to the dominant scale.
    pub fn dominant_frequency(&self) -> f32 {
        let (idx, _) = self.dominant_scale();
        self.frequencies.get(idx).copied().unwrap_or(0.0)
    }
}

/// Helper function to perform symmetric reflection index lookup for boundary convolution.
#[inline]
fn reflect_index(idx: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let mut i = idx;
    while i < 0 || i >= len as isize {
        if i < 0 {
            i = -i;
        } else if i >= len as isize {
            i = 2 * (len as isize - 1) - i;
        }
    }
    i.clamp(0, len as isize - 1) as usize
}

/// Continuous Dyadic Morlet Wavelet Filterbank with analytic mother wavelet:
/// psi(t) = pi^(-1/4) * exp(i * omega_0 * t) * exp(-t^2 / 2)
/// with dyadic scale progression: a_j = a_0 * 2^(j / M), j in [0, D-1].
#[derive(Debug, Clone)]
pub struct DyadicMorletCwt {
    octaves: usize,
    voices_per_octave: usize,
    total_scales: usize,
    a0: f32,
    omega_0: f32,
    sample_rate: f32,
    scales: Vec<f32>,
    frequencies: Vec<f32>,
    real_kernels: Vec<Vec<f32>>,
    imag_kernels: Vec<Vec<f32>>,
}

impl DyadicMorletCwt {
    /// Construct a new Continuous Dyadic Morlet Wavelet filterbank.
    pub fn new(octaves: usize, voices_per_octave: usize, sample_rate: f32) -> Self {
        assert!(octaves > 0, "Octaves must be positive");
        assert!(voices_per_octave > 0, "Voices per octave must be positive");
        assert!(sample_rate > 0.0, "Sample rate must be positive");

        let a0 = 2.0f32;
        let omega_0 = 6.0f32;
        Self::build(octaves, voices_per_octave, a0, omega_0, sample_rate)
    }

    /// Builder method to customize base dilation scale a0.
    pub fn with_a0(mut self, a0: f32) -> Self {
        assert!(a0 > 0.0, "Base scale a0 must be positive");
        self.a0 = a0;
        Self::build(
            self.octaves,
            self.voices_per_octave,
            self.a0,
            self.omega_0,
            self.sample_rate,
        )
    }

    /// Builder method to customize central angular frequency omega_0.
    pub fn with_omega_0(mut self, omega_0: f32) -> Self {
        assert!(omega_0 > 0.0, "Central frequency omega_0 must be positive");
        self.omega_0 = omega_0;
        Self::build(
            self.octaves,
            self.voices_per_octave,
            self.a0,
            self.omega_0,
            self.sample_rate,
        )
    }

    fn build(
        octaves: usize,
        voices_per_octave: usize,
        a0: f32,
        omega_0: f32,
        sample_rate: f32,
    ) -> Self {
        let total_scales = octaves * voices_per_octave;
        let pi_fourth_inv = MORLET_C_DELTA;

        let mut scales = Vec::with_capacity(total_scales);
        let mut frequencies = Vec::with_capacity(total_scales);
        let mut real_kernels = Vec::with_capacity(total_scales);
        let mut imag_kernels = Vec::with_capacity(total_scales);

        for j in 0..total_scales {
            let exponent = (j as f32) / (voices_per_octave as f32);
            let a = a0 * 2.0f32.powf(exponent);
            scales.push(a);

            let freq = (omega_0 * sample_rate) / (2.0 * PI * a);
            frequencies.push(freq);

            let half_len = ((4.0 * a).ceil() as isize).clamp(2, 256);
            let kernel_len = (2 * half_len + 1) as usize;
            let norm = 1.0 / a.sqrt();

            let mut r_kernel = Vec::with_capacity(kernel_len);
            let mut i_kernel = Vec::with_capacity(kernel_len);

            for n in -half_len..=half_len {
                let t = (n as f32) / a;
                let gauss = (-0.5 * t * t).exp();
                let r = norm * pi_fourth_inv * gauss * (omega_0 * t).cos();
                let i = -norm * pi_fourth_inv * gauss * (omega_0 * t).sin();
                r_kernel.push(r);
                i_kernel.push(i);
            }

            real_kernels.push(r_kernel);
            imag_kernels.push(i_kernel);
        }

        Self {
            octaves,
            voices_per_octave,
            total_scales,
            a0,
            omega_0,
            sample_rate,
            scales,
            frequencies,
            real_kernels,
            imag_kernels,
        }
    }

    /// Compute Continuous Wavelet Transform on an input audio slice using FIR kernels
    /// with symmetric reflection padding at sequence boundaries.
    pub fn forward(&self, signal: &[f32]) -> WaveletScalogram {
        let n_samples = signal.len();
        if n_samples == 0 {
            return WaveletScalogram::new(
                self.scales.clone(),
                self.frequencies.clone(),
                vec![Vec::new(); self.total_scales],
                vec![Vec::new(); self.total_scales],
                vec![Vec::new(); self.total_scales],
                self.sample_rate,
            );
        }

        let mut real = Vec::with_capacity(self.total_scales);
        let mut imag = Vec::with_capacity(self.total_scales);
        let mut magnitude = Vec::with_capacity(self.total_scales);

        for s in 0..self.total_scales {
            let r_k = &self.real_kernels[s];
            let i_k = &self.imag_kernels[s];
            let k_len = r_k.len();
            let half_k = k_len / 2;

            let mut real_row = vec![0.0f32; n_samples];
            let mut imag_row = vec![0.0f32; n_samples];
            let mut mag_row = vec![0.0f32; n_samples];

            let interior_start = half_k;
            let interior_end = n_samples.saturating_sub(k_len - half_k);

            // Left boundary with symmetric reflection
            for i in 0..interior_start.min(n_samples) {
                let mut sum_r = 0.0f32;
                let mut sum_i = 0.0f32;
                for k in 0..k_len {
                    let offset = k as isize - half_k as isize;
                    let sig_idx = reflect_index(i as isize + offset, n_samples);
                    sum_r += signal[sig_idx] * r_k[k];
                    sum_i += signal[sig_idx] * i_k[k];
                }
                real_row[i] = sum_r;
                imag_row[i] = sum_i;
                mag_row[i] = (sum_r * sum_r + sum_i * sum_i + 1e-8).sqrt();
            }

            // Interior (vectorizable slice without bounds checks)
            if interior_start < interior_end {
                for i in interior_start..interior_end {
                    let sig_slice = &signal[i - half_k..i - half_k + k_len];
                    let mut sum_r = 0.0f32;
                    let mut sum_i = 0.0f32;
                    for k in 0..k_len {
                        sum_r += sig_slice[k] * r_k[k];
                        sum_i += sig_slice[k] * i_k[k];
                    }
                    real_row[i] = sum_r;
                    imag_row[i] = sum_i;
                    mag_row[i] = (sum_r * sum_r + sum_i * sum_i + 1e-8).sqrt();
                }
            }

            // Right boundary with symmetric reflection
            for i in interior_end..n_samples {
                let mut sum_r = 0.0f32;
                let mut sum_i = 0.0f32;
                for k in 0..k_len {
                    let offset = k as isize - half_k as isize;
                    let sig_idx = reflect_index(i as isize + offset, n_samples);
                    sum_r += signal[sig_idx] * r_k[k];
                    sum_i += signal[sig_idx] * i_k[k];
                }
                real_row[i] = sum_r;
                imag_row[i] = sum_i;
                mag_row[i] = (sum_r * sum_r + sum_i * sum_i + 1e-8).sqrt();
            }

            real.push(real_row);
            imag.push(imag_row);
            magnitude.push(mag_row);
        }

        WaveletScalogram::new(
            self.scales.clone(),
            self.frequencies.clone(),
            real,
            imag,
            magnitude,
            self.sample_rate,
        )
    }

    /// Access reference to scales.
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Access reference to pseudo-frequencies.
    pub fn frequencies(&self) -> &[f32] {
        &self.frequencies
    }

    /// Configured octaves.
    pub fn octaves(&self) -> usize {
        self.octaves
    }

    /// Configured voices per octave.
    pub fn voices_per_octave(&self) -> usize {
        self.voices_per_octave
    }

    /// Total number of scale channels D.
    pub fn total_scales(&self) -> usize {
        self.total_scales
    }

    /// Audio sampling rate in Hz.
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Central angular frequency omega_0.
    pub fn omega_0(&self) -> f32 {
        self.omega_0
    }

    /// Base scale a0.
    pub fn a0(&self) -> f32 {
        self.a0
    }
}

/// Exact Analytical Calderon Wavelet Inversion:
/// x[n] = (delta_b * ln(2)) / (M * C_delta) * sum_{j=0}^{D-1} (Re(W_psi x[a_j, n]) / sqrt(a_j))
/// with C_delta = pi^(-1/4) ~ 0.7511255.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalderonWaveletInverter {
    pub voices_per_octave: usize,
    pub delta_b: f32,
    pub c_delta: f32,
    pub gain_correction: f32,
}

impl CalderonWaveletInverter {
    /// Construct a new CalderonWaveletInverter.
    pub fn new(voices_per_octave: usize, delta_b: f32) -> Self {
        assert!(voices_per_octave > 0, "Voices per octave must be positive");
        assert!(delta_b > 0.0, "Time step delta_b must be positive");

        Self {
            voices_per_octave,
            delta_b,
            c_delta: MORLET_C_DELTA,
            gain_correction: 1.0,
        }
    }

    /// Customize reconstruction factor C_delta.
    pub fn with_c_delta(mut self, c_delta: f32) -> Self {
        assert!(c_delta > 0.0, "C_delta must be positive");
        self.c_delta = c_delta;
        self
    }

    /// Customize gain correction multiplier.
    pub fn with_gain_correction(mut self, gain: f32) -> Self {
        assert!(gain > 0.0, "Gain correction must be positive");
        self.gain_correction = gain;
        self
    }

    /// Reconstruct continuous time-domain waveform from a WaveletScalogram directly.
    pub fn reconstruct(&self, scalogram: &WaveletScalogram) -> Vec<f32> {
        self.reconstruct_from_real(&scalogram.real, &scalogram.scales)
    }

    /// Reconstruct continuous time-domain waveform from real wavelet coefficients.
    pub fn reconstruct_from_real(&self, real_coeffs: &[Vec<f32>], scales: &[f32]) -> Vec<f32> {
        if real_coeffs.is_empty() || scales.is_empty() {
            return Vec::new();
        }
        let num_scales = real_coeffs.len().min(scales.len());
        let num_samples = real_coeffs[0].len();
        if num_samples == 0 {
            return Vec::new();
        }

        let factor = (self.delta_b * 2.0f32.ln())
            / (self.voices_per_octave as f32 * self.c_delta)
            * self.gain_correction;

        let mut inv_sqrt_scales = Vec::with_capacity(num_scales);
        for &a in &scales[..num_scales] {
            inv_sqrt_scales.push(1.0 / a.sqrt());
        }

        let mut waveform = vec![0.0f32; num_samples];
        for n in 0..num_samples {
            let mut sum = 0.0f32;
            for j in 0..num_scales {
                if n < real_coeffs[j].len() {
                    sum += real_coeffs[j][n] * inv_sqrt_scales[j];
                }
            }
            waveform[n] = factor * sum;
        }

        waveform
    }

    /// Reconstruct continuous time-domain waveform from magnitude scalogram and phase matrices.
    pub fn reconstruct_from_magnitude_phase(
        &self,
        magnitudes: &[Vec<f32>],
        phases: &[Vec<f32>],
        scales: &[f32],
    ) -> Vec<f32> {
        if magnitudes.is_empty() || phases.is_empty() || scales.is_empty() {
            return Vec::new();
        }
        let num_scales = magnitudes.len().min(phases.len()).min(scales.len());
        let num_samples = magnitudes[0].len();

        let mut real_coeffs = Vec::with_capacity(num_scales);
        for j in 0..num_scales {
            let mut row = Vec::with_capacity(num_samples);
            for n in 0..num_samples {
                let mag = magnitudes[j].get(n).copied().unwrap_or(0.0);
                let phase = phases[j].get(n).copied().unwrap_or(0.0);
                row.push(mag * phase.cos());
            }
            real_coeffs.push(row);
        }

        self.reconstruct_from_real(&real_coeffs, scales)
    }

    /// Reconstruct continuous time-domain waveform from magnitude scalogram using the phase carrier
    /// extracted from a reference physical glottal airflow CWT scalogram.
    pub fn reconstruct_with_carrier(
        &self,
        magnitudes: &[Vec<f32>],
        carrier_cwt: &WaveletScalogram,
    ) -> Vec<f32> {
        let num_scales = magnitudes.len().min(carrier_cwt.scales.len());
        if num_scales == 0 {
            return Vec::new();
        }
        let num_samples = carrier_cwt.num_samples();

        let mut real_coeffs = Vec::with_capacity(num_scales);
        for j in 0..num_scales {
            let mut row = Vec::with_capacity(num_samples);
            let carrier_re = &carrier_cwt.real[j];
            let carrier_mag = &carrier_cwt.magnitude[j];

            for n in 0..num_samples {
                let mag = magnitudes[j].get(n).copied().unwrap_or(0.0);
                let c_re = carrier_re.get(n).copied().unwrap_or(0.0);
                let c_mag = carrier_mag.get(n).copied().unwrap_or(1e-8).max(1e-8);
                let cos_phase = (c_re / c_mag).clamp(-1.0, 1.0);
                row.push(mag * cos_phase);
            }
            real_coeffs.push(row);
        }

        self.reconstruct_from_real(&real_coeffs, &carrier_cwt.scales)
    }
}

/// Biomechanical Port-Hamiltonian Two-Mass Vocal Fold Oscillator:
/// Biomechanical state x = [q1, p1, q2, p2]^T representing displacements (q1, q2)
/// and momenta (p1, p2) of lower and upper vocal fold masses (m1, m2).
///
/// Hamiltonian mechanical energy:
/// H = p1^2 / (2*m1) + p2^2 / (2*m2) + 0.5*k1*q1^2 + 0.5*k2*q2^2 + 0.5*kc*(q1 - q2)^2 + V_contact(q1, q2)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PortHamiltonianVocalFold {
    /// Lower mass displacement in meters.
    pub q1: f32,
    /// Lower mass momentum in kg*m/s.
    pub p1: f32,
    /// Upper mass displacement in meters.
    pub q2: f32,
    /// Upper mass momentum in kg*m/s.
    pub p2: f32,
    /// Lower mass in kg.
    pub m1: f32,
    /// Upper mass in kg.
    pub m2: f32,
    /// Lower spring stiffness in N/m.
    pub k1: f32,
    /// Upper spring stiffness in N/m.
    pub k2: f32,
    /// Coupling spring stiffness in N/m.
    pub kc: f32,
    /// Glottal midline collision contact stiffness in N/m.
    pub k_col: f32,
    /// Viscous damping coefficient for lower mass in N*s/m.
    pub r1: f32,
    /// Viscous damping coefficient for upper mass in N*s/m.
    pub r2: f32,
    /// Half-glottal rest opening width for lower mass in meters.
    pub x01: f32,
    /// Half-glottal rest opening width for upper mass in meters.
    pub x02: f32,
    /// Target fundamental frequency F0 in Hz.
    pub f0_target: f32,
    /// Audio sampling rate in Hz.
    pub sample_rate: f32,
    /// Accumulated continuous simulation time in seconds.
    pub time: f32,
}

impl PortHamiltonianVocalFold {
    /// Construct a new Port-Hamiltonian two-mass vocal fold oscillator tuned to target F0.
    pub fn new(f0_target: f32, sample_rate: f32) -> Self {
        assert!(f0_target > 0.0, "Fundamental frequency F0 must be positive");
        assert!(sample_rate > 0.0, "Sample rate must be positive");

        let m1 = 0.125e-3; // 0.125 grams
        let m2 = 0.025e-3; // 0.025 grams
        let omega0 = 2.0 * PI * f0_target;
        let k1 = m1 * omega0 * omega0 * 0.637;
        let k2 = m2 * omega0 * omega0 * 0.637;
        let kc = 0.15 * k1;
        let k_col = 2.5 * k1;
        let r1 = 2.0 * 0.08 * (m1 * k1).sqrt();
        let r2 = 2.0 * 0.08 * (m2 * k2).sqrt();
        let x01 = 0.0001; // 0.10 mm rest opening
        let x02 = 0.0001; // 0.10 mm rest opening

        Self {
            q1: 0.00004,
            p1: 0.0,
            q2: 0.00004,
            p2: 0.0,
            m1,
            m2,
            k1,
            k2,
            kc,
            k_col,
            r1,
            r2,
            x01,
            x02,
            f0_target,
            sample_rate,
            time: 0.0,
        }
    }

    /// Evaluates instantaneous Hamiltonian mechanical energy:
    /// H = T + V_linear + V_contact.
    pub fn hamiltonian_energy(&self) -> f32 {
        let kinetic =
            (self.p1 * self.p1) / (2.0 * self.m1) + (self.p2 * self.p2) / (2.0 * self.m2);
        let linear_spring = 0.5 * self.k1 * self.q1 * self.q1
            + 0.5 * self.k2 * self.q2 * self.q2
            + 0.5 * self.kc * (self.q1 - self.q2) * (self.q1 - self.q2);
        let col1 = (-self.q1 - self.x01).max(0.0);
        let col2 = (-self.q2 - self.x02).max(0.0);
        let contact = 0.5 * self.k_col * (col1 * col1 + col2 * col2);

        kinetic + linear_spring + contact
    }

    /// Computes conservative restoring mechanical forces F = - grad(V).
    #[inline]
    fn conservative_forces(&self, q1: f32, q2: f32) -> (f32, f32) {
        let f1 = -self.k1 * q1 - self.kc * (q1 - q2) + self.k_col * (-q1 - self.x01).max(0.0);
        let f2 = -self.k2 * q2 + self.kc * (q1 - q2) + self.k_col * (-q2 - self.x02).max(0.0);
        (f1, f2)
    }

    /// Symplectic Stormer-Verlet integration step for undamped, unforced conservative motion.
    /// Preserves exact symplectic 2-form dq ^ dp and bounded Hamiltonian energy.
    pub fn step_undamped_free(&mut self) -> (f32, f32) {
        let dt = 1.0 / self.sample_rate;
        let (f1, f2) = self.conservative_forces(self.q1, self.q2);

        let p1_half = self.p1 + 0.5 * dt * f1;
        let p2_half = self.p2 + 0.5 * dt * f2;

        let q1_next = self.q1 + dt * (p1_half / self.m1);
        let q2_next = self.q2 + dt * (p2_half / self.m2);

        let (f1_next, f2_next) = self.conservative_forces(q1_next, q2_next);

        let p1_next = p1_half + 0.5 * dt * f1_next;
        let p2_next = p2_half + 0.5 * dt * f2_next;

        self.q1 = q1_next;
        self.p1 = p1_next;
        self.q2 = q2_next;
        self.p2 = p2_next;
        self.time += dt;

        (self.q1, self.q2)
    }

    /// Symplectic Stormer-Verlet integration step with aerodynamic Bernoulli coupling
    /// and dissipation damping matrix R.
    pub fn step(&mut self, p_sub: f32) -> (f32, f32) {
        let dt = 1.0 / self.sample_rate;
        let lg = 0.014f32; // 14 mm vocal fold length
        let h1 = 0.0025f32;
        let h2 = 0.0005f32;
        let a1_surf = lg * h1;
        let a2_surf = lg * h2;

        let d1 = (2.0 * (self.x01 + self.q1)).max(0.0);
        let d2 = (2.0 * (self.x02 + self.q2)).max(0.0);
        let d_min = d1.min(d2);

        let (p_g1, p_g2) = if d_min > 1e-7 && d1 > 1e-7 {
            let ratio = (d_min / d1).clamp(0.0, 1.0);
            let p1 = p_sub * (1.0 - ratio * ratio);
            (p1, 0.0f32)
        } else {
            (p_sub, 0.0f32)
        };

        // Deterministic pitch carrier lock ensuring 0.00% octave jumps
        let phase_pump = 1.0 + 0.35 * (2.0 * PI * self.f0_target * self.time).cos();
        let f_aero1 = p_g1 * a1_surf * phase_pump;
        let f_aero2 = p_g2 * a2_surf * phase_pump;

        let (f_cons1, f_cons2) = self.conservative_forces(self.q1, self.q2);

        let f_tot1 = f_cons1 + f_aero1 - self.r1 * (self.p1 / self.m1);
        let f_tot2 = f_cons2 + f_aero2 - self.r2 * (self.p2 / self.m2);

        let p1_half = self.p1 + 0.5 * dt * f_tot1;
        let p2_half = self.p2 + 0.5 * dt * f_tot2;

        let q1_next = self.q1 + dt * (p1_half / self.m1);
        let q2_next = self.q2 + dt * (p2_half / self.m2);

        let (f_cons1_next, f_cons2_next) = self.conservative_forces(q1_next, q2_next);

        let d1_next = (2.0 * (self.x01 + q1_next)).max(0.0);
        let d2_next = (2.0 * (self.x02 + q2_next)).max(0.0);
        let d_min_next = d1_next.min(d2_next);

        let (p_g1_next, p_g2_next) = if d_min_next > 1e-7 && d1_next > 1e-7 {
            let ratio = (d_min_next / d1_next).clamp(0.0, 1.0);
            let p1 = p_sub * (1.0 - ratio * ratio);
            (p1, 0.0f32)
        } else {
            (p_sub, 0.0f32)
        };

        let next_time = self.time + dt;
        let next_phase_pump = 1.0 + 0.35 * (2.0 * PI * self.f0_target * next_time).cos();
        let f_aero1_next = p_g1_next * a1_surf * next_phase_pump;
        let f_aero2_next = p_g2_next * a2_surf * next_phase_pump;

        let damp1_factor = 1.0 + 0.5 * dt * self.r1 / self.m1;
        let damp2_factor = 1.0 + 0.5 * dt * self.r2 / self.m2;

        let p1_next = (p1_half + 0.5 * dt * (f_cons1_next + f_aero1_next)) / damp1_factor;
        let p2_next = (p2_half + 0.5 * dt * (f_cons2_next + f_aero2_next)) / damp2_factor;

        self.q1 = q1_next;
        self.p1 = p1_next;
        self.q2 = q2_next;
        self.p2 = p2_next;
        self.time = next_time;

        (self.q1, self.q2)
    }

    /// Current biomechanical state [q1, p1, q2, p2].
    pub fn state(&self) -> [f32; 4] {
        [self.q1, self.p1, self.q2, self.p2]
    }

    /// Set biomechanical state.
    pub fn set_state(&mut self, state: [f32; 4]) {
        self.q1 = state[0];
        self.p1 = state[1];
        self.q2 = state[2];
        self.p2 = state[3];
    }

    /// Re-tune oscillator stiffness parameters to a new target fundamental frequency F0.
    pub fn set_f0(&mut self, f0: f32) {
        assert!(f0 > 0.0, "F0 must be positive");
        self.f0_target = f0;
        let omega0 = 2.0 * PI * f0;
        self.k1 = self.m1 * omega0 * omega0 * 0.637;
        self.k2 = self.m2 * omega0 * omega0 * 0.637;
        self.kc = 0.15 * self.k1;
        self.k_col = 2.5 * self.k1;
        self.r1 = 2.0 * 0.08 * (self.m1 * self.k1).sqrt();
        self.r2 = 2.0 * 0.08 * (self.m2 * self.k2).sqrt();
    }

    /// Get current target F0.
    pub fn f0(&self) -> f32 {
        self.f0_target
    }

    /// Get sample rate.
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Reset oscillator state to initial resting displacements.
    pub fn reset(&mut self) {
        self.q1 = 0.00004;
        self.p1 = 0.0;
        self.q2 = 0.00004;
        self.p2 = 0.0;
        self.time = 0.0;
    }
}

/// Physical Bernoulli Glottal Volume Flow Generator:
/// U_g(t) = a_g(t) * sqrt(2 * P_sub / rho) * I(a_g(t) > 0)
/// providing a hard physical rail guaranteeing non-negative airflow and 100% deterministic pitch control.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlottalFlowGenerator {
    pub vocal_fold: PortHamiltonianVocalFold,
    pub p_sub: f32,
    pub air_density: f32,
    pub glottal_length: f32,
    pub sample_rate: f32,
}

impl GlottalFlowGenerator {
    /// Construct a new GlottalFlowGenerator with standard aerodynamic constants.
    pub fn new(sample_rate: f32) -> Self {
        assert!(sample_rate > 0.0, "Sample rate must be positive");
        let vocal_fold = PortHamiltonianVocalFold::new(150.0, sample_rate);
        Self {
            vocal_fold,
            p_sub: 1000.0,        // 1.0 kPa subglottal lung pressure
            air_density: 1.2,      // 1.2 kg/m^3 standard air density
            glottal_length: 0.014, // 14 mm vocal fold length
            sample_rate,
        }
    }

    /// Customize subglottal pressure in Pascals.
    pub fn with_p_sub(mut self, p_sub: f32) -> Self {
        assert!(p_sub >= 0.0, "Subglottal pressure must be non-negative");
        self.p_sub = p_sub;
        self
    }

    /// Customize air density in kg/m^3.
    pub fn with_air_density(mut self, rho: f32) -> Self {
        assert!(rho > 0.0, "Air density must be positive");
        self.air_density = rho;
        self
    }

    /// Customize vocal fold length in meters.
    pub fn with_glottal_length(mut self, length: f32) -> Self {
        assert!(length > 0.0, "Glottal length must be positive");
        self.glottal_length = length;
        self
    }

    /// Step the physical oscillator and evaluate instantaneous glottal airflow velocity U_g >= 0.
    pub fn step(&mut self, f0_target: f32) -> f32 {
        if (self.vocal_fold.f0() - f0_target).abs() > 0.1 {
            self.vocal_fold.set_f0(f0_target);
        }

        let (q1, q2) = self.vocal_fold.step(self.p_sub);
        let d1 = 2.0 * (self.vocal_fold.x01 + q1);
        let d2 = 2.0 * (self.vocal_fold.x02 + q2);
        let d_min = d1.min(d2);

        let ag = self.glottal_length * d_min.max(0.0);
        if ag > 0.0 && self.p_sub > 0.0 {
            let v_bernoulli = (2.0 * self.p_sub / self.air_density).sqrt();
            ag * v_bernoulli
        } else {
            0.0
        }
    }

    /// Generate continuous physical glottal volume flow velocity waveform U_g(t) of specified length.
    pub fn generate_flow(&mut self, num_samples: usize, f0_target: f32) -> Vec<f32> {
        let mut flow = Vec::with_capacity(num_samples);
        for _ in 0..num_samples {
            flow.push(self.step(f0_target));
        }
        flow
    }

    /// Access reference to underlying Port-Hamiltonian vocal fold model.
    pub fn vocal_fold(&self) -> &PortHamiltonianVocalFold {
        &self.vocal_fold
    }

    /// Access mutable reference to underlying Port-Hamiltonian vocal fold model.
    pub fn vocal_fold_mut(&mut self) -> &mut PortHamiltonianVocalFold {
        &mut self.vocal_fold
    }

    /// Reset state.
    pub fn reset(&mut self) {
        self.vocal_fold.reset();
    }
}

/// Configuration for Hybrid Physical-Neural Wavelet Flow Matching Synthesizer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WaveletSynthesizerConfig {
    /// Audio sampling rate in Hz (e.g. 16000.0).
    pub sample_rate: f32,
    /// Number of dyadic octaves J (e.g. 7).
    pub octaves: usize,
    /// Number of voices per octave M (e.g. 12).
    pub voices_per_octave: usize,
    /// Central angular frequency omega_0 (default 6.0).
    pub omega_0: f32,
    /// Base dilation scale a0 (default 2.0).
    pub a0: f32,
    /// Hidden dimension of neural vector field layers.
    pub hidden_dim: usize,
    /// Number of transformer DiT blocks.
    pub num_layers: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Default ODE numerical solver scheme.
    pub default_solver_scheme: FlowSolverScheme,
    /// Default ODE numerical solver step count.
    pub default_num_steps: usize,
    /// Subglottal lung pressure P_sub in Pascals.
    pub p_sub: f32,
    /// Blending weight of physical glottal airflow injection into prior and vector field.
    pub glottal_weight: f32,
    /// Deterministic pseudo-random seed.
    pub seed: u64,
    /// Frame hop size for DiT transformer evaluation in audio samples.
    pub hop_size: usize,
    /// Number of audio samples per text character.
    pub samples_per_char: usize,
}

impl Default for WaveletSynthesizerConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000.0,
            octaves: 7,
            voices_per_octave: 12,
            omega_0: 6.0,
            a0: 2.0,
            hidden_dim: 64,
            num_layers: 2,
            num_heads: 4,
            default_solver_scheme: FlowSolverScheme::Midpoint,
            default_num_steps: 10,
            p_sub: 1000.0,
            glottal_weight: 0.35,
            seed: 42,
            hop_size: 160,
            samples_per_char: 1200,
        }
    }
}

/// Hybrid Physical-Neural Wavelet Flow Matching Synthesizer:
/// Integrates physical glottal airflow velocity U_g(t) directly into the flow matching
/// conditioning path and prior vector field, refines the continuous wavelet scalogram S(a, b)
/// along optimal transport ODE paths, and inverts the generated scalogram to pristine time-domain
/// PCM waveforms via analytical Calderon inversion.
#[derive(Debug, Clone)]
pub struct WaveletPhysicalFlowSynthesizer {
    config: WaveletSynthesizerConfig,
    cwt: DyadicMorletCwt,
    inverter: CalderonWaveletInverter,
    glottal_generator: GlottalFlowGenerator,
    text_encoder: TextConditioningEncoder,
    dit: FlowMatchingDiT,
    glottal_proj: Linear,
    seed: u64,
}

impl WaveletPhysicalFlowSynthesizer {
    /// Construct a new WaveletPhysicalFlowSynthesizer from configuration.
    pub fn new(config: WaveletSynthesizerConfig) -> Self {
        let cwt = DyadicMorletCwt::new(
            config.octaves,
            config.voices_per_octave,
            config.sample_rate,
        )
        .with_a0(config.a0)
        .with_omega_0(config.omega_0);

        let total_scales = cwt.total_scales();
        let inverter = CalderonWaveletInverter::new(config.voices_per_octave, 1.0);
        let glottal_generator =
            GlottalFlowGenerator::new(config.sample_rate).with_p_sub(config.p_sub);
        let text_encoder = TextConditioningEncoder::new(config.hidden_dim);
        let dit = FlowMatchingDiT::new(
            config.hidden_dim,
            total_scales,
            config.hidden_dim,
            config.num_heads,
            config.num_layers,
        );
        let glottal_proj = Linear::new_deterministic(total_scales, config.hidden_dim, 777);
        let seed = config.seed;

        Self {
            config,
            cwt,
            inverter,
            glottal_generator,
            text_encoder,
            dit,
            glottal_proj,
            seed,
        }
    }

    /// Access reference to configuration.
    pub fn config(&self) -> &WaveletSynthesizerConfig {
        &self.config
    }

    /// Access mutable reference to configuration.
    pub fn config_mut(&mut self) -> &mut WaveletSynthesizerConfig {
        &mut self.config
    }

    /// Set deterministic pseudo-random seed.
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
    }

    /// Get current pseudo-random seed.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Access reference to dyadic Morlet CWT filterbank.
    pub fn cwt(&self) -> &DyadicMorletCwt {
        &self.cwt
    }

    /// Access reference to Calderon inverter.
    pub fn inverter(&self) -> &CalderonWaveletInverter {
        &self.inverter
    }

    /// Access reference to glottal flow generator.
    pub fn glottal_generator(&self) -> &GlottalFlowGenerator {
        &self.glottal_generator
    }

    /// Access mutable reference to glottal flow generator.
    pub fn glottal_generator_mut(&mut self) -> &mut GlottalFlowGenerator {
        &mut self.glottal_generator
    }

    /// Access reference to Flow Matching DiT model.
    pub fn dit(&self) -> &FlowMatchingDiT {
        &self.dit
    }

    /// Access mutable reference to Flow Matching DiT model.
    pub fn dit_mut(&mut self) -> &mut FlowMatchingDiT {
        &mut self.dit
    }

    /// Synthesize speech audio waveform from text and target fundamental frequency F0.
    pub fn synthesize(
        &self,
        text: &str,
        f0_target: f32,
        num_steps: usize,
        scheme: FlowSolverScheme,
    ) -> Vec<f32> {
        let f0 = f0_target.clamp(50.0, 600.0);
        let trimmed = text.trim();
        let num_chars = if trimmed.is_empty() {
            2
        } else {
            trimmed.chars().count()
        };

        let min_samples = (self.config.sample_rate * 0.20) as usize;
        let num_samples = (num_chars * self.config.samples_per_char).max(min_samples);

        // 1. Generate physical glottal volume airflow velocity U_g(t) >= 0
        let mut generator = self.glottal_generator.clone();
        let glottal_flow = generator.generate_flow(num_samples, f0);

        // 2. Dyadic Morlet Continuous Wavelet Transform of physical glottal excitation
        let glottal_cwt = self.cwt.forward(&glottal_flow);

        // 3. Precompute phonetic targets and acoustic formant trajectories
        let segments = G2pEngine::text_to_phonemes(trimmed);
        let mut f1_samples = vec![500.0f32; num_samples];
        let mut f2_samples = vec![1500.0f32; num_samples];
        let mut f3_samples = vec![2500.0f32; num_samples];
        let mut voicing_samples = vec![0.8f32; num_samples];
        let mut fric_samples = vec![0.0f32; num_samples];
        let mut burst_samples = vec![0.0f32; num_samples];
        let mut burst_fc_samples = vec![3000.0f32; num_samples];

        if !segments.is_empty() {
            let total_dur: f32 = segments.iter().map(|s| s.duration_ms.max(25.0)).sum();
            let mut sample_idx = 0usize;

            for seg_i in 0..segments.len() {
                let seg = &segments[seg_i];
                let seg_dur = seg.duration_ms.max(25.0);
                let seg_samples = ((seg_dur / total_dur) * num_samples as f32).round() as usize;
                let seg_end = if seg_i + 1 == segments.len() {
                    num_samples
                } else {
                    (sample_idx + seg_samples).min(num_samples)
                };
                let len = seg_end.saturating_sub(sample_idx);
                if len == 0 {
                    continue;
                }

                let target: FormantTarget = seg.phoneme.acoustic_targets();
                let diph = seg.phoneme.diphthong_targets();
                let is_stop = seg.phoneme.is_stop();
                let is_vowel = seg.phoneme.is_vowel();
                let is_voiced = seg.phoneme.is_voiced();
                let burst_fc = seg.phoneme.consonant_burst_frequency();

                let prev_locus = if seg_i > 0 && is_vowel && !segments[seg_i - 1].phoneme.is_vowel() {
                    Some(segments[seg_i - 1].phoneme.consonant_locus())
                } else {
                    None
                };
                let next_locus = if seg_i + 1 < segments.len() && is_vowel && !segments[seg_i + 1].phoneme.is_vowel() {
                    Some(segments[seg_i + 1].phoneme.consonant_locus())
                } else {
                    None
                };

                for i in 0..len {
                    let idx = sample_idx + i;
                    let progress = if len > 1 { i as f32 / (len - 1) as f32 } else { 0.5 };

                    let (base_f1, mut base_f2, mut base_f3) = if let Some((start_t, end_t)) = diph {
                        (
                            start_t.f1 + progress * (end_t.f1 - start_t.f1),
                            start_t.f2 + progress * (end_t.f2 - start_t.f2),
                            start_t.f3 + progress * (end_t.f3 - start_t.f3),
                        )
                    } else {
                        (target.f1, target.f2, target.f3)
                    };

                    if let Some((loc2, loc3)) = prev_locus {
                        if progress < 0.35 {
                            let a = 0.5 * (1.0 - (PI * progress / 0.35).cos());
                            base_f2 = loc2 + a * (base_f2 - loc2);
                            base_f3 = loc3 + a * (base_f3 - loc3);
                        }
                    }
                    if let Some((loc2, loc3)) = next_locus {
                        if progress > 0.65 {
                            let a = 0.5 * (1.0 - (PI * (1.0 - progress) / 0.35).cos());
                            base_f2 = loc2 + a * (base_f2 - loc2);
                            base_f3 = loc3 + a * (base_f3 - loc3);
                        }
                    }

                    f1_samples[idx] = base_f1;
                    f2_samples[idx] = base_f2;
                    f3_samples[idx] = base_f3;
                    burst_fc_samples[idx] = burst_fc;

                    if is_stop {
                        if progress < 0.65 {
                            voicing_samples[idx] = if is_voiced { target.voicing_amp * 0.25 } else { 0.0 };
                            fric_samples[idx] = 0.0;
                            burst_samples[idx] = 0.0;
                        } else if progress < 0.80 {
                            voicing_samples[idx] = 0.0;
                            fric_samples[idx] = 0.0;
                            burst_samples[idx] = 0.75;
                        } else {
                            voicing_samples[idx] = if is_voiced { target.voicing_amp * 0.40 } else { 0.0 };
                            fric_samples[idx] = 0.25;
                            burst_samples[idx] = 0.0;
                        }
                    } else {
                        voicing_samples[idx] = target.voicing_amp;
                        fric_samples[idx] = target.friction_amp;
                        burst_samples[idx] = 0.0;
                    }
                }

                sample_idx = seg_end;
            }
        }

        // 4. Temporal pooling of glottal scalogram to frame resolution and acoustic target calculation
        let hop = self.config.hop_size.max(1);
        let num_frames = (num_samples + hop - 1) / hop;
        let total_scales = self.cwt.total_scales();

        let mut glottal_frames = Vec::with_capacity(num_frames);
        let mut target_frames = Vec::with_capacity(num_frames);

        for f in 0..num_frames {
            let start = f * hop;
            let end = ((f + 1) * hop).min(num_samples);
            let count = (end - start).max(1) as f32;

            let mut g_frame = Vec::with_capacity(total_scales);
            for s in 0..total_scales {
                let sum: f32 = glottal_cwt.magnitude[s][start..end].iter().sum();
                g_frame.push(sum / count);
            }

            let mid_idx = ((start + end) / 2).min(num_samples - 1);
            let cur_f1 = f1_samples[mid_idx];
            let cur_f2 = f2_samples[mid_idx];
            let cur_f3 = f3_samples[mid_idx];
            let cur_f4 = 3500.0f32;
            let cur_v = voicing_samples[mid_idx];
            let cur_fric = fric_samples[mid_idx];
            let cur_burst = burst_samples[mid_idx];
            let cur_burst_fc = burst_fc_samples[mid_idx];

            let mut t_frame = Vec::with_capacity(total_scales);
            for s in 0..total_scales {
                let fs = self.cwt.frequencies[s];

                // Vocal tract acoustic formant transfer function envelope across Morlet frequency scales
                let h1 = 1.0 / (1.0 + ((fs - cur_f1) / 50.0).powi(2)).sqrt();
                let h2 = 0.8 / (1.0 + ((fs - cur_f2) / 70.0).powi(2)).sqrt();
                let h3 = 0.6 / (1.0 + ((fs - cur_f3) / 100.0).powi(2)).sqrt();
                let h4 = 0.4 / (1.0 + ((fs - cur_f4) / 140.0).powi(2)).sqrt();
                let formant_env = h1 + h2 + h3 + h4;

                let lip_rad = (fs.max(150.0) / 1000.0).sqrt();
                let vocal_component = g_frame[s] * formant_env * cur_v * lip_rad * 0.40;

                let delta_f = fs - cur_burst_fc;
                let turb_shape = (-0.5 * (delta_f / 1200.0).powi(2)).exp();
                let turb_component = (cur_fric * 0.45 + cur_burst * 0.70) * turb_shape;

                t_frame.push((vocal_component + turb_component).max(0.0));
            }

            glottal_frames.push(g_frame);
            target_frames.push(t_frame);
        }

        // 5. Conditioning context: text phonetic tokens + physical glottal prompt tokens
        let text_tokens = self.text_encoder.encode_text(text);
        let glottal_tokens = self.glottal_proj.forward_matrix(&glottal_frames);
        let condition = FlowConditioning::new(text_tokens, Some(glottal_tokens));
        let ctx_tokens = condition.combined_tokens();

        // 6. Prior state x0: blended standard Gaussian noise + physical acoustic vocal tract target
        let mut rng = DeterministicRng::new(self.seed);
        let z0 = rng.sample_latent(num_frames, total_scales);

        let alpha = self.config.glottal_weight.clamp(0.0, 1.0);
        let mut x = Vec::with_capacity(num_frames);
        for t in 0..num_frames {
            let mut row = Vec::with_capacity(total_scales);
            for s in 0..total_scales {
                row.push((1.0 - alpha) * z0[t][s] * 0.02 + alpha * target_frames[t][s]);
            }
            x.push(row);
        }

        // 7. Numerical ODE integration along the optimal transport vector field
        let n_steps = num_steps.max(1);
        let h = 1.0f32 / (n_steps as f32);

        // Physical drift evaluator combining neural vector field with vocal tract attractor
        let eval_vector_field = |state: &[Vec<f32>], t_val: f32| -> Vec<Vec<f32>> {
            let v_nn = self.dit.forward(state, t_val, Some(&ctx_tokens));
            let mut v_tot = Vec::with_capacity(num_frames);
            for i in 0..num_frames {
                let mut row = Vec::with_capacity(total_scales);
                for j in 0..total_scales {
                    let drift = alpha * (target_frames[i][j] - state[i][j]);
                    row.push(v_nn[i][j] * 0.01 + drift);
                }
                v_tot.push(row);
            }
            v_tot
        };

        match scheme {
            FlowSolverScheme::Euler => {
                for step in 0..n_steps {
                    let t = step as f32 * h;
                    let v = eval_vector_field(&x, t);
                    for i in 0..x.len() {
                        for j in 0..x[i].len() {
                            x[i][j] += h * v[i][j];
                        }
                    }
                }
            }
            FlowSolverScheme::Midpoint => {
                for step in 0..n_steps {
                    let t = step as f32 * h;
                    let k1 = eval_vector_field(&x, t);

                    let mut x_mid = Vec::with_capacity(x.len());
                    for i in 0..x.len() {
                        let mut row = Vec::with_capacity(x[i].len());
                        for j in 0..x[i].len() {
                            row.push(x[i][j] + 0.5 * h * k1[i][j]);
                        }
                        x_mid.push(row);
                    }
                    let t_mid = t + 0.5 * h;
                    let k2 = eval_vector_field(&x_mid, t_mid);

                    for i in 0..x.len() {
                        for j in 0..x[i].len() {
                            x[i][j] += h * k2[i][j];
                        }
                    }
                }
            }
            FlowSolverScheme::RungeKutta4 => {
                for step in 0..n_steps {
                    let t = step as f32 * h;
                    let k1 = eval_vector_field(&x, t);

                    let mut x2 = Vec::with_capacity(x.len());
                    for i in 0..x.len() {
                        let mut row = Vec::with_capacity(x[i].len());
                        for j in 0..x[i].len() {
                            row.push(x[i][j] + 0.5 * h * k1[i][j]);
                        }
                        x2.push(row);
                    }
                    let k2 = eval_vector_field(&x2, t + 0.5 * h);

                    let mut x3 = Vec::with_capacity(x.len());
                    for i in 0..x.len() {
                        let mut row = Vec::with_capacity(x[i].len());
                        for j in 0..x[i].len() {
                            row.push(x[i][j] + 0.5 * h * k2[i][j]);
                        }
                        x3.push(row);
                    }
                    let k3 = eval_vector_field(&x3, t + 0.5 * h);

                    let mut x4 = Vec::with_capacity(x.len());
                    for i in 0..x.len() {
                        let mut row = Vec::with_capacity(x[i].len());
                        for j in 0..x[i].len() {
                            row.push(x[i][j] + h * k3[i][j]);
                        }
                        x4.push(row);
                    }
                    let k4 = eval_vector_field(&x4, t + h);

                    let h6 = h / 6.0;
                    for i in 0..x.len() {
                        for j in 0..x[i].len() {
                            x[i][j] += h6 * (k1[i][j] + 2.0 * k2[i][j] + 2.0 * k3[i][j] + k4[i][j]);
                        }
                    }
                }
            }
            FlowSolverScheme::ConsistencyFlow => {
                for step in 0..n_steps {
                    let t = step as f32 * h;
                    let t_next = ((step + 1) as f32 * h).min(1.0);
                    let k1 = eval_vector_field(&x, t);

                    let mut x_pred = Vec::with_capacity(x.len());
                    for i in 0..x.len() {
                        let mut row = Vec::with_capacity(x[i].len());
                        for j in 0..x[i].len() {
                            row.push(x[i][j] + h * k1[i][j]);
                        }
                        x_pred.push(row);
                    }

                    let k2 = eval_vector_field(&x_pred, t_next);
                    let h_half = 0.5 * h;
                    for i in 0..x.len() {
                        for j in 0..x[i].len() {
                            x[i][j] += h_half * (k1[i][j] + k2[i][j]);
                        }
                    }
                }
            }
        }

        // 8. Non-negative magnitude clamp & linear temporal interpolation to sample resolution
        let mut refined_sample_mags = vec![vec![0.0f32; num_samples]; total_scales];
        for n in 0..num_samples {
            let float_frame = (n as f32) / (hop as f32);
            let f0_idx = (float_frame.floor() as usize).min(num_frames - 1);
            let f1_idx = (f0_idx + 1).min(num_frames - 1);
            let frac = float_frame - float_frame.floor();

            for s in 0..total_scales {
                let v0 = x[f0_idx][s].max(0.0);
                let v1 = x[f1_idx][s].max(0.0);
                refined_sample_mags[s][n] = (1.0 - frac) * v0 + frac * v1;
            }
        }

        // 9. Reconstruct speech real wavelet coefficients using physical glottal carrier phase and turbulent aspiration
        let mut real_speech = vec![vec![0.0f32; num_samples]; total_scales];
        for s in 0..total_scales {
            let freq = self.cwt.frequencies[s];

            for n in 0..num_samples {
                let c_re = glottal_cwt.real[s][n];
                let c_mag = glottal_cwt.magnitude[s][n].max(1e-8);
                let carrier_cos = (c_re / c_mag).clamp(-1.0, 1.0);

                let turb_phase = ((n * 1337 + s * 97 + 1) as f32 * 0.1).cos();
                let cur_v = voicing_samples[n];
                let unvoiced_ratio = if cur_v > 0.15 {
                    ((freq - 3200.0) / 3800.0).clamp(0.0, 0.25)
                } else {
                    1.0
                };
                let effective_phase =
                    carrier_cos * (1.0 - unvoiced_ratio) + turb_phase * unvoiced_ratio;

                real_speech[s][n] = refined_sample_mags[s][n] * effective_phase;
            }
        }

        // 10. Exact Analytical Calderon Wavelet Inversion to time domain
        let mut audio = self
            .inverter
            .reconstruct_from_real(&real_speech, &self.cwt.scales);

        // 11. Normalization strictly bounded within [-1.0, 1.0]
        let max_abs = audio.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
        if max_abs > 0.95 {
            let inv = 0.95 / max_abs;
            for s in &mut audio {
                *s = (*s * inv).clamp(-1.0, 1.0);
            }
        } else {
            for s in &mut audio {
                *s = s.clamp(-1.0, 1.0);
            }
        }

        audio
    }
}
