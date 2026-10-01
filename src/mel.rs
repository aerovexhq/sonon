//! Mel-scale filterbank and acoustic feature representation for speech and phrase recognition.

use std::f32::consts::PI;

/// Mel filterbank triangular weights.
#[derive(Debug, Clone)]
pub struct MelFilterbank {
    num_filters: usize,
    fft_size: usize,
    sample_rate: f32,
    filter_weights: Vec<Vec<(usize, f32)>>,
}

impl MelFilterbank {
    /// Construct a triangular Mel filterbank covering [low_freq, high_freq] Hz.
    pub fn new(
        num_filters: usize,
        fft_size: usize,
        sample_rate: f32,
        low_freq: f32,
        high_freq: f32,
    ) -> Self {
        assert!(num_filters > 0, "Number of filters must be positive");
        assert!(fft_size > 0, "FFT size must be positive");
        assert!(sample_rate > 0.0, "Sample rate must be positive");
        assert!(
            high_freq > low_freq && high_freq <= sample_rate / 2.0,
            "Invalid frequency bounds"
        );

        let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
        let mel_to_hz = |mel: f32| 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0);

        let min_mel = hz_to_mel(low_freq);
        let max_mel = hz_to_mel(high_freq);

        let mut mel_points = Vec::with_capacity(num_filters + 2);
        for i in 0..=(num_filters + 1) {
            let mel = min_mel + (i as f32) * (max_mel - min_mel) / ((num_filters + 1) as f32);
            mel_points.push(mel_to_hz(mel));
        }

        let num_bins = fft_size / 2 + 1;
        let bin_width = sample_rate / (fft_size as f32);

        let mut filter_weights = Vec::with_capacity(num_filters);
        for m in 1..=num_filters {
            let left_hz = mel_points[m - 1];
            let center_hz = mel_points[m];
            let right_hz = mel_points[m + 1];

            let mut weights = Vec::new();
            for k in 0..num_bins {
                let freq = (k as f32) * bin_width;
                if freq >= left_hz && freq <= center_hz {
                    let weight = (freq - left_hz) / (center_hz - left_hz);
                    if weight > 1e-6 {
                        weights.push((k, weight));
                    }
                } else if freq > center_hz && freq <= right_hz {
                    let weight = (right_hz - freq) / (right_hz - center_hz);
                    if weight > 1e-6 {
                        weights.push((k, weight));
                    }
                }
            }
            filter_weights.push(weights);
        }

        Self {
            num_filters,
            fft_size,
            sample_rate,
            filter_weights,
        }
    }

    /// Apply filterbank to power spectrum, producing linear channel energies.
    pub fn compute_energies(&self, power_spectrum: &[f32]) -> Vec<f32> {
        let mut energies = Vec::with_capacity(self.num_filters);
        for filter in &self.filter_weights {
            let mut sum = 0.0f32;
            for &(bin, weight) in filter {
                if bin < power_spectrum.len() {
                    sum += power_spectrum[bin] * weight;
                }
            }
            energies.push(sum);
        }
        energies
    }

    /// Apply filterbank to power spectrum, producing log-energies.
    pub fn compute_log_energies(&self, power_spectrum: &[f32]) -> Vec<f32> {
        let mut log_energies = Vec::with_capacity(self.num_filters);
        for filter in &self.filter_weights {
            let mut sum = 0.0f32;
            for &(bin, weight) in filter {
                if bin < power_spectrum.len() {
                    sum += power_spectrum[bin] * weight;
                }
            }
            // Clamp to prevent log(0)
            let log_energy = (sum.max(1e-10)).ln();
            log_energies.push(log_energy);
        }
        log_energies
    }

    /// Compute Mel-Frequency Cepstral Coefficients (MFCC) via DCT-II.
    pub fn compute_mfcc(&self, log_energies: &[f32], num_ceps: usize) -> Vec<f32> {
        let n = log_energies.len();
        let mut mfcc = Vec::with_capacity(num_ceps);
        let norm = (2.0 / n as f32).sqrt();

        for i in 0..num_ceps {
            let mut sum = 0.0f32;
            for j in 0..n {
                let angle = PI * (i as f32) * (j as f32 + 0.5) / (n as f32);
                sum += log_energies[j] * angle.cos();
            }
            mfcc.push(sum * norm);
        }
        mfcc
    }

    pub fn num_filters(&self) -> usize {
        self.num_filters
    }

    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }
}
