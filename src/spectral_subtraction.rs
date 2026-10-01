#![deny(unsafe_code)]

/// Configuration parameters for Spectral Subtraction Noise Suppression.
#[derive(Debug, Clone, Copy)]
pub struct SpectralSubtractionConfig {
    /// Over-subtraction factor alpha (typically 1.2 to 2.5). Higher values aggressively
    /// remove noise peaks but require spectral floor clamping.
    pub alpha: f32,
    /// Spectral floor gamma (typically 0.02 to 0.10, i.e., -17 dB to -10 dB).
    /// Prevents isolated spectral peaks that produce "musical noise" artifacts.
    pub spectral_floor: f32,
    /// Recursive noise floor adaptation rate during non-speech intervals (beta, typically 0.02 to 0.10).
    pub noise_adapt_rate: f32,
}

impl Default for SpectralSubtractionConfig {
    fn default() -> Self {
        Self {
            alpha: 1.5,
            spectral_floor: 0.05,
            noise_adapt_rate: 0.05,
        }
    }
}

/// Recursive Spectral Subtraction Noise Suppressor.
///
/// Operates on the one-sided power spectral density (PSD) frames produced by the STFT.
/// Continuously tracks stationary background noise during non-speech intervals
/// (guided by Voice Activity Detection) and applies generalized spectral over-subtraction
/// with spectral floor clamping:
///
/// P_clean[k] = max(P[k] - alpha * N[k],  spectral_floor * P[k])
#[derive(Debug, Clone)]
pub struct SpectralSubtractionSuppressor {
    num_bins: usize,
    config: SpectralSubtractionConfig,
    noise_floor: Vec<f32>,
    is_initialized: bool,
}

impl SpectralSubtractionSuppressor {
    /// Constructs a new Spectral Subtraction Suppressor.
    ///
    /// # Arguments
    /// * `num_bins` - Number of unique frequency bins in the one-sided power spectrum (e.g., N/2 + 1).
    /// * `config` - Algorithm tuning parameters.
    pub fn new(num_bins: usize, config: SpectralSubtractionConfig) -> Self {
        Self {
            num_bins,
            config,
            noise_floor: vec![1e-6; num_bins],
            is_initialized: false,
        }
    }

    /// Recursively updates the estimated background noise spectrum using a non-speech frame.
    pub fn update_noise_floor(&mut self, power_spectrum: &[f32]) {
        let n = self.num_bins.min(power_spectrum.len());
        if !self.is_initialized {
            for i in 0..n {
                self.noise_floor[i] = power_spectrum[i].max(1e-9);
            }
            self.is_initialized = true;
            return;
        }

        let beta = self.config.noise_adapt_rate;
        let one_minus_beta = 1.0 - beta;

        for i in 0..n {
            let p = power_spectrum[i].max(1e-9);
            self.noise_floor[i] = one_minus_beta * self.noise_floor[i] + beta * p;
        }
    }

    /// Processes a power spectral density frame in place.
    ///
    /// If `is_speech` is false, updates the noise floor estimate.
    /// If `is_speech` is true, performs spectral subtraction to attenuate background noise.
    pub fn process_spectrum(&mut self, power_spectrum: &mut [f32], is_speech: bool) {
        let n = self.num_bins.min(power_spectrum.len());

        if !is_speech {
            self.update_noise_floor(&power_spectrum[..n]);
        }

        if !self.is_initialized {
            return;
        }

        let alpha = self.config.alpha;
        let floor = self.config.spectral_floor;

        for i in 0..n {
            let original_p = power_spectrum[i];
            let noise_p = self.noise_floor[i];

            let subtracted = original_p - alpha * noise_p;
            let floor_p = floor * original_p;

            power_spectrum[i] = subtracted.max(floor_p).max(1e-10);
        }
    }

    /// Returns the current estimated noise floor spectrum.
    pub fn noise_floor(&self) -> &[f32] {
        &self.noise_floor
    }

    /// Resets the noise floor estimate.
    pub fn reset(&mut self) {
        self.noise_floor.fill(1e-6);
        self.is_initialized = false;
    }
}
