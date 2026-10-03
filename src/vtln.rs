#![deny(unsafe_code)]

//! Vocal Tract Length Normalization (VTLN) frequency warping for edge acoustic feature extraction.
//!
//! Normalizes speaker physiological differences arising from vocal tract length ($L \sim 17.5$ cm for adult
//! males, $15.0$ cm for adult females, $10-12$ cm for children), reducing inter-speaker formant dispersion
//! on streaming MFCC filterbanks.
//!
//! Implements piecewise linear and bilinear frequency warping functions $f' = g_\alpha(f)$ with
//! monotonic Nyquist boundary preservation and grid-search optimal warping factor estimation.

use serde::{Deserialize, Serialize};

/// Configuration parameters for Vocal Tract Length Normalization (VTLN).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VtlnConfig {
    /// Frequency warping factor $\alpha \in [0.75, 1.35]$ ($\alpha = 1.0$ is identity).
    pub alpha: f32,
    /// Inflection point factor as a fraction of Nyquist frequency (typically $c = 0.875$, i.e. $7/8$).
    pub inflection_factor: f32,
}

impl Default for VtlnConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            inflection_factor: 0.875,
        }
    }
}

/// Piecewise linear and bilinear Vocal Tract Length Normalization frequency warping.
#[derive(Debug, Clone)]
pub struct VtlnWarping;

impl VtlnWarping {
    /// Forward piecewise linear VTLN frequency warping $f' = g_\alpha(f)$.
    ///
    /// Maps input physical frequency $f \in [0, f_{\text{Nyq}}]$ to normalized frequency $f'$.
    ///
    /// For $\alpha \le 1.0$:
    /// $$f_0 = c \cdot f_{\text{Nyq}}$$
    /// $$g_\alpha(f) = \begin{cases} \alpha f, & f \le f_0 \\ \alpha f_0 + \frac{f_{\text{Nyq}} - \alpha f_0}{f_{\text{Nyq}} - f_0} (f - f_0), & f > f_0 \end{cases}$$
    ///
    /// For $\alpha > 1.0$:
    /// $$f_0 = \frac{c}{\alpha} \cdot f_{\text{Nyq}}$$
    /// with identical piecewise continuity ensuring monotonic mapping strictly bounded by $f_{\text{Nyq}}$.
    pub fn warp_freq(f: f32, alpha: f32, nyquist: f32) -> f32 {
        if (alpha - 1.0).abs() < 1e-4 {
            return f.clamp(0.0, nyquist);
        }

        let c = 0.875f32;
        let f0 = if alpha <= 1.0 {
            c * nyquist
        } else {
            (c / alpha) * nyquist
        };
        let f0_prime = alpha * f0;

        let f_clamped = f.clamp(0.0, nyquist);
        if f_clamped <= f0 {
            (alpha * f_clamped).clamp(0.0, nyquist)
        } else {
            let slope = (nyquist - f0_prime) / (nyquist - f0);
            (f0_prime + slope * (f_clamped - f0)).clamp(0.0, nyquist)
        }
    }

    /// Inverse piecewise linear VTLN frequency warping $f = g_\alpha^{-1}(f')$.
    ///
    /// Maps normalized canonical target frequency $f'$ back to physical speaker frequency $f$.
    pub fn inv_warp_freq(f_prime: f32, alpha: f32, nyquist: f32) -> f32 {
        if (alpha - 1.0).abs() < 1e-4 {
            return f_prime.clamp(0.0, nyquist);
        }

        let c = 0.875f32;
        let f0 = if alpha <= 1.0 {
            c * nyquist
        } else {
            (c / alpha) * nyquist
        };
        let f0_prime = alpha * f0;

        let fp_clamped = f_prime.clamp(0.0, nyquist);
        if fp_clamped <= f0_prime {
            (fp_clamped / alpha).clamp(0.0, nyquist)
        } else {
            let slope = (nyquist - f0) / (nyquist - f0_prime);
            (f0 + slope * (fp_clamped - f0_prime)).clamp(0.0, nyquist)
        }
    }

    /// Warps a discrete power spectrum by resampling FFT bins using linear interpolation.
    pub fn warp_power_spectrum(
        power: &[f32],
        alpha: f32,
        sample_rate: f32,
    ) -> Vec<f32> {
        if power.is_empty() {
            return Vec::new();
        }
        if (alpha - 1.0).abs() < 1e-4 {
            return power.to_vec();
        }

        let num_bins = power.len();
        let nyquist = sample_rate * 0.5;
        let bin_width = nyquist / (num_bins - 1) as f32;

        let mut warped = vec![0.0f32; num_bins];

        for k in 0..num_bins {
            let f_target = k as f32 * bin_width;
            // The frequency in the source spectrum that corresponds to this target frequency:
            let f_source = Self::inv_warp_freq(f_target, alpha, nyquist);
            let source_bin = f_source / bin_width;

            let k0 = (source_bin.floor() as usize).min(num_bins - 1);
            let k1 = (k0 + 1).min(num_bins - 1);
            let frac = source_bin - k0 as f32;

            warped[k] = power[k0] * (1.0 - frac) + power[k1] * frac;
        }

        warped
    }
}

/// Dynamic estimator for speaker-specific vocal tract length warping factor $\alpha$.
#[derive(Debug, Clone)]
pub struct VtlnWarpEstimator {
    canonical_centroid_hz: f32,
    smoothed_alpha: f32,
    smoothing_factor: f32,
}

impl Default for VtlnWarpEstimator {
    fn default() -> Self {
        Self {
            canonical_centroid_hz: 1350.0, // Canonical vowel formant centroid for average adult voice
            smoothed_alpha: 1.0,
            smoothing_factor: 0.15,
        }
    }
}

impl VtlnWarpEstimator {
    /// Construct a new VTLN warp estimator with custom canonical reference centroid.
    pub fn new(canonical_centroid_hz: f32) -> Self {
        Self {
            canonical_centroid_hz: canonical_centroid_hz.max(500.0),
            smoothed_alpha: 1.0,
            smoothing_factor: 0.15,
        }
    }

    /// Estimate instantaneous vocal tract warping factor $\alpha$ from speech spectral centroid.
    ///
    /// Evaluates the power-weighted center of gravity in the speech vowel formant region (300 - 3500 Hz):
    /// $$C = \frac{\sum_k f_k \cdot P[k]}{\sum_k P[k]}$$
    /// $$\alpha \approx \left(\frac{C_{\text{canon}}}{C}\right)^{0.65}$$
    pub fn estimate_from_centroid(
        &mut self,
        power_spectrum: &[f32],
        sample_rate: f32,
    ) -> f32 {
        if power_spectrum.is_empty() {
            return self.smoothed_alpha;
        }

        let num_bins = power_spectrum.len();
        let bin_width = (sample_rate * 0.5) / (num_bins - 1) as f32;

        let min_bin = ((300.0 / bin_width).floor() as usize).min(num_bins - 1);
        let max_bin = ((3500.0 / bin_width).ceil() as usize).min(num_bins - 1);

        let mut weighted_sum = 0.0f32;
        let mut total_power = 0.0f32;

        for k in min_bin..=max_bin {
            let p = power_spectrum[k];
            let f = k as f32 * bin_width;
            weighted_sum += f * p;
            total_power += p;
        }

        if total_power < 1e-6 {
            return self.smoothed_alpha;
        }

        let centroid = weighted_sum / total_power;

        // Scaling relation between formant frequency shift and VTLN alpha
        // Longer vocal tract (male) -> lower centroid -> alpha < 1.0
        // Shorter vocal tract (female/child) -> higher centroid -> alpha > 1.0
        let ratio = centroid / self.canonical_centroid_hz;
        let instant_alpha = ratio.powf(0.65).clamp(0.78, 1.28);

        self.smoothed_alpha = self.smoothed_alpha * (1.0 - self.smoothing_factor)
            + instant_alpha * self.smoothing_factor;

        self.smoothed_alpha
    }

    /// Grid-search optimal warping factor $\alpha^* \in [\alpha_{\min}, \alpha_{\max}]$ minimizing
    /// distance to an enrolled template:
    /// $$\alpha^* = \arg\min_\alpha d(\mathbf{T}, \mathbf{X}_\alpha)$$
    pub fn grid_search_optimal_alpha<F>(
        candidate_alphas: &[f32],
        mut evaluate_distance_fn: F,
    ) -> (f32, f32)
    where
        F: FnMut(f32) -> f32,
    {
        assert!(!candidate_alphas.is_empty(), "Candidate alphas must not be empty");

        let mut best_alpha = candidate_alphas[0];
        let mut min_dist = f32::INFINITY;

        for &alpha in candidate_alphas {
            let dist = evaluate_distance_fn(alpha);
            if dist < min_dist {
                min_dist = dist;
                best_alpha = alpha;
            }
        }

        (best_alpha, min_dist)
    }

    /// Current smoothed vocal tract length normalization factor.
    pub fn current_alpha(&self) -> f32 {
        self.smoothed_alpha
    }

    /// Reset internal smoothed state to canonical 1.0.
    pub fn reset(&mut self) {
        self.smoothed_alpha = 1.0;
    }
}
