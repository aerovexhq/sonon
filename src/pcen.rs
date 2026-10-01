//! Per-Channel Energy Normalization (PCEN) for dynamic range compression and noise AGC.
//!
//! Replaces static logarithmic energy with adaptive feed-forward automatic gain control,
//! providing robust noise suppression against drone motor RPM shifts and variable speaker proximity.

/// Configuration parameters for Per-Channel Energy Normalization.
#[derive(Debug, Clone, PartialEq)]
pub struct PcenConfig {
    /// Recursive smoothing coefficient `s` (typically 0.025, corresponding to ~400ms time constant).
    pub smoothing: f32,
    /// Gain normalization exponent `alpha` (typically 0.6 to 0.98).
    pub alpha: f32,
    /// Dynamic range offset / bias `delta` (typically 2.0).
    pub delta: f32,
    /// Power compression exponent `r` (typically 0.25 to 0.33).
    pub r: f32,
    /// Numerical stabilizer `eps` to prevent division by zero (typically 1e-6).
    pub eps: f32,
}

impl Default for PcenConfig {
    fn default() -> Self {
        Self {
            smoothing: 0.025,
            alpha: 0.98,
            delta: 2.0,
            r: 0.25,
            eps: 1e-6,
        }
    }
}

/// State-tracking filter computing Per-Channel Energy Normalization across streaming audio frames.
#[derive(Debug, Clone)]
pub struct PcenFilter {
    config: PcenConfig,
    num_channels: usize,
    smoother_state: Vec<f32>,
    initialized: bool,
}

impl PcenFilter {
    /// Create a new PCEN filter for the given number of frequency channels.
    pub fn new(num_channels: usize, config: PcenConfig) -> Self {
        assert!(num_channels > 0, "Number of channels must be positive");
        assert!(
            config.smoothing > 0.0 && config.smoothing <= 1.0,
            "Smoothing factor must be in (0, 1]"
        );
        assert!(config.r > 0.0, "Compression exponent must be positive");

        Self {
            config,
            num_channels,
            smoother_state: vec![0.0; num_channels],
            initialized: false,
        }
    }

    /// Process a single frame of linear filterbank energies in-place or returning normalized energies.
    pub fn process_frame(&mut self, energies: &[f32]) -> Vec<f32> {
        assert_eq!(
            energies.len(),
            self.num_channels,
            "Frame length must match number of PCEN channels"
        );

        let mut output = Vec::with_capacity(self.num_channels);
        let s = self.config.smoothing;
        let alpha = self.config.alpha;
        let delta = self.config.delta;
        let r = self.config.r;
        let eps = self.config.eps;
        let delta_r = delta.powf(r);

        for (i, &energy) in energies.iter().enumerate() {
            let energy_clamped = energy.max(0.0);

            // Update recursive low-pass noise smoother
            if !self.initialized {
                self.smoother_state[i] = energy_clamped;
            } else {
                self.smoother_state[i] = (1.0 - s) * self.smoother_state[i] + s * energy_clamped;
            }

            // Compute normalized ratio: E = ( M / (eps + M_smooth)^alpha + delta )^r - delta^r
            let denominator = (eps + self.smoother_state[i]).powf(alpha);
            let normalized = energy_clamped / denominator;
            let pcen_val = (normalized + delta).powf(r) - delta_r;

            output.push(pcen_val.max(0.0));
        }

        self.initialized = true;
        output
    }

    /// Reset internal recursive smoother state.
    pub fn reset(&mut self) {
        self.smoother_state.fill(0.0);
        self.initialized = false;
    }

    /// Return current config.
    pub fn config(&self) -> &PcenConfig {
        &self.config
    }

    /// Return number of channels.
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }
}
