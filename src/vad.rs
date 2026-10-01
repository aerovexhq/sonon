//! Voice Activity Detector (VAD) with adaptive noise floor and hangover smoothing.

/// Real-time energy-based Voice Activity Detector.
#[derive(Debug, Clone)]
pub struct EnergyVad {
    noise_floor: f32,
    threshold_factor: f32,
    alpha_noise: f32,
    hangover_frames: usize,
    hangover_counter: usize,
    is_active: bool,
    calibrated: bool,
}

impl EnergyVad {
    /// Create a new VAD instance with standard robotics thresholds.
    pub fn new(threshold_factor: f32, alpha_noise: f32, hangover_frames: usize) -> Self {
        Self {
            noise_floor: 1e-4,
            threshold_factor,
            alpha_noise,
            hangover_frames,
            hangover_counter: 0,
            is_active: false,
            calibrated: false,
        }
    }

    /// Process a frame of audio samples and return whether voice activity is detected.
    pub fn process_frame(&mut self, frame: &[f32]) -> bool {
        if frame.is_empty() {
            return false;
        }

        // Compute Root Mean Square (RMS) energy
        let sum_sq: f32 = frame.iter().map(|&x| x * x).sum();
        let energy = (sum_sq / (frame.len() as f32)).sqrt();

        if !self.calibrated {
            self.noise_floor = energy.max(1e-6);
            self.calibrated = true;
            return false;
        }

        let threshold = self.noise_floor * self.threshold_factor;

        if energy > threshold {
            self.hangover_counter = self.hangover_frames;
            self.is_active = true;
        } else if self.hangover_counter > 0 {
            self.hangover_counter -= 1;
            self.is_active = true;
        } else {
            self.is_active = false;
            // Update adaptive noise floor during silence
            self.noise_floor = self.alpha_noise * self.noise_floor + (1.0 - self.alpha_noise) * energy;
            self.noise_floor = self.noise_floor.max(1e-6);
        }

        self.is_active
    }

    /// Return estimated noise floor energy.
    pub fn noise_floor(&self) -> f32 {
        self.noise_floor
    }

    /// Check if speech is currently active.
    pub fn is_speech_active(&self) -> bool {
        self.is_active
    }

    /// Reset noise floor and hangover counter.
    pub fn reset(&mut self) {
        self.noise_floor = 1e-4;
        self.hangover_counter = 0;
        self.is_active = false;
    }
}
