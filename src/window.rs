//! Windowing functions for audio frame preprocessing.

use std::f32::consts::PI;

/// Supported windowing function types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowType {
    Rectangular,
    Hann,
    Hamming,
    Blackman,
}

/// Precomputed window coefficient buffer.
#[derive(Debug, Clone)]
pub struct Window {
    coefficients: Vec<f32>,
    window_type: WindowType,
}

impl Window {
    /// Create a precomputed window of length `size`.
    pub fn new(window_type: WindowType, size: usize) -> Self {
        assert!(size > 0, "Window size must be greater than zero");
        let mut coefficients = Vec::with_capacity(size);

        match window_type {
            WindowType::Rectangular => {
                coefficients.resize(size, 1.0);
            }
            WindowType::Hann => {
                let n = (size - 1).max(1) as f32;
                for i in 0..size {
                    let val = 0.5 * (1.0 - (2.0 * PI * (i as f32) / n).cos());
                    coefficients.push(val);
                }
            }
            WindowType::Hamming => {
                let n = (size - 1).max(1) as f32;
                for i in 0..size {
                    let val = 0.54 - 0.46 * (2.0 * PI * (i as f32) / n).cos();
                    coefficients.push(val);
                }
            }
            WindowType::Blackman => {
                let n = (size - 1).max(1) as f32;
                for i in 0..size {
                    let x = 2.0 * PI * (i as f32) / n;
                    let val = 0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos();
                    coefficients.push(val);
                }
            }
        }

        Self {
            coefficients,
            window_type,
        }
    }

    /// Apply precomputed window to an input frame in-place.
    pub fn apply(&self, frame: &mut [f32]) {
        assert_eq!(
            frame.len(),
            self.coefficients.len(),
            "Frame size must match precomputed window size"
        );
        for (sample, &coeff) in frame.iter_mut().zip(self.coefficients.iter()) {
            *sample *= coeff;
        }
    }

    /// Return precomputed coefficients slice.
    pub fn coefficients(&self) -> &[f32] {
        &self.coefficients
    }

    /// Return window length.
    pub fn len(&self) -> usize {
        self.coefficients.len()
    }

    /// Return window type.
    pub fn window_type(&self) -> WindowType {
        self.window_type
    }
}
