//! Short-Time Fourier Transform (STFT) and Power Spectral Density estimation.

use std::f32::consts::PI;

/// Real-to-complex in-place Fast Fourier Transform (Radix-2 Cooley-Tukey).
#[derive(Debug, Clone)]
pub struct FftProcessor {
    size: usize,
}

impl FftProcessor {
    /// Create a new FFT processor for power-of-2 size.
    pub fn new(size: usize) -> Self {
        assert!(
            size > 0 && (size & (size - 1)) == 0,
            "FFT size must be a power of two"
        );
        Self { size }
    }

    /// Compute in-place Radix-2 Cooley-Tukey complex FFT.
    pub fn fft_in_place(&self, real: &mut [f32], imag: &mut [f32]) {
        let n = self.size;
        assert_eq!(real.len(), n, "Real slice must match FFT size");
        assert_eq!(imag.len(), n, "Imaginary slice must match FFT size");

        // Bit-reversal permutation
        let mut j = 0;
        for i in 0..(n - 1) {
            if i < j {
                real.swap(i, j);
                imag.swap(i, j);
            }
            let mut k = n >> 1;
            while k <= j {
                j -= k;
                k >>= 1;
            }
            j += k;
        }

        // Cooley-Tukey Radix-2 butterfly computation
        let mut len = 2;
        while len <= n {
            let angle = -2.0 * PI / (len as f32);
            let wlen_r = angle.cos();
            let wlen_i = angle.sin();

            let mut i = 0;
            while i < n {
                let mut w_r = 1.0;
                let mut w_i = 0.0;
                for k in 0..(len / 2) {
                    let u_r = real[i + k];
                    let u_i = imag[i + k];
                    let v_r = real[i + k + len / 2] * w_r - imag[i + k + len / 2] * w_i;
                    let v_i = real[i + k + len / 2] * w_i + imag[i + k + len / 2] * w_r;

                    real[i + k] = u_r + v_r;
                    imag[i + k] = u_i + v_i;
                    real[i + k + len / 2] = u_r - v_r;
                    imag[i + k + len / 2] = u_i - v_i;

                    let next_w_r = w_r * wlen_r - w_i * wlen_i;
                    let next_w_i = w_r * wlen_i + w_i * wlen_r;
                    w_r = next_w_r;
                    w_i = next_w_i;
                }
                i += len;
            }
            len <<= 1;
        }
    }

    /// Compute power spectral density for a real input signal frame.
    /// Returns a vector of length `size / 2 + 1` with magnitude squared.
    pub fn power_spectrum(&self, input: &[f32]) -> Vec<f32> {
        let n = self.size;
        assert_eq!(input.len(), n, "Input length must match FFT size");

        let mut real = input.to_vec();
        let mut imag = vec![0.0; n];

        self.fft_in_place(&mut real, &mut imag);

        // Compute one-sided power spectrum
        let num_bins = n / 2 + 1;
        let mut power = Vec::with_capacity(num_bins);
        let norm = 1.0 / (n as f32);
        for k in 0..num_bins {
            let p = (real[k] * real[k] + imag[k] * imag[k]) * norm;
            power.push(p);
        }
        power
    }

    /// Return FFT size.
    pub fn size(&self) -> usize {
        self.size
    }
}
