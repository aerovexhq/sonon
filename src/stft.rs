//! Short-Time Fourier Transform (STFT) and Power Spectral Density estimation.

use std::f32::consts::PI;

/// Real-to-complex in-place Fast Fourier Transform (Radix-2 Cooley-Tukey).
#[derive(Debug, Clone)]
pub struct FftProcessor {
    size: usize,
    bit_rev: Vec<usize>,
    twiddles: Vec<(f32, f32)>,
}

impl FftProcessor {
    /// Create a new FFT processor for power-of-2 size with precomputed twiddle factors and bit-reversals.
    pub fn new(size: usize) -> Self {
        assert!(
            size > 0 && (size & (size - 1)) == 0,
            "FFT size must be a power of two"
        );

        let bits = size.trailing_zeros();
        let mut bit_rev = Vec::with_capacity(size);
        for i in 0..size {
            let mut rev = 0;
            let mut temp = i;
            for _ in 0..bits {
                rev = (rev << 1) | (temp & 1);
                temp >>= 1;
            }
            bit_rev.push(rev);
        }

        let mut twiddles = Vec::with_capacity(size / 2);
        for k in 0..(size / 2) {
            let angle = -2.0 * PI * (k as f32) / (size as f32);
            twiddles.push((angle.cos(), angle.sin()));
        }

        Self {
            size,
            bit_rev,
            twiddles,
        }
    }

    /// Compute in-place Radix-2 Cooley-Tukey complex FFT.
    pub fn fft_in_place(&self, real: &mut [f32], imag: &mut [f32]) {
        let n = self.size;
        assert_eq!(real.len(), n, "Real slice must match FFT size");
        assert_eq!(imag.len(), n, "Imaginary slice must match FFT size");

        // Precomputed bit-reversal permutation
        for i in 0..n {
            let j = self.bit_rev[i];
            if i < j {
                real.swap(i, j);
                imag.swap(i, j);
            }
        }

        // Cooley-Tukey Radix-2 butterfly computation using precomputed twiddle factors
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let step = n / len;

            let mut i = 0;
            while i < n {
                for k in 0..half {
                    let (w_r, w_i) = self.twiddles[k * step];
                    let u_r = real[i + k];
                    let u_i = imag[i + k];
                    let v_r = real[i + k + half] * w_r - imag[i + k + half] * w_i;
                    let v_i = real[i + k + half] * w_i + imag[i + k + half] * w_r;

                    real[i + k] = u_r + v_r;
                    imag[i + k] = u_i + v_i;
                    real[i + k + half] = u_r - v_r;
                    imag[i + k + half] = u_i - v_i;
                }
                i += len;
            }
            len <<= 1;
        }
    }

    /// Compute in-place Radix-2 Cooley-Tukey complex Inverse Fast Fourier Transform (IFFT).
    pub fn ifft_in_place(&self, real: &mut [f32], imag: &mut [f32]) {
        let n = self.size;
        assert_eq!(real.len(), n, "Real slice must match FFT size");
        assert_eq!(imag.len(), n, "Imaginary slice must match FFT size");

        for im in imag.iter_mut() {
            *im = -*im;
        }

        self.fft_in_place(real, imag);

        let scale = 1.0 / (n as f32);
        for (re, im) in real.iter_mut().zip(imag.iter_mut()) {
            *re *= scale;
            *im = -*im * scale;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_ifft_roundtrip() {
        let size = 128;
        let fft = FftProcessor::new(size);
        let orig: Vec<f32> = (0..size).map(|i| (i as f32 * 0.1).sin() + 0.5 * (i as f32 * 0.35).cos()).collect();
        let mut real = orig.clone();
        let mut imag = vec![0.0f32; size];

        fft.fft_in_place(&mut real, &mut imag);
        fft.ifft_in_place(&mut real, &mut imag);

        for i in 0..size {
            assert!((real[i] - orig[i]).abs() < 1e-5, "Mismatch at index {}: {} vs {}", i, real[i], orig[i]);
            assert!(imag[i].abs() < 1e-5, "Imaginary part not zero at index {}: {}", i, imag[i]);
        }
    }
}
