//! Universal Neural Vocoder & Waveform Synthesizer (BigVGAN-v2 + Anti-Aliased SnakeBeta)
//! implemented in pure safe Rust for modules/sonon.
//!
//! Provides:
//! 1. SnakeBeta Periodic Non-Linearity: Decoupled pitch excitation and formant resonance.
//! 2. Physics-Informed Anti-Aliasing Low-Pass Filter: Kaiser-windowed zero-phase FIR filter.
//! 3. Anti-Aliased Multi-Period ResBlock (AntiAliasedAmpBlock): Dilated 1D convolutions with Anti-Aliased SnakeBeta.
//! 4. Universal Waveform Generator (BigVganVocoder, VocoderConfig): Multi-stage upsampling vocoder.
//! 5. Discriminators and Loss Suite: MultiPeriodDiscriminator, MultiResolutionStftDiscriminator, VocoderLossEvaluator.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

use crate::mel::MelFilterbank;
use crate::stft::FftProcessor;

// ============================================================================
// 1. Math Utilities and Special Functions
// ============================================================================

/// Compute zeroth-order modified Bessel function of the first kind I_0(x)
/// using a 25-term Taylor series expansion.
///
/// I_0(x) = sum_{k=0}^{25} ((x^2 / 4)^k) / (k!)^2.
pub fn bessel_i0(x: f32) -> f32 {
    let mut sum = 1.0f32;
    let mut term = 1.0f32;
    let x2_4 = (x * x) * 0.25;
    for k in 1..=25 {
        let k_f = k as f32;
        term *= x2_4 / (k_f * k_f);
        sum += term;
        if term < 1e-12 * sum {
            break;
        }
    }
    sum
}

/// Compute normalized sinc function: sinc(x) = sin(PI * x) / (PI * x), with sinc(0) = 1.0.
#[inline]
pub fn normalized_sinc(x: f32) -> f32 {
    if x.abs() < 1e-7 {
        1.0
    } else {
        let px = PI * x;
        px.sin() / px
    }
}

/// Generate Kaiser window of given length N and shape parameter beta.
///
/// w(n) = I_0(beta * sqrt(1 - ((2n)/(N-1) - 1)^2)) / I_0(beta).
pub fn kaiser_window(size: usize, beta: f32) -> Vec<f32> {
    if size == 0 {
        return Vec::new();
    }
    if size == 1 {
        return vec![1.0];
    }
    let i0_beta = bessel_i0(beta);
    let n_minus_1 = (size - 1) as f32;
    let mut window = Vec::with_capacity(size);
    for n in 0..size {
        let u = (2.0 * n as f32) / n_minus_1 - 1.0;
        let arg = (1.0 - u * u).max(0.0).sqrt();
        let val = bessel_i0(beta * arg) / i0_beta;
        window.push(val);
    }
    window
}

/// Helper function to reflect indices symmetrically across boundaries [0, len - 1].
#[inline]
pub fn reflect_index(idx: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let mut i = idx;
    let max = (len - 1) as isize;
    while i < 0 || i > max {
        if i < 0 {
            i = -i;
        } else if i > max {
            i = 2 * max - i;
        }
    }
    i as usize
}

// ============================================================================
// 2. SnakeBeta Periodic Non-Linearity
// ============================================================================

/// Scalar SnakeBeta periodic non-linearity activation:
///
/// f_{alpha, beta}(x) = x + (1 / (beta + 1e-6)) * sin^2(alpha * (beta + 1e-6) * x).
#[inline]
pub fn snake_beta_scalar(x: f32, alpha: f32, beta: f32) -> f32 {
    let b = beta + 1e-6;
    let s = (alpha * b * x).sin();
    x + (1.0 / b) * s * s
}

/// SnakeBeta Periodic Non-Linearity activation layer.
///
/// Decouples pitch excitation frequency modulation (alpha) from amplitude formant
/// resonance (beta), generating rich infinite harmonic series without numerical collapse.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnakeBeta {
    pub channels: usize,
    pub alpha: Vec<f32>,
    pub beta: Vec<f32>,
}

impl SnakeBeta {
    /// Construct a new SnakeBeta activation with unit parameters.
    pub fn new(channels: usize) -> Self {
        Self {
            channels,
            alpha: vec![1.0; channels],
            beta: vec![1.0; channels],
        }
    }

    /// Construct a new SnakeBeta activation with explicit parameter vectors.
    pub fn new_with_params(alpha: Vec<f32>, beta: Vec<f32>) -> Self {
        assert_eq!(
            alpha.len(),
            beta.len(),
            "Alpha and beta parameter vectors must have identical lengths"
        );
        let channels = alpha.len();
        Self {
            channels,
            alpha,
            beta,
        }
    }

    /// Construct a new SnakeBeta activation with deterministic initialization.
    pub fn new_deterministic(channels: usize, seed_salt: usize) -> Self {
        let mut alpha = Vec::with_capacity(channels);
        let mut beta = Vec::with_capacity(channels);
        for i in 0..channels {
            let angle = (i + seed_salt * 31337 + 1) as f32 * 0.137;
            alpha.push(0.75 + 0.5 * angle.sin().abs());
            beta.push(0.75 + 0.5 * (angle * 1.618).cos().abs());
        }
        Self {
            channels,
            alpha,
            beta,
        }
    }

    /// In-place forward pass on channel vectors of shape [channels][time].
    pub fn forward_channels(&self, channels: &mut [Vec<f32>]) {
        assert_eq!(
            channels.len(),
            self.channels,
            "Channel dimension mismatch in SnakeBeta forward_channels"
        );
        for (c, ch_vec) in channels.iter_mut().enumerate() {
            let a = self.alpha[c];
            let b = self.beta[c];
            for val in ch_vec.iter_mut() {
                *val = snake_beta_scalar(*val, a, b);
            }
        }
    }

    /// Forward pass accepting either [channels][time] or [time][channels].
    pub fn forward(&self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if x.is_empty() {
            return Vec::new();
        }
        if x.len() == self.channels {
            // Layout: [channels][time]
            let mut out = x.to_vec();
            self.forward_channels(&mut out);
            out
        } else if x[0].len() == self.channels {
            // Layout: [time][channels]
            let mut out = Vec::with_capacity(x.len());
            for frame in x {
                let mut new_frame = Vec::with_capacity(self.channels);
                for c in 0..self.channels {
                    new_frame.push(snake_beta_scalar(frame[c], self.alpha[c], self.beta[c]));
                }
                out.push(new_frame);
            }
            out
        } else {
            panic!(
                "Input shape [{}, {}] does not match SnakeBeta channels ({})",
                x.len(),
                x[0].len(),
                self.channels
            );
        }
    }
}

// ============================================================================
// 3. Physics-Informed Anti-Aliasing Low-Pass Filter
// ============================================================================

/// Physics-Informed Anti-Aliasing Low-Pass FIR Filter.
///
/// Uses Kaiser windowed sinc kernel normalized to unity gain, and performs
/// zero-phase depthwise 1D FIR convolution with symmetric edge padding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KaiserLowPassFilter {
    pub kernel_size: usize,
    pub cutoff_freq: f32,
    pub beta: f32,
    pub kernel: Vec<f32>,
}

impl KaiserLowPassFilter {
    /// Construct a new Kaiser low-pass filter with specified kernel size, cutoff, and beta.
    pub fn new(kernel_size: usize, cutoff_freq: f32, beta: f32) -> Self {
        let size = if kernel_size % 2 == 0 {
            kernel_size + 1
        } else {
            kernel_size.max(3)
        };
        let w = kaiser_window(size, beta);
        let center = (size - 1) as f32 * 0.5;
        let mut kernel = Vec::with_capacity(size);
        for n in 0..size {
            let d = n as f32 - center;
            let sinc_val = 2.0 * cutoff_freq * normalized_sinc(2.0 * cutoff_freq * d);
            kernel.push(sinc_val * w[n]);
        }
        let sum: f32 = kernel.iter().sum();
        if sum.abs() > 1e-8 {
            for v in kernel.iter_mut() {
                *v /= sum;
            }
        }
        Self {
            kernel_size: size,
            cutoff_freq,
            beta,
            kernel,
        }
    }

    /// Filter a single 1D signal slice using zero-phase symmetric edge padding.
    pub fn filter_1d(&self, signal: &[f32]) -> Vec<f32> {
        let len = signal.len();
        if len == 0 {
            return Vec::new();
        }
        let k_size = self.kernel.len();
        let pad = (k_size - 1) / 2;
        let mut output = Vec::with_capacity(len);
        for t in 0..len {
            let mut sum = 0.0f32;
            for k in 0..k_size {
                let idx = reflect_index(t as isize + k as isize - pad as isize, len);
                sum += self.kernel[k] * signal[idx];
            }
            output.push(sum);
        }
        output
    }

    /// Filter depthwise across channel vectors [channels][time].
    pub fn filter_channels(&self, channels: &[Vec<f32>]) -> Vec<Vec<f32>> {
        channels.iter().map(|ch| self.filter_1d(ch)).collect()
    }

    /// In-place depthwise channel filtering.
    pub fn filter_channels_in_place(&self, channels: &mut [Vec<f32>]) {
        for ch in channels.iter_mut() {
            *ch = self.filter_1d(ch);
        }
    }
}

// ============================================================================
// 4. 1D Convolutions
// ============================================================================

/// 1D Dilated Convolution layer with learnable/configurable weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conv1d {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub dilation: usize,
    pub padding: usize,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
}

impl Conv1d {
    /// Construct a new Conv1d with explicit weights and biases.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        padding: usize,
        weights: Vec<f32>,
        bias: Vec<f32>,
    ) -> Self {
        assert_eq!(
            weights.len(),
            out_channels * in_channels * kernel_size,
            "Conv1d weight vector length mismatch"
        );
        assert_eq!(bias.len(), out_channels, "Conv1d bias vector length mismatch");
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation,
            padding,
            weights,
            bias,
        }
    }

    /// Construct a new Conv1d with deterministic pseudo-random initialization.
    pub fn new_deterministic(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        padding: usize,
        seed_salt: usize,
    ) -> Self {
        let num_weights = out_channels * in_channels * kernel_size;
        let scale = (2.0 / (in_channels * kernel_size + out_channels) as f32).sqrt();
        let mut weights = Vec::with_capacity(num_weights);
        for i in 0..num_weights {
            let angle = (i + seed_salt * 54321 + 1) as f32 * 0.1987;
            weights.push(angle.sin() * (angle * 1.324).cos() * scale);
        }
        let bias = vec![0.0; out_channels];
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation,
            padding,
            weights,
            bias,
        }
    }

    /// Forward pass on tensor shaped [in_channels][in_len].
    /// Returns tensor shaped [out_channels][out_len].
    pub fn forward(&self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if x.is_empty() || x[0].is_empty() {
            return vec![Vec::new(); self.out_channels];
        }
        assert_eq!(x.len(), self.in_channels, "Conv1d input channel mismatch");
        let in_len = x[0].len();
        let eff_k = (self.kernel_size - 1) * self.dilation + 1;
        let total_padded = in_len + 2 * self.padding;
        if total_padded < eff_k {
            return vec![Vec::new(); self.out_channels];
        }
        let out_len = (total_padded - eff_k) / self.stride + 1;
        let mut output = vec![vec![0.0f32; out_len]; self.out_channels];

        for out_c in 0..self.out_channels {
            let b = self.bias[out_c];
            let out_slice = &mut output[out_c];
            for o_idx in 0..out_len {
                let mut sum = b;
                let center_in = (o_idx * self.stride) as isize - self.padding as isize;
                for in_c in 0..self.in_channels {
                    let in_slice = &x[in_c];
                    let w_base = (out_c * self.in_channels + in_c) * self.kernel_size;
                    for k in 0..self.kernel_size {
                        let in_pos = center_in + (k * self.dilation) as isize;
                        if in_pos >= 0 && (in_pos as usize) < in_len {
                            sum += self.weights[w_base + k] * in_slice[in_pos as usize];
                        }
                    }
                }
                out_slice[o_idx] = sum;
            }
        }
        output
    }
}

/// Transposed 1D Convolution layer for waveform upsampling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvTranspose1d {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
}

impl ConvTranspose1d {
    /// Construct a new ConvTranspose1d with deterministic pseudo-random initialization.
    pub fn new_deterministic(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        kernel_size: usize,
        seed_salt: usize,
    ) -> Self {
        let padding = kernel_size.saturating_sub(stride) / 2;
        let num_weights = in_channels * out_channels * kernel_size;
        let scale = (2.0 / (in_channels * kernel_size + out_channels) as f32).sqrt();
        let mut weights = Vec::with_capacity(num_weights);
        for i in 0..num_weights {
            let angle = (i + seed_salt * 76543 + 3) as f32 * 0.1618;
            weights.push(angle.sin() * (angle * 2.718).cos() * scale);
        }
        let bias = vec![0.0; out_channels];
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            weights,
            bias,
        }
    }

    /// Forward pass on input [in_channels][in_len].
    /// Output length is strictly in_len * stride.
    pub fn forward(&self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if x.is_empty() || x[0].is_empty() {
            return vec![Vec::new(); self.out_channels];
        }
        assert_eq!(
            x.len(),
            self.in_channels,
            "ConvTranspose1d input channel mismatch"
        );
        let in_len = x[0].len();
        let out_len = in_len * self.stride;
        let mut output = vec![vec![0.0f32; out_len]; self.out_channels];

        for out_c in 0..self.out_channels {
            let b = self.bias[out_c];
            output[out_c].fill(b);
        }

        for in_c in 0..self.in_channels {
            let in_slice = &x[in_c];
            for out_c in 0..self.out_channels {
                let out_slice = &mut output[out_c];
                let w_base = (in_c * self.out_channels + out_c) * self.kernel_size;
                for (t, &val) in in_slice.iter().enumerate() {
                    if val.abs() < 1e-12 {
                        continue;
                    }
                    let base_out = (t * self.stride) as isize - self.padding as isize;
                    for k in 0..self.kernel_size {
                        let out_pos = base_out + k as isize;
                        if out_pos >= 0 && (out_pos as usize) < out_len {
                            out_slice[out_pos as usize] += val * self.weights[w_base + k];
                        }
                    }
                }
            }
        }
        output
    }
}

// ============================================================================
// 5. Anti-Aliased Multi-Period ResBlock (AntiAliasedAmpBlock)
// ============================================================================

/// Anti-Aliased Multi-Period ResBlock (AntiAliasedAmpBlock).
///
/// Combines SnakeBeta, Kaiser Low-Pass Filters, and dilated 1D convolutions:
/// residual + conv_2(lpf_2(SnakeBeta(conv_1(lpf_1(SnakeBeta(x)))))).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntiAliasedAmpBlock {
    pub channels: usize,
    pub kernel_size: usize,
    pub dilation: usize,
    pub snake1: SnakeBeta,
    pub lpf1: KaiserLowPassFilter,
    pub conv1: Conv1d,
    pub snake2: SnakeBeta,
    pub lpf2: KaiserLowPassFilter,
    pub conv2: Conv1d,
}

impl AntiAliasedAmpBlock {
    /// Construct a new AntiAliasedAmpBlock with specified channel size, kernel, dilation, and filter settings.
    pub fn new(
        channels: usize,
        kernel_size: usize,
        dilation: usize,
        lpf_size: usize,
        lpf_cutoff: f32,
        lpf_beta: f32,
        seed_salt: usize,
    ) -> Self {
        let snake1 = SnakeBeta::new_deterministic(channels, seed_salt + 1);
        let lpf1 = KaiserLowPassFilter::new(lpf_size, lpf_cutoff, lpf_beta);
        let pad1 = ((kernel_size - 1) * dilation) / 2;
        let conv1 = Conv1d::new_deterministic(
            channels,
            channels,
            kernel_size,
            1,
            dilation,
            pad1,
            seed_salt + 2,
        );

        let snake2 = SnakeBeta::new_deterministic(channels, seed_salt + 3);
        let lpf2 = KaiserLowPassFilter::new(lpf_size, lpf_cutoff, lpf_beta);
        let pad2 = (kernel_size - 1) / 2;
        let conv2 = Conv1d::new_deterministic(
            channels,
            channels,
            kernel_size,
            1,
            1,
            pad2,
            seed_salt + 4,
        );

        Self {
            channels,
            kernel_size,
            dilation,
            snake1,
            lpf1,
            conv1,
            snake2,
            lpf2,
            conv2,
        }
    }

    /// Forward pass on tensor shaped [channels][time].
    /// Preserves channel dimensions and sequence length with dilated receptive field.
    pub fn forward(&self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if x.is_empty() {
            return Vec::new();
        }
        let s1 = self.snake1.forward(x);
        let l1 = self.lpf1.filter_channels(&s1);
        let c1 = self.conv1.forward(&l1);
        let s2 = self.snake2.forward(&c1);
        let l2 = self.lpf2.filter_channels(&s2);
        let c2 = self.conv2.forward(&l2);

        let mut output = x.to_vec();
        for c in 0..self.channels.min(c2.len()) {
            for t in 0..output[c].len().min(c2[c].len()) {
                output[c][t] += c2[c][t];
            }
        }
        output
    }
}

/// Upsampling stage containing transposed convolution and Multi-Receptive Field (MRF) AMP blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpsampleStage {
    pub conv_transpose: ConvTranspose1d,
    pub amp_blocks: Vec<AntiAliasedAmpBlock>,
}

impl UpsampleStage {
    /// Forward pass: upsamples sequence length by stride and applies ensemble MRF fusion.
    pub fn forward(&self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let x_up = self.conv_transpose.forward(x);
        if self.amp_blocks.is_empty() {
            return x_up;
        }
        let out_c = x_up.len();
        let out_t = x_up[0].len();
        let mut fused = vec![vec![0.0f32; out_t]; out_c];
        for block in &self.amp_blocks {
            let b_out = block.forward(&x_up);
            for c in 0..out_c {
                for t in 0..out_t {
                    fused[c][t] += b_out[c][t];
                }
            }
        }
        let num_blocks = self.amp_blocks.len() as f32;
        for c in 0..out_c {
            for t in 0..out_t {
                fused[c][t] /= num_blocks;
            }
        }
        fused
    }
}

// ============================================================================
// 6. Universal Waveform Generator (BigVganVocoder, VocoderConfig)
// ============================================================================

/// Configuration parameters for Universal BigVGAN-v2 Neural Vocoder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocoderConfig {
    pub in_mel_channels: usize,
    pub initial_channels: usize,
    pub upsample_rates: Vec<usize>,
    pub upsample_kernel_sizes: Vec<usize>,
    pub resblock_kernel_sizes: Vec<usize>,
    pub resblock_dilations: Vec<Vec<usize>>,
    pub lpf_filter_size: usize,
    pub lpf_cutoff: f32,
    pub lpf_beta: f32,
    pub seed: u64,
}

impl Default for VocoderConfig {
    fn default() -> Self {
        Self {
            in_mel_channels: 80,
            initial_channels: 128,
            upsample_rates: vec![8, 4, 4, 2],
            upsample_kernel_sizes: vec![16, 8, 8, 4],
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilations: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            lpf_filter_size: 15,
            lpf_cutoff: 0.45,
            lpf_beta: 6.0,
            seed: 42,
        }
    }
}

impl VocoderConfig {
    /// Compute total upsampling hop size (product of upsample rates).
    pub fn total_hop_size(&self) -> usize {
        self.upsample_rates.iter().product()
    }

    /// Fast configuration for unit tests and rapid embedded testing.
    pub fn fast_test_config() -> Self {
        Self {
            in_mel_channels: 80,
            initial_channels: 32,
            upsample_rates: vec![4, 2],
            upsample_kernel_sizes: vec![8, 4],
            resblock_kernel_sizes: vec![3, 5],
            resblock_dilations: vec![vec![1, 2], vec![1, 2]],
            lpf_filter_size: 7,
            lpf_cutoff: 0.45,
            lpf_beta: 6.0,
            seed: 42,
        }
    }
}

/// Universal BigVGAN-v2 Neural Vocoder & Waveform Synthesizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BigVganVocoder {
    pub config: VocoderConfig,
    pub pre_conv: Conv1d,
    pub upsample_stages: Vec<UpsampleStage>,
    pub post_snake: SnakeBeta,
    pub post_conv: Conv1d,
}

impl BigVganVocoder {
    /// Construct a new BigVGAN-v2 vocoder from configuration.
    pub fn new(config: VocoderConfig) -> Self {
        let pre_conv = Conv1d::new_deterministic(
            config.in_mel_channels,
            config.initial_channels,
            7,
            1,
            1,
            3,
            (config.seed + 10) as usize,
        );

        let mut curr_channels = config.initial_channels;
        let mut upsample_stages = Vec::with_capacity(config.upsample_rates.len());

        for (stage_idx, &rate) in config.upsample_rates.iter().enumerate() {
            let next_channels = (curr_channels / 2).max(16);
            let k_size = config
                .upsample_kernel_sizes
                .get(stage_idx)
                .copied()
                .unwrap_or(rate * 2);
            let conv_t = ConvTranspose1d::new_deterministic(
                curr_channels,
                next_channels,
                rate,
                k_size,
                (config.seed + (stage_idx as u64 + 1) * 100) as usize,
            );

            let mut blocks = Vec::new();
            for (k_idx, &k) in config.resblock_kernel_sizes.iter().enumerate() {
                let dilations = config
                    .resblock_dilations
                    .get(k_idx)
                    .cloned()
                    .unwrap_or_else(|| vec![1, 3, 5]);
                for (d_idx, &d) in dilations.iter().enumerate() {
                    let salt = (config.seed as usize)
                        + stage_idx * 1000
                        + k_idx * 100
                        + d_idx * 10;
                    let block = AntiAliasedAmpBlock::new(
                        next_channels,
                        k,
                        d,
                        config.lpf_filter_size,
                        config.lpf_cutoff,
                        config.lpf_beta,
                        salt,
                    );
                    blocks.push(block);
                }
            }

            upsample_stages.push(UpsampleStage {
                conv_transpose: conv_t,
                amp_blocks: blocks,
            });
            curr_channels = next_channels;
        }

        let post_snake =
            SnakeBeta::new_deterministic(curr_channels, (config.seed + 9999) as usize);
        let post_conv = Conv1d::new_deterministic(
            curr_channels,
            1,
            7,
            1,
            1,
            3,
            (config.seed + 10000) as usize,
        );

        Self {
            config,
            pre_conv,
            upsample_stages,
            post_snake,
            post_conv,
        }
    }

    /// Access reference to vocoder configuration.
    pub fn config(&self) -> &VocoderConfig {
        &self.config
    }

    /// Synthesize continuous audio PCM waveform from acoustic mel spectrogram frames.
    ///
    /// Accepts mel frames formatted as [time][channels] or [channels][time].
    /// Output audio samples are guaranteed strictly bounded in [-1.0, 1.0].
    pub fn synthesize_waveform(&self, mel_frames: &[Vec<f32>]) -> Vec<f32> {
        if mel_frames.is_empty() {
            return Vec::new();
        }

        let mel_tensor = if mel_frames.len() == self.config.in_mel_channels {
            mel_frames.to_vec()
        } else {
            let num_frames = mel_frames.len();
            let mut tensor = vec![vec![0.0f32; num_frames]; self.config.in_mel_channels];
            for t in 0..num_frames {
                let frame = &mel_frames[t];
                for c in 0..self.config.in_mel_channels.min(frame.len()) {
                    tensor[c][t] = frame[c];
                }
            }
            tensor
        };

        let mut x = self.pre_conv.forward(&mel_tensor);
        for stage in &self.upsample_stages {
            x = stage.forward(&x);
        }

        let s = self.post_snake.forward(&x);
        let wav_channels = self.post_conv.forward(&s);

        if wav_channels.is_empty() || wav_channels[0].is_empty() {
            return Vec::new();
        }

        wav_channels[0]
            .iter()
            .map(|&val| val.tanh().clamp(-1.0, 1.0))
            .collect()
    }
}

// ============================================================================
// 7. Discriminators & Loss Suite
// ============================================================================

/// Output of a single Period Discriminator forward pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeriodDiscriminatorOutput {
    pub period: usize,
    pub score: f32,
    pub logits: Vec<f32>,
    pub features: Vec<Vec<f32>>,
}

/// 2D Periodic Convolutional Discriminator for a single prime period p.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeriodDiscriminator {
    pub period: usize,
    pub layer_weights: Vec<Vec<f32>>,
    pub layer_biases: Vec<Vec<f32>>,
}

impl PeriodDiscriminator {
    /// Construct a new PeriodDiscriminator for prime period p.
    pub fn new(period: usize, seed_salt: usize) -> Self {
        // 3 hidden layers + 1 final projection layer
        let mut layer_weights = Vec::new();
        let mut layer_biases = Vec::new();

        let channel_configs = [(1, 16, 5), (16, 32, 5), (32, 64, 3), (64, 1, 3)];
        for (idx, &(in_c, out_c, k_h)) in channel_configs.iter().enumerate() {
            let num_w = in_c * out_c * k_h;
            let scale = (2.0 / (in_c * k_h + out_c) as f32).sqrt();
            let mut weights = Vec::with_capacity(num_w);
            for w_i in 0..num_w {
                let angle = (w_i + (seed_salt + idx * 7) * 411 + 1) as f32 * 0.231;
                weights.push(angle.sin() * (angle * 1.414).cos() * scale);
            }
            let biases = vec![0.0; out_c];
            layer_weights.push(weights);
            layer_biases.push(biases);
        }

        Self {
            period,
            layer_weights,
            layer_biases,
        }
    }

    /// Forward pass on 1D waveform: extracts 2D periodic grid (T/p, p) and applies periodic convolutions.
    pub fn forward(&self, waveform: &[f32]) -> PeriodDiscriminatorOutput {
        if waveform.is_empty() {
            return PeriodDiscriminatorOutput {
                period: self.period,
                score: 0.0,
                logits: Vec::new(),
                features: Vec::new(),
            };
        }

        let p = self.period;
        let pad_len = (p - (waveform.len() % p)) % p;
        let mut padded = waveform.to_vec();
        padded.resize(waveform.len() + pad_len, 0.0);
        let h = padded.len() / p;

        // Current grid: [channels][h][p]
        let mut curr_channels = vec![vec![vec![0.0f32; p]; h]; 1];
        for h_i in 0..h {
            for w_i in 0..p {
                curr_channels[0][h_i][w_i] = padded[h_i * p + w_i];
            }
        }

        let channel_configs = [
            (1, 16, 5, 2),  // in_c, out_c, k_h, stride_h
            (16, 32, 5, 2),
            (32, 64, 3, 1),
            (64, 1, 3, 1),
        ];

        let mut features = Vec::new();

        for (l_idx, &(in_c, out_c, k_h, stride_h)) in channel_configs.iter().enumerate() {
            let curr_h = curr_channels[0].len();
            let pad_h = (k_h - 1) / 2;
            let out_h = if curr_h + 2 * pad_h >= k_h {
                (curr_h + 2 * pad_h - k_h) / stride_h + 1
            } else {
                1
            };

            let mut next_channels = vec![vec![vec![0.0f32; p]; out_h]; out_c];
            let weights = &self.layer_weights[l_idx];
            let biases = &self.layer_biases[l_idx];

            for o_c in 0..out_c {
                let b = biases[o_c];
                for oh in 0..out_h {
                    let base_in_h = (oh * stride_h) as isize - pad_h as isize;
                    for w_i in 0..p {
                        let mut sum = b;
                        for ic in 0..in_c {
                            let in_grid = &curr_channels[ic];
                            let w_base = (o_c * in_c + ic) * k_h;
                            for kh_idx in 0..k_h {
                                let ih = base_in_h + kh_idx as isize;
                                if ih >= 0 && (ih as usize) < curr_h {
                                    sum += weights[w_base + kh_idx] * in_grid[ih as usize][w_i];
                                }
                            }
                        }
                        // LeakyReLU activation for hidden layers, linear for final
                        if l_idx < channel_configs.len() - 1 {
                            next_channels[o_c][oh][w_i] = if sum > 0.0 { sum } else { 0.1 * sum };
                        } else {
                            next_channels[o_c][oh][w_i] = sum;
                        }
                    }
                }
            }

            // Record flattened feature representation
            let mut flat_feat = Vec::with_capacity(out_c * out_h * p);
            for oc_grid in &next_channels {
                for row in oc_grid {
                    flat_feat.extend_from_slice(row);
                }
            }
            features.push(flat_feat);
            curr_channels = next_channels;
        }

        // Final score logits
        let mut logits = Vec::new();
        for row in &curr_channels[0] {
            logits.extend_from_slice(row);
        }
        let score = if logits.is_empty() {
            0.0
        } else {
            logits.iter().sum::<f32>() / (logits.len() as f32)
        };

        PeriodDiscriminatorOutput {
            period: self.period,
            score,
            logits,
            features,
        }
    }
}

/// Output of Multi-Period Discriminator ensemble.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiPeriodOutput {
    pub period_scores: Vec<f32>,
    pub period_logits: Vec<Vec<f32>>,
    pub features: Vec<Vec<Vec<f32>>>,
    pub mean_score: f32,
}

/// Multi-Period Discriminator (MPD) evaluating prime periods p in {2, 3, 5, 7, 11}.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiPeriodDiscriminator {
    pub discriminators: Vec<PeriodDiscriminator>,
}

impl Default for MultiPeriodDiscriminator {
    fn default() -> Self {
        Self::new(&[2, 3, 5, 7, 11])
    }
}

impl MultiPeriodDiscriminator {
    /// Construct a new MultiPeriodDiscriminator with specified period list.
    pub fn new(periods: &[usize]) -> Self {
        let discriminators = periods
            .iter()
            .enumerate()
            .map(|(idx, &p)| PeriodDiscriminator::new(p, idx * 100 + 42))
            .collect();
        Self { discriminators }
    }

    /// Forward pass evaluating waveform across all prime periods.
    pub fn forward(&self, waveform: &[f32]) -> MultiPeriodOutput {
        let mut period_scores = Vec::with_capacity(self.discriminators.len());
        let mut period_logits = Vec::with_capacity(self.discriminators.len());
        let mut features = Vec::with_capacity(self.discriminators.len());

        for disc in &self.discriminators {
            let out = disc.forward(waveform);
            period_scores.push(out.score);
            period_logits.push(out.logits);
            features.push(out.features);
        }

        let mean_score = if period_scores.is_empty() {
            0.0
        } else {
            period_scores.iter().sum::<f32>() / (period_scores.len() as f32)
        };

        MultiPeriodOutput {
            period_scores,
            period_logits,
            features,
            mean_score,
        }
    }
}

/// Output of Multi-Resolution STFT Discriminator forward pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiResolutionStftOutput {
    pub resolution_scores: Vec<f32>,
    pub resolution_logits: Vec<Vec<f32>>,
    pub features: Vec<Vec<Vec<f32>>>,
    pub phase_coherences: Vec<f32>,
    pub mean_score: f32,
}

/// Multi-Resolution STFT Discriminator (MRSD) evaluating spectral magnitude and phase coherence.
#[derive(Debug, Clone)]
pub struct MultiResolutionStftDiscriminator {
    pub resolutions: Vec<(usize, usize)>, // (fft_size, hop_size)
    pub processors: Vec<FftProcessor>,
}

impl Default for MultiResolutionStftDiscriminator {
    fn default() -> Self {
        Self::new(&[(512, 64), (1024, 128), (2048, 256)])
    }
}

impl MultiResolutionStftDiscriminator {
    /// Construct a new MultiResolutionStftDiscriminator with custom resolutions.
    pub fn new(resolutions: &[(usize, usize)]) -> Self {
        let processors = resolutions
            .iter()
            .map(|&(fft_size, _)| FftProcessor::new(fft_size))
            .collect();
        Self {
            resolutions: resolutions.to_vec(),
            processors,
        }
    }

    /// Forward pass evaluating STFT magnitude and phase coherence across resolutions.
    pub fn forward(&self, waveform: &[f32]) -> MultiResolutionStftOutput {
        let mut resolution_scores = Vec::with_capacity(self.resolutions.len());
        let mut resolution_logits = Vec::with_capacity(self.resolutions.len());
        let mut features = Vec::with_capacity(self.resolutions.len());
        let mut phase_coherences = Vec::with_capacity(self.resolutions.len());

        for (idx, &(fft_size, hop_size)) in self.resolutions.iter().enumerate() {
            let fft = &self.processors[idx];
            if waveform.len() < fft_size {
                resolution_scores.push(0.0);
                resolution_logits.push(Vec::new());
                features.push(Vec::new());
                phase_coherences.push(1.0);
                continue;
            }

            let num_frames = (waveform.len() - fft_size) / hop_size + 1;
            let num_bins = fft_size / 2 + 1;
            let mut magnitudes = vec![vec![0.0f32; num_bins]; num_frames];
            let mut phase_angles = vec![vec![0.0f32; num_bins]; num_frames];

            // Hann window
            let mut window = Vec::with_capacity(fft_size);
            for n in 0..fft_size {
                let w = 0.5 - 0.5 * (2.0 * PI * n as f32 / (fft_size - 1) as f32).cos();
                window.push(w);
            }

            for t in 0..num_frames {
                let offset = t * hop_size;
                let mut real = vec![0.0f32; fft_size];
                let mut imag = vec![0.0f32; fft_size];
                for n in 0..fft_size {
                    real[n] = waveform[offset + n] * window[n];
                }
                fft.fft_in_place(&mut real, &mut imag);

                for k in 0..num_bins {
                    let r = real[k];
                    let i = imag[k];
                    magnitudes[t][k] = (r * r + i * i).sqrt();
                    phase_angles[t][k] = i.atan2(r);
                }
            }

            // Phase coherence metric: average phase derivative stability across time
            let mut coherence_sum = 0.0f32;
            let mut coherence_count = 0usize;
            if num_frames > 1 {
                for t in 1..num_frames {
                    for k in 1..num_bins {
                        let d_phase = (phase_angles[t][k] - phase_angles[t - 1][k]).cos();
                        coherence_sum += d_phase;
                        coherence_count += 1;
                    }
                }
            }
            let phase_coherence = if coherence_count > 0 {
                (coherence_sum / (coherence_count as f32)).clamp(-1.0, 1.0)
            } else {
                1.0
            };
            phase_coherences.push(phase_coherence);

            // Discriminator scoring over spectral magnitudes
            let mut logits = Vec::with_capacity(num_frames);
            let mut layer_features = Vec::new();
            let mut flat_spec = Vec::new();

            for t in 0..num_frames {
                let frame_mag = &magnitudes[t];
                let mean_log_mag = frame_mag.iter().map(|&m| (m + 1e-5).ln()).sum::<f32>()
                    / (num_bins as f32);
                logits.push(mean_log_mag.tanh());
                flat_spec.extend_from_slice(frame_mag);
            }
            layer_features.push(flat_spec);

            let score = if logits.is_empty() {
                0.0
            } else {
                logits.iter().sum::<f32>() / (logits.len() as f32)
            };

            resolution_scores.push(score);
            resolution_logits.push(logits);
            features.push(layer_features);
        }

        let mean_score = if resolution_scores.is_empty() {
            0.0
        } else {
            resolution_scores.iter().sum::<f32>() / (resolution_scores.len() as f32)
        };

        MultiResolutionStftOutput {
            resolution_scores,
            resolution_logits,
            features,
            phase_coherences,
            mean_score,
        }
    }
}

/// Comprehensive loss report returned by VocoderLossEvaluator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VocoderLossReport {
    pub generator_adversarial_loss: f32,
    pub discriminator_adversarial_loss: f32,
    pub feature_matching_loss: f32,
    pub mel_spectral_loss: f32,
    pub total_generator_loss: f32,
    pub total_discriminator_loss: f32,
}

/// Vocoder Loss Evaluator computing LSGAN adversarial score, feature matching distance,
/// and Mel spectral reconstruction loss.
#[derive(Debug, Clone)]
pub struct VocoderLossEvaluator {
    pub mpd: MultiPeriodDiscriminator,
    pub mrsd: MultiResolutionStftDiscriminator,
    pub mel_filterbank: MelFilterbank,
    pub fft: FftProcessor,
    pub lambda_fm: f32,
    pub lambda_mel: f32,
}

impl Default for VocoderLossEvaluator {
    fn default() -> Self {
        Self::new(80, 1024, 16000.0, 0.0, 8000.0)
    }
}

impl VocoderLossEvaluator {
    /// Construct a new VocoderLossEvaluator with specified Mel filterbank parameters.
    pub fn new(
        num_mel_bins: usize,
        fft_size: usize,
        sample_rate: f32,
        low_freq: f32,
        high_freq: f32,
    ) -> Self {
        let mpd = MultiPeriodDiscriminator::default();
        let mrsd = MultiResolutionStftDiscriminator::default();
        let mel_filterbank =
            MelFilterbank::new(num_mel_bins, fft_size, sample_rate, low_freq, high_freq);
        let fft = FftProcessor::new(fft_size);

        Self {
            mpd,
            mrsd,
            mel_filterbank,
            fft,
            lambda_fm: 2.0,
            lambda_mel: 45.0,
        }
    }

    /// Compute Mel log-energy spectrogram for a waveform slice.
    pub fn compute_mel_spectrogram(&self, waveform: &[f32], hop_size: usize) -> Vec<Vec<f32>> {
        let fft_size = self.fft.size();
        if waveform.len() < fft_size {
            return Vec::new();
        }
        let num_frames = (waveform.len() - fft_size) / hop_size + 1;
        let mut mel_frames = Vec::with_capacity(num_frames);

        for t in 0..num_frames {
            let offset = t * hop_size;
            let frame_slice = &waveform[offset..offset + fft_size];
            let power = self.fft.power_spectrum(frame_slice);
            let log_mel = self.mel_filterbank.compute_log_energies(&power);
            mel_frames.push(log_mel);
        }
        mel_frames
    }

    /// Evaluate complete loss suite between real ground-truth and synthesized audio waveforms.
    pub fn evaluate_losses(
        &self,
        real_audio: &[f32],
        fake_audio: &[f32],
        mel_hop_size: usize,
    ) -> VocoderLossReport {
        // 1. Discriminator evaluations
        let real_mpd = self.mpd.forward(real_audio);
        let fake_mpd = self.mpd.forward(fake_audio);
        let real_mrsd = self.mrsd.forward(real_audio);
        let fake_mrsd = self.mrsd.forward(fake_audio);

        // 2. LSGAN Adversarial Losses
        // Generator wants fake scores to equal 1.0
        let mut gen_adv_sum = 0.0f32;
        let mut gen_adv_count = 0usize;
        for score in fake_mpd.period_scores.iter().chain(fake_mrsd.resolution_scores.iter()) {
            let diff = score - 1.0;
            gen_adv_sum += diff * diff;
            gen_adv_count += 1;
        }
        let generator_adversarial_loss = if gen_adv_count > 0 {
            gen_adv_sum / (gen_adv_count as f32)
        } else {
            0.0
        };

        // Discriminator wants real scores to equal 1.0, and fake scores to equal 0.0
        let mut disc_adv_sum = 0.0f32;
        let mut disc_adv_count = 0usize;
        for score in real_mpd.period_scores.iter().chain(real_mrsd.resolution_scores.iter()) {
            let diff = score - 1.0;
            disc_adv_sum += diff * diff;
            disc_adv_count += 1;
        }
        for score in fake_mpd.period_scores.iter().chain(fake_mrsd.resolution_scores.iter()) {
            let diff = *score;
            disc_adv_sum += diff * diff;
            disc_adv_count += 1;
        }
        let discriminator_adversarial_loss = if disc_adv_count > 0 {
            disc_adv_sum / (disc_adv_count as f32)
        } else {
            0.0
        };

        // 3. Feature Matching Loss
        let mut fm_sum = 0.0f32;
        let mut fm_count = 0usize;
        for (r_disc, f_disc) in real_mpd.features.iter().zip(fake_mpd.features.iter()) {
            for (r_feat, f_feat) in r_disc.iter().zip(f_disc.iter()) {
                let n = r_feat.len().min(f_feat.len());
                if n > 0 {
                    let mut layer_diff = 0.0f32;
                    for i in 0..n {
                        layer_diff += (r_feat[i] - f_feat[i]).abs();
                    }
                    fm_sum += layer_diff / (n as f32);
                    fm_count += 1;
                }
            }
        }
        for (r_disc, f_disc) in real_mrsd.features.iter().zip(fake_mrsd.features.iter()) {
            for (r_feat, f_feat) in r_disc.iter().zip(f_disc.iter()) {
                let n = r_feat.len().min(f_feat.len());
                if n > 0 {
                    let mut layer_diff = 0.0f32;
                    for i in 0..n {
                        layer_diff += (r_feat[i] - f_feat[i]).abs();
                    }
                    fm_sum += layer_diff / (n as f32);
                    fm_count += 1;
                }
            }
        }
        let feature_matching_loss = if fm_count > 0 {
            fm_sum / (fm_count as f32)
        } else {
            0.0
        };

        // 4. Mel Spectral Reconstruction Loss
        let real_mel = self.compute_mel_spectrogram(real_audio, mel_hop_size);
        let fake_mel = self.compute_mel_spectrogram(fake_audio, mel_hop_size);
        let num_m_frames = real_mel.len().min(fake_mel.len());
        let mut mel_diff_sum = 0.0f32;
        let mut mel_bins_count = 0usize;

        for t in 0..num_m_frames {
            let r_f = &real_mel[t];
            let f_f = &fake_mel[t];
            let num_b = r_f.len().min(f_f.len());
            for b in 0..num_b {
                mel_diff_sum += (r_f[b] - f_f[b]).abs();
                mel_bins_count += 1;
            }
        }
        let mel_spectral_loss = if mel_bins_count > 0 {
            mel_diff_sum / (mel_bins_count as f32)
        } else {
            0.0
        };

        let total_generator_loss = generator_adversarial_loss
            + self.lambda_fm * feature_matching_loss
            + self.lambda_mel * mel_spectral_loss;

        let total_discriminator_loss = discriminator_adversarial_loss;

        VocoderLossReport {
            generator_adversarial_loss,
            discriminator_adversarial_loss,
            feature_matching_loss,
            mel_spectral_loss,
            total_generator_loss,
            total_discriminator_loss,
        }
    }
}
