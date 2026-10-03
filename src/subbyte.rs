//! Ultra-low-bitrate acoustic quantization and sub-byte weight packing for edge KWS.
//!
//! Provides non-uniform Lloyd-Max scalar quantization (1-bit, 2-bit, 4-bit) with per-dimension
//! standardization, packed bit-stream storage (4 values/byte for 2-bit, 2 values/byte for 4-bit),
//! Asymmetric Distance Computation (ADC) with precomputed query lookup tables (LUT),
//! and memory-bounded Sakoe-Chiba Dynamic Time Warping (DTW) for extreme microcontroller constraints.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};

/// Supported sub-byte quantization bit-widths for edge KWS dictionaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SubByteBitWidth {
    /// 1-bit quantization (2 reconstruction levels, 8 values per byte).
    OneBit,
    /// 2-bit non-uniform quantization (4 reconstruction levels, 4 values per byte).
    TwoBit,
    /// 4-bit non-uniform quantization (16 reconstruction levels, 2 values per byte).
    FourBit,
}

impl SubByteBitWidth {
    /// Number of reconstruction levels ($2^B$).
    #[inline]
    pub fn num_levels(self) -> usize {
        match self {
            SubByteBitWidth::OneBit => 2,
            SubByteBitWidth::TwoBit => 4,
            SubByteBitWidth::FourBit => 16,
        }
    }

    /// Bits per quantized scalar value.
    #[inline]
    pub fn bits(self) -> usize {
        match self {
            SubByteBitWidth::OneBit => 1,
            SubByteBitWidth::TwoBit => 2,
            SubByteBitWidth::FourBit => 4,
        }
    }

    /// Bit mask for extracting a single symbol.
    #[inline]
    pub fn mask(self) -> u8 {
        match self {
            SubByteBitWidth::OneBit => 0x01,
            SubByteBitWidth::TwoBit => 0x03,
            SubByteBitWidth::FourBit => 0x0F,
        }
    }

    /// Number of packed scalar values stored per byte.
    #[inline]
    pub fn values_per_byte(self) -> usize {
        8 / self.bits()
    }
}

/// Standardized scalar codebook containing normalized Lloyd-Max centroids and per-dimension affine scales.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubByteCodebook {
    /// Quantization bit-width.
    pub bit_width: SubByteBitWidth,
    /// Sorted normalized reconstruction centroids: $z_0 < z_1 < \dots < z_{K-1}$.
    pub normalized_centroids: Vec<f32>,
    /// Normalized decision boundaries: $t_1 < t_2 < \dots < t_{K-1}$.
    pub normalized_boundaries: Vec<f32>,
    /// Per-dimension feature mean: $\mu_d$.
    pub means: Vec<f32>,
    /// Per-dimension feature standard deviation scale: $\sigma_d$.
    pub scales: Vec<f32>,
}

impl SubByteCodebook {
    /// Construct a uniform normalized codebook spanning `[-2.5, 2.5]`.
    pub fn uniform(bit_width: SubByteBitWidth, dim: usize) -> Self {
        let k = bit_width.num_levels();
        let min_val = -2.5f32;
        let max_val = 2.5f32;
        let step = (max_val - min_val) / (k as f32);

        let mut centroids = Vec::with_capacity(k);
        for i in 0..k {
            centroids.push(min_val + (i as f32 + 0.5) * step);
        }

        let mut boundaries = Vec::with_capacity(k.saturating_sub(1));
        for i in 1..k {
            boundaries.push((centroids[i - 1] + centroids[i]) * 0.5);
        }

        Self {
            bit_width,
            normalized_centroids: centroids,
            normalized_boundaries: boundaries,
            means: vec![0.0f32; dim],
            scales: vec![1.0f32; dim],
        }
    }

    /// Quantize a continuous scalar value for feature dimension `dim` into a discrete codebook index in `0..K`.
    #[inline]
    pub fn quantize_scalar(&self, dim: usize, val: f32) -> u8 {
        let mean = if dim < self.means.len() { self.means[dim] } else { 0.0 };
        let scale = if dim < self.scales.len() { self.scales[dim] } else { 1.0 };
        let z = (val - mean) / scale;

        let mut idx = 0u8;
        for &bound in &self.normalized_boundaries {
            if z > bound {
                idx += 1;
            } else {
                break;
            }
        }
        idx
    }

    /// Dequantize a codebook index for feature dimension `dim` into its continuous reconstruction centroid.
    #[inline]
    pub fn dequantize_index(&self, dim: usize, idx: u8) -> f32 {
        let mean = if dim < self.means.len() { self.means[dim] } else { 0.0 };
        let scale = if dim < self.scales.len() { self.scales[dim] } else { 1.0 };
        let i = (idx as usize).min(self.normalized_centroids.len().saturating_sub(1));
        mean + scale * self.normalized_centroids[i]
    }
}

/// Lloyd-Max iterative codebook trainer for empirical speech distributions.
#[derive(Debug, Clone)]
pub struct LloydMaxTrainer {
    max_iterations: usize,
    convergence_tolerance: f32,
}

impl Default for LloydMaxTrainer {
    fn default() -> Self {
        Self {
            max_iterations: 40,
            convergence_tolerance: 1e-5,
        }
    }
}

impl LloydMaxTrainer {
    /// Construct a new trainer with specified iteration parameters.
    pub fn new(max_iterations: usize, convergence_tolerance: f32) -> Self {
        Self {
            max_iterations,
            convergence_tolerance,
        }
    }

    /// Train a non-uniform Lloyd-Max scalar codebook on a 1D slice of empirical data.
    pub fn train_1d(&self, samples: &[f32], bit_width: SubByteBitWidth) -> SubByteCodebook {
        if samples.is_empty() {
            return SubByteCodebook::uniform(bit_width, 1);
        }
        let sum: f32 = samples.iter().sum();
        let mean = sum / (samples.len() as f32);
        let var: f32 = samples.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / (samples.len() as f32);
        let std = var.sqrt().max(1e-4);
        let standardized: Vec<f32> = samples.iter().map(|&x| (x - mean) / std).collect();
        let (centroids, boundaries) = self.train_standardized_1d(&standardized, bit_width);
        SubByteCodebook {
            bit_width,
            normalized_centroids: centroids,
            normalized_boundaries: boundaries,
            means: vec![mean],
            scales: vec![std],
        }
    }

    /// Train a non-uniform Lloyd-Max scalar codebook on continuous speech feature frames.
    pub fn train_from_frames(&self, features: &[Vec<f32>], bit_width: SubByteBitWidth) -> SubByteCodebook {
        if features.is_empty() {
            return SubByteCodebook::uniform(bit_width, 13);
        }

        let dim = features[0].len();
        let num_frames = features.len();

        // 1. Calculate per-dimension mean and standard deviation
        let mut means = vec![0.0f32; dim];
        let mut scales = vec![1.0f32; dim];

        for d in 0..dim {
            let sum: f32 = features.iter().map(|f| f[d]).sum();
            let mean = sum / (num_frames as f32);
            let var: f32 = features
                .iter()
                .map(|f| (f[d] - mean) * (f[d] - mean))
                .sum::<f32>()
                / (num_frames as f32);
            let std = var.sqrt().max(1e-4);
            means[d] = mean;
            scales[d] = std;
        }

        // 2. Standardize all feature scalars: z = (x_d - mean_d) / std_d
        let mut standardized_samples = Vec::with_capacity(num_frames * dim);
        for f in features {
            for d in 0..dim {
                let z = (f[d] - means[d]) / scales[d];
                standardized_samples.push(z);
            }
        }

        // 3. Train optimal 1D Lloyd-Max centroids on standardized samples
        let (centroids, boundaries) = self.train_standardized_1d(&standardized_samples, bit_width);

        SubByteCodebook {
            bit_width,
            normalized_centroids: centroids,
            normalized_boundaries: boundaries,
            means,
            scales,
        }
    }

    /// Train 1D Lloyd-Max centroids and boundaries on pre-standardized samples.
    pub fn train_standardized_1d(&self, samples: &[f32], bit_width: SubByteBitWidth) -> (Vec<f32>, Vec<f32>) {
        let k = bit_width.num_levels();
        if samples.is_empty() {
            let cb = SubByteCodebook::uniform(bit_width, 1);
            return (cb.normalized_centroids, cb.normalized_boundaries);
        }

        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        let min_val = sorted[0];
        let max_val = sorted[n - 1];

        if (max_val - min_val).abs() < 1e-6 {
            let cb = SubByteCodebook::uniform(bit_width, 1);
            return (cb.normalized_centroids, cb.normalized_boundaries);
        }

        // Initialize centroids via quantile percentiles
        let mut centroids = Vec::with_capacity(k);
        for i in 0..k {
            let q_idx = ((i as f32 + 0.5) / (k as f32) * (n as f32)).floor() as usize;
            centroids.push(sorted[q_idx.min(n - 1)]);
        }

        let mut boundaries = vec![0.0f32; k.saturating_sub(1)];

        for _iter in 0..self.max_iterations {
            // Update decision boundaries: t_k = (c_{k-1} + c_k) / 2
            for i in 0..boundaries.len() {
                boundaries[i] = (centroids[i] + centroids[i + 1]) * 0.5;
            }

            // Assign samples to clusters and compute conditional expectations
            let mut cluster_sums = vec![0.0f64; k];
            let mut cluster_counts = vec![0usize; k];

            for &x in &sorted {
                let mut c_idx = 0usize;
                for (b_idx, &bound) in boundaries.iter().enumerate() {
                    if x > bound {
                        c_idx = b_idx + 1;
                    } else {
                        break;
                    }
                }
                cluster_sums[c_idx] += x as f64;
                cluster_counts[c_idx] += 1;
            }

            // Update centroids: c_k = sum(X_k) / |X_k|
            let mut max_shift = 0.0f32;
            for i in 0..k {
                if cluster_counts[i] > 0 {
                    let new_c = (cluster_sums[i] / (cluster_counts[i] as f64)) as f32;
                    let shift = (new_c - centroids[i]).abs();
                    if shift > max_shift {
                        max_shift = shift;
                    }
                    centroids[i] = new_c;
                }
            }

            if max_shift < self.convergence_tolerance {
                break;
            }
        }

        // Final boundary update
        for i in 0..boundaries.len() {
            boundaries[i] = (centroids[i] + centroids[i + 1]) * 0.5;
        }

        (centroids, boundaries)
    }
}

/// Compact sub-byte packed acoustic feature frame.
///
/// For 2-bit quantization, 16 features occupy only 4 bytes (128 bits $\to$ 32 bits).
/// For 4-bit quantization, 16 features occupy only 8 bytes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubBytePackedFrame {
    /// Packed byte array storing compact indices.
    pub packed_bytes: Vec<u8>,
    /// Original continuous feature dimension.
    pub dimension: usize,
    /// Bit-width format.
    pub bit_width: SubByteBitWidth,
}

impl SubBytePackedFrame {
    /// Pack a slice of discrete codebook indices `0..K` into a sub-byte bitstream.
    pub fn from_indices(indices: &[u8], bit_width: SubByteBitWidth) -> Self {
        let dim = indices.len();
        let vpb = bit_width.values_per_byte();
        let num_bytes = dim.div_ceil(vpb);
        let mut packed_bytes = vec![0u8; num_bytes];
        let bits = bit_width.bits();
        let mask = bit_width.mask();

        for (i, &idx) in indices.iter().enumerate() {
            let byte_idx = i / vpb;
            let offset_in_byte = (i % vpb) * bits;
            let val = idx & mask;
            packed_bytes[byte_idx] |= val << offset_in_byte;
        }

        Self {
            packed_bytes,
            dimension: dim,
            bit_width,
        }
    }

    /// Quantize a continuous floating-point frame using a codebook and pack into sub-byte storage.
    pub fn quantize_and_pack(frame: &[f32], codebook: &SubByteCodebook) -> Self {
        let indices: Vec<u8> = frame
            .iter()
            .enumerate()
            .map(|(d, &x)| codebook.quantize_scalar(d, x))
            .collect();
        Self::from_indices(&indices, codebook.bit_width)
    }

    /// Extract a single discrete index at feature index `idx`.
    #[inline(always)]
    pub fn get_index(&self, idx: usize) -> u8 {
        if idx >= self.dimension {
            return 0;
        }
        let vpb = self.bit_width.values_per_byte();
        let bits = self.bit_width.bits();
        let mask = self.bit_width.mask();

        let byte_idx = idx / vpb;
        let offset = (idx % vpb) * bits;
        (self.packed_bytes[byte_idx] >> offset) & mask
    }

    /// Unpack all discrete indices into a vector.
    pub fn unpack_indices(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.dimension);
        for i in 0..self.dimension {
            out.push(self.get_index(i));
        }
        out
    }

    /// Reconstruct continuous floating-point frame using codebook centroids.
    pub fn dequantize(&self, codebook: &SubByteCodebook) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.dimension);
        for i in 0..self.dimension {
            let q = self.get_index(i);
            out.push(codebook.dequantize_index(i, q));
        }
        out
    }

    /// Return total raw byte size of the packed frame.
    #[inline]
    pub fn byte_len(&self) -> usize {
        self.packed_bytes.len()
    }
}

/// Enrolled wake-word template stored entirely in compact sub-byte packed memory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubBytePhraseTemplate {
    /// Keyword phrase name.
    pub name: String,
    /// Chronological sequence of sub-byte packed frames.
    pub frames: Vec<SubBytePackedFrame>,
    /// Trained scalar codebook with per-dimension scaling.
    pub codebook: SubByteCodebook,
    /// Calibrated DTW decision threshold.
    pub threshold: f32,
    /// Sakoe-Chiba corridor band radius.
    pub band_radius: usize,
    /// Feature vector dimension.
    pub dimension: usize,
}

impl SubBytePhraseTemplate {
    /// Enroll a continuous float feature sequence by training a Lloyd-Max codebook and packing frames.
    pub fn from_features(
        name: impl Into<String>,
        features: &[Vec<f32>],
        bit_width: SubByteBitWidth,
        threshold: f32,
        band_radius: usize,
    ) -> Self {
        let name_str = name.into();
        if features.is_empty() {
            return Self {
                name: name_str,
                frames: Vec::new(),
                codebook: SubByteCodebook::uniform(bit_width, 13),
                threshold,
                band_radius,
                dimension: 0,
            };
        }

        let dim = features[0].len();
        let trainer = LloydMaxTrainer::default();
        let codebook = trainer.train_from_frames(features, bit_width);

        let frames: Vec<SubBytePackedFrame> = features
            .iter()
            .map(|f| SubBytePackedFrame::quantize_and_pack(f, &codebook))
            .collect();

        Self {
            name: name_str,
            frames,
            codebook,
            threshold,
            band_radius,
            dimension: dim,
        }
    }

    /// Total memory footprint in bytes of the packed template dictionary.
    pub fn memory_footprint_bytes(&self) -> usize {
        let frames_bytes: usize = self.frames.iter().map(|f| f.byte_len()).sum();
        let codebook_bytes = self.codebook.normalized_centroids.len() * 4
            + self.codebook.normalized_boundaries.len() * 4
            + self.codebook.means.len() * 4
            + self.codebook.scales.len() * 4;
        frames_bytes + codebook_bytes
    }

    /// Memory compression ratio compared to 32-bit uncompressed float storage.
    pub fn compression_ratio(&self) -> f32 {
        let uncompressed_bytes = self.frames.len() * self.dimension * 4;
        if uncompressed_bytes == 0 {
            return 1.0;
        }
        (uncompressed_bytes as f32) / (self.memory_footprint_bytes().max(1) as f32)
    }

    /// Frame length of the template.
    #[inline]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Is the template empty?
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}

/// Asymmetric Distance Computation (ADC) accelerator for sub-byte quantized DTW.
///
/// In ADC, incoming streaming speech query frames $\mathbf{x} \in \mathbb{R}^D$ remain unquantized
/// or 8-bit, while the template dictionary is stored in sub-byte packed indices $q_{j, d}$.
///
/// For each query frame $\mathbf{x}$, an exact lookup table (LUT) of size $D \times K$ is precomputed:
/// $$\text{LUT}[d, k] = (x_d - c_{d, k})^2$$
///
/// The distance between query frame $\mathbf{x}$ and packed template frame $\mathbf{q}_j$ is then computed
/// using zero floating-point arithmetic during matching via pure table accumulation:
/// $$D(\mathbf{x}, \mathbf{q}_j) = \sqrt{\sum_{d=0}^{D-1} \text{LUT}[d, \mathbf{q}_{j, d}]}$$
#[derive(Debug, Clone)]
pub struct SubByteDtwMatcher {
    templates: Vec<SubBytePhraseTemplate>,
}

impl Default for SubByteDtwMatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl SubByteDtwMatcher {
    /// Construct a new empty sub-byte DTW matcher.
    pub fn new() -> Self {
        Self {
            templates: Vec::new(),
        }
    }

    /// Add an enrolled sub-byte phrase template.
    pub fn add_template(&mut self, template: SubBytePhraseTemplate) {
        self.templates.push(template);
    }

    /// Clear all enrolled templates.
    pub fn clear_templates(&mut self) {
        self.templates.clear();
    }

    /// Access enrolled templates.
    pub fn templates(&self) -> &[SubBytePhraseTemplate] {
        &self.templates
    }

    /// Access mutable reference to enrolled templates.
    pub fn templates_mut(&mut self) -> &mut [SubBytePhraseTemplate] {
        &mut self.templates
    }

    /// Precompute Asymmetric Distance Computation (ADC) distance LUT for a query frame.
    ///
    /// Table shape: `[dim * K]`, indexed by `d * K + k`.
    pub fn build_adc_lut(query_frame: &[f32], codebook: &SubByteCodebook) -> Vec<f32> {
        let dim = query_frame.len();
        let k = codebook.normalized_centroids.len();
        let mut lut = vec![0.0f32; dim * k];

        for (d, &q_val) in query_frame.iter().enumerate() {
            let offset = d * k;
            let mean = if d < codebook.means.len() { codebook.means[d] } else { 0.0 };
            let scale = if d < codebook.scales.len() { codebook.scales[d] } else { 1.0 };

            for c_idx in 0..k {
                let recon = mean + scale * codebook.normalized_centroids[c_idx];
                let diff = q_val - recon;
                lut[offset + c_idx] = diff * diff;
            }
        }

        lut
    }

    /// Compute frame Euclidean distance using Asymmetric Distance Computation (ADC) LUT.
    #[inline(always)]
    pub fn frame_distance_adc(lut: &[f32], packed_frame: &SubBytePackedFrame, k: usize) -> f32 {
        let mut sum_sq = 0.0f32;
        let dim = packed_frame.dimension;

        match packed_frame.bit_width {
            SubByteBitWidth::FourBit => {
                // 2 values per byte
                let mut d = 0;
                for &byte in &packed_frame.packed_bytes {
                    let idx0 = (byte & 0x0F) as usize;
                    sum_sq += lut[d * k + idx0];
                    d += 1;
                    if d >= dim {
                        break;
                    }

                    let idx1 = ((byte >> 4) & 0x0F) as usize;
                    sum_sq += lut[d * k + idx1];
                    d += 1;
                    if d >= dim {
                        break;
                    }
                }
            }
            SubByteBitWidth::TwoBit => {
                // 4 values per byte
                let mut d = 0;
                for &byte in &packed_frame.packed_bytes {
                    let idx0 = (byte & 0x03) as usize;
                    sum_sq += lut[d * k + idx0];
                    d += 1;
                    if d >= dim {
                        break;
                    }

                    let idx1 = ((byte >> 2) & 0x03) as usize;
                    sum_sq += lut[d * k + idx1];
                    d += 1;
                    if d >= dim {
                        break;
                    }

                    let idx2 = ((byte >> 4) & 0x03) as usize;
                    sum_sq += lut[d * k + idx2];
                    d += 1;
                    if d >= dim {
                        break;
                    }

                    let idx3 = ((byte >> 6) & 0x03) as usize;
                    sum_sq += lut[d * k + idx3];
                    d += 1;
                    if d >= dim {
                        break;
                    }
                }
            }
            SubByteBitWidth::OneBit => {
                // 8 values per byte
                let mut d = 0;
                for &byte in &packed_frame.packed_bytes {
                    for bit in 0..8 {
                        let idx = ((byte >> bit) & 0x01) as usize;
                        sum_sq += lut[d * k + idx];
                        d += 1;
                        if d >= dim {
                            break;
                        }
                    }
                    if d >= dim {
                        break;
                    }
                }
            }
        }

        sum_sq.sqrt()
    }

    /// Compute Sakoe-Chiba corridor DTW distance using Asymmetric Distance Computation (ADC)
    /// between a continuous query sequence and a sub-byte packed template.
    pub fn compute_distance_adc_banded(
        query_seq: &[Vec<f32>],
        template: &SubBytePhraseTemplate,
    ) -> f32 {
        let n = query_seq.len();
        let m = template.frames.len();
        let band_radius = template.band_radius;

        if n == 0 || m == 0 || n.abs_diff(m) > band_radius {
            return f32::INFINITY;
        }

        let k = template.codebook.normalized_centroids.len();

        // Precompute ADC lookup tables for all query frames: O(N * D * K)
        let query_luts: Vec<Vec<f32>> = query_seq
            .iter()
            .map(|frame| Self::build_adc_lut(frame, &template.codebook))
            .collect();

        let mut prev = vec![f32::INFINITY; m + 1];
        let mut curr = vec![f32::INFINITY; m + 1];
        prev[0] = 0.0;

        for i in 1..=n {
            curr.fill(f32::INFINITY);
            let j_start = 1.max(i.saturating_sub(band_radius));
            let j_end = m.min(i + band_radius);

            let lut = &query_luts[i - 1];

            for j in j_start..=j_end {
                let dist = Self::frame_distance_adc(lut, &template.frames[j - 1], k);
                let min_prev = prev[j].min(curr[j - 1]).min(prev[j - 1]);
                if min_prev.is_finite() {
                    curr[j] = dist + min_prev;
                }
            }
            std::mem::swap(&mut prev, &mut curr);
        }

        if !prev[m].is_finite() {
            return f32::INFINITY;
        }

        prev[m] / ((n + m) as f32)
    }

    /// Match a streaming window of query frames against enrolled sub-byte templates.
    pub fn match_streaming_window(
        &self,
        history: &[Vec<f32>],
        noise_floor: f32,
        config: &crate::dtw::StreamingDtwConfig,
    ) -> Option<crate::dtw::StreamingMatchResult> {
        if history.is_empty() || self.templates.is_empty() {
            return None;
        }

        let history_len = history.len();
        let mut best_result: Option<crate::dtw::StreamingMatchResult> = None;
        let mut best_dist = f32::INFINITY;

        for template in &self.templates {
            let m = template.frames.len();
            if m == 0 {
                continue;
            }

            let eff_thresh = template.threshold
                * (1.0
                    + config.noise_adapt_alpha
                        * (config.noise_adapt_beta * noise_floor.max(0.0)).tanh());

            let min_len = ((m as f32 * config.min_scale).round() as usize).max(4);
            let max_len = ((m as f32 * config.max_scale).round() as usize).min(history_len);

            if history_len < min_len {
                continue;
            }

            let mut len = min_len;
            while len <= max_len {
                let start_idx = history_len - len;
                let candidate = &history[start_idx..history_len];

                let dist = Self::compute_distance_adc_banded(candidate, template);

                if dist <= eff_thresh && dist < best_dist {
                    best_dist = dist;
                    let conf = (1.0 - (dist / eff_thresh)).clamp(0.0, 1.0);
                    best_result = Some(crate::dtw::StreamingMatchResult {
                        keyword: template.name.clone(),
                        distance: dist,
                        effective_threshold: eff_thresh,
                        confidence: conf,
                        matched_frames: len,
                        template_frames: m,
                    });
                }

                len += config.step;
            }
        }

        best_result
    }
}
