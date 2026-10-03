//! Dynamic Time Warping (DTW) alignment, Sakoe-Chiba band pruning, and DBA template matcher.

use serde::{Deserialize, Serialize};

/// Distance metric between two feature vectors.
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Feature vectors must have equal dimension");
    let sum: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| (x - y) * (x - y)).sum();
    sum.sqrt()
}

/// Weighted Euclidean distance metric between two feature vectors using reliability weights.
pub fn weighted_euclidean_distance(a: &[f32], b: &[f32], weights: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Feature vectors must have equal dimension");
    assert_eq!(a.len(), weights.len(), "Weights must match feature dimension");
    let sum: f32 = a
        .iter()
        .zip(b.iter())
        .zip(weights.iter())
        .map(|((&x, &y), &w)| w * (x - y) * (x - y))
        .sum();
    sum.sqrt()
}

/// Enrolled phrase template for few-shot keyword spotting.
#[derive(Debug, Clone)]
pub struct PhraseTemplate {
    pub name: String,
    pub features: Vec<Vec<f32>>,
    pub threshold: f32,
    pub band_radius: usize,
}

/// Configuration for streaming continuous Sakoe-Chiba DTW keyword matching.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreamingDtwConfig {
    /// Minimum search scale factor relative to template length (e.g. 0.70 allows faster speech).
    pub min_scale: f32,
    /// Maximum search scale factor relative to template length (e.g. 1.35 allows slower speech).
    pub max_scale: f32,
    /// Search candidate length step (1 for fine resolution, 2 for accelerated step).
    pub step: usize,
    /// Noise adaptation scale factor alpha: T_eff = T * (1 + alpha * tanh(beta * noise_floor)).
    pub noise_adapt_alpha: f32,
    /// Noise adaptation sensitivity beta.
    pub noise_adapt_beta: f32,
    /// Refractory period in frames to suppress duplicate firings for a single utterance.
    pub refractory_frames: usize,
}

impl Default for StreamingDtwConfig {
    fn default() -> Self {
        Self {
            min_scale: 0.75,
            max_scale: 1.30,
            step: 1,
            noise_adapt_alpha: 0.25,
            noise_adapt_beta: 0.15,
            refractory_frames: 15,
        }
    }
}

/// Result of a streaming keyword spotting match.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamingMatchResult {
    pub keyword: String,
    pub distance: f32,
    pub effective_threshold: f32,
    pub confidence: f32,
    pub matched_frames: usize,
    pub template_frames: usize,
}

/// Dynamic Time Warping matcher with Sakoe-Chiba band pruning and multi-exemplar DBA fusion.
#[derive(Debug, Clone)]
pub struct DtwMatcher {
    templates: Vec<PhraseTemplate>,
}

impl DtwMatcher {
    /// Create an empty DTW matcher.
    pub fn new() -> Self {
        Self {
            templates: Vec::new(),
        }
    }

    /// Enroll a phrase template with associated recognition distance threshold and default band radius.
    pub fn add_template(&mut self, name: impl Into<String>, features: Vec<Vec<f32>>, threshold: f32) {
        let band_radius = (features.len() / 4).max(8);
        self.add_template_banded(name, features, threshold, band_radius);
    }

    /// Enroll a phrase template with explicit Sakoe-Chiba corridor band radius `R`.
    pub fn add_template_banded(
        &mut self,
        name: impl Into<String>,
        features: Vec<Vec<f32>>,
        threshold: f32,
        band_radius: usize,
    ) {
        self.templates.push(PhraseTemplate {
            name: name.into(),
            features,
            threshold,
            band_radius,
        });
    }

    /// Enroll a phrase template using multiple audio exemplars, fusing them via DBA and
    /// automatically calibrating the distance threshold.
    pub fn add_template_exemplars(
        &mut self,
        name: impl Into<String>,
        exemplars: &[Vec<Vec<f32>>],
        band_radius: usize,
        margin_factor: f32,
    ) -> f32 {
        assert!(!exemplars.is_empty(), "Must provide at least one exemplar");
        let centroid = dtw_barycenter_averaging(exemplars, 5, band_radius);
        let threshold = calibrate_threshold(exemplars, band_radius, margin_factor);
        self.add_template_banded(name, centroid, threshold, band_radius);
        threshold
    }

    /// Compute unconstrained symmetric DTW distance between two feature sequences.
    pub fn compute_distance(seq1: &[Vec<f32>], seq2: &[Vec<f32>]) -> f32 {
        let max_len = seq1.len().max(seq2.len());
        Self::compute_distance_banded(seq1, seq2, max_len)
    }

    /// Compute Sakoe-Chiba corridor-constrained DTW distance between two feature sequences.
    ///
    /// Restricts warping path to `|i - j| <= band_radius`, reducing complexity from O(N*M) to O(N*R).
    pub fn compute_distance_banded(seq1: &[Vec<f32>], seq2: &[Vec<f32>], band_radius: usize) -> f32 {
        let n = seq1.len();
        let m = seq2.len();

        if n == 0 || m == 0 {
            return f32::INFINITY;
        }

        // If length difference exceeds band radius, path is physically unreachable
        if n.abs_diff(m) > band_radius {
            return f32::INFINITY;
        }

        let mut prev = vec![f32::INFINITY; m + 1];
        let mut curr = vec![f32::INFINITY; m + 1];
        prev[0] = 0.0;

        for i in 1..=n {
            curr.fill(f32::INFINITY);
            let j_start = 1.max(i.saturating_sub(band_radius));
            let j_end = m.min(i + band_radius);

            for j in j_start..=j_end {
                let dist = euclidean_distance(&seq1[i - 1], &seq2[j - 1]);
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

        // Path-length normalized distance
        prev[m] / ((n + m) as f32)
    }

    /// Match an observation window against enrolled phrase templates using each template's band radius.
    /// Returns best matching template name and distance if within threshold.
    pub fn match_window(&self, observation: &[Vec<f32>]) -> Option<(String, f32)> {
        let mut best_match: Option<(String, f32)> = None;
        let mut best_dist = f32::INFINITY;

        for template in &self.templates {
            let dist = Self::compute_distance_banded(
                observation,
                &template.features,
                template.band_radius,
            );
            if dist <= template.threshold && dist < best_dist {
                best_dist = dist;
                best_match = Some((template.name.clone(), dist));
            }
        }

        best_match
    }

    /// Match a streaming chronological feature history against enrolled templates,
    /// evaluating candidate window lengths ending at the current frame and adapting to background noise floor.
    pub fn match_streaming_window(
        &self,
        history: &[Vec<f32>],
        noise_floor: f32,
        config: &StreamingDtwConfig,
    ) -> Option<StreamingMatchResult> {
        if history.is_empty() || self.templates.is_empty() {
            return None;
        }

        let history_len = history.len();
        let mut best_result: Option<StreamingMatchResult> = None;
        let mut best_dist = f32::INFINITY;

        for template in &self.templates {
            let m = template.features.len();
            if m == 0 {
                continue;
            }

            // Adaptive threshold based on background acoustic noise floor
            let eff_thresh = template.threshold
                * (1.0 + config.noise_adapt_alpha * (config.noise_adapt_beta * noise_floor.max(0.0)).tanh());

            let min_len = ((m as f32 * config.min_scale).round() as usize).max(4);
            let max_len = ((m as f32 * config.max_scale).round() as usize).min(history_len);

            if history_len < min_len {
                continue;
            }

            let mut len = min_len;
            while len <= max_len {
                let start_idx = history_len - len;
                let candidate = &history[start_idx..history_len];
                let dist = Self::compute_distance_banded(
                    candidate,
                    &template.features,
                    template.band_radius,
                );

                if dist <= eff_thresh && dist < best_dist {
                    best_dist = dist;
                    let conf = (1.0 - (dist / eff_thresh)).clamp(0.0, 1.0);
                    best_result = Some(StreamingMatchResult {
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

    /// Return count of enrolled templates.
    pub fn template_count(&self) -> usize {
        self.templates.len()
    }

    /// Clear all enrolled templates.
    pub fn clear_templates(&mut self) {
        self.templates.clear();
    }

    /// Access enrolled templates slice.
    pub fn templates(&self) -> &[PhraseTemplate] {
        &self.templates
    }
}

impl Default for DtwMatcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract alignment warping path between two sequences under Sakoe-Chiba band constraint.
/// Returns vector of aligned index pairs `(i, j)` with 0-indexed coordinates.
pub fn extract_warping_path_banded(
    seq1: &[Vec<f32>],
    seq2: &[Vec<f32>],
    band_radius: usize,
) -> Vec<(usize, usize)> {
    let n = seq1.len();
    let m = seq2.len();
    if n == 0 || m == 0 || n.abs_diff(m) > band_radius {
        return Vec::new();
    }

    let mut cost = vec![vec![f32::INFINITY; m + 1]; n + 1];
    cost[0][0] = 0.0;

    for i in 1..=n {
        let j_start = 1.max(i.saturating_sub(band_radius));
        let j_end = m.min(i + band_radius);
        for j in j_start..=j_end {
            let dist = euclidean_distance(&seq1[i - 1], &seq2[j - 1]);
            let min_prev = cost[i - 1][j].min(cost[i][j - 1]).min(cost[i - 1][j - 1]);
            if min_prev.is_finite() {
                cost[i][j] = dist + min_prev;
            }
        }
    }

    if !cost[n][m].is_finite() {
        return Vec::new();
    }

    // Backtrack from (n, m) down to (1, 1)
    let mut path = Vec::with_capacity(n + m);
    let mut i = n;
    let mut j = m;

    while i > 1 || j > 1 {
        path.push((i - 1, j - 1));
        if i == 1 {
            j -= 1;
        } else if j == 1 {
            i -= 1;
        } else {
            let diag = cost[i - 1][j - 1];
            let up = cost[i - 1][j];
            let left = cost[i][j - 1];

            if diag <= up && diag <= left {
                i -= 1;
                j -= 1;
            } else if up <= left {
                i -= 1;
            } else {
                j -= 1;
            }
        }
    }
    path.push((0, 0));

    path.reverse();
    path
}

/// Fuses multiple voice exemplars into a single robust centroid reference template using
/// Dynamic Time Warping Barycenter Averaging (DBA).
pub fn dtw_barycenter_averaging(
    exemplars: &[Vec<Vec<f32>>],
    max_iters: usize,
    band_radius: usize,
) -> Vec<Vec<f32>> {
    if exemplars.is_empty() {
        return Vec::new();
    }
    if exemplars.len() == 1 {
        return exemplars[0].clone();
    }

    // Select the medoid exemplar (the one with minimum total DTW distance to all others) as initial centroid
    let mut best_medoid_idx = 0;
    let mut min_total_dist = f32::INFINITY;

    for (idx, cand) in exemplars.iter().enumerate() {
        let mut total_dist = 0.0f32;
        for other in exemplars {
            total_dist += DtwMatcher::compute_distance_banded(cand, other, band_radius);
        }
        if total_dist < min_total_dist {
            min_total_dist = total_dist;
            best_medoid_idx = idx;
        }
    }

    let mut centroid = exemplars[best_medoid_idx].clone();
    let num_features = centroid[0].len();
    let centroid_len = centroid.len();

    for _ in 0..max_iters {
        let mut sums = vec![vec![0.0f32; num_features]; centroid_len];
        let mut counts = vec![0usize; centroid_len];

        for exemplar in exemplars {
            let path = extract_warping_path_banded(exemplar, &centroid, band_radius);
            for &(i_ex, j_cent) in &path {
                for f in 0..num_features {
                    sums[j_cent][f] += exemplar[i_ex][f];
                }
                counts[j_cent] += 1;
            }
        }

        // Update centroid coordinates as arithmetic averages
        for j in 0..centroid_len {
            if counts[j] > 0 {
                let count_f = counts[j] as f32;
                for f in 0..num_features {
                    centroid[j][f] = sums[j][f] / count_f;
                }
            }
        }
    }

    centroid
}

/// Calibrates an adaptive recognition threshold based on inter-exemplar DTW self-similarity.
pub fn calibrate_threshold(
    exemplars: &[Vec<Vec<f32>>],
    band_radius: usize,
    margin_factor: f32,
) -> f32 {
    if exemplars.len() <= 1 {
        return 3.5; // Standard default single-exemplar threshold
    }

    let mut max_pairwise_dist = 0.0f32;
    let mut total_dist = 0.0f32;
    let mut pairs = 0usize;

    for i in 0..exemplars.len() {
        for j in (i + 1)..exemplars.len() {
            let d = DtwMatcher::compute_distance_banded(&exemplars[i], &exemplars[j], band_radius);
            if d.is_finite() {
                max_pairwise_dist = max_pairwise_dist.max(d);
                total_dist += d;
                pairs += 1;
            }
        }
    }

    if pairs == 0 {
        return 3.5;
    }

    let mean_dist = total_dist / (pairs as f32);
    // Add margin factor (e.g. 1.25x to 1.5x) to accommodate ambient variation
    (mean_dist * margin_factor).max(max_pairwise_dist * 1.25).max(0.40)
}

/// Binary classification confusion matrix for wake-word and acoustic phrase spotting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ConfusionMatrix {
    /// True positive occurrences (keyword correctly detected).
    pub true_positives: usize,
    /// False positive occurrences (negative/foil falsely flagged as keyword).
    pub false_positives: usize,
    /// True negative occurrences (negative/foil correctly rejected).
    pub true_negatives: usize,
    /// False negative occurrences (keyword failed to trigger).
    pub false_negatives: usize,
}

impl ConfusionMatrix {
    /// Construct a new confusion matrix with raw counts.
    pub fn new(tp: usize, fp: usize, tn: usize, fn_count: usize) -> Self {
        Self {
            true_positives: tp,
            false_positives: fp,
            true_negatives: tn,
            false_negatives: fn_count,
        }
    }

    /// Return total evaluation count.
    pub fn total_samples(&self) -> usize {
        self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
    }

    /// Compute Precision: TP / (TP + FP).
    pub fn precision(&self) -> f32 {
        let denom = self.true_positives + self.false_positives;
        if denom == 0 {
            1.0
        } else {
            (self.true_positives as f32) / (denom as f32)
        }
    }

    /// Compute Recall (Sensitivity / True Positive Rate): TP / (TP + FN).
    pub fn recall(&self) -> f32 {
        let denom = self.true_positives + self.false_negatives;
        if denom == 0 {
            1.0
        } else {
            (self.true_positives as f32) / (denom as f32)
        }
    }

    /// Compute F1 Score harmonic mean: 2 * (P * R) / (P + R).
    pub fn f1_score(&self) -> f32 {
        let p = self.precision();
        let r = self.recall();
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * (p * r) / (p + r)
        }
    }

    /// Compute False Positive Rate (FPR / Fallout): FP / (FP + TN).
    pub fn false_positive_rate(&self) -> f32 {
        let denom = self.false_positives + self.true_negatives;
        if denom == 0 {
            0.0
        } else {
            (self.false_positives as f32) / (denom as f32)
        }
    }

    /// Compute False Negative Rate (FNR / Miss Rate): FN / (FN + TP).
    pub fn false_negative_rate(&self) -> f32 {
        let denom = self.false_negatives + self.true_positives;
        if denom == 0 {
            0.0
        } else {
            (self.false_negatives as f32) / (denom as f32)
        }
    }

    /// Compute Classification Accuracy: (TP + TN) / Total.
    pub fn accuracy(&self) -> f32 {
        let total = self.total_samples();
        if total == 0 {
            1.0
        } else {
            ((self.true_positives + self.true_negatives) as f32) / (total as f32)
        }
    }
}

/// Online background acoustic noise tracker estimating feature variance and dynamic reliability weights.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcousticNoiseClusterTracker {
    num_features: usize,
    mean_noise: Vec<f32>,
    var_noise: Vec<f32>,
    alpha: f32,
    sample_count: usize,
    weights: Vec<f32>,
}

impl AcousticNoiseClusterTracker {
    /// Construct a new noise tracker for a given feature dimension (e.g., 13 or 26).
    pub fn new(num_features: usize, alpha: f32) -> Self {
        Self {
            num_features,
            mean_noise: vec![0.0; num_features],
            var_noise: vec![1.0; num_features],
            alpha: alpha.clamp(0.001, 0.5),
            sample_count: 0,
            weights: vec![1.0; num_features],
        }
    }

    /// Update noise statistics with an observed background non-speech feature vector.
    pub fn update_noise(&mut self, frame: &[f32]) {
        if frame.len() != self.num_features {
            return;
        }

        if self.sample_count == 0 {
            self.mean_noise.copy_from_slice(frame);
            self.var_noise.fill(0.01);
            self.sample_count = 1;
            return;
        }

        self.sample_count += 1;
        let alpha = self.alpha;

        for k in 0..self.num_features {
            let x = frame[k];
            let prev_m = self.mean_noise[k];
            let new_m = (1.0 - alpha) * prev_m + alpha * x;
            let diff = x - new_m;
            let new_var = (1.0 - alpha) * self.var_noise[k] + alpha * (diff * diff);

            self.mean_noise[k] = new_m;
            self.var_noise[k] = new_var.max(1e-5);
        }

        // Calculate reliability weights: higher noise variance -> lower weight
        let mut max_w = 0.0f32;
        for k in 0..self.num_features {
            let sigma = self.var_noise[k].sqrt();
            let w = 1.0 / (1.0 + 2.0 * sigma);
            self.weights[k] = w;
            if w > max_w {
                max_w = w;
            }
        }

        if max_w > 1e-4 {
            for k in 0..self.num_features {
                self.weights[k] = (self.weights[k] / max_w).clamp(0.15, 1.0);
            }
        }
    }

    /// Access active feature reliability weights.
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Access running background mean feature vector.
    pub fn mean_noise(&self) -> &[f32] {
        &self.mean_noise
    }

    /// Access running background feature variance vector.
    pub fn var_noise(&self) -> &[f32] {
        &self.var_noise
    }

    /// Total background frames observed.
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Estimate spectral flatness of background noise: geometric mean / arithmetic mean.
    pub fn spectral_flatness(&self) -> f32 {
        if self.mean_noise.is_empty() {
            return 1.0;
        }
        let pos_energies: Vec<f32> = self.mean_noise.iter().map(|&x| x.abs() + 1e-4).collect();
        let sum: f32 = pos_energies.iter().sum();
        let am = sum / (pos_energies.len() as f32);

        let log_sum: f32 = pos_energies.iter().map(|&x| x.ln()).sum();
        let gm = (log_sum / (pos_energies.len() as f32)).exp();

        if am > 1e-6 {
            (gm / am).clamp(0.0, 1.0)
        } else {
            1.0
        }
    }

    /// Compute noise-weighted Euclidean distance between two feature vectors.
    pub fn compute_weighted_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        weighted_euclidean_distance(a, b, &self.weights)
    }

    /// Reset noise tracker state.
    pub fn reset(&mut self) {
        self.mean_noise.fill(0.0);
        self.var_noise.fill(1.0);
        self.weights.fill(1.0);
        self.sample_count = 0;
    }
}

/// Quantized 8-bit representation of an audio feature frame with affine scaling parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantizedFrame {
    pub values: Vec<i8>,
    pub scale: f32,
    pub zero_point: i32,
}

impl QuantizedFrame {
    /// Quantize a floating-point feature vector into an 8-bit signed integer frame.
    pub fn from_f32(vec: &[f32]) -> Self {
        if vec.is_empty() {
            return Self {
                values: Vec::new(),
                scale: 1.0,
                zero_point: 0,
            };
        }
        let min_val = vec.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = vec.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let range = (max_val - min_val).max(1e-5);
        let scale = range / 254.0;
        let zero_point = (-min_val / scale).round() as i32 - 128;

        let values = vec
            .iter()
            .map(|&x| {
                let q = (x / scale).round() as i32 + zero_point;
                q.clamp(-128, 127) as i8
            })
            .collect();

        Self {
            values,
            scale,
            zero_point,
        }
    }

    /// Dequantize back to floating-point representation.
    pub fn to_f32(&self) -> Vec<f32> {
        self.values
            .iter()
            .map(|&q| (q as i32 - self.zero_point) as f32 * self.scale)
            .collect()
    }
}

/// Quantized enrolled phrase template for 8-bit integer memory-efficient matching.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantizedPhraseTemplate {
    pub name: String,
    pub frames: Vec<Vec<i8>>,
    pub scale: f32,
    pub threshold: f32,
    pub band_radius: usize,
}

/// Fast 8-bit integer Dynamic Time Warping matcher with zero-allocation rolling buffers.
#[derive(Debug, Clone, Default)]
pub struct QuantizedDtwMatcher {
    templates: Vec<QuantizedPhraseTemplate>,
}

impl QuantizedDtwMatcher {
    /// Create an empty 8-bit quantized DTW matcher.
    pub fn new() -> Self {
        Self {
            templates: Vec::new(),
        }
    }

    /// Enroll a template by quantizing its 32-bit floating-point features into 8-bit integer frames.
    pub fn add_template(
        &mut self,
        name: impl Into<String>,
        features: &[Vec<f32>],
        threshold: f32,
        band_radius: usize,
    ) {
        if features.is_empty() {
            return;
        }

        // Global min/max across all template frames for unified affine scaling
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for frame in features {
            for &val in frame {
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }
        }

        let range = (max_val - min_val).max(1e-5);
        let scale = range / 254.0;
        let zero_point = (-min_val / scale).round() as i32 - 128;

        let quantized_frames: Vec<Vec<i8>> = features
            .iter()
            .map(|frame| {
                frame
                    .iter()
                    .map(|&x| {
                        let q = (x / scale).round() as i32 + zero_point;
                        q.clamp(-128, 127) as i8
                    })
                    .collect()
            })
            .collect();

        self.templates.push(QuantizedPhraseTemplate {
            name: name.into(),
            frames: quantized_frames,
            scale,
            threshold,
            band_radius,
        });
    }

    /// Compute L1 Manhattan distance between two 8-bit feature frames.
    #[inline(always)]
    pub fn frame_distance_l1(a: &[i8], b: &[i8]) -> i32 {
        let mut sum = 0i32;
        let n = a.len().min(b.len());
        for i in 0..n {
            sum += (a[i] as i32 - b[i] as i32).abs();
        }
        sum
    }

    /// Compute Sakoe-Chiba corridor DTW distance between two 8-bit quantized feature sequences
    /// using only two rolling 1D cost buffers (zero 2D matrix heap allocation).
    pub fn compute_distance_banded_q(
        seq1: &[Vec<i8>],
        seq2: &[Vec<i8>],
        band_radius: usize,
    ) -> i32 {
        let n = seq1.len();
        let m = seq2.len();

        if n == 0 || m == 0 || n.abs_diff(m) > band_radius {
            return i32::MAX / 2;
        }

        let mut prev = vec![i32::MAX / 2; m + 1];
        let mut curr = vec![i32::MAX / 2; m + 1];
        prev[0] = 0;

        for i in 1..=n {
            curr.fill(i32::MAX / 2);
            let j_start = 1.max(i.saturating_sub(band_radius));
            let j_end = m.min(i + band_radius);

            let frame1 = &seq1[i - 1];

            for j in j_start..=j_end {
                let dist = Self::frame_distance_l1(frame1, &seq2[j - 1]);
                let min_prev = prev[j].min(curr[j - 1]).min(prev[j - 1]);
                if min_prev < i32::MAX / 2 {
                    curr[j] = min_prev.saturating_add(dist);
                }
            }
            std::mem::swap(&mut prev, &mut curr);
        }

        if prev[m] >= i32::MAX / 2 {
            return i32::MAX / 2;
        }

        prev[m] / ((n + m) as i32)
    }

    /// Compute float-equivalent DTW distance between two 8-bit quantized feature sequences.
    pub fn compute_distance_float_equivalent(
        seq1: &[Vec<i8>],
        seq2: &[Vec<i8>],
        band_radius: usize,
        scale: f32,
    ) -> f32 {
        let q_dist = Self::compute_distance_banded_q(seq1, seq2, band_radius);
        if q_dist >= i32::MAX / 2 {
            f32::INFINITY
        } else {
            (q_dist as f32) * scale
        }
    }

    /// Match a streaming window of quantized observation frames against enrolled templates.
    pub fn match_streaming_window_q(
        &self,
        history: &[Vec<i8>],
        scale: f32,
        noise_floor: f32,
        config: &StreamingDtwConfig,
    ) -> Option<StreamingMatchResult> {
        if history.is_empty() || self.templates.is_empty() {
            return None;
        }

        let history_len = history.len();
        let mut best_result: Option<StreamingMatchResult> = None;
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

                let dist_q = Self::compute_distance_banded_q(
                    candidate,
                    &template.frames,
                    template.band_radius,
                );

                if dist_q < i32::MAX / 2 {
                    let dist = (dist_q as f32) * scale;
                    if dist <= eff_thresh && dist < best_dist {
                        best_dist = dist;
                        let conf = (1.0 - (dist / eff_thresh)).clamp(0.0, 1.0);
                        best_result = Some(StreamingMatchResult {
                            keyword: template.name.clone(),
                            distance: dist,
                            effective_threshold: eff_thresh,
                            confidence: conf,
                            matched_frames: len,
                            template_frames: m,
                        });
                    }
                }

                len += config.step;
            }
        }

        best_result
    }

    /// Access enrolled quantized templates.
    pub fn templates(&self) -> &[QuantizedPhraseTemplate] {
        &self.templates
    }

    /// Total count of enrolled templates.
    pub fn template_count(&self) -> usize {
        self.templates.len()
    }
}
