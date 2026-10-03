//! Continual few-shot domain adaptation and acoustic active learning for wake-word spotting.
//!
//! Provides bounded streaming exemplar memory consolidation with Fisher information pruning,
//! incremental Online Dynamic Time Warping Barycenter Averaging (Online DBA),
//! anchor drift protection against catastrophic forgetting, and an edge active learning trigger
//! for borderline confidence speech segments.

#![deny(unsafe_code)]

use crate::dtw::{euclidean_distance, DtwMatcher};
use serde::{Deserialize, Serialize};

/// Acoustic exemplar with Fisher information importance score and metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcousticExemplar {
    /// 13-dimensional acoustic feature frames.
    pub features: Vec<Vec<f32>>,
    /// Evaluated Fisher information importance score.
    pub importance_score: f32,
    /// Acquisition timestamp in seconds.
    pub timestamp_sec: f32,
    /// Estimated Signal-to-Noise Ratio (SNR) in dB.
    pub snr_db: f32,
    /// DTW distance to canonical centroid at admission.
    pub admission_distance: f32,
    /// True if this is an immutable enrollment anchor protecting against catastrophic forgetting.
    pub is_anchor: bool,
}

impl AcousticExemplar {
    /// Create a new acoustic exemplar.
    pub fn new(
        features: Vec<Vec<f32>>,
        importance_score: f32,
        timestamp_sec: f32,
        snr_db: f32,
        admission_distance: f32,
        is_anchor: bool,
    ) -> Self {
        Self {
            features,
            importance_score,
            timestamp_sec,
            snr_db,
            admission_distance,
            is_anchor,
        }
    }

    /// Number of feature frames.
    pub fn frame_count(&self) -> usize {
        self.features.len()
    }
}

/// Fisher Information Matrix evaluator for acoustic speech trajectories.
#[derive(Debug, Clone)]
pub struct FisherInformationEvaluator;

impl FisherInformationEvaluator {
    /// Compute diagonal Fisher information score of an exemplar relative to a reference centroid.
    ///
    /// Evaluates:
    /// \[
    /// \mathcal{I}(\mathbf{X}) = \frac{1}{T \cdot D} \sum_{t=1}^T \sum_{d=0}^{D-1} \left(\frac{x_t[d] - c_t[d]}{\sigma_t^2[d] + \epsilon}\right)^2
    /// \]
    /// where frames are time-aligned via Sakoe-Chiba DTW.
    pub fn compute_importance(
        exemplar_features: &[Vec<f32>],
        reference_features: &[Vec<f32>],
        feature_variances: &[f32],
        band_radius: usize,
    ) -> f32 {
        if exemplar_features.is_empty() || reference_features.is_empty() {
            return 0.0;
        }

        let n = exemplar_features.len();
        let m = reference_features.len();
        let d = 13;

        if n.abs_diff(m) > band_radius {
            return 0.1;
        }

        // Align exemplar with reference using banded DTW
        let mut prev = vec![f32::INFINITY; m + 1];
        let mut curr = vec![f32::INFINITY; m + 1];
        prev[0] = 0.0;

        // Store backpointers for alignment path reconstruction
        let mut path_matrix = vec![0usize; (n + 1) * (m + 1)];

        for i in 1..=n {
            curr.fill(f32::INFINITY);
            let j_start = 1.max(i.saturating_sub(band_radius));
            let j_end = m.min(i + band_radius);

            for j in j_start..=j_end {
                let cost = euclidean_distance(&exemplar_features[i - 1], &reference_features[j - 1]);
                let (min_cost, dir) = if prev[j - 1] <= prev[j] && prev[j - 1] <= curr[j - 1] {
                    (prev[j - 1], 0) // diagonal
                } else if prev[j] <= curr[j - 1] {
                    (prev[j], 1)     // up
                } else {
                    (curr[j - 1], 2) // left
                };

                if min_cost.is_finite() {
                    curr[j] = cost + min_cost;
                    path_matrix[i * (m + 1) + j] = dir;
                }
            }
            std::mem::swap(&mut prev, &mut curr);
        }

        // Backtrack optimal alignment pairs (i - 1, j - 1)
        let mut i = n;
        let mut j = m;
        let mut aligned_pairs = Vec::with_capacity(n + m);

        while i > 0 && j > 0 {
            aligned_pairs.push((i - 1, j - 1));
            let dir = path_matrix[i * (m + 1) + j];
            match dir {
                0 => { i -= 1; j -= 1; }
                1 => { i -= 1; }
                _ => { j -= 1; }
            }
        }

        if aligned_pairs.is_empty() {
            return 0.1;
        }

        // Accumulate Fisher information metric
        let mut fisher_sum = 0.0f32;
        let eps = 1e-3f32;

        for &(e_idx, r_idx) in &aligned_pairs {
            let e_frame = &exemplar_features[e_idx];
            let r_frame = &reference_features[r_idx];

            for dim in 0..d {
                let e_val = if dim < e_frame.len() { e_frame[dim] } else { 0.0 };
                let r_val = if dim < r_frame.len() { r_frame[dim] } else { 0.0 };
                let var = if dim < feature_variances.len() { feature_variances[dim] } else { 1.0 };
                let denom = var.max(eps);

                let diff = (e_val - r_val) / denom;
                fisher_sum += diff * diff;
            }
        }

        let num_elements = (aligned_pairs.len() * d) as f32;
        (fisher_sum / num_elements).clamp(0.01, 100.0)
    }

    /// Compute pairwise DTW acoustic diversity distance between two exemplars.
    pub fn acoustic_distance(
        seq1: &[Vec<f32>],
        seq2: &[Vec<f32>],
        band_radius: usize,
    ) -> f32 {
        let d = DtwMatcher::compute_distance_banded(seq1, seq2, band_radius);
        if d.is_finite() {
            d
        } else {
            100.0
        }
    }
}

/// Bounded streaming exemplar memory buffer with Fisher information and diversity pruning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExemplarMemoryBuffer {
    /// Keyword identifier.
    keyword: String,
    /// Bounded capacity of the memory dictionary.
    capacity: usize,
    /// Enrolled exemplar collection.
    exemplars: Vec<AcousticExemplar>,
    /// Running feature dimension variances (length 13).
    feature_variances: Vec<f32>,
    /// Band radius for DTW calculations.
    band_radius: usize,
    /// Diversity weighting factor alpha in [0.0, 1.0].
    diversity_alpha: f32,
}

impl ExemplarMemoryBuffer {
    /// Create a new bounded exemplar buffer with specified capacity and band radius.
    pub fn new(keyword: impl Into<String>, capacity: usize, band_radius: usize) -> Self {
        assert!(capacity >= 2, "Capacity must be at least 2");
        Self {
            keyword: keyword.into(),
            capacity,
            exemplars: Vec::with_capacity(capacity),
            feature_variances: vec![1.0; 13],
            band_radius,
            diversity_alpha: 0.65,
        }
    }

    /// Set initial immutable anchor template.
    pub fn set_anchor_exemplar(&mut self, features: Vec<Vec<f32>>, snr_db: f32) {
        let anchor = AcousticExemplar::new(
            features,
            10.0, // High anchor importance
            0.0,
            snr_db,
            0.0,
            true, // is_anchor = true
        );

        if let Some(pos) = self.exemplars.iter().position(|e| e.is_anchor) {
            self.exemplars[pos] = anchor;
        } else if self.exemplars.len() < self.capacity {
            self.exemplars.insert(0, anchor);
        } else {
            self.exemplars[0] = anchor;
        }
        self.update_feature_variances();
    }

    /// Insert or consolidate a new candidate exemplar into memory.
    ///
    /// If buffer is full, prunes the non-anchor exemplar with the lowest composite score:
    /// \[
    /// \text{Score}(i) = \alpha \cdot \mathcal{I}_i + (1 - \alpha) \cdot \min_{j \neq i} \text{dist}(X_i, X_j)
    /// \]
    /// guaranteeing anchor preservation and maximum acoustic diversity.
    pub fn insert_exemplar(
        &mut self,
        features: Vec<Vec<f32>>,
        timestamp_sec: f32,
        snr_db: f32,
        reference_centroid: &[Vec<f32>],
    ) -> bool {
        if features.is_empty() {
            return false;
        }

        let admission_dist = DtwMatcher::compute_distance_banded(&features, reference_centroid, self.band_radius);
        if !admission_dist.is_finite() {
            return false;
        }

        let importance = FisherInformationEvaluator::compute_importance(
            &features,
            reference_centroid,
            &self.feature_variances,
            self.band_radius,
        );

        let candidate = AcousticExemplar::new(
            features,
            importance,
            timestamp_sec,
            snr_db,
            admission_dist,
            false,
        );

        if self.exemplars.len() < self.capacity {
            self.exemplars.push(candidate);
            self.update_feature_variances();
            return true;
        }

        // Buffer is at capacity: evaluate retention scores for all non-anchor candidates
        let mut candidates_to_score = Vec::new();
        for (idx, ex) in self.exemplars.iter().enumerate() {
            if !ex.is_anchor {
                candidates_to_score.push(idx);
            }
        }

        if candidates_to_score.is_empty() {
            // All slots are anchors; cannot admit
            return false;
        }

        // Find candidate with minimum retention score
        let mut min_score = f32::INFINITY;
        let mut prune_idx = candidates_to_score[0];

        for &i in &candidates_to_score {
            let ex_i = &self.exemplars[i];
            let mut min_div = f32::INFINITY;

            for (j, ex_j) in self.exemplars.iter().enumerate() {
                if i != j {
                    let d = FisherInformationEvaluator::acoustic_distance(
                        &ex_i.features,
                        &ex_j.features,
                        self.band_radius,
                    );
                    if d < min_div {
                        min_div = d;
                    }
                }
            }

            let score = self.diversity_alpha * ex_i.importance_score + (1.0 - self.diversity_alpha) * min_div;
            if score < min_score {
                min_score = score;
                prune_idx = i;
            }
        }

        // Evaluate candidate's retention score against weakest in buffer
        let mut candidate_min_div = f32::INFINITY;
        for (j, ex_j) in self.exemplars.iter().enumerate() {
            if j != prune_idx {
                let d = FisherInformationEvaluator::acoustic_distance(
                    &candidate.features,
                    &ex_j.features,
                    self.band_radius,
                );
                if d < candidate_min_div {
                    candidate_min_div = d;
                }
            }
        }
        let candidate_score = self.diversity_alpha * candidate.importance_score
            + (1.0 - self.diversity_alpha) * candidate_min_div;

        if candidate_score > min_score {
            self.exemplars[prune_idx] = candidate;
            self.update_feature_variances();
            true
        } else {
            false
        }
    }

    /// Update running empirical variance across all exemplar feature frames.
    fn update_feature_variances(&mut self) {
        let d = 13;
        let mut sum = [0.0f32; 13];
        let mut sum_sq = [0.0f32; 13];
        let mut total_frames = 0usize;

        for ex in &self.exemplars {
            for frame in &ex.features {
                total_frames += 1;
                for dim in 0..d {
                    let val = if dim < frame.len() { frame[dim] } else { 0.0 };
                    sum[dim] += val;
                    sum_sq[dim] += val * val;
                }
            }
        }

        if total_frames > 1 {
            let n = total_frames as f32;
            for dim in 0..d {
                let mean = sum[dim] / n;
                let var = (sum_sq[dim] / n) - (mean * mean);
                self.feature_variances[dim] = var.max(0.1);
            }
        }
    }

    /// Access reference to stored exemplars.
    pub fn exemplars(&self) -> &[AcousticExemplar] {
        &self.exemplars
    }

    /// Number of active exemplars in memory.
    pub fn len(&self) -> usize {
        self.exemplars.len()
    }

    /// True if memory buffer contains no exemplars.
    pub fn is_empty(&self) -> bool {
        self.exemplars.is_empty()
    }

    /// Capacity of memory buffer.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Immutable anchor exemplar if present.
    pub fn anchor(&self) -> Option<&AcousticExemplar> {
        self.exemplars.iter().find(|e| e.is_anchor)
    }
}

/// Incremental Online DTW Barycenter Averaging (Online DBA) updater with anchor drift bounds.
#[derive(Debug, Clone)]
pub struct OnlineDbaUpdater {
    /// Exponential moving average adaptation learning rate eta in [0.01, 0.30].
    learning_rate: f32,
    /// Maximum allowed drift radius from original enrollment anchor in DTW distance.
    max_drift_radius: f32,
    /// Band radius for Sakoe-Chiba DTW alignment.
    band_radius: usize,
}

impl Default for OnlineDbaUpdater {
    fn default() -> Self {
        Self::new(0.12, 4.5, 12)
    }
}

impl OnlineDbaUpdater {
    /// Construct a new Online DBA updater.
    pub fn new(learning_rate: f32, max_drift_radius: f32, band_radius: usize) -> Self {
        Self {
            learning_rate: learning_rate.clamp(0.01, 0.40),
            max_drift_radius: max_drift_radius.max(0.5),
            band_radius,
        }
    }

    /// Incrementally update the canonical centroid template given a new confirmed observation.
    ///
    /// Applies anchor drift projection if updated template diverges further than `max_drift_radius`
    /// from the immutable enrollment anchor.
    pub fn update_centroid(
        &self,
        current_centroid: &[Vec<f32>],
        new_observation: &[Vec<f32>],
        anchor_centroid: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, String> {
        if current_centroid.is_empty() {
            return Err("Current centroid is empty".to_string());
        }
        if new_observation.is_empty() {
            return Err("New observation is empty".to_string());
        }

        let n = current_centroid.len();
        let m = new_observation.len();
        let d = 13;

        if n.abs_diff(m) > self.band_radius {
            return Err(format!("Observation length {} incompatible with centroid {}", m, n));
        }

        // Align centroid with observation
        let mut prev = vec![f32::INFINITY; m + 1];
        let mut curr = vec![f32::INFINITY; m + 1];
        prev[0] = 0.0;

        let mut path_matrix = vec![0usize; (n + 1) * (m + 1)];

        for i in 1..=n {
            curr.fill(f32::INFINITY);
            let j_start = 1.max(i.saturating_sub(self.band_radius));
            let j_end = m.min(i + self.band_radius);

            for j in j_start..=j_end {
                let cost = euclidean_distance(&current_centroid[i - 1], &new_observation[j - 1]);
                let (min_cost, dir) = if prev[j - 1] <= prev[j] && prev[j - 1] <= curr[j - 1] {
                    (prev[j - 1], 0)
                } else if prev[j] <= curr[j - 1] {
                    (prev[j], 1)
                } else {
                    (curr[j - 1], 2)
                };

                if min_cost.is_finite() {
                    curr[j] = cost + min_cost;
                    path_matrix[i * (m + 1) + j] = dir;
                }
            }
            std::mem::swap(&mut prev, &mut curr);
        }

        // Accumulate mapped frames from new_observation to each centroid frame i in 0..n
        let mut mapped_sums = vec![vec![0.0f32; d]; n];
        let mut mapped_counts = vec![0usize; n];

        let mut i = n;
        let mut j = m;

        while i > 0 && j > 0 {
            let c_idx = i - 1;
            let o_idx = j - 1;

            let obs_frame = &new_observation[o_idx];
            for dim in 0..d {
                let val = if dim < obs_frame.len() { obs_frame[dim] } else { 0.0 };
                mapped_sums[c_idx][dim] += val;
            }
            mapped_counts[c_idx] += 1;

            let dir = path_matrix[i * (m + 1) + j];
            match dir {
                0 => { i -= 1; j -= 1; }
                1 => { i -= 1; }
                _ => { j -= 1; }
            }
        }

        // Apply exponential moving average update
        let eta = self.learning_rate;
        let mut updated = Vec::with_capacity(n);

        for c_idx in 0..n {
            let orig = &current_centroid[c_idx];
            let mut new_frame = vec![0.0f32; d];

            if mapped_counts[c_idx] > 0 {
                let inv_k = 1.0 / (mapped_counts[c_idx] as f32);
                for dim in 0..d {
                    let avg_obs = mapped_sums[c_idx][dim] * inv_k;
                    let o_val = if dim < orig.len() { orig[dim] } else { 0.0 };
                    new_frame[dim] = (1.0 - eta) * o_val + eta * avg_obs;
                }
            } else {
                for dim in 0..d {
                    new_frame[dim] = if dim < orig.len() { orig[dim] } else { 0.0 };
                }
            }
            updated.push(new_frame);
        }

        // Anchor drift protection: verify distance to anchor
        if !anchor_centroid.is_empty() {
            let drift = DtwMatcher::compute_distance_banded(&updated, anchor_centroid, self.band_radius);
            if drift > self.max_drift_radius {
                // Project back towards anchor: convex combination
                let beta = ((drift - self.max_drift_radius) / drift).clamp(0.0, 0.85);
                for idx in 0..n {
                    for dim in 0..d {
                        let anc_val = if idx < anchor_centroid.len() && dim < anchor_centroid[idx].len() {
                            anchor_centroid[idx][dim]
                        } else {
                            updated[idx][dim]
                        };
                        updated[idx][dim] = (1.0 - beta) * updated[idx][dim] + beta * anc_val;
                    }
                }
            }
        }

        Ok(updated)
    }
}

/// Candidate speech segment flagged by edge active learning trigger for review or consolidation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActiveLearningCandidate {
    /// Unique sequential candidate ID.
    pub candidate_id: u64,
    /// Targeted keyword name.
    pub keyword: String,
    /// Measured DTW distance to canonical reference.
    pub match_distance: f32,
    /// Decision threshold at time of trigger.
    pub decision_threshold: f32,
    /// Normalized uncertainty score in [0.0, 1.0] (1.0 = exactly at threshold boundary).
    pub uncertainty_score: f32,
    /// Estimated Signal-to-Noise Ratio (dB).
    pub snr_db: f32,
    /// Utterance timestamp in seconds.
    pub timestamp_sec: f32,
    /// Extracted continuous acoustic feature frames.
    pub features: Vec<Vec<f32>>,
    /// Raw time-domain audio samples if preserved.
    pub raw_audio: Vec<f32>,
    /// True if confidence is borderline and requires operator/self-supervised verification.
    pub requires_verification: bool,
}

/// Configuration parameters for Edge Active Learning trigger.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActiveLearningConfig {
    /// Relative uncertainty band around decision threshold (e.g. 0.15 = +/-15%).
    pub uncertainty_band_ratio: f32,
    /// Minimum Signal-to-Noise Ratio (dB) to avoid learning from pure ambient noise.
    pub min_snr_db: f32,
    /// Minimum uncertainty score to flag candidate for logging.
    pub trigger_uncertainty_threshold: f32,
    /// Confidence threshold above which candidate is auto-admitted to memory without verification.
    pub auto_admit_confidence: f32,
    /// Maximum pending active learning candidates stored in ring buffer.
    pub candidate_buffer_capacity: usize,
}

impl Default for ActiveLearningConfig {
    fn default() -> Self {
        Self {
            uncertainty_band_ratio: 0.18,
            min_snr_db: 5.0,
            trigger_uncertainty_threshold: 0.25,
            auto_admit_confidence: 0.88,
            candidate_buffer_capacity: 16,
        }
    }
}

/// Edge active learning trigger identifying borderline or informative speech segments.
#[derive(Debug, Clone)]
pub struct ActiveLearningTrigger {
    config: ActiveLearningConfig,
    next_candidate_id: u64,
    pending_candidates: Vec<ActiveLearningCandidate>,
}

impl Default for ActiveLearningTrigger {
    fn default() -> Self {
        Self::new(ActiveLearningConfig::default())
    }
}

impl ActiveLearningTrigger {
    /// Construct a new active learning trigger with specified configuration.
    pub fn new(config: ActiveLearningConfig) -> Self {
        let cap = config.candidate_buffer_capacity;
        Self {
            config,
            next_candidate_id: 1,
            pending_candidates: Vec::with_capacity(cap),
        }
    }

    /// Evaluate an observed speech window and DTW match distance for active learning trigger.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_window(
        &mut self,
        keyword: &str,
        match_dist: f32,
        threshold: f32,
        snr_db: f32,
        timestamp_sec: f32,
        features: Vec<Vec<f32>>,
        raw_audio: Vec<f32>,
    ) -> Option<ActiveLearningCandidate> {
        if snr_db < self.config.min_snr_db || !match_dist.is_finite() || threshold <= 0.0 {
            return None;
        }

        let band_width = threshold * self.config.uncertainty_band_ratio;
        let delta = (match_dist - threshold).abs();

        if delta > band_width {
            return None;
        }

        // Uncertainty score U in [0.0, 1.0], peaked at threshold boundary
        let uncertainty = (1.0 - delta / band_width).clamp(0.0, 1.0);
        if uncertainty < self.config.trigger_uncertainty_threshold {
            return None;
        }

        let confidence = (1.0 - match_dist / (threshold * 1.5)).clamp(0.0, 1.0);
        let requires_verification = confidence < self.config.auto_admit_confidence;

        let candidate = ActiveLearningCandidate {
            candidate_id: self.next_candidate_id,
            keyword: keyword.to_string(),
            match_distance: match_dist,
            decision_threshold: threshold,
            uncertainty_score: uncertainty,
            snr_db,
            timestamp_sec,
            features,
            raw_audio,
            requires_verification,
        };

        self.next_candidate_id = self.next_candidate_id.wrapping_add(1);

        if self.pending_candidates.len() >= self.config.candidate_buffer_capacity {
            self.pending_candidates.remove(0);
        }
        self.pending_candidates.push(candidate.clone());

        Some(candidate)
    }

    /// Access reference to pending active learning candidates.
    pub fn pending_candidates(&self) -> &[ActiveLearningCandidate] {
        &self.pending_candidates
    }

    /// Clear pending candidate buffer.
    pub fn clear_pending(&mut self) {
        self.pending_candidates.clear();
    }
}

/// Telemetry status report for continual few-shot adaptation and active learning.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdaptationTelemetry {
    /// Keyword identifier.
    pub keyword: String,
    /// Number of active exemplars in memory.
    pub exemplar_count: usize,
    /// Maximum capacity of memory buffer.
    pub memory_capacity: usize,
    /// Total incremental online DBA updates applied.
    pub total_updates_applied: u64,
    /// Distance from current canonical centroid to immutable anchor.
    pub drift_from_anchor: f32,
    /// Maximum allowed drift radius.
    pub max_drift_radius: f32,
    /// Average Fisher information across active exemplars.
    pub mean_fisher_importance: f32,
    /// Number of active learning candidates logged.
    pub pending_candidates_count: usize,
}

/// Unified continual few-shot domain adaptation engine for Sonon KWS.
#[derive(Debug, Clone)]
pub struct ContinualAdaptationEngine {
    memory_buffer: ExemplarMemoryBuffer,
    online_dba: OnlineDbaUpdater,
    active_learning: ActiveLearningTrigger,
    canonical_centroid: Vec<Vec<f32>>,
    anchor_centroid: Vec<Vec<f32>>,
    calibrated_threshold: f32,
    total_updates_applied: u64,
    band_radius: usize,
}

impl ContinualAdaptationEngine {
    /// Construct a new continual adaptation engine for an enrolled keyword.
    pub fn new(
        keyword: impl Into<String>,
        initial_template: Vec<Vec<f32>>,
        threshold: f32,
        memory_capacity: usize,
        band_radius: usize,
    ) -> Self {
        let kw_str = keyword.into();
        let mut memory = ExemplarMemoryBuffer::new(kw_str.clone(), memory_capacity, band_radius);
        memory.set_anchor_exemplar(initial_template.clone(), 20.0);

        Self {
            memory_buffer: memory,
            online_dba: OnlineDbaUpdater::new(0.10, 4.0, band_radius),
            active_learning: ActiveLearningTrigger::default(),
            canonical_centroid: initial_template.clone(),
            anchor_centroid: initial_template,
            calibrated_threshold: threshold,
            total_updates_applied: 0,
            band_radius,
        }
    }

    /// Process a streaming wake-word spotting event for continual online adaptation.
    ///
    /// If match is high-confidence, automatically updates the canonical centroid via Online DBA
    /// and consolidates the exemplar into memory buffer.
    /// If match is borderline, triggers Active Learning candidate extraction.
    pub fn process_observation(
        &mut self,
        features: &[Vec<f32>],
        match_dist: f32,
        snr_db: f32,
        timestamp_sec: f32,
        raw_audio: &[f32],
    ) -> Result<Option<ActiveLearningCandidate>, String> {
        let thresh = self.calibrated_threshold;
        let kw = self.memory_buffer.keyword.clone();

        // 1. Evaluate Active Learning trigger on borderline speech
        let al_candidate = self.active_learning.evaluate_window(
            &kw,
            match_dist,
            thresh,
            snr_db,
            timestamp_sec,
            features.to_vec(),
            raw_audio.to_vec(),
        );

        // 2. High-confidence adaptation: distance is strictly within threshold and good SNR
        if match_dist <= thresh && snr_db >= 5.0 && !features.is_empty() {
            // Update canonical centroid incrementally
            let updated = self.online_dba.update_centroid(
                &self.canonical_centroid,
                features,
                &self.anchor_centroid,
            )?;
            self.canonical_centroid = updated;
            self.total_updates_applied = self.total_updates_applied.wrapping_add(1);

            // Consolidate into exemplar memory buffer
            self.memory_buffer.insert_exemplar(
                features.to_vec(),
                timestamp_sec,
                snr_db,
                &self.canonical_centroid,
            );
        }

        Ok(al_candidate)
    }

    /// Admitting an operator-confirmed active learning candidate into memory.
    pub fn admit_candidate(&mut self, candidate: &ActiveLearningCandidate) -> Result<(), String> {
        if candidate.features.is_empty() {
            return Err("Candidate features are empty".to_string());
        }

        let updated = self.online_dba.update_centroid(
            &self.canonical_centroid,
            &candidate.features,
            &self.anchor_centroid,
        )?;
        self.canonical_centroid = updated;
        self.total_updates_applied = self.total_updates_applied.wrapping_add(1);

        self.memory_buffer.insert_exemplar(
            candidate.features.clone(),
            candidate.timestamp_sec,
            candidate.snr_db,
            &self.canonical_centroid,
        );

        Ok(())
    }

    /// Current canonical centroid template for KWS matching.
    pub fn canonical_centroid(&self) -> &[Vec<f32>] {
        &self.canonical_centroid
    }

    /// Current calibrated decision threshold.
    pub fn calibrated_threshold(&self) -> f32 {
        self.calibrated_threshold
    }

    /// Generate telemetry snapshot.
    pub fn telemetry(&self) -> AdaptationTelemetry {
        let drift = DtwMatcher::compute_distance_banded(
            &self.canonical_centroid,
            &self.anchor_centroid,
            self.band_radius,
        );

        let mean_fisher = if self.memory_buffer.is_empty() {
            0.0
        } else {
            self.memory_buffer
                .exemplars()
                .iter()
                .map(|e| e.importance_score)
                .sum::<f32>()
                / (self.memory_buffer.len() as f32)
        };

        AdaptationTelemetry {
            keyword: self.memory_buffer.keyword.clone(),
            exemplar_count: self.memory_buffer.len(),
            memory_capacity: self.memory_buffer.capacity(),
            total_updates_applied: self.total_updates_applied,
            drift_from_anchor: if drift.is_finite() { drift } else { 0.0 },
            max_drift_radius: self.online_dba.max_drift_radius,
            mean_fisher_importance: mean_fisher,
            pending_candidates_count: self.active_learning.pending_candidates().len(),
        }
    }
}
