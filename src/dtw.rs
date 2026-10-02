//! Dynamic Time Warping (DTW) alignment, Sakoe-Chiba band pruning, and DBA template matcher.

/// Distance metric between two feature vectors.
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Feature vectors must have equal dimension");
    let sum: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| (x - y) * (x - y)).sum();
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
            return f32::INFINITY;
        }

        // Path-length normalized distance
        cost[n][m] / ((n + m) as f32)
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
    (mean_dist * margin_factor).max(max_pairwise_dist * 1.15).max(2.0)
}
