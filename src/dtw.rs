//! Dynamic Time Warping (DTW) alignment and phrase spotting template matcher.

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
}

/// Dynamic Time Warping matcher for streaming phrase spotting.
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

    /// Enroll a phrase template with associated recognition distance threshold.
    pub fn add_template(&mut self, name: impl Into<String>, features: Vec<Vec<f32>>, threshold: f32) {
        self.templates.push(PhraseTemplate {
            name: name.into(),
            features,
            threshold,
        });
    }

    /// Compute symmetric DTW distance between two feature sequences.
    pub fn compute_distance(seq1: &[Vec<f32>], seq2: &[Vec<f32>]) -> f32 {
        let n = seq1.len();
        let m = seq2.len();

        if n == 0 || m == 0 {
            return f32::INFINITY;
        }

        let mut cost = vec![vec![f32::INFINITY; m + 1]; n + 1];
        cost[0][0] = 0.0;

        for i in 1..=n {
            for j in 1..=m {
                let dist = euclidean_distance(&seq1[i - 1], &seq2[j - 1]);
                let min_prev = cost[i - 1][j].min(cost[i][j - 1]).min(cost[i - 1][j - 1]);
                cost[i][j] = dist + min_prev;
            }
        }

        // Return path-length normalized distance
        cost[n][m] / ((n + m) as f32)
    }

    /// Match an observation window against enrolled phrase templates.
    /// Returns best matching template name and distance if within threshold.
    pub fn match_window(&self, observation: &[Vec<f32>]) -> Option<(String, f32)> {
        let mut best_match: Option<(String, f32)> = None;
        let mut best_dist = f32::INFINITY;

        for template in &self.templates {
            let dist = Self::compute_distance(observation, &template.features);
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
}

impl Default for DtwMatcher {
    fn default() -> Self {
        Self::new()
    }
}
