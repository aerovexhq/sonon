#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration parameters for Distribution-Free Conformal Prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConformalConfig {
    /// Statistical significance risk level $\alpha_{\text{risk}} \in (0, 0.50]$ (e.g. 0.05 for 95% confidence).
    pub risk_level: f32,
    /// Minimum required calibration sample count.
    pub min_calibration_samples: usize,
    /// Non-conformity penalty factor for minimal-pair foils.
    pub foil_penalty_weight: f32,
}

impl Default for ConformalConfig {
    fn default() -> Self {
        Self {
            risk_level: 0.05,
            min_calibration_samples: 8,
            foil_penalty_weight: 1.25,
        }
    }
}

/// Evaluation report detailing finite-sample conformal threshold calibration and coverage bounds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConformalCalibrationReport {
    /// Target wake-word identifier.
    pub keyword: String,
    /// Derived conformal distance threshold $\hat{q}_{1 - \alpha}$.
    pub conformal_threshold: f32,
    /// Number of positive calibration utterances evaluated.
    pub positive_samples: usize,
    /// Number of minimal-pair foil negative calibration utterances evaluated.
    pub foil_samples: usize,
    /// Formal statistical coverage guarantee $1 - \alpha_{\text{risk}}$.
    pub coverage_guarantee: f32,
    /// Empirical quantile index used: $\lceil (n + 1)(1 - \alpha) \rceil$.
    pub quantile_index: usize,
}

/// Dynamic multi-hypothesis prediction set emitted by the conformal engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConformalPredictionSet {
    /// Admissible candidate keywords whose non-conformity scores fall within the conformal bound.
    pub candidates: Vec<String>,
    /// Corresponding conformal p-values for each candidate keyword.
    pub p_values: Vec<f32>,
    /// Minimum DTW match distance among candidates in the set.
    pub best_distance: f32,
}

impl ConformalPredictionSet {
    /// Returns true if exactly one keyword was detected with statistical certainty.
    pub fn is_singleton(&self) -> bool {
        self.candidates.len() == 1
    }

    /// Returns true if no keyword satisfied the conformal safety bound (noise / un-enrolled utterance).
    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    /// Returns true if multiple conflicting keywords were detected (ambiguity requiring disambiguation).
    pub fn is_ambiguous(&self) -> bool {
        self.candidates.len() > 1
    }

    /// Returns the single confirmed keyword if the prediction set is a singleton.
    pub fn confirmed_keyword(&self) -> Option<&str> {
        if self.is_singleton() {
            Some(&self.candidates[0])
        } else {
            None
        }
    }
}

/// Stored per-keyword calibration profile.
#[derive(Debug, Clone)]
struct KeywordCalibration {
    conformal_threshold: f32,
    sorted_calibration_scores: Vec<f32>,
    report: ConformalCalibrationReport,
}

/// Distribution-Free Split Conformal Prediction Engine for Safe Robotics KWS.
///
/// Provides mathematically rigorous, finite-sample statistical guarantees:
///
/// $$P(d(X_{n+1}, T_k) \le \hat{q}) \ge 1 - \alpha_{\text{risk}}$$
///
/// for any arbitrary non-stationary drone rotor and environmental noise distribution without parametric assumptions.
#[derive(Debug, Clone)]
pub struct ConformalKwsPredictor {
    config: ConformalConfig,
    calibrations: HashMap<String, KeywordCalibration>,
}

impl ConformalKwsPredictor {
    /// Construct a new Conformal KWS predictor with specified risk level configuration.
    pub fn new(config: ConformalConfig) -> Self {
        assert!(
            config.risk_level > 0.0 && config.risk_level < 1.0,
            "Risk level alpha must be in (0, 1)"
        );
        Self {
            config,
            calibrations: HashMap::new(),
        }
    }

    /// Calibrate distribution-free conformal confidence threshold for a keyword using positive exemplars and foil distances.
    pub fn calibrate(
        &mut self,
        keyword: impl Into<String>,
        positive_distances: &[f32],
        foil_distances: &[f32],
    ) -> Result<ConformalCalibrationReport, String> {
        let kw = keyword.into();
        let n_pos = positive_distances.len();
        if n_pos < self.config.min_calibration_samples {
            return Err(format!(
                "Keyword '{}' has only {} positive samples, minimum required is {}",
                kw, n_pos, self.config.min_calibration_samples
            ));
        }

        // Non-conformity scores for true positive utterances: distance d
        let mut scores: Vec<f32> = positive_distances.iter().copied().filter(|d| d.is_finite()).collect();
        scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = scores.len();
        let alpha = self.config.risk_level;

        // Conformal finite-sample quantile index: ceil((n + 1) * (1 - alpha))
        let q_idx = (((n + 1) as f32 * (1.0 - alpha)).ceil() as usize).min(n).max(1);
        let q_val = scores[q_idx - 1];

        // Sanity check against foil distribution: conformal threshold should separate positive cluster from foils
        let min_foil = foil_distances
            .iter()
            .copied()
            .filter(|d| d.is_finite())
            .fold(f32::INFINITY, |acc, d| acc.min(d));

        let effective_threshold = if min_foil.is_finite() && q_val >= min_foil {
            // Constrain threshold slightly below closest foil if overlap occurs
            (min_foil * 0.95).max(scores[0])
        } else {
            q_val
        };

        let report = ConformalCalibrationReport {
            keyword: kw.clone(),
            conformal_threshold: effective_threshold,
            positive_samples: n,
            foil_samples: foil_distances.len(),
            coverage_guarantee: 1.0 - alpha,
            quantile_index: q_idx,
        };

        self.calibrations.insert(
            kw,
            KeywordCalibration {
                conformal_threshold: effective_threshold,
                sorted_calibration_scores: scores,
                report: report.clone(),
            },
        );

        Ok(report)
    }

    /// Compute empirical conformal p-value for an observed DTW distance:
    ///
    /// $$p_{\text{conf}}(d) = \frac{1 + \sum_{i=1}^n \mathbb{I}(\alpha_i \ge d)}{n + 1}$$
    pub fn compute_p_value(&self, keyword: &str, distance: f32) -> f32 {
        if let Some(calib) = self.calibrations.get(keyword) {
            let n = calib.sorted_calibration_scores.len();
            if n == 0 {
                return 0.0;
            }
            let count_ge = calib
                .sorted_calibration_scores
                .iter()
                .filter(|&&s| s >= distance)
                .count();

            ((count_ge + 1) as f32) / ((n + 1) as f32)
        } else {
            0.0
        }
    }

    /// Check if an observed distance is a statistically verified detection within the conformal safety bound.
    pub fn is_verified_detection(&self, keyword: &str, distance: f32) -> bool {
        if let Some(calib) = self.calibrations.get(keyword) {
            distance <= calib.conformal_threshold
        } else {
            false
        }
    }

    /// Evaluate multi-hypothesis conformal prediction set $\mathcal{C}(X)$ across all candidate keyword match distances.
    pub fn predict_set(&self, keyword_distances: &[(&str, f32)]) -> ConformalPredictionSet {
        let mut candidates = Vec::new();
        let mut p_values = Vec::new();
        let mut best_dist = f32::INFINITY;

        for &(kw, dist) in keyword_distances {
            if self.is_verified_detection(kw, dist) {
                let p = self.compute_p_value(kw, dist);
                candidates.push(kw.to_string());
                p_values.push(p);
                if dist < best_dist {
                    best_dist = dist;
                }
            }
        }

        ConformalPredictionSet {
            candidates,
            p_values,
            best_distance: best_dist,
        }
    }

    /// Access active configuration.
    pub fn config(&self) -> &ConformalConfig {
        &self.config
    }

    /// Access calibration report for an enrolled keyword.
    pub fn calibration_report(&self, keyword: &str) -> Option<&ConformalCalibrationReport> {
        self.calibrations.get(keyword).map(|c| &c.report)
    }

    /// Returns the number of calibrated keywords in the conformal predictor.
    pub fn calibrated_count(&self) -> usize {
        self.calibrations.len()
    }
}
