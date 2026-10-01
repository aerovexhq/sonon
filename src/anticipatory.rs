#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};

/// Two-Phase Speculative Flight Actuator Interlock State.
///
/// Designed to interface directly with the Kestrel autopilot / MAVLink executive:
/// - PreArm: Triggered at 70% phrase completion when Wald SPRT crosses upper confidence boundary.
///   Instructs flight controller to pre-spool ESC motors and lock braking surfaces (-180 ms latency gain).
/// - Commit: Triggered when the final 30% suffix acoustic confirms, committing the high-G maneuver.
/// - Rollback: Triggered if suffix fails or an impostor acoustic occurs, restoring nominal trim safely.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum InterlockState {
    /// Nominal listening mode awaiting speech onsets.
    Listening,
    /// Speculative Pre-Arm state triggered at >= 70% duration.
    PreArm { phrase_id: u32, timestamp_sec: f64, log_likelihood_ratio: f32 },
    /// Final maneuver commit triggered after suffix confirmation.
    Commit { phrase_id: u32, timestamp_sec: f64, log_likelihood_ratio: f32 },
    /// Rollback event if suffix fails verification.
    Rollback { phrase_id: u32, timestamp_sec: f64 },
}

/// Configuration for Abraham Wald's Sequential Probability Ratio Test (SPRT).
#[derive(Debug, Clone, Copy)]
pub struct WaldSprtConfig {
    /// Target False Alarm Probability alpha (e.g. 0.001 -> 0.1%).
    pub alpha_target: f32,
    /// Target False Rejection Probability beta (e.g. 0.01 -> 1.0%).
    pub beta_target: f32,
    /// Phrase completion ratio required to trigger PreArm (typically 0.70 = 70%).
    pub prefix_ratio: f32,
}

impl Default for WaldSprtConfig {
    fn default() -> Self {
        Self {
            alpha_target: 0.001,
            beta_target: 0.01,
            prefix_ratio: 0.70,
        }
    }
}

/// Anticipatory Prefix-CTC & SPRT Early-Exit Decoder.
///
/// Evaluates cumulative log-likelihood ratios frame-by-frame and fires
/// pre-emptive flight commands at 70% phrase completion before the speaker
/// vocalizes trailing phonemes.
#[derive(Debug, Clone)]
pub struct AnticipatoryPrefixDecoder {
    config: WaldSprtConfig,
    upper_boundary_a: f32,
    lower_boundary_b: f32,
    cumulative_llr: f32,
    current_state: InterlockState,
    pre_armed_phrase_id: Option<u32>,
}

impl AnticipatoryPrefixDecoder {
    /// Constructs a new Anticipatory Prefix Decoder with Wald SPRT boundaries.
    pub fn new(config: WaldSprtConfig) -> Self {
        let alpha = config.alpha_target.clamp(1e-5, 0.49);
        let beta = config.beta_target.clamp(1e-5, 0.49);

        // Wald's Stopping Boundaries:
        // Upper threshold A = ln((1 - beta) / alpha)
        // Lower threshold B = ln(beta / (1 - alpha))
        let upper_boundary_a = ((1.0 - beta) / alpha).ln();
        let lower_boundary_b = (beta / (1.0 - alpha)).ln();

        Self {
            config,
            upper_boundary_a,
            lower_boundary_b,
            cumulative_llr: 0.0,
            current_state: InterlockState::Listening,
            pre_armed_phrase_id: None,
        }
    }

    /// Evaluates a single streaming frame observation.
    ///
    /// # Arguments
    /// * `frame_llr` - Instantaneous frame log-likelihood ratio: ln(P(x | WakeWord) / P(x | Noise)).
    /// * `frame_idx` - Current 0-indexed frame count of active utterance.
    /// * `total_expected_frames` - Nominal total duration of enrolled keyword in frames.
    /// * `phrase_id` - Identifier for the enrolled phrase template.
    /// * `timestamp_sec` - Current timestamp in seconds.
    pub fn feed_frame(
        &mut self,
        frame_llr: f32,
        frame_idx: usize,
        total_expected_frames: usize,
        phrase_id: u32,
        timestamp_sec: f64,
    ) -> InterlockState {
        self.cumulative_llr += frame_llr;
        let progress_ratio = (frame_idx as f32) / (total_expected_frames.max(1) as f32);

        match self.current_state {
            InterlockState::Listening => {
                // If cumulative LLR crosses upper boundary A at or beyond prefix_ratio (e.g. 70%)
                if progress_ratio >= self.config.prefix_ratio && self.cumulative_llr >= self.upper_boundary_a {
                    self.current_state = InterlockState::PreArm {
                        phrase_id,
                        timestamp_sec,
                        log_likelihood_ratio: self.cumulative_llr,
                    };
                    self.pre_armed_phrase_id = Some(phrase_id);
                } else if self.cumulative_llr <= self.lower_boundary_b {
                    // Rejection boundary: reset accumulation
                    self.cumulative_llr = 0.0;
                }
            }
            InterlockState::PreArm { phrase_id: armed_id, .. } => {
                // Suffix verification: utterance has reached completion
                if progress_ratio >= 1.0 {
                    if self.cumulative_llr >= self.upper_boundary_a && armed_id == phrase_id {
                        self.current_state = InterlockState::Commit {
                            phrase_id,
                            timestamp_sec,
                            log_likelihood_ratio: self.cumulative_llr,
                        };
                    } else {
                        // Suffix failed or diverged: rollback
                        self.current_state = InterlockState::Rollback {
                            phrase_id,
                            timestamp_sec,
                        };
                    }
                } else if self.cumulative_llr <= self.lower_boundary_b {
                    // Early rollback if LLR drops precipitously during suffix
                    self.current_state = InterlockState::Rollback {
                        phrase_id,
                        timestamp_sec,
                    };
                }
            }
            InterlockState::Commit { .. } | InterlockState::Rollback { .. } => {
                // State stays committed until explicit reset
            }
        }

        self.current_state
    }

    /// Returns the current state of the flight interlock state machine.
    pub fn current_state(&self) -> InterlockState {
        self.current_state
    }

    /// Returns upper decision threshold A.
    pub fn upper_boundary(&self) -> f32 {
        self.upper_boundary_a
    }

    /// Returns lower decision threshold B.
    pub fn lower_boundary(&self) -> f32 {
        self.lower_boundary_b
    }

    /// Returns cumulative LLR accumulated so far.
    pub fn cumulative_llr(&self) -> f32 {
        self.cumulative_llr
    }

    /// Resets the decoder state back to nominal Listening mode.
    pub fn reset(&mut self) {
        self.cumulative_llr = 0.0;
        self.current_state = InterlockState::Listening;
        self.pre_armed_phrase_id = None;
    }
}
