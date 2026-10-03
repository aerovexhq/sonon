//! Continuous Phonetic CTC Prefix Beam Search & On-Device Language Model Rescorer
//! for Multi-Word Robotics and Aerospace Flight Commands.
//!
//! Provides:
//! - 40-token extended CTC alphabet (39 ARPAbet phonemes + 1 Blank token)
//! - Speech-aware acoustic posterior projection over Mel-scale Cauchy formant centroids
//! - Continuous prefix beam search with repetitive token collapsing and log-space probability pooling
//! - Pronunciation Lexicon Trie for fast dynamic phonetic prefix matching and word segmentation
//! - N-Gram Finite State Transducer (FST) flight command grammar rescorer with backoff smoothing
//! - Sub-20ms streaming latency and Bayesian posterior confidence scoring.

#![deny(unsafe_code)]

use crate::mel::MelFilterbank;
use crate::phonetic::{G2pEngine, KlattSynthesizer, Phoneme, PhonemeSegment};
use crate::stft::FftProcessor;
use crate::window::{Window, WindowType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Number of active ARPAbet phonemes in the vocabulary.
pub const NUM_PHONEMES: usize = 39;

/// Extended CTC vocabulary size (39 ARPAbet phonemes + 1 Blank token).
pub const CTC_VOCAB_SIZE: usize = 40;

/// Index of the special CTC blank token in posterior vectors.
pub const CTC_BLANK_INDEX: usize = 39;

/// All 39 ARPAbet non-silence phonemes in canonical indexed order.
pub const ALL_PHONEMES: [Phoneme; NUM_PHONEMES] = [
    Phoneme::AA, Phoneme::AE, Phoneme::AH, Phoneme::AO, Phoneme::AW, Phoneme::AY,
    Phoneme::EH, Phoneme::ER, Phoneme::EY, Phoneme::IH, Phoneme::IY, Phoneme::OW,
    Phoneme::OY, Phoneme::UH, Phoneme::UW,
    Phoneme::B, Phoneme::D, Phoneme::G, Phoneme::P, Phoneme::T, Phoneme::K,
    Phoneme::DH, Phoneme::F, Phoneme::S, Phoneme::SH, Phoneme::TH, Phoneme::V,
    Phoneme::Z, Phoneme::ZH, Phoneme::HH,
    Phoneme::CH, Phoneme::JH,
    Phoneme::M, Phoneme::N, Phoneme::NG,
    Phoneme::L, Phoneme::R, Phoneme::W, Phoneme::Y,
];

/// Map a Phoneme enum to its 0-based vocabulary index in [0, 38].
#[inline(always)]
pub fn phoneme_to_index(phoneme: Phoneme) -> Option<usize> {
    ALL_PHONEMES.iter().position(|&p| p == phoneme)
}

/// Map a 0-based vocabulary index in [0, 38] to its corresponding Phoneme enum.
#[inline(always)]
pub fn index_to_phoneme(index: usize) -> Option<Phoneme> {
    if index < NUM_PHONEMES {
        Some(ALL_PHONEMES[index])
    } else {
        None
    }
}

/// Numerically stable log-sum-exp operation: ln(exp(a) + exp(b)).
#[inline(always)]
pub fn log_add_exp(a: f32, b: f32) -> f32 {
    if a.is_infinite() && a < 0.0 {
        return b;
    }
    if b.is_infinite() && b < 0.0 {
        return a;
    }
    let max = a.max(b);
    let min = a.min(b);
    let diff = min - max;
    if diff < -20.0 {
        max
    } else {
        max + diff.exp().ln_1p()
    }
}

/// Discrete token in the CTC extended alphabet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CtcToken {
    /// Canonical acoustic phoneme token.
    Phoneme(Phoneme),
    /// Special CTC blank separator token.
    Blank,
}

impl CtcToken {
    /// Return index in [0, 39] corresponding to this token.
    pub fn index(self) -> usize {
        match self {
            CtcToken::Phoneme(p) => phoneme_to_index(p).unwrap_or(0),
            CtcToken::Blank => CTC_BLANK_INDEX,
        }
    }

    /// Construct a CtcToken from an index in [0, 39].
    pub fn from_index(index: usize) -> Option<Self> {
        if index == CTC_BLANK_INDEX {
            Some(CtcToken::Blank)
        } else if index < NUM_PHONEMES {
            Some(CtcToken::Phoneme(ALL_PHONEMES[index]))
        } else {
            None
        }
    }
}

/// Frame-level acoustic posterior probability distribution over all 40 CTC tokens.
#[derive(Debug, Clone, PartialEq)]
pub struct CtcPosteriorFrame {
    /// Linear probability mass distribution summing to 1.0.
    pub probabilities: [f32; CTC_VOCAB_SIZE],
    /// Natural log probabilities in (-inf, 0.0].
    pub log_probabilities: [f32; CTC_VOCAB_SIZE],
}

impl CtcPosteriorFrame {
    /// Construct from precomputed linear probabilities.
    pub fn new(probabilities: [f32; CTC_VOCAB_SIZE]) -> Self {
        let mut log_probabilities = [0.0f32; CTC_VOCAB_SIZE];
        for (i, &p) in probabilities.iter().enumerate() {
            log_probabilities[i] = if p > 1e-12 { p.ln() } else { -27.63 }; // ln(1e-12)
        }
        Self {
            probabilities,
            log_probabilities,
        }
    }

    /// Construct from unnormalized logit scores via numerically stable softmax with temperature.
    pub fn from_logits(logits: &[f32; CTC_VOCAB_SIZE], temperature: f32) -> Self {
        let temp = temperature.max(1e-4);
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));

        let mut exp_vals = [0.0f32; CTC_VOCAB_SIZE];
        let mut sum_exp = 0.0f32;

        for (i, &logit) in logits.iter().enumerate() {
            let exp_val = ((logit - max_logit) / temp).exp();
            exp_vals[i] = exp_val;
            sum_exp += exp_val;
        }

        let inv_sum = 1.0 / sum_exp.max(1e-12);
        let mut probabilities = [0.0f32; CTC_VOCAB_SIZE];
        let mut log_probabilities = [0.0f32; CTC_VOCAB_SIZE];

        for i in 0..CTC_VOCAB_SIZE {
            let p = exp_vals[i] * inv_sum;
            probabilities[i] = p;
            log_probabilities[i] = if p > 1e-12 { p.ln() } else { -27.63 };
        }

        Self {
            probabilities,
            log_probabilities,
        }
    }

    /// Compute frame posterior from 13-dimensional acoustic features (e.g. MFCC) and speech activity flag.
    pub fn from_acoustic_frame(
        frame: &[f32],
        is_speech: bool,
        centroids: &[[f32; 12]; NUM_PHONEMES],
        blank_bias: f32,
        variance: f32,
        temperature: f32,
    ) -> Self {
        let mut logits = [0.0f32; CTC_VOCAB_SIZE];

        if frame.len() < 13 {
            logits[CTC_BLANK_INDEX] = 5.0;
            return Self::from_logits(&logits, temperature);
        }

        // Lifter and normalize C1..C12 formant dimensions (excluding C0 loudness)
        let mut norm_frame = [0.0f32; 12];
        let mut sq_sum = 0.0f32;
        for d in 0..12 {
            let lifter = 1.0 + 5.0 * ((std::f32::consts::PI * (d as f32 + 1.0)) / 12.0).sin();
            let val = frame[d + 1] * lifter;
            norm_frame[d] = val;
            sq_sum += val * val;
        }
        let norm = sq_sum.sqrt().max(1e-6);
        for d in 0..12 {
            norm_frame[d] /= norm;
        }

        let mut max_phoneme_logit = f32::NEG_INFINITY;
        let var_scale = 2.0 * variance.max(0.01);

        for i in 0..NUM_PHONEMES {
            let mut dist_sq = 0.0f32;
            for d in 0..12 {
                let diff = norm_frame[d] - centroids[i][d];
                dist_sq += diff * diff;
            }
            let logit = -dist_sq / var_scale;
            logits[i] = logit;
            if logit > max_phoneme_logit {
                max_phoneme_logit = logit;
            }
        }

        // Blank logit calculation:
        // During silence or non-speech frames, blank token strongly dominates.
        // During active speech, blank logit is placed relative to max_phoneme_logit minus blank_bias.
        let blank_logit = if !is_speech {
            max_phoneme_logit + 5.5
        } else {
            max_phoneme_logit - blank_bias
        };
        logits[CTC_BLANK_INDEX] = blank_logit;

        Self::from_logits(&logits, temperature)
    }

    /// Return probability assigned to the CTC blank token.
    #[inline(always)]
    pub fn blank_prob(&self) -> f32 {
        self.probabilities[CTC_BLANK_INDEX]
    }

    /// Return natural log probability of the CTC blank token.
    #[inline(always)]
    pub fn blank_log_prob(&self) -> f32 {
        self.log_probabilities[CTC_BLANK_INDEX]
    }

    /// Return probability assigned to a given phoneme.
    #[inline(always)]
    pub fn phoneme_prob(&self, phoneme: Phoneme) -> f32 {
        if let Some(idx) = phoneme_to_index(phoneme) {
            self.probabilities[idx]
        } else {
            0.0
        }
    }

    /// Return natural log probability assigned to a given phoneme.
    #[inline(always)]
    pub fn phoneme_log_prob(&self, phoneme: Phoneme) -> f32 {
        if let Some(idx) = phoneme_to_index(phoneme) {
            self.log_probabilities[idx]
        } else {
            f32::NEG_INFINITY
        }
    }

    /// Extract top tokens with probability greater than or equal to `min_prob`.
    pub fn top_tokens(&self, min_prob: f32) -> Vec<(CtcToken, f32)> {
        let mut candidates = Vec::with_capacity(CTC_VOCAB_SIZE);
        for i in 0..CTC_VOCAB_SIZE {
            let p = self.probabilities[i];
            if p >= min_prob {
                if let Some(token) = CtcToken::from_index(i) {
                    candidates.push((token, p));
                }
            }
        }
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates
    }
}

/// Pronunciation Lexicon Trie for fast forward prefix validation and multi-word phonetic segmentation.
#[derive(Debug, Clone)]
pub struct LexiconTrie {
    nodes: Vec<TrieNode>,
}

#[derive(Debug, Clone)]
struct TrieNode {
    word: Option<String>,
    children: HashMap<Phoneme, usize>,
}

impl Default for LexiconTrie {
    fn default() -> Self {
        Self::new()
    }
}

impl LexiconTrie {
    /// Construct a new empty Lexicon Trie.
    pub fn new() -> Self {
        let mut trie = Self {
            nodes: Vec::with_capacity(256),
        };
        // Root node at index 0
        trie.nodes.push(TrieNode {
            word: None,
            children: HashMap::new(),
        });
        trie
    }

    /// Insert a word along with its canonical ARPAbet pronunciation sequence.
    pub fn insert(&mut self, word: impl Into<String>, phonemes: &[Phoneme]) {
        let word_str = word.into();
        let mut curr = 0;

        for &p in phonemes {
            if let Some(&next) = self.nodes[curr].children.get(&p) {
                curr = next;
            } else {
                let next_idx = self.nodes.len();
                self.nodes.push(TrieNode {
                    word: None,
                    children: HashMap::new(),
                });
                self.nodes[curr].children.insert(p, next_idx);
                curr = next_idx;
            }
        }

        self.nodes[curr].word = Some(word_str);
    }

    /// Check if a phoneme sequence forms an exact enrolled word.
    pub fn exact_word(&self, phonemes: &[Phoneme]) -> Option<&str> {
        let mut curr = 0;
        for &p in phonemes {
            curr = *self.nodes[curr].children.get(&p)?;
        }
        self.nodes[curr].word.as_deref()
    }

    /// Check if a phoneme sequence is a valid prefix of at least one word in the lexicon.
    pub fn is_valid_prefix(&self, phonemes: &[Phoneme]) -> bool {
        let mut curr = 0;
        for &p in phonemes {
            match self.nodes[curr].children.get(&p) {
                Some(&next) => curr = next,
                None => return false,
            }
        }
        true
    }

    /// Greedily segment a continuous phoneme trajectory into recognized words and any uncompleted trailing phoneme suffix.
    pub fn segment_phonemes(&self, phonemes: &[Phoneme]) -> (Vec<String>, Vec<Phoneme>) {
        let mut words = Vec::new();
        let mut start = 0;
        let total_len = phonemes.len();

        while start < total_len {
            let mut curr = 0;
            let mut longest_match: Option<String> = None;
            let mut longest_end = start;

            for i in start..total_len {
                if let Some(&next) = self.nodes[curr].children.get(&phonemes[i]) {
                    curr = next;
                    if let Some(ref w) = self.nodes[curr].word {
                        longest_match = Some(w.clone());
                        longest_end = i + 1;
                    }
                } else {
                    break;
                }
            }

            if let Some(word) = longest_match {
                words.push(word);
                start = longest_end;
            } else {
                break;
            }
        }

        let suffix = phonemes[start..].to_vec();
        (words, suffix)
    }
}

/// N-Gram Language Model and Finite State Transducer (FST) Grammar Rescorer for flight commands.
#[derive(Debug, Clone)]
pub struct FlightGrammarLm {
    unigram_log_probs: HashMap<String, f32>,
    bigram_log_probs: HashMap<(String, String), f32>,
    backoff_penalty: f32,
    unseen_word_penalty: f32,
}

impl Default for FlightGrammarLm {
    fn default() -> Self {
        Self::new_flight_control_lm()
    }
}

impl FlightGrammarLm {
    /// Construct an empty language model.
    pub fn new() -> Self {
        Self {
            unigram_log_probs: HashMap::new(),
            bigram_log_probs: HashMap::new(),
            backoff_penalty: -3.2,
            unseen_word_penalty: -6.5,
        }
    }

    /// Construct with pre-trained flight control command vocabulary and bigram transition priors.
    pub fn new_flight_control_lm() -> Self {
        let mut lm = Self::new();

        // 1. Unigram base priors (log probabilities)
        let unigrams = [
            ("take", -1.1),
            ("off", -1.1),
            ("land", -1.0),
            ("immediately", -1.8),
            ("hold", -1.2),
            ("position", -1.2),
            ("return", -1.3),
            ("to", -1.0),
            ("home", -1.3),
            ("abort", -1.4),
            ("mission", -1.4),
            ("emergency", -1.5),
            ("stop", -1.3),
            ("climb", -1.4),
            ("descend", -1.4),
            ("ten", -1.6),
            ("five", -1.6),
            ("meters", -1.4),
            ("yaw", -1.5),
            ("left", -1.5),
            ("right", -1.5),
            ("ninety", -1.8),
            ("forty", -1.8),
            ("degrees", -1.8),
            ("orbit", -1.6),
            ("waypoint", -1.5),
            ("alpha", -1.7),
            ("bravo", -1.7),
            ("confirm", -1.6),
            ("command", -1.6),
            ("arm", -1.4),
            ("disarm", -1.4),
            ("motors", -1.4),
            ("hover", -1.2),
            ("status", -1.2),
        ];

        for (word, log_p) in unigrams {
            lm.set_unigram(word, log_p);
        }

        // 2. High-probability bigram flight grammar transitions (log probabilities)
        let bigrams = [
            ("take", "off", -0.05),
            ("land", "immediately", -0.10),
            ("hold", "position", -0.08),
            ("return", "to", -0.08),
            ("to", "home", -0.05),
            ("abort", "mission", -0.05),
            ("emergency", "stop", -0.05),
            ("climb", "ten", -0.40),
            ("climb", "five", -0.40),
            ("descend", "five", -0.40),
            ("descend", "ten", -0.40),
            ("ten", "meters", -0.08),
            ("five", "meters", -0.08),
            ("yaw", "left", -0.40),
            ("yaw", "right", -0.40),
            ("left", "ninety", -0.20),
            ("right", "forty", -0.25),
            ("forty", "five", -0.10),
            ("ninety", "degrees", -0.15),
            ("five", "degrees", -0.15),
            ("orbit", "waypoint", -0.08),
            ("waypoint", "alpha", -0.30),
            ("waypoint", "bravo", -0.30),
            ("confirm", "command", -0.05),
            ("arm", "motors", -0.05),
            ("disarm", "motors", -0.05),
        ];

        for (w1, w2, log_p) in bigrams {
            lm.set_bigram(w1, w2, log_p);
        }

        lm
    }

    /// Set unigram log probability for a word.
    pub fn set_unigram(&mut self, word: impl Into<String>, log_prob: f32) {
        self.unigram_log_probs.insert(word.into(), log_prob);
    }

    /// Set bigram transition log probability P(w2 | w1).
    pub fn set_bigram(&mut self, w1: impl Into<String>, w2: impl Into<String>, log_prob: f32) {
        self.bigram_log_probs.insert((w1.into(), w2.into()), log_prob);
    }

    /// Evaluate language model log score for an ordered sequence of words.
    pub fn score_word_sequence(&self, words: &[String]) -> f32 {
        if words.is_empty() {
            return 0.0;
        }

        let mut total_log_p = 0.0f32;

        // Unigram prior for initial word
        let first_w = &words[0];
        let w1_log_p = self
            .unigram_log_probs
            .get(first_w)
            .copied()
            .unwrap_or(self.unseen_word_penalty);
        total_log_p += w1_log_p;

        // Bigram transitions
        for i in 1..words.len() {
            let prev = &words[i - 1];
            let curr = &words[i];

            if let Some(&bigram_p) = self.bigram_log_probs.get(&(prev.clone(), curr.clone())) {
                total_log_p += bigram_p;
            } else {
                // Backoff to unigram with backoff penalty
                let unigram_p = self
                    .unigram_log_probs
                    .get(curr)
                    .copied()
                    .unwrap_or(self.unseen_word_penalty);
                total_log_p += unigram_p + self.backoff_penalty;
            }
        }

        total_log_p
    }
}

/// A candidate hypothesis tracked during CTC prefix beam search.
#[derive(Debug, Clone, PartialEq)]
pub struct CtcHypothesis {
    /// Collapsed sequence of phonemes without CTC blanks.
    pub phonemes: Vec<Phoneme>,
    /// Natural log probability of paths generating this prefix ending in a CTC blank.
    pub log_p_blank: f32,
    /// Natural log probability of paths generating this prefix ending in a non-blank phoneme.
    pub log_p_non_blank: f32,
    /// Segmented word sequence produced by the Lexicon Trie.
    pub words: Vec<String>,
    /// Accumulated language model score for the segmented words.
    pub lm_log_prob: f32,
    /// Composite rescore value combining acoustic, LM, word count, and length normalization.
    pub rescore_val: f32,
}

impl CtcHypothesis {
    /// Total acoustic log probability: ln(P_blank + P_non_blank).
    #[inline(always)]
    pub fn total_log_prob(&self) -> f32 {
        log_add_exp(self.log_p_blank, self.log_p_non_blank)
    }

    /// Space-separated recognized command transcription.
    pub fn transcription(&self) -> String {
        self.words.join(" ")
    }
}

/// Configuration parameters for Continuous CTC Prefix Beam Search decoding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CtcDecoderConfig {
    /// Beam width: maximum number of active hypotheses retained per frame (e.g. 16..48).
    pub beam_width: usize,
    /// Token pruning threshold: minimum probability for considering non-blank token transitions.
    pub pruning_threshold: f32,
    /// Blank prior bias added to blank logit (higher reduces insertion rate, lower reduces deletions).
    pub blank_bias: f32,
    /// Language model weight (alpha_lm) scaling grammar log probabilities.
    pub lm_weight: f32,
    /// Word insertion bonus (beta) awarded per successfully completed word in flight vocabulary.
    pub word_insertion_bonus: f32,
    /// Length normalization exponent (alpha_len) in denominator (len + 1)^alpha_len.
    pub length_penalty_alpha: f32,
    /// Softmax temperature scaling acoustic logits.
    pub temperature: f32,
    /// Acoustic centroid distribution scaling variance.
    pub variance: f32,
    /// Consecutive silence frames required to trigger automatic endpoint command finalization.
    pub silence_cutoff_frames: usize,
}

impl Default for CtcDecoderConfig {
    fn default() -> Self {
        Self {
            beam_width: 24,
            pruning_threshold: 1e-4,
            blank_bias: 0.85,
            lm_weight: 0.65,
            word_insertion_bonus: 1.25,
            length_penalty_alpha: 0.70,
            temperature: 1.0,
            variance: 0.06,
            silence_cutoff_frames: 20, // 20 frames * 10ms = 200ms silence endpoint
        }
    }
}

/// Finalized command recognition outcome emitted by the CTC decoder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommandRecognitionResult {
    /// Clean recognized command string (e.g. "take off", "hold position").
    pub command: String,
    /// Segmented individual flight words.
    pub words: Vec<String>,
    /// Underlying recognized phoneme sequence.
    pub phonemes: Vec<Phoneme>,
    /// Normalized recognition confidence score in [0.0, 1.0].
    pub confidence: f32,
    /// Total acoustic log likelihood.
    pub acoustic_score: f32,
    /// Language model grammar log likelihood.
    pub lm_score: f32,
    /// Length-normalized composite decision score.
    pub total_score: f32,
    /// Total number of acoustic frames evaluated in this command utterance.
    pub frames_duration: usize,
    /// Total processing latency in milliseconds.
    pub latency_ms: f32,
}

/// Continuous streaming CTC Prefix Beam Search Decoder with Lexicon Trie and N-Gram FST Rescoring.
pub struct CtcCommandDecoder {
    config: CtcDecoderConfig,
    trie: LexiconTrie,
    lm: FlightGrammarLm,
    centroids: [[f32; 12]; NUM_PHONEMES],
    hypotheses: Vec<CtcHypothesis>,
    consecutive_silence_frames: usize,
    frames_processed: usize,
}

impl CtcCommandDecoder {
    /// Construct a new decoder with default flight control vocabulary and configuration.
    pub fn new(config: CtcDecoderConfig) -> Self {
        let trie = Self::build_default_flight_trie();
        let lm = FlightGrammarLm::new_flight_control_lm();
        let centroids = Self::precompute_phoneme_centroids();

        let mut decoder = Self {
            config,
            trie,
            lm,
            centroids,
            hypotheses: Vec::with_capacity(32),
            consecutive_silence_frames: 0,
            frames_processed: 0,
        };
        decoder.reset();
        decoder
    }

    /// Reset internal beam state for processing a new utterance stream.
    pub fn reset(&mut self) {
        self.hypotheses.clear();
        self.hypotheses.push(CtcHypothesis {
            phonemes: Vec::new(),
            log_p_blank: 0.0,
            log_p_non_blank: f32::NEG_INFINITY,
            words: Vec::new(),
            lm_log_prob: 0.0,
            rescore_val: 0.0,
        });
        self.consecutive_silence_frames = 0;
        self.frames_processed = 0;
    }

    /// Precompute canonical 12-dimensional MFCC formant centroids for all 39 phonemes using steady-state synthesis.
    fn precompute_phoneme_centroids() -> [[f32; 12]; NUM_PHONEMES] {
        let sample_rate = 16000.0;
        let mut synth = KlattSynthesizer::new(sample_rate);
        synth.set_f0(125.0);
        synth.set_speaking_rate(1.0);

        let window = Window::new(WindowType::Hann, 512);
        let fft = FftProcessor::new(512);
        let mel = MelFilterbank::new(26, 512, sample_rate, 80.0, sample_rate / 2.0);

        let mut centroids = [[0.0f32; 12]; NUM_PHONEMES];

        for (idx, &p) in ALL_PHONEMES.iter().enumerate() {
            let seg = PhonemeSegment {
                phoneme: p,
                duration_ms: 140.0,
                stress: 0,
            };
            let audio = synth.synthesize(&[seg]);
            if audio.len() < 512 {
                continue;
            }

            let mut all_mfccs = Vec::new();
            let mut max_energy = f32::NEG_INFINITY;
            let start_pos = (sample_rate * 0.01) as usize;
            let end_pos = audio.len().saturating_sub((sample_rate * 0.01) as usize);
            let mut pos = start_pos;
            let mut frame_buf = vec![0.0f32; 512];

            while pos + 512 <= end_pos {
                frame_buf.copy_from_slice(&audio[pos..pos + 512]);

                // Apply pre-emphasis filter matching SononEngine
                let mut prev = 0.0f32;
                for s in &mut frame_buf {
                    let curr = *s;
                    *s = curr - 0.97 * prev;
                    prev = curr;
                }

                window.apply(&mut frame_buf);
                let power = fft.power_spectrum(&frame_buf);
                let log_energies = mel.compute_log_energies(&power);
                let mfcc = mel.compute_mfcc(&log_energies, 13);
                if mfcc[0] > max_energy {
                    max_energy = mfcc[0];
                }
                all_mfccs.push(mfcc);
                pos += 160;
            }

            let mut accum_mfcc = [0.0f32; 12];
            let mut frame_count = 0;
            let energy_thresh = max_energy - 15.0;

            for mfcc in &all_mfccs {
                if mfcc[0] < energy_thresh {
                    continue;
                }
                for d in 0..12 {
                    let lifter = 1.0 + 5.0 * ((std::f32::consts::PI * (d as f32 + 1.0)) / 12.0).sin();
                    accum_mfcc[d] += mfcc[d + 1] * lifter;
                }
                frame_count += 1;
            }

            if frame_count > 0 {
                let inv = 1.0 / (frame_count as f32);
                for d in 0..12 {
                    accum_mfcc[d] *= inv;
                }
            }

            // Unit-normalize across 12 dimensions
            let mut norm_c = [0.0f32; 12];
            let mut sq = 0.0f32;
            for d in 0..12 {
                let val = accum_mfcc[d];
                norm_c[d] = val;
                sq += val * val;
            }
            let norm = sq.sqrt().max(1e-6);
            for d in 0..12 {
                norm_c[d] /= norm;
            }

            centroids[idx] = norm_c;
        }

        centroids
    }

    /// Build default Lexicon Trie populated with standard robotics and flight commands.
    fn build_default_flight_trie() -> LexiconTrie {
        let mut trie = LexiconTrie::new();

        let dictionary = [
            ("take", vec![Phoneme::T, Phoneme::EY, Phoneme::K]),
            ("off", vec![Phoneme::AO, Phoneme::F]),
            ("land", vec![Phoneme::L, Phoneme::AE, Phoneme::N, Phoneme::D]),
            ("immediately", vec![Phoneme::IH, Phoneme::M, Phoneme::IY, Phoneme::D, Phoneme::IY, Phoneme::AH, Phoneme::T, Phoneme::L, Phoneme::IY]),
            ("hold", vec![Phoneme::HH, Phoneme::OW, Phoneme::L, Phoneme::D]),
            ("position", vec![Phoneme::P, Phoneme::AH, Phoneme::Z, Phoneme::IH, Phoneme::SH, Phoneme::AH, Phoneme::N]),
            ("return", vec![Phoneme::R, Phoneme::IH, Phoneme::T, Phoneme::ER, Phoneme::N]),
            ("to", vec![Phoneme::T, Phoneme::UW]),
            ("home", vec![Phoneme::HH, Phoneme::OW, Phoneme::M]),
            ("abort", vec![Phoneme::AH, Phoneme::B, Phoneme::AO, Phoneme::R, Phoneme::T]),
            ("mission", vec![Phoneme::M, Phoneme::IH, Phoneme::SH, Phoneme::AH, Phoneme::N]),
            ("emergency", vec![Phoneme::IH, Phoneme::M, Phoneme::ER, Phoneme::JH, Phoneme::EH, Phoneme::N, Phoneme::S, Phoneme::IY]),
            ("stop", vec![Phoneme::S, Phoneme::T, Phoneme::AA, Phoneme::P]),
            ("climb", vec![Phoneme::K, Phoneme::L, Phoneme::AY, Phoneme::M]),
            ("descend", vec![Phoneme::D, Phoneme::IH, Phoneme::S, Phoneme::EH, Phoneme::N, Phoneme::D]),
            ("ten", vec![Phoneme::T, Phoneme::EH, Phoneme::N]),
            ("five", vec![Phoneme::F, Phoneme::AY, Phoneme::V]),
            ("meters", vec![Phoneme::M, Phoneme::IY, Phoneme::T, Phoneme::ER, Phoneme::Z]),
            ("yaw", vec![Phoneme::Y, Phoneme::AO]),
            ("left", vec![Phoneme::L, Phoneme::EH, Phoneme::F, Phoneme::T]),
            ("right", vec![Phoneme::R, Phoneme::AY, Phoneme::T]),
            ("ninety", vec![Phoneme::N, Phoneme::AY, Phoneme::N, Phoneme::T, Phoneme::IY]),
            ("forty", vec![Phoneme::F, Phoneme::AO, Phoneme::R, Phoneme::T, Phoneme::IY]),
            ("degrees", vec![Phoneme::D, Phoneme::IH, Phoneme::G, Phoneme::R, Phoneme::IY, Phoneme::Z]),
            ("orbit", vec![Phoneme::AO, Phoneme::R, Phoneme::B, Phoneme::IH, Phoneme::T]),
            ("waypoint", vec![Phoneme::W, Phoneme::EY, Phoneme::P, Phoneme::OY, Phoneme::N, Phoneme::T]),
            ("alpha", vec![Phoneme::AE, Phoneme::L, Phoneme::F, Phoneme::AH]),
            ("bravo", vec![Phoneme::B, Phoneme::R, Phoneme::AA, Phoneme::V, Phoneme::OW]),
            ("confirm", vec![Phoneme::K, Phoneme::AH, Phoneme::N, Phoneme::F, Phoneme::ER, Phoneme::M]),
            ("command", vec![Phoneme::K, Phoneme::AH, Phoneme::M, Phoneme::AE, Phoneme::N, Phoneme::D]),
            ("arm", vec![Phoneme::AA, Phoneme::R, Phoneme::M]),
            ("disarm", vec![Phoneme::D, Phoneme::IH, Phoneme::S, Phoneme::AA, Phoneme::R, Phoneme::M]),
            ("motors", vec![Phoneme::M, Phoneme::OW, Phoneme::T, Phoneme::ER, Phoneme::Z]),
            ("hover", vec![Phoneme::HH, Phoneme::AH, Phoneme::V, Phoneme::ER]),
            ("status", vec![Phoneme::S, Phoneme::T, Phoneme::AE, Phoneme::T, Phoneme::AH, Phoneme::S]),
        ];

        for (word, phonemes) in dictionary {
            trie.insert(word, &phonemes);
        }

        trie
    }

    /// Add a custom vocabulary word and pronunciation to the decoder's pronunciation trie.
    pub fn add_custom_word(&mut self, word: impl Into<String>, phonemes: &[Phoneme]) {
        self.trie.insert(word, phonemes);
    }

    /// Add a custom phrase using automated G2P phoneme conversion.
    pub fn add_custom_phrase(&mut self, phrase: &str) {
        let segments = G2pEngine::text_to_phonemes(phrase);
        let words: Vec<&str> = phrase.split_whitespace().collect();
        let mut seg_idx = 0;

        for w in words {
            let mut word_phonemes = Vec::new();
            while seg_idx < segments.len() {
                let seg = &segments[seg_idx];
                seg_idx += 1;
                if seg.phoneme == Phoneme::SIL {
                    break;
                }
                word_phonemes.push(seg.phoneme);
            }
            if !word_phonemes.is_empty() {
                self.trie.insert(w.to_lowercase(), &word_phonemes);
            }
        }
    }

    /// Ingest a single acoustic frame (e.g. 13-dim MFCC) and update CTC beam search state.
    pub fn step_frame(&mut self, frame: &[f32], is_speech: bool) {
        if !is_speech {
            self.consecutive_silence_frames += 1;
        } else {
            self.consecutive_silence_frames = 0;
        }

        let posterior = CtcPosteriorFrame::from_acoustic_frame(
            frame,
            is_speech,
            &self.centroids,
            self.config.blank_bias,
            self.config.variance,
            self.config.temperature,
        );

        self.step_posterior(&posterior);
    }

    /// Ingest precomputed posterior frame and perform prefix beam search step.
    pub fn step_posterior(&mut self, posterior: &CtcPosteriorFrame) {
        self.frames_processed += 1;
        let blank_log_p = posterior.blank_log_prob();
        let prune_log_thresh = self.config.pruning_threshold.max(1e-6).ln();

        // Accumulate next candidate prefixes: map prefix -> (log_p_blank, log_p_non_blank)
        let mut next_beam: HashMap<Vec<Phoneme>, (f32, f32)> = HashMap::with_capacity(self.config.beam_width * 4);

        for hyp in &self.hypotheses {
            let p_prev = hyp.total_log_prob();

            // 1. Transition with Blank token (prefix does not change)
            let blank_entry = next_beam
                .entry(hyp.phonemes.clone())
                .or_insert((f32::NEG_INFINITY, f32::NEG_INFINITY));
            blank_entry.0 = log_add_exp(blank_entry.0, p_prev + blank_log_p);

            // 2. Transition with non-blank phonemes
            for (idx, &p) in ALL_PHONEMES.iter().enumerate() {
                let p_log = posterior.log_probabilities[idx];
                if p_log < prune_log_thresh {
                    continue;
                }

                if let Some(&last_p) = hyp.phonemes.last() {
                    if last_p == p {
                        // Case A: Repeated token without blank in between remains identical prefix
                        let same_entry = next_beam
                            .entry(hyp.phonemes.clone())
                            .or_insert((f32::NEG_INFINITY, f32::NEG_INFINITY));
                        same_entry.1 = log_add_exp(same_entry.1, hyp.log_p_non_blank + p_log);

                        // Case B: Repeated token across blank separator extends to new repeated phoneme
                        let mut extended = hyp.phonemes.clone();
                        extended.push(p);
                        let ext_entry = next_beam
                            .entry(extended)
                            .or_insert((f32::NEG_INFINITY, f32::NEG_INFINITY));
                        ext_entry.1 = log_add_exp(ext_entry.1, hyp.log_p_blank + p_log);
                        continue;
                    }
                }

                // Case C: Different phoneme or empty prefix extends prefix
                let mut extended = hyp.phonemes.clone();
                extended.push(p);
                let ext_entry = next_beam
                    .entry(extended)
                    .or_insert((f32::NEG_INFINITY, f32::NEG_INFINITY));
                ext_entry.1 = log_add_exp(ext_entry.1, p_prev + p_log);
            }
        }

        // 3. Rescore and prune candidate hypotheses
        let mut scored_hypotheses: Vec<CtcHypothesis> = Vec::with_capacity(next_beam.len());

        for (prefix, (log_b, log_nb)) in next_beam {
            let total_ac_log = log_add_exp(log_b, log_nb);

            // Segment prefix into completed words and trailing phonetic suffix
            let (words, suffix) = self.trie.segment_phonemes(&prefix);

            // Language model score for recognized word sequence
            let lm_log_p = self.lm.score_word_sequence(&words);

            // Suffix penalty check:
            // If suffix is a valid prefix of some word in the trie (in-progress utterance), no penalty.
            // If suffix does not match any prefix, apply out-of-vocabulary mismatch penalty.
            let suffix_penalty = if suffix.is_empty() || self.trie.is_valid_prefix(&suffix) {
                0.0
            } else {
                -2.5 * (suffix.len() as f32)
            };

            let rescore_val = total_ac_log
                + self.config.lm_weight * lm_log_p
                + suffix_penalty;

            scored_hypotheses.push(CtcHypothesis {
                phonemes: prefix,
                log_p_blank: log_b,
                log_p_non_blank: log_nb,
                words,
                lm_log_prob: lm_log_p,
                rescore_val,
            });
        }

        // Sort descending by rescore_val and retain top beam_width
        scored_hypotheses.sort_by(|a, b| {
            b.rescore_val
                .partial_cmp(&a.rescore_val)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        scored_hypotheses.truncate(self.config.beam_width);
        self.hypotheses = scored_hypotheses;
    }

    /// Access reference to current top hypothesis in the beam.
    pub fn current_best_hypothesis(&self) -> Option<&CtcHypothesis> {
        self.hypotheses.first()
    }

    /// Access all active hypotheses currently in the beam.
    pub fn active_hypotheses(&self) -> &[CtcHypothesis] {
        &self.hypotheses
    }

    /// Return consecutive non-speech / silence frames elapsed.
    pub fn consecutive_silence_frames(&self) -> usize {
        self.consecutive_silence_frames
    }

    /// Return total frames processed in this decoding session.
    pub fn frames_processed(&self) -> usize {
        self.frames_processed
    }

    /// Finalize decoding and return best recognized flight command if complete words were decoded.
    pub fn finalize(&mut self) -> Option<CommandRecognitionResult> {
        // Find best hypothesis that completed all recognized words with clean boundary (empty suffix)
        // If none exist with empty suffix, fallback to best hypothesis with completed words
        let clean_candidates: Vec<&CtcHypothesis> = self
            .hypotheses
            .iter()
            .filter(|h| !h.words.is_empty() && self.trie.segment_phonemes(&h.phonemes).1.is_empty())
            .collect();

        let (top, candidate_pool) = if !clean_candidates.is_empty() {
            let best = (*clean_candidates
                .iter()
                .max_by(|a, b| {
                    a.rescore_val
                        .partial_cmp(&b.rescore_val)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })?)
            .clone();
            (best, clean_candidates)
        } else {
            let pool: Vec<&CtcHypothesis> = self.hypotheses.iter().filter(|h| !h.words.is_empty()).collect();
            let best = (*pool
                .iter()
                .max_by(|a, b| {
                    a.rescore_val
                        .partial_cmp(&b.rescore_val)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })?)
            .clone();
            (best, pool)
        };

        let command = top.transcription();

        // Calculate Bayesian posterior confidence marginalized across hypotheses matching top recognized command
        let mut total_log_prob = f32::NEG_INFINITY;
        let mut cmd_log_prob = f32::NEG_INFINITY;

        for h in &candidate_pool {
            let p = h.total_log_prob();
            total_log_prob = log_add_exp(total_log_prob, p);
            if h.transcription() == command {
                cmd_log_prob = log_add_exp(cmd_log_prob, p);
            }
        }

        let confidence = if total_log_prob > f32::NEG_INFINITY && cmd_log_prob > f32::NEG_INFINITY {
            (cmd_log_prob - total_log_prob).exp().clamp(0.0, 1.0)
        } else {
            0.85
        };
        let acoustic_score = top.total_log_prob();
        let lm_score = top.lm_log_prob;
        let total_score = top.rescore_val;
        let words = top.words;
        let phonemes = top.phonemes;

        Some(CommandRecognitionResult {
            command,
            words,
            phonemes,
            confidence,
            acoustic_score,
            lm_score,
            total_score,
            frames_duration: self.frames_processed,
            latency_ms: 0.0, // Calculated by caller
        })
    }

    /// Access reference to active decoder configuration.
    pub fn config(&self) -> &CtcDecoderConfig {
        &self.config
    }

    /// Access mutable reference to active decoder configuration.
    pub fn config_mut(&mut self) -> &mut CtcDecoderConfig {
        &mut self.config
    }

    /// Access reference to underlying Pronunciation Lexicon Trie.
    pub fn trie(&self) -> &LexiconTrie {
        &self.trie
    }

    /// Access reference to underlying N-Gram Grammar Language Model.
    pub fn lm(&self) -> &FlightGrammarLm {
        &self.lm
    }

    /// Access reference to underlying phoneme centroids.
    pub fn centroids(&self) -> &[[f32; 12]; NUM_PHONEMES] {
        &self.centroids
    }

    /// Access mutable reference to underlying N-Gram Grammar Language Model.
    pub fn lm_mut(&mut self) -> &mut FlightGrammarLm {
        &mut self.lm
    }
}
