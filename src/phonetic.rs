//! Zero-shot text-to-template phonetic engine, rule-based Grapheme-to-Phoneme (G2P),
//! and Klatt acoustic formant synthesizer for voice command enrollment without human audio.

#![deny(unsafe_code)]

use std::f32::consts::PI;

/// Standard ARPAbet phoneme representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phoneme {
    // Vowels
    AA, // father, odd
    AE, // at, bat, plank
    AH, // hut, cut, but
    AO, // ought, caught
    AW, // cow, out
    AY, // hide, bite
    EH, // red, bed
    ER, // hurt, bird
    EY, // ate, bait
    IH, // it, bit
    IY, // eat, beat
    OW, // oat, boat
    OY, // toy, boy
    UH, // hood, could
    UW, // two, blue

    // Stops (Plosives)
    B,
    D,
    G,
    P,
    T,
    K,

    // Fricatives
    DH, // thee, them
    F,  // fee, fall
    S,  // see, stop
    SH, // she, shot
    TH, // thin, bath
    V,  // voice, vee
    Z,  // zoo, zap
    ZH, // measure, vision
    HH, // he, help

    // Affricates
    CH, // cheer, chew
    JH, // joy, just

    // Nasals
    M,
    N,
    NG, // sing, ring, plank

    // Liquids & Semivowels
    L,
    R,
    W,
    Y,

    // Silence
    SIL,
}

/// A timed phoneme segment with duration and stress.
#[derive(Debug, Clone, PartialEq)]
pub struct PhonemeSegment {
    pub phoneme: Phoneme,
    pub duration_ms: f32,
    pub stress: u8,
}

/// Acoustic formant targets for Klatt synthesis.
#[derive(Debug, Clone, Copy)]
pub struct FormantTarget {
    pub f1: f32,
    pub f2: f32,
    pub f3: f32,
    pub b1: f32,
    pub b2: f32,
    pub b3: f32,
    pub voicing_amp: f32,
    pub aspiration_amp: f32,
    pub friction_amp: f32,
    pub default_duration_ms: f32,
}

impl Phoneme {
    /// Return standard acoustic formant frequencies (Hz), bandwidths (Hz), and source amplitudes.
    pub fn acoustic_targets(self) -> FormantTarget {
        match self {
            Phoneme::AA => FormantTarget {
                f1: 730.0, f2: 1090.0, f3: 2440.0,
                b1: 130.0, b2: 110.0, b3: 170.0,
                voicing_amp: 1.0, aspiration_amp: 0.0, friction_amp: 0.0,
                default_duration_ms: 180.0,
            },
            Phoneme::AE => FormantTarget {
                f1: 660.0, f2: 1720.0, f3: 2410.0,
                b1: 120.0, b2: 120.0, b3: 180.0,
                voicing_amp: 1.0, aspiration_amp: 0.0, friction_amp: 0.0,
                default_duration_ms: 180.0,
            },
            Phoneme::AH => FormantTarget {
                f1: 520.0, f2: 1190.0, f3: 2390.0,
                b1: 110.0, b2: 100.0, b3: 160.0,
                voicing_amp: 1.0, aspiration_amp: 0.0, friction_amp: 0.0,
                default_duration_ms: 130.0,
            },
            Phoneme::AO => FormantTarget {
                f1: 570.0, f2: 840.0, f3: 2410.0,
                b1: 110.0, b2: 100.0, b3: 160.0,
                voicing_amp: 1.0, aspiration_amp: 0.0, friction_amp: 0.0,
                default_duration_ms: 180.0,
            },
            Phoneme::AW => FormantTarget {
                f1: 700.0, f2: 1200.0, f3: 2400.0,
                b1: 120.0, b2: 110.0, b3: 170.0,
                voicing_amp: 1.0, aspiration_amp: 0.0, friction_amp: 0.0,
                default_duration_ms: 200.0,
            },
            Phoneme::AY => FormantTarget {
                f1: 700.0, f2: 1600.0, f3: 2500.0,
                b1: 120.0, b2: 120.0, b3: 180.0,
                voicing_amp: 1.0, aspiration_amp: 0.0, friction_amp: 0.0,
                default_duration_ms: 200.0,
            },
            Phoneme::EH => FormantTarget {
                f1: 530.0, f2: 1840.0, f3: 2480.0,
                b1: 110.0, b2: 110.0, b3: 170.0,
                voicing_amp: 1.0, aspiration_amp: 0.0, friction_amp: 0.0,
                default_duration_ms: 140.0,
            },
            Phoneme::ER => FormantTarget {
                f1: 490.0, f2: 1350.0, f3: 1690.0,
                b1: 120.0, b2: 100.0, b3: 150.0,
                voicing_amp: 1.0, aspiration_amp: 0.0, friction_amp: 0.0,
                default_duration_ms: 170.0,
            },
            Phoneme::EY => FormantTarget {
                f1: 480.0, f2: 2000.0, f3: 2550.0,
                b1: 100.0, b2: 110.0, b3: 170.0,
                voicing_amp: 1.0, aspiration_amp: 0.0, friction_amp: 0.0,
                default_duration_ms: 190.0,
            },
            Phoneme::IH => FormantTarget {
                f1: 390.0, f2: 1990.0, f3: 2550.0,
                b1: 100.0, b2: 110.0, b3: 170.0,
                voicing_amp: 1.0, aspiration_amp: 0.0, friction_amp: 0.0,
                default_duration_ms: 120.0,
            },
            Phoneme::IY => FormantTarget {
                f1: 270.0, f2: 2290.0, f3: 3010.0,
                b1: 90.0, b2: 120.0, b3: 200.0,
                voicing_amp: 1.0, aspiration_amp: 0.0, friction_amp: 0.0,
                default_duration_ms: 170.0,
            },
            Phoneme::OW => FormantTarget {
                f1: 500.0, f2: 1000.0, f3: 2350.0,
                b1: 110.0, b2: 100.0, b3: 160.0,
                voicing_amp: 1.0, aspiration_amp: 0.0, friction_amp: 0.0,
                default_duration_ms: 190.0,
            },
            Phoneme::OY => FormantTarget {
                f1: 550.0, f2: 1400.0, f3: 2400.0,
                b1: 110.0, b2: 110.0, b3: 170.0,
                voicing_amp: 1.0, aspiration_amp: 0.0, friction_amp: 0.0,
                default_duration_ms: 200.0,
            },
            Phoneme::UH => FormantTarget {
                f1: 440.0, f2: 1020.0, f3: 2240.0,
                b1: 100.0, b2: 100.0, b3: 160.0,
                voicing_amp: 1.0, aspiration_amp: 0.0, friction_amp: 0.0,
                default_duration_ms: 120.0,
            },
            Phoneme::UW => FormantTarget {
                f1: 300.0, f2: 870.0, f3: 2240.0,
                b1: 90.0, b2: 90.0, b3: 160.0,
                voicing_amp: 1.0, aspiration_amp: 0.0, friction_amp: 0.0,
                default_duration_ms: 160.0,
            },

            // Stops
            Phoneme::B => FormantTarget {
                f1: 200.0, f2: 1100.0, f3: 2200.0,
                b1: 150.0, b2: 150.0, b3: 200.0,
                voicing_amp: 0.5, aspiration_amp: 0.0, friction_amp: 0.1,
                default_duration_ms: 50.0,
            },
            Phoneme::D => FormantTarget {
                f1: 250.0, f2: 1700.0, f3: 2600.0,
                b1: 150.0, b2: 150.0, b3: 200.0,
                voicing_amp: 0.5, aspiration_amp: 0.0, friction_amp: 0.15,
                default_duration_ms: 50.0,
            },
            Phoneme::G => FormantTarget {
                f1: 250.0, f2: 2000.0, f3: 2400.0,
                b1: 150.0, b2: 150.0, b3: 200.0,
                voicing_amp: 0.5, aspiration_amp: 0.0, friction_amp: 0.15,
                default_duration_ms: 50.0,
            },
            Phoneme::P => FormantTarget {
                f1: 300.0, f2: 1000.0, f3: 2200.0,
                b1: 200.0, b2: 200.0, b3: 250.0,
                voicing_amp: 0.0, aspiration_amp: 0.4, friction_amp: 0.3,
                default_duration_ms: 50.0,
            },
            Phoneme::T => FormantTarget {
                f1: 300.0, f2: 1800.0, f3: 2800.0,
                b1: 200.0, b2: 200.0, b3: 250.0,
                voicing_amp: 0.0, aspiration_amp: 0.4, friction_amp: 0.4,
                default_duration_ms: 50.0,
            },
            Phoneme::K => FormantTarget {
                f1: 300.0, f2: 2100.0, f3: 2500.0,
                b1: 200.0, b2: 200.0, b3: 250.0,
                voicing_amp: 0.0, aspiration_amp: 0.45, friction_amp: 0.4,
                default_duration_ms: 60.0,
            },

            // Fricatives
            Phoneme::DH => FormantTarget {
                f1: 350.0, f2: 1600.0, f3: 2600.0,
                b1: 150.0, b2: 150.0, b3: 200.0,
                voicing_amp: 0.6, aspiration_amp: 0.1, friction_amp: 0.25,
                default_duration_ms: 70.0,
            },
            Phoneme::F => FormantTarget {
                f1: 300.0, f2: 1400.0, f3: 2500.0,
                b1: 200.0, b2: 250.0, b3: 300.0,
                voicing_amp: 0.0, aspiration_amp: 0.2, friction_amp: 0.5,
                default_duration_ms: 100.0,
            },
            Phoneme::S => FormantTarget {
                f1: 300.0, f2: 1800.0, f3: 4000.0,
                b1: 200.0, b2: 300.0, b3: 400.0,
                voicing_amp: 0.0, aspiration_amp: 0.1, friction_amp: 0.8,
                default_duration_ms: 110.0,
            },
            Phoneme::SH => FormantTarget {
                f1: 350.0, f2: 1900.0, f3: 3200.0,
                b1: 180.0, b2: 250.0, b3: 300.0,
                voicing_amp: 0.0, aspiration_amp: 0.2, friction_amp: 0.7,
                default_duration_ms: 110.0,
            },
            Phoneme::TH => FormantTarget {
                f1: 300.0, f2: 1500.0, f3: 2800.0,
                b1: 200.0, b2: 250.0, b3: 350.0,
                voicing_amp: 0.0, aspiration_amp: 0.2, friction_amp: 0.45,
                default_duration_ms: 90.0,
            },
            Phoneme::V => FormantTarget {
                f1: 350.0, f2: 1400.0, f3: 2500.0,
                b1: 140.0, b2: 160.0, b3: 200.0,
                voicing_amp: 0.7, aspiration_amp: 0.05, friction_amp: 0.35,
                default_duration_ms: 80.0,
            },
            Phoneme::Z => FormantTarget {
                f1: 350.0, f2: 1800.0, f3: 3800.0,
                b1: 140.0, b2: 200.0, b3: 300.0,
                voicing_amp: 0.7, aspiration_amp: 0.05, friction_amp: 0.55,
                default_duration_ms: 90.0,
            },
            Phoneme::ZH => FormantTarget {
                f1: 380.0, f2: 1800.0, f3: 3000.0,
                b1: 150.0, b2: 200.0, b3: 250.0,
                voicing_amp: 0.7, aspiration_amp: 0.05, friction_amp: 0.5,
                default_duration_ms: 90.0,
            },
            Phoneme::HH => FormantTarget {
                f1: 450.0, f2: 1500.0, f3: 2500.0,
                b1: 250.0, b2: 250.0, b3: 300.0,
                voicing_amp: 0.0, aspiration_amp: 0.6, friction_amp: 0.2,
                default_duration_ms: 80.0,
            },

            // Affricates
            Phoneme::CH => FormantTarget {
                f1: 300.0, f2: 1800.0, f3: 3200.0,
                b1: 180.0, b2: 220.0, b3: 300.0,
                voicing_amp: 0.0, aspiration_amp: 0.25, friction_amp: 0.6,
                default_duration_ms: 90.0,
            },
            Phoneme::JH => FormantTarget {
                f1: 350.0, f2: 1800.0, f3: 3000.0,
                b1: 160.0, b2: 200.0, b3: 250.0,
                voicing_amp: 0.6, aspiration_amp: 0.1, friction_amp: 0.45,
                default_duration_ms: 90.0,
            },

            // Nasals
            Phoneme::M => FormantTarget {
                f1: 280.0, f2: 1100.0, f3: 2300.0,
                b1: 100.0, b2: 120.0, b3: 180.0,
                voicing_amp: 0.8, aspiration_amp: 0.0, friction_amp: 0.0,
                default_duration_ms: 90.0,
            },
            Phoneme::N => FormantTarget {
                f1: 300.0, f2: 1500.0, f3: 2400.0,
                b1: 100.0, b2: 120.0, b3: 180.0,
                voicing_amp: 0.8, aspiration_amp: 0.0, friction_amp: 0.0,
                default_duration_ms: 90.0,
            },
            Phoneme::NG => FormantTarget {
                f1: 320.0, f2: 1800.0, f3: 2350.0,
                b1: 110.0, b2: 130.0, b3: 180.0,
                voicing_amp: 0.8, aspiration_amp: 0.0, friction_amp: 0.0,
                default_duration_ms: 100.0,
            },

            // Liquids & Semivowels
            Phoneme::L => FormantTarget {
                f1: 380.0, f2: 1100.0, f3: 2700.0,
                b1: 100.0, b2: 110.0, b3: 180.0,
                voicing_amp: 0.9, aspiration_amp: 0.0, friction_amp: 0.0,
                default_duration_ms: 90.0,
            },
            Phoneme::R => FormantTarget {
                f1: 420.0, f2: 1300.0, f3: 1650.0,
                b1: 100.0, b2: 100.0, b3: 150.0,
                voicing_amp: 0.9, aspiration_amp: 0.0, friction_amp: 0.0,
                default_duration_ms: 90.0,
            },
            Phoneme::W => FormantTarget {
                f1: 300.0, f2: 700.0, f3: 2200.0,
                b1: 90.0, b2: 90.0, b3: 160.0,
                voicing_amp: 0.9, aspiration_amp: 0.0, friction_amp: 0.0,
                default_duration_ms: 90.0,
            },
            Phoneme::Y => FormantTarget {
                f1: 280.0, f2: 2200.0, f3: 2900.0,
                b1: 90.0, b2: 120.0, b3: 190.0,
                voicing_amp: 0.9, aspiration_amp: 0.0, friction_amp: 0.0,
                default_duration_ms: 90.0,
            },

            // Silence
            Phoneme::SIL => FormantTarget {
                f1: 200.0, f2: 1000.0, f3: 2000.0,
                b1: 200.0, b2: 200.0, b3: 200.0,
                voicing_amp: 0.0, aspiration_amp: 0.0, friction_amp: 0.0,
                default_duration_ms: 60.0,
            },
        }
    }
}

/// Rule-based Grapheme-to-Phoneme (G2P) engine.
pub struct G2pEngine;

impl G2pEngine {
    /// Convert an English phrase text into a sequence of timed ARPAbet phoneme segments.
    pub fn text_to_phonemes(text: &str) -> Vec<PhonemeSegment> {
        let mut segments = Vec::new();
        let words: Vec<&str> = text
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphabetic()))
            .filter(|w| !w.is_empty())
            .collect();

        for (idx, word) in words.iter().enumerate() {
            let word_lower = word.to_lowercase();
            let phonemes = Self::word_to_phonemes(&word_lower);

            for (p_idx, &p) in phonemes.iter().enumerate() {
                let targets = p.acoustic_targets();
                let stress = if p_idx == 0 || (p == Phoneme::AE || p == Phoneme::EY || p == Phoneme::OW) {
                    1
                } else {
                    0
                };
                segments.push(PhonemeSegment {
                    phoneme: p,
                    duration_ms: targets.default_duration_ms,
                    stress,
                });
            }

            // Insert inter-word boundary silence pause if not last word
            if idx + 1 < words.len() {
                segments.push(PhonemeSegment {
                    phoneme: Phoneme::SIL,
                    duration_ms: 40.0,
                    stress: 0,
                });
            }
        }

        segments
    }

    /// Map a single lowercased word to ARPAbet phonemes using dictionary lookup + deterministic letter-to-sound rules.
    fn word_to_phonemes(word: &str) -> Vec<Phoneme> {
        // High-priority robotics and flight control lexicon
        match word {
            "abort" => vec![Phoneme::AH, Phoneme::B, Phoneme::AO, Phoneme::R, Phoneme::T],
            "land" => vec![Phoneme::L, Phoneme::AE, Phoneme::N, Phoneme::D],
            "plank" => vec![Phoneme::P, Phoneme::L, Phoneme::AE, Phoneme::NG, Phoneme::K],
            "take" => vec![Phoneme::T, Phoneme::EY, Phoneme::K],
            "off" => vec![Phoneme::AO, Phoneme::F],
            "takeoff" => vec![Phoneme::T, Phoneme::EY, Phoneme::K, Phoneme::AO, Phoneme::F],
            "hold" => vec![Phoneme::HH, Phoneme::OW, Phoneme::L, Phoneme::D],
            "hover" => vec![Phoneme::HH, Phoneme::AH, Phoneme::V, Phoneme::ER],
            "arm" => vec![Phoneme::AA, Phoneme::R, Phoneme::M],
            "disarm" => vec![Phoneme::D, Phoneme::IH, Phoneme::S, Phoneme::AA, Phoneme::R, Phoneme::M],
            "return" => vec![Phoneme::R, Phoneme::IH, Phoneme::T, Phoneme::ER, Phoneme::N],
            "stop" => vec![Phoneme::S, Phoneme::T, Phoneme::AA, Phoneme::P],
            "emergency" => vec![Phoneme::IH, Phoneme::M, Phoneme::ER, Phoneme::JH, Phoneme::EH, Phoneme::N, Phoneme::S, Phoneme::IY],
            "launch" => vec![Phoneme::L, Phoneme::AO, Phoneme::N, Phoneme::CH],
            "kill" => vec![Phoneme::K, Phoneme::IH, Phoneme::L],
            "up" => vec![Phoneme::AH, Phoneme::P],
            "down" => vec![Phoneme::D, Phoneme::AW, Phoneme::N],
            "forward" => vec![Phoneme::F, Phoneme::AO, Phoneme::R, Phoneme::W, Phoneme::ER, Phoneme::D],
            "backward" => vec![Phoneme::B, Phoneme::AE, Phoneme::K, Phoneme::W, Phoneme::ER, Phoneme::D],
            "left" => vec![Phoneme::L, Phoneme::EH, Phoneme::F, Phoneme::T],
            "right" => vec![Phoneme::R, Phoneme::AY, Phoneme::T],
            "faster" => vec![Phoneme::F, Phoneme::AE, Phoneme::S, Phoneme::T, Phoneme::ER],
            "slower" => vec![Phoneme::S, Phoneme::L, Phoneme::OW, Phoneme::ER],
            "status" => vec![Phoneme::S, Phoneme::T, Phoneme::AE, Phoneme::T, Phoneme::AH, Phoneme::S],
            "battery" => vec![Phoneme::B, Phoneme::AE, Phoneme::T, Phoneme::ER, Phoneme::IY],
            "altitude" => vec![Phoneme::AE, Phoneme::L, Phoneme::T, Phoneme::AH, Phoneme::T, Phoneme::UW, Phoneme::D],
            "heading" => vec![Phoneme::HH, Phoneme::EH, Phoneme::D, Phoneme::IH, Phoneme::NG],
            "waypoint" => vec![Phoneme::W, Phoneme::EY, Phoneme::P, Phoneme::OY, Phoneme::N, Phoneme::T],
            "loiter" => vec![Phoneme::L, Phoneme::OY, Phoneme::T, Phoneme::ER],
            "auto" => vec![Phoneme::AO, Phoneme::T, Phoneme::OW],
            "manual" => vec![Phoneme::M, Phoneme::AE, Phoneme::N, Phoneme::Y, Phoneme::UW, Phoneme::AH, Phoneme::L],
            "stabilize" => vec![Phoneme::S, Phoneme::T, Phoneme::EY, Phoneme::B, Phoneme::AH, Phoneme::L, Phoneme::AY, Phoneme::Z],
            _ => Self::rule_based_g2p(word),
        }
    }

    /// Deterministic fallback letter-chunk G2P converter for arbitrary English vocabulary.
    fn rule_based_g2p(word: &str) -> Vec<Phoneme> {
        let chars: Vec<char> = word.chars().collect();
        let mut phonemes = Vec::new();
        let len = chars.len();
        let mut i = 0;

        while i < len {
            // Check 3-letter digraphs
            if i + 3 <= len {
                let tri: String = chars[i..i + 3].iter().collect();
                match tri.as_str() {
                    "igh" => {
                        phonemes.push(Phoneme::AY);
                        i += 3;
                        continue;
                    }
                    "ing" => {
                        phonemes.push(Phoneme::IH);
                        phonemes.push(Phoneme::NG);
                        i += 3;
                        continue;
                    }
                    "tch" => {
                        phonemes.push(Phoneme::CH);
                        i += 3;
                        continue;
                    }
                    _ => {}
                }
            }

            // Check 2-letter digraphs
            if i + 2 <= len {
                let pair: String = chars[i..i + 2].iter().collect();
                let matched = match pair.as_str() {
                    "th" => Some(Phoneme::TH),
                    "ch" => Some(Phoneme::CH),
                    "sh" => Some(Phoneme::SH),
                    "ph" => Some(Phoneme::F),
                    "wh" => Some(Phoneme::W),
                    "ee" => Some(Phoneme::IY),
                    "oo" => Some(Phoneme::UW),
                    "ea" => Some(Phoneme::IY),
                    "ai" | "ay" => Some(Phoneme::EY),
                    "oa" => Some(Phoneme::OW),
                    "ou" | "ow" => Some(Phoneme::AW),
                    "oy" | "oi" => Some(Phoneme::OY),
                    "ck" => Some(Phoneme::K),
                    "ng" => Some(Phoneme::NG),
                    "qu" => {
                        phonemes.push(Phoneme::K);
                        phonemes.push(Phoneme::W);
                        None
                    }
                    _ => None,
                };

                if let Some(p) = matched {
                    phonemes.push(p);
                    i += 2;
                    continue;
                } else if pair == "qu" {
                    i += 2;
                    continue;
                }
            }

            // Single letter phoneme mapping
            let c = chars[i];
            match c {
                'a' => {
                    // Check silent-e pattern (e.g., "gate" -> EY)
                    if i + 2 < len && chars[i + 2] == 'e' && (i + 3 == len || !chars[i + 3].is_alphabetic()) {
                        phonemes.push(Phoneme::EY);
                    } else {
                        phonemes.push(Phoneme::AE);
                    }
                }
                'b' => phonemes.push(Phoneme::B),
                'c' => {
                    if i + 1 < len && (chars[i + 1] == 'e' || chars[i + 1] == 'i' || chars[i + 1] == 'y') {
                        phonemes.push(Phoneme::S);
                    } else {
                        phonemes.push(Phoneme::K);
                    }
                }
                'd' => phonemes.push(Phoneme::D),
                'e' => {
                    if i + 1 == len {
                        // Silent terminal 'e'
                    } else {
                        phonemes.push(Phoneme::EH);
                    }
                }
                'f' => phonemes.push(Phoneme::F),
                'g' => phonemes.push(Phoneme::G),
                'h' => phonemes.push(Phoneme::HH),
                'i' => {
                    if i + 2 < len && chars[i + 2] == 'e' && (i + 3 == len || !chars[i + 3].is_alphabetic()) {
                        phonemes.push(Phoneme::AY);
                    } else {
                        phonemes.push(Phoneme::IH);
                    }
                }
                'j' => phonemes.push(Phoneme::JH),
                'k' => phonemes.push(Phoneme::K),
                'l' => phonemes.push(Phoneme::L),
                'm' => phonemes.push(Phoneme::M),
                'n' => phonemes.push(Phoneme::N),
                'o' => {
                    if i + 2 < len && chars[i + 2] == 'e' {
                        phonemes.push(Phoneme::OW);
                    } else {
                        phonemes.push(Phoneme::AA);
                    }
                }
                'p' => phonemes.push(Phoneme::P),
                'r' => phonemes.push(Phoneme::R),
                's' => phonemes.push(Phoneme::S),
                't' => phonemes.push(Phoneme::T),
                'u' => phonemes.push(Phoneme::AH),
                'v' => phonemes.push(Phoneme::V),
                'w' => phonemes.push(Phoneme::W),
                'x' => {
                    phonemes.push(Phoneme::K);
                    phonemes.push(Phoneme::S);
                }
                'y' => {
                    if i + 1 == len {
                        phonemes.push(Phoneme::IY);
                    } else {
                        phonemes.push(Phoneme::Y);
                    }
                }
                'z' => phonemes.push(Phoneme::Z),
                _ => {}
            }
            i += 1;
        }

        phonemes
    }
}

/// 2nd-order digital IIR formant resonator.
#[derive(Debug, Clone, Copy)]
struct FormantResonator {
    a1: f32,
    a2: f32,
    b0: f32,
    y1: f32,
    y2: f32,
}

impl FormantResonator {
    fn new(freq: f32, bw: f32, sample_rate: f32) -> Self {
        let r = (-PI * bw / sample_rate).exp();
        let theta = 2.0 * PI * freq / sample_rate;
        let a1 = -2.0 * r * theta.cos();
        let a2 = r * r;
        let b0 = 1.0 - r; // Normalized unit peak gain

        Self {
            a1,
            a2,
            b0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    #[inline(always)]
    fn process(&mut self, input: f32) -> f32 {
        let out = self.b0 * input - self.a1 * self.y1 - self.a2 * self.y2;
        self.y2 = self.y1;
        self.y1 = out;
        out
    }
}

/// Klatt acoustic formant speech synthesizer in pure safe Rust.
pub struct KlattSynthesizer {
    sample_rate: f32,
    f0_base: f32,
}

impl KlattSynthesizer {
    /// Construct a new Klatt synthesizer with specified sample rate (e.g., 16000.0).
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            f0_base: 125.0, // Standard male fundamental pitch
        }
    }

    /// Set fundamental pitch frequency $F_0$ in Hz.
    pub fn set_f0(&mut self, f0: f32) {
        self.f0_base = f0.clamp(60.0, 400.0);
    }

    /// Synthesize continuous raw audio samples from a list of phoneme segments.
    pub fn synthesize(&self, segments: &[PhonemeSegment]) -> Vec<f32> {
        let mut total_samples = 0usize;
        for s in segments {
            total_samples += ((s.duration_ms / 1000.0) * self.sample_rate) as usize;
        }

        let mut audio = Vec::with_capacity(total_samples);
        if segments.is_empty() {
            return audio;
        }

        let mut glottal_phase = 0.0f32;
        let mut noise_state = 123456789u64;

        for s in segments {
            let seg_samples = ((s.duration_ms / 1000.0) * self.sample_rate).max(1.0) as usize;
            let target = s.phoneme.acoustic_targets();

            // Set up formant resonators for F1, F2, F3
            let mut res1 = FormantResonator::new(target.f1, target.b1, self.sample_rate);
            let mut res2 = FormantResonator::new(target.f2, target.b2, self.sample_rate);
            let mut res3 = FormantResonator::new(target.f3, target.b3, self.sample_rate);

            // Subtle pitch intonation contour: slight declination across segment
            let f0 = self.f0_base * if s.stress > 0 { 1.15 } else { 1.0 };
            let f0_step = (f0 / self.sample_rate) * 2.0 * PI;

            for n in 0..seg_samples {
                // 1. Voicing excitation (Rosenberg glottal pulse)
                let voiced_source = if target.voicing_amp > 0.0 {
                    glottal_phase += f0_step;
                    if glottal_phase >= 2.0 * PI {
                        glottal_phase -= 2.0 * PI;
                    }
                    // Glottal waveform
                    let p = glottal_phase / (2.0 * PI);
                    if p < 0.4 {
                        0.5 * (1.0 - (PI * p / 0.4).cos())
                    } else if p < 0.6 {
                        ((PI * (p - 0.4) / 0.4).cos()).max(0.0)
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                // 2. Unvoiced noise excitation (LCG white noise)
                noise_state = noise_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let unvoiced_noise = (((noise_state >> 33) as f32) / (u32::MAX as f32) - 0.5) * 2.0;

                // 3. Combined source excitation
                let excitation = target.voicing_amp * voiced_source
                    + target.aspiration_amp * unvoiced_noise * 0.35;

                // 4. Formant resonator cascade/parallel synthesis
                let f1_out = res1.process(excitation);
                let f2_out = res2.process(excitation);
                let f3_out = res3.process(excitation);

                // High-frequency friction noise for sibilants and fricatives
                let friction_out = target.friction_amp * unvoiced_noise * 0.4;

                let mixed = 0.50 * f1_out + 0.35 * f2_out + 0.20 * f3_out + friction_out;

                // Soft boundary envelope (fade in / fade out at edges of segment)
                let env = if n < 40 {
                    (n as f32) / 40.0
                } else if n + 40 >= seg_samples {
                    ((seg_samples - n) as f32) / 40.0
                } else {
                    1.0
                };

                audio.push((mixed * env * 0.6).clamp(-1.0, 1.0));
            }
        }

        audio
    }
}
