//! Zero-shot text-to-template phonetic engine, rule-based Grapheme-to-Phoneme (G2P),
//! and Klatt acoustic formant synthesizer for voice command enrollment without human audio.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::f32::consts::PI;
use std::path::Path;

/// Standard ARPAbet phoneme representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

/// Regional accent / dialect profile for acoustic vowel formant adaptation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VocalAccent {
    /// General American English standard formant targets.
    GeneralAmerican,
    /// Received Pronunciation (British English) with back-vowel and non-rhotic shifts.
    ReceivedPronunciation,
    /// International / non-native English with generalized vowel centralization.
    International,
    /// Australian English with vowel raising ([AE] -> [EH]) and broad diphthong glides.
    Australian,
    /// Indian English with retroflex F3 depression and monophthongized vowels.
    IndianEnglish,
    /// East Asian L2 English with vowel centralization and epenthesis.
    EastAsian,
    /// Spanish-accented English with 5-vowel phoneme collapse and short VOT.
    SpanishAccented,
}

impl VocalAccent {
    /// List of all supported regional vocal accents.
    pub fn all_supported() -> &'static [VocalAccent] {
        &[
            VocalAccent::GeneralAmerican,
            VocalAccent::ReceivedPronunciation,
            VocalAccent::International,
            VocalAccent::Australian,
            VocalAccent::IndianEnglish,
            VocalAccent::EastAsian,
            VocalAccent::SpanishAccented,
        ]
    }
}

impl Default for VocalAccent {
    fn default() -> Self {
        Self::GeneralAmerican
    }
}

/// Prosodic pitch intonation contour governing macro-F0 trajectory over an utterance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IntonationContour {
    /// Natural declarative statement: gentle arch peaking mid-utterance, descending cadence.
    Declarative,
    /// Interrogative / questioning inflection: pitch elevates progressively toward utterance end.
    Interrogative,
    /// Authoritative robotics command: crisp high pitch attack with rapid decisive fall.
    AuthoritativeCommand,
    /// Urgent alert: elevated pitch baseline with intensified vibrato tremor.
    UrgentAlert,
}

impl Default for IntonationContour {
    fn default() -> Self {
        Self::Declarative
    }
}

impl IntonationContour {
    /// Evaluates pitch multiplier given normalized utterance progress `t_norm` in [0, 1] and absolute time `t_sec`.
    pub fn evaluate(&self, t_norm: f32, t_sec: f32) -> f32 {
        match self {
            IntonationContour::Declarative => {
                1.06 + 0.14 * (PI * t_norm).sin() - 0.16 * t_norm
            }
            IntonationContour::Interrogative => {
                0.95 + 0.35 * t_norm * t_norm
            }
            IntonationContour::AuthoritativeCommand => {
                1.22 - 0.38 * t_norm
            }
            IntonationContour::UrgentAlert => {
                1.20 + 0.08 * (2.0 * PI * 8.5 * t_sec).sin()
            }
        }
    }
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

    /// Returns true if this phoneme is a vowel.
    pub fn is_vowel(self) -> bool {
        matches!(
            self,
            Phoneme::AA
                | Phoneme::AE
                | Phoneme::AH
                | Phoneme::AO
                | Phoneme::AW
                | Phoneme::AY
                | Phoneme::EH
                | Phoneme::ER
                | Phoneme::EY
                | Phoneme::IH
                | Phoneme::IY
                | Phoneme::OW
                | Phoneme::OY
                | Phoneme::UH
                | Phoneme::UW
        )
    }

    /// Returns true if this phoneme is a plosive stop consonant.
    pub fn is_stop(self) -> bool {
        matches!(
            self,
            Phoneme::B | Phoneme::D | Phoneme::G | Phoneme::P | Phoneme::T | Phoneme::K
        )
    }

    /// Characteristic consonant burst / frication center frequency in Hz.
    pub fn consonant_burst_frequency(self) -> f32 {
        match self {
            Phoneme::S | Phoneme::Z => 5800.0,
            Phoneme::SH | Phoneme::ZH | Phoneme::CH | Phoneme::JH => 3200.0,
            Phoneme::T | Phoneme::D => 4200.0,
            Phoneme::K | Phoneme::G => 1900.0,
            Phoneme::P | Phoneme::B => 800.0,
            Phoneme::F | Phoneme::V | Phoneme::TH | Phoneme::DH | Phoneme::HH => 2400.0,
            _ => 3000.0,
        }
    }

    /// Returns initial and terminal formant targets for diphthong glides.
    pub fn diphthong_targets(self) -> Option<(FormantTarget, FormantTarget)> {
        match self {
            Phoneme::EY => {
                let mut start = self.acoustic_targets();
                start.f1 = 540.0;
                start.f2 = 1750.0;
                let mut end = self.acoustic_targets();
                end.f1 = 300.0;
                end.f2 = 2200.0;
                Some((start, end))
            }
            Phoneme::AY => {
                let mut start = self.acoustic_targets();
                start.f1 = 750.0;
                start.f2 = 1150.0;
                let mut end = self.acoustic_targets();
                end.f1 = 320.0;
                end.f2 = 2150.0;
                Some((start, end))
            }
            Phoneme::OW => {
                let mut start = self.acoustic_targets();
                start.f1 = 580.0;
                start.f2 = 950.0;
                let mut end = self.acoustic_targets();
                end.f1 = 340.0;
                end.f2 = 900.0;
                Some((start, end))
            }
            Phoneme::AW => {
                let mut start = self.acoustic_targets();
                start.f1 = 750.0;
                start.f2 = 1200.0;
                let mut end = self.acoustic_targets();
                end.f1 = 350.0;
                end.f2 = 950.0;
                Some((start, end))
            }
            Phoneme::OY => {
                let mut start = self.acoustic_targets();
                start.f1 = 580.0;
                start.f2 = 950.0;
                let mut end = self.acoustic_targets();
                end.f1 = 320.0;
                end.f2 = 2150.0;
                Some((start, end))
            }
            _ => None,
        }
    }

    /// Return standard acoustic formant frequencies with accent-specific adaptations.
    pub fn acoustic_targets_with_accent(self, accent: VocalAccent) -> FormantTarget {
        let mut target = self.acoustic_targets();
        match accent {
            VocalAccent::GeneralAmerican => target,
            VocalAccent::ReceivedPronunciation => {
                match self {
                    Phoneme::AA => {
                        target.f1 *= 0.92;
                        target.f2 *= 0.95;
                    }
                    Phoneme::AE => {
                        target.f1 *= 0.90;
                        target.f2 *= 1.05;
                    }
                    Phoneme::AO => {
                        target.f1 *= 0.88;
                        target.f2 *= 0.90;
                    }
                    Phoneme::ER => {
                        target.f3 *= 1.25;
                    }
                    _ => {}
                }
                target
            }
            VocalAccent::International => {
                if self.is_vowel() {
                    target.f1 = target.f1 * 0.85 + 500.0 * 0.15;
                    target.f2 = target.f2 * 0.85 + 1500.0 * 0.15;
                }
                target
            }
            VocalAccent::Australian => {
                match self {
                    Phoneme::AE => {
                        target.f1 *= 0.84;
                        target.f2 *= 1.08;
                    }
                    Phoneme::AA => {
                        target.f1 *= 0.90;
                        target.f2 *= 0.88;
                    }
                    Phoneme::IY => {
                        target.f1 *= 1.25;
                        target.f2 *= 0.93;
                    }
                    Phoneme::ER => {
                        target.f3 *= 1.20;
                    }
                    _ => {}
                }
                target
            }
            VocalAccent::IndianEnglish => {
                match self {
                    Phoneme::T | Phoneme::D | Phoneme::N | Phoneme::L => {
                        target.f3 *= 0.82;
                    }
                    Phoneme::EY => {
                        target.f1 = 500.0;
                        target.f2 = 1850.0;
                    }
                    Phoneme::OW => {
                        target.f1 = 500.0;
                        target.f2 = 950.0;
                    }
                    Phoneme::W => {
                        target.f2 *= 1.15;
                    }
                    Phoneme::V => {
                        target.friction_amp *= 0.6;
                    }
                    _ => {}
                }
                target
            }
            VocalAccent::EastAsian => {
                if self.is_vowel() {
                    target.f1 = target.f1 * 0.82 + 500.0 * 0.18;
                    target.f2 = target.f2 * 0.82 + 1500.0 * 0.18;
                }
                match self {
                    Phoneme::AE => {
                        target.f1 = 680.0;
                        target.f2 = 1350.0;
                    }
                    Phoneme::UW => {
                        target.f2 = 1350.0;
                    }
                    Phoneme::ER => {
                        target.f3 = 2400.0;
                    }
                    Phoneme::P | Phoneme::T | Phoneme::K | Phoneme::B | Phoneme::D | Phoneme::G => {
                        target.default_duration_ms *= 0.85;
                    }
                    _ => {}
                }
                target
            }
            VocalAccent::SpanishAccented => {
                match self {
                    Phoneme::IH => {
                        target.f1 = 310.0;
                        target.f2 = 2200.0;
                    }
                    Phoneme::UH => {
                        target.f1 = 340.0;
                        target.f2 = 900.0;
                    }
                    Phoneme::AE => {
                        target.f1 = 700.0;
                        target.f2 = 1250.0;
                    }
                    Phoneme::AO => {
                        target.f1 = 520.0;
                        target.f2 = 950.0;
                    }
                    Phoneme::P | Phoneme::T | Phoneme::K => {
                        target.aspiration_amp = 0.0;
                    }
                    _ => {}
                }
                target
            }
        }
    }

    /// Returns initial and terminal formant targets for diphthong glides with accent adaptation.
    pub fn diphthong_targets_with_accent(
        self,
        accent: VocalAccent,
    ) -> Option<(FormantTarget, FormantTarget)> {
        let (mut start, mut end) = self.diphthong_targets()?;
        match accent {
            VocalAccent::GeneralAmerican => Some((start, end)),
            VocalAccent::ReceivedPronunciation => {
                match self {
                    Phoneme::OW => {
                        start.f2 = 1350.0;
                    }
                    Phoneme::AY => {
                        start.f1 = 700.0;
                        start.f2 = 1250.0;
                    }
                    _ => {}
                }
                Some((start, end))
            }
            VocalAccent::International => {
                start.f1 = start.f1 * 0.90 + 500.0 * 0.10;
                start.f2 = start.f2 * 0.90 + 1500.0 * 0.10;
                end.f1 = end.f1 * 0.90 + 500.0 * 0.10;
                end.f2 = end.f2 * 0.90 + 1500.0 * 0.10;
                Some((start, end))
            }
            VocalAccent::Australian => {
                match self {
                    Phoneme::EY => {
                        start.f1 = 680.0;
                        start.f2 = 1550.0;
                    }
                    Phoneme::OW => {
                        start.f1 = 520.0;
                        start.f2 = 1400.0;
                    }
                    Phoneme::AY => {
                        start.f1 = 750.0;
                        start.f2 = 1100.0;
                    }
                    _ => {}
                }
                Some((start, end))
            }
            VocalAccent::IndianEnglish => {
                match self {
                    Phoneme::EY => {
                        start.f1 = 500.0;
                        start.f2 = 1850.0;
                        end.f1 = 500.0;
                        end.f2 = 1850.0;
                    }
                    Phoneme::OW => {
                        start.f1 = 500.0;
                        start.f2 = 950.0;
                        end.f1 = 500.0;
                        end.f2 = 950.0;
                    }
                    _ => {}
                }
                Some((start, end))
            }
            VocalAccent::EastAsian => {
                start.f1 = start.f1 * 0.85 + 500.0 * 0.15;
                start.f2 = start.f2 * 0.85 + 1500.0 * 0.15;
                end.f1 = end.f1 * 0.85 + 500.0 * 0.15;
                end.f2 = end.f2 * 0.85 + 1500.0 * 0.15;
                Some((start, end))
            }
            VocalAccent::SpanishAccented => {
                match self {
                    Phoneme::EY => {
                        start.f1 = 480.0;
                        start.f2 = 1900.0;
                        end.f1 = 300.0;
                        end.f2 = 2200.0;
                    }
                    Phoneme::OW => {
                        start.f1 = 500.0;
                        start.f2 = 950.0;
                        end.f1 = 340.0;
                        end.f2 = 900.0;
                    }
                    _ => {}
                }
                Some((start, end))
            }
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
            "falcon" => vec![Phoneme::F, Phoneme::AE, Phoneme::L, Phoneme::K, Phoneme::AH, Phoneme::N],
            "jarvis" => vec![Phoneme::JH, Phoneme::AA, Phoneme::R, Phoneme::V, Phoneme::IH, Phoneme::S],
            "alpha" => vec![Phoneme::AE, Phoneme::L, Phoneme::F, Phoneme::AH],
            "bravo" => vec![Phoneme::B, Phoneme::R, Phoneme::AA, Phoneme::V, Phoneme::OW],
            "home" => vec![Phoneme::HH, Phoneme::OW, Phoneme::M],
            "recon" => vec![Phoneme::R, Phoneme::IY, Phoneme::K, Phoneme::AA, Phoneme::N],
            "patrol" => vec![Phoneme::P, Phoneme::AH, Phoneme::T, Phoneme::R, Phoneme::OW, Phoneme::L],
            "position" => vec![Phoneme::P, Phoneme::AH, Phoneme::Z, Phoneme::IH, Phoneme::SH, Phoneme::AH, Phoneme::N],
            "reboot" => vec![Phoneme::R, Phoneme::IY, Phoneme::B, Phoneme::UW, Phoneme::T],
            "engage" => vec![Phoneme::EH, Phoneme::N, Phoneme::G, Phoneme::EY, Phoneme::JH],
            "disengage" => vec![Phoneme::D, Phoneme::IH, Phoneme::S, Phoneme::EH, Phoneme::N, Phoneme::G, Phoneme::EY, Phoneme::JH],
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

/// 2nd-order digital IIR formant resonator with normalized DC unity gain.
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
        let mut res = Self {
            a1: 0.0,
            a2: 0.0,
            b0: 1.0,
            y1: 0.0,
            y2: 0.0,
        };
        res.set_coefficients(freq, bw, sample_rate);
        res
    }

    #[inline(always)]
    fn set_coefficients(&mut self, freq: f32, bw: f32, sample_rate: f32) {
        let f_clamped = freq.clamp(60.0, sample_rate * 0.48);
        let b_clamped = bw.clamp(30.0, 1500.0);
        let r = (-PI * b_clamped / sample_rate).exp();
        let theta = 2.0 * PI * f_clamped / sample_rate;
        self.a1 = -2.0 * r * theta.cos();
        self.a2 = r * r;
        self.b0 = (1.0 - 2.0 * r * theta.cos() + r * r).max(1e-6); // Unity DC gain (1.0)
    }

    #[inline(always)]
    fn process(&mut self, input: f32) -> f32 {
        let out = self.b0 * input - self.a1 * self.y1 - self.a2 * self.y2;
        self.y2 = self.y1;
        self.y1 = out;
        out
    }
}

/// Bandpass resonator for fricative, aspiration, and burst consonants.
#[derive(Debug, Clone, Copy)]
struct BandpassResonator {
    a1: f32,
    a2: f32,
    b0: f32,
    y1: f32,
    y2: f32,
}

impl BandpassResonator {
    fn new(freq: f32, bw: f32, sample_rate: f32) -> Self {
        let mut res = Self {
            a1: 0.0,
            a2: 0.0,
            b0: 0.1,
            y1: 0.0,
            y2: 0.0,
        };
        res.set_coefficients(freq, bw, sample_rate);
        res
    }

    #[inline(always)]
    fn set_coefficients(&mut self, freq: f32, bw: f32, sample_rate: f32) {
        let f_clamped = freq.clamp(200.0, sample_rate * 0.48);
        let b_clamped = bw.clamp(50.0, 3000.0);
        let r = (-PI * b_clamped / sample_rate).exp();
        let theta = 2.0 * PI * f_clamped / sample_rate;
        self.a1 = -2.0 * r * theta.cos();
        self.a2 = r * r;
        self.b0 = (1.0 - r).max(1e-5); // Peak normalized gain
    }

    #[inline(always)]
    fn process(&mut self, input: f32) -> f32 {
        let out = self.b0 * input - self.a1 * self.y1 - self.a2 * self.y2;
        self.y2 = self.y1;
        self.y1 = out;
        out
    }
}

/// Analytical Liljencrants-Fant (LF-1985) parametric glottal flow model.
/// Parameterized by Fant (1995) shape parameter Rd.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LiljencrantsFantPulse {
    pub rd: f32,
    pub tp: f32,
    pub te: f32,
    pub ta: f32,
    pub alpha: f32,
    pub wg: f32,
    pub ee: f32,
    pub epsilon: f32,
}

impl LiljencrantsFantPulse {
    /// Construct an LF pulse model from Fant (1995) shape parameter Rd.
    /// - Rd = 0.5: pressed / tense voice
    /// - Rd = 1.0: modal / normal voice
    /// - Rd = 2.0: lax / breathy voice
    pub fn from_rd(rd: f32) -> Self {
        let rd_clamped = rd.clamp(0.3, 2.7);
        let ra = (-0.01 + 0.048 * rd_clamped).clamp(0.01, 0.12);
        let rk = (0.224 + 0.118 * rd_clamped).clamp(0.20, 0.55);
        let rg = ((0.5 + 1.2 * rk) / (0.11 * rd_clamped / (0.5 + 1.2 * rk) + rk)).clamp(0.7, 1.8);

        let tp = (1.0 / (2.0 * rg)).clamp(0.40, 0.70);
        let te = (tp * (1.0 + rk)).clamp(tp + 0.05, 0.85);
        let ta = ra.clamp(0.01, 0.15);
        let wg = PI / tp;
        let alpha = 0.05 / te;
        let ee = 1.0;
        let epsilon = (1.0 / ta).clamp(5.0, 100.0);

        Self {
            rd: rd_clamped,
            tp,
            te,
            ta,
            alpha,
            wg,
            ee,
            epsilon,
        }
    }

    /// Evaluate glottal flow derivative wave at phase p in [0.0, 1.0).
    #[inline(always)]
    pub fn evaluate(&self, p: f32) -> f32 {
        if p < self.te {
            (self.alpha * p).exp() * (self.wg * p).sin() - 0.35 * (2.0 * self.wg * p).sin()
        } else if p < (self.te + self.ta * 2.0).min(0.95) {
            let dt = p - self.te;
            -self.ee * (-self.epsilon * dt).exp()
        } else {
            0.0
        }
    }
}

/// Advanced Klatt & Liljencrants-Fant (LF) acoustic formant speech synthesizer in pure safe Rust.
pub struct KlattSynthesizer {
    sample_rate: f32,
    f0_base: f32,
    speaking_rate: f32,
    vocal_tract_scale: f32,
    breathiness: f32,
    glottal_rd: f32,
    accent: VocalAccent,
    intonation_contour: IntonationContour,
}

impl KlattSynthesizer {
    /// Construct a new Klatt synthesizer with specified sample rate (e.g., 16000.0).
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            f0_base: 125.0, // Standard male fundamental pitch in Hz
            speaking_rate: 1.0,
            vocal_tract_scale: 1.0,
            breathiness: 0.03,
            glottal_rd: 1.0, // Modal voice
            accent: VocalAccent::GeneralAmerican,
            intonation_contour: IntonationContour::Declarative,
        }
    }

    /// Set fundamental pitch frequency $F_0$ in Hz.
    pub fn set_f0(&mut self, f0: f32) {
        self.f0_base = f0.clamp(60.0, 400.0);
    }

    /// Set speaking rate (speed multiplier, e.g. 0.8 to 1.5).
    pub fn set_speaking_rate(&mut self, rate: f32) {
        self.speaking_rate = rate.clamp(0.5, 2.5);
    }

    /// Set vocal tract length scaling factor (0.8 = deep/long tract, 1.2 = high/short tract).
    pub fn set_vocal_tract_scale(&mut self, scale: f32) {
        self.vocal_tract_scale = scale.clamp(0.6, 1.5);
    }

    /// Set vocal aspiration breathiness level (0.0 to 0.4).
    pub fn set_breathiness(&mut self, breathiness: f32) {
        self.breathiness = breathiness.clamp(0.0, 0.4);
    }

    /// Set Liljencrants-Fant glottal shape factor Rd (0.3 to 2.7).
    pub fn set_glottal_rd(&mut self, rd: f32) {
        self.glottal_rd = rd.clamp(0.3, 2.7);
    }

    /// Set regional vocal accent.
    pub fn set_accent(&mut self, accent: VocalAccent) {
        self.accent = accent;
    }

    /// Return active regional vocal accent.
    pub fn accent(&self) -> VocalAccent {
        self.accent
    }

    /// Set macro-prosodic pitch intonation contour.
    pub fn set_intonation_contour(&mut self, contour: IntonationContour) {
        self.intonation_contour = contour;
    }

    /// Return active macro-prosodic pitch intonation contour.
    pub fn intonation_contour(&self) -> IntonationContour {
        self.intonation_contour
    }

    /// Synthesize continuous raw audio samples from a list of phoneme segments.
    pub fn synthesize(&self, segments: &[PhonemeSegment]) -> Vec<f32> {
        if segments.is_empty() {
            return Vec::new();
        }

        // 1. Calculate sample lengths per segment with speed scaling
        let mut seg_spans = Vec::with_capacity(segments.len());
        let mut total_samples = 0usize;
        for s in segments {
            let dur_ms = (s.duration_ms / self.speaking_rate).max(20.0);
            let samples = ((dur_ms / 1000.0) * self.sample_rate).max(16.0) as usize;
            let start = total_samples;
            let end = total_samples + samples;
            seg_spans.push((s, start, end));
            total_samples = end;
        }

        if total_samples == 0 {
            return Vec::new();
        }

        // 2. Continuous parameter trajectory vectors
        let mut f1_traj = vec![500.0f32; total_samples];
        let mut f2_traj = vec![1500.0f32; total_samples];
        let mut f3_traj = vec![2500.0f32; total_samples];
        let mut b1_traj = vec![100.0f32; total_samples];
        let mut b2_traj = vec![120.0f32; total_samples];
        let mut b3_traj = vec![180.0f32; total_samples];
        let mut voicing_traj = vec![0.0f32; total_samples];
        let mut fric_traj = vec![0.0f32; total_samples];
        let mut fric_fc_traj = vec![3000.0f32; total_samples];
        let mut burst_amp_traj = vec![0.0f32; total_samples];
        let mut burst_fc_traj = vec![3000.0f32; total_samples];
        let mut stress_traj = vec![0.0f32; total_samples];

        let scale = self.vocal_tract_scale;

        // Populate segment nominal values
        for &(seg, start, end) in &seg_spans {
            let len = end - start;
            let targets = seg.phoneme.acoustic_targets_with_accent(self.accent);
            let diph = seg.phoneme.diphthong_targets_with_accent(self.accent);

            for i in 0..len {
                let idx = start + i;
                let progress = (i as f32) / (len as f32);

                if let Some((start_target, end_target)) = diph {
                    // Diphthong continuous glide
                    f1_traj[idx] = (start_target.f1 + progress * (end_target.f1 - start_target.f1)) * scale;
                    f2_traj[idx] = (start_target.f2 + progress * (end_target.f2 - start_target.f2)) * scale;
                    f3_traj[idx] = (start_target.f3 + progress * (end_target.f3 - start_target.f3)) * scale;
                } else {
                    f1_traj[idx] = targets.f1 * scale;
                    f2_traj[idx] = targets.f2 * scale;
                    f3_traj[idx] = targets.f3 * scale;
                }

                b1_traj[idx] = targets.b1;
                b2_traj[idx] = targets.b2;
                b3_traj[idx] = targets.b3;
                stress_traj[idx] = if seg.stress > 0 { 1.0 } else { 0.0 };

                if seg.phoneme.is_stop() {
                    // Stop consonants: closure (silence) for 65%, burst for 15%, release for 20%
                    let burst_start = (len as f32 * 0.65) as usize;
                    let burst_end = (len as f32 * 0.80) as usize;
                    if i < burst_start {
                        voicing_traj[idx] = targets.voicing_amp * 0.3; // Low voicing bar for voiced stops
                        fric_traj[idx] = 0.0;
                    } else if i < burst_end {
                        voicing_traj[idx] = 0.0;
                        burst_amp_traj[idx] = 0.75;
                        burst_fc_traj[idx] = seg.phoneme.consonant_burst_frequency();
                    } else {
                        voicing_traj[idx] = targets.voicing_amp * 0.5;
                        fric_traj[idx] = 0.0;
                    }
                } else {
                    voicing_traj[idx] = targets.voicing_amp;
                    fric_traj[idx] = targets.friction_amp;
                    fric_fc_traj[idx] = seg.phoneme.consonant_burst_frequency();
                }
            }
        }

        // 3. Coarticulation Smoothing across segment boundaries (30 ms window)
        let smooth_samples = ((0.030 * self.sample_rate) as usize).min(total_samples / 4);
        for k in 1..seg_spans.len() {
            let boundary = seg_spans[k].1;
            let w_start = boundary.saturating_sub(smooth_samples / 2);
            let w_end = (boundary + smooth_samples / 2).min(total_samples);
            let w_len = w_end - w_start;

            if w_len > 1 {
                let f1_a = f1_traj[w_start];
                let f1_b = f1_traj[w_end - 1];
                let f2_a = f2_traj[w_start];
                let f2_b = f2_traj[w_end - 1];
                let f3_a = f3_traj[w_start];
                let f3_b = f3_traj[w_end - 1];
                let v_a = voicing_traj[w_start];
                let v_b = voicing_traj[w_end - 1];

                for j in 0..w_len {
                    let tau = (j as f32) / (w_len as f32);
                    let alpha = 0.5 * (1.0 - (PI * tau).cos()); // S-curve fade
                    let idx = w_start + j;
                    f1_traj[idx] = f1_a + alpha * (f1_b - f1_a);
                    f2_traj[idx] = f2_a + alpha * (f2_b - f2_a);
                    f3_traj[idx] = f3_a + alpha * (f3_b - f3_a);
                    voicing_traj[idx] = v_a + alpha * (v_b - v_a);
                }
            }
        }

        // 4. Cascade Vocal Tract Resonators (Continuous state across entire utterance)
        let mut res1 = FormantResonator::new(f1_traj[0], b1_traj[0], self.sample_rate);
        let mut res2 = FormantResonator::new(f2_traj[0], b2_traj[0], self.sample_rate);
        let mut res3 = FormantResonator::new(f3_traj[0], b3_traj[0], self.sample_rate);
        let mut res4 = FormantResonator::new(3500.0 * scale, 280.0, self.sample_rate);
        let mut res_fric = BandpassResonator::new(fric_fc_traj[0], 1200.0, self.sample_rate);
        let mut res_burst = BandpassResonator::new(3000.0, 1000.0, self.sample_rate);

        let lf_pulse = LiljencrantsFantPulse::from_rd(self.glottal_rd);
        let mut glottal_phase = 0.0f32;
        let mut noise_state = 12345678901234567u64;
        let mut y_rad_prev = 0.0f32;
        let mut audio = Vec::with_capacity(total_samples);

        for n in 0..total_samples {
            let t = (n as f32) / self.sample_rate;
            let t_norm = (n as f32) / (total_samples as f32);

            // Natural Macro-Prosody Intonation Arc:
            let intonation = self.intonation_contour.evaluate(t_norm, t);
            let stress_boost = 1.0 + 0.18 * stress_traj[n];
            let micro_jitter = 1.0 + 0.0035 * (2.0 * PI * 6.1 * t).sin();
            let f0 = self.f0_base * intonation * stress_boost * micro_jitter;
            let f0_step = (f0 / self.sample_rate) * 2.0 * PI;

            // Glottal phase advancement
            glottal_phase += f0_step;
            if glottal_phase >= 2.0 * PI {
                glottal_phase -= 2.0 * PI;
            }

            // Liljencrants-Fant (LF) parametric glottal flow excitation
            let p = glottal_phase / (2.0 * PI);
            let glottal_wave = lf_pulse.evaluate(p);

            // 64-bit LCG white noise generator
            noise_state = noise_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let unvoiced_noise = (((noise_state >> 33) as f32) / (u32::MAX as f32) - 0.5) * 2.0;

            // Voicing source excitation with aspiration breathiness
            let v_amp = voicing_traj[n];
            let asp_amp = self.breathiness;
            // Update vocal tract resonators dynamically at 0.5ms control rate (every 8 samples)
            if n % 8 == 0 {
                res1.set_coefficients(f1_traj[n], b1_traj[n], self.sample_rate);
                res2.set_coefficients(f2_traj[n], b2_traj[n], self.sample_rate);
                res3.set_coefficients(f3_traj[n], b3_traj[n], self.sample_rate);
                if fric_traj[n] > 0.001 {
                    res_fric.set_coefficients(fric_fc_traj[n], 1200.0, self.sample_rate);
                }
                if burst_amp_traj[n] > 0.001 {
                    res_burst.set_coefficients(burst_fc_traj[n], 900.0, self.sample_rate);
                }
            }

            let excitation = v_amp * glottal_wave + 0.02 * v_amp * unvoiced_noise + asp_amp * unvoiced_noise;

            // Cascade vocal tract processing: R1 -> R2 -> R3 -> R4
            let r1 = res1.process(excitation);
            let r2 = res2.process(r1);
            let r3 = res3.process(r2);
            let r4 = res4.process(r3);

            // Lip radiation impedance filter: 1 - 0.95 z^-1
            let vocal_rad = r4 - 0.95 * y_rad_prev;
            y_rad_prev = r4;

            // Fricative branch
            let fric_amp = fric_traj[n];
            let fric_val = if fric_amp > 0.001 {
                fric_amp * res_fric.process(unvoiced_noise) * 0.45
            } else {
                0.0
            };

            // Stop burst branch
            let burst_amp = burst_amp_traj[n];
            let burst_val = if burst_amp > 0.001 {
                burst_amp * res_burst.process(unvoiced_noise) * 0.65
            } else {
                0.0
            };

            let mixed = 0.40 * vocal_rad + fric_val + burst_val;

            // Master envelope fade-in (first 80 samples) and fade-out (last 80 samples)
            let master_env = if n < 80 {
                (n as f32) / 80.0
            } else if n + 80 >= total_samples {
                ((total_samples - n) as f32) / 80.0
            } else {
                1.0
            };

            audio.push(mixed * master_env);
        }

        // Peak normalization to 0.85 (-1.4 dBFS)
        let peak = audio.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
        if peak > 0.001 {
            let norm_gain = 0.85 / peak;
            for sample in &mut audio {
                *sample *= norm_gain;
            }
        }

        audio
    }
}

/// Configuration specifying parameter variations for generating diverse synthetic acoustic training exemplars.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExemplarVariationConfig {
    /// Pitch frequencies in Hz to cycle across (e.g. male, female, youth, adult).
    pub pitch_frequencies: Vec<f32>,
    /// Speaking rate multipliers (e.g. 0.85 deliberate, 1.0 standard, 1.22 rapid).
    pub speaking_rates: Vec<f32>,
    /// Vocal tract length scale factors (e.g. 0.90 youth/female, 1.0 modal, 1.12 deep adult).
    pub vocal_tract_scales: Vec<f32>,
    /// Liljencrants-Fant glottal Rd shape factors (e.g. 0.7 pressed/authoritative, 1.0 modal, 1.4 relaxed).
    pub glottal_rd_factors: Vec<f32>,
    /// Breathiness aspiration amplitudes (e.g. 0.02 crisp, 0.05 breathy).
    pub breathiness_levels: Vec<f32>,
    /// Regional vocal accents to cycle across (e.g. General American, British RP, International).
    pub accents: Vec<VocalAccent>,
    /// Pitch intonation contours to cycle across (e.g. Declarative, Command, Interrogative).
    pub contours: Vec<IntonationContour>,
}

impl Default for ExemplarVariationConfig {
    fn default() -> Self {
        Self {
            pitch_frequencies: vec![105.0, 135.0, 175.0, 215.0],
            speaking_rates: vec![0.88, 1.0, 1.18],
            vocal_tract_scales: vec![0.92, 1.0, 1.10],
            glottal_rd_factors: vec![0.75, 1.0, 1.35],
            breathiness_levels: vec![0.02, 0.05],
            accents: vec![
                VocalAccent::GeneralAmerican,
                VocalAccent::ReceivedPronunciation,
                VocalAccent::International,
            ],
            contours: vec![
                IntonationContour::Declarative,
                IntonationContour::AuthoritativeCommand,
                IntonationContour::Interrogative,
            ],
        }
    }
}

/// Metadata record for an instantiated synthetic training exemplar.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SyntheticExemplarMetadata {
    /// Synthesized raw audio samples normalized to [-1.0, 1.0].
    pub audio: Vec<f32>,
    /// Synthesized text phrase.
    pub phrase: String,
    /// Base fundamental pitch in Hz.
    pub pitch_f0: f32,
    /// Speaking rate multiplier.
    pub speaking_rate: f32,
    /// Vocal tract length scale.
    pub vocal_tract_scale: f32,
    /// Liljencrants-Fant glottal shape factor Rd.
    pub glottal_rd: f32,
    /// Aspiration breathiness level.
    pub breathiness: f32,
    /// Regional vocal accent.
    pub accent: VocalAccent,
    /// Prosodic intonation contour.
    pub contour: IntonationContour,
}

/// Generator of diverse, realistic acoustic exemplars for zero-shot wake-word enrollment and training datasets.
#[derive(Debug, Clone)]
pub struct SyntheticExemplarGenerator {
    sample_rate: f32,
    config: ExemplarVariationConfig,
}

impl SyntheticExemplarGenerator {
    /// Construct a new synthetic exemplar generator with default variation profiles.
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            config: ExemplarVariationConfig::default(),
        }
    }

    /// Construct with custom variation profiles.
    pub fn with_config(sample_rate: f32, config: ExemplarVariationConfig) -> Self {
        Self {
            sample_rate,
            config,
        }
    }

    /// Access reference to variation configuration.
    pub fn config(&self) -> &ExemplarVariationConfig {
        &self.config
    }

    /// Access mutable reference to variation configuration.
    pub fn config_mut(&mut self) -> &mut ExemplarVariationConfig {
        &mut self.config
    }

    /// Generate `count` diverse synthetic audio exemplars with comprehensive acoustic metadata.
    pub fn generate_exemplars_with_metadata(
        &self,
        phrase: &str,
        count: usize,
    ) -> Vec<SyntheticExemplarMetadata> {
        if count == 0 {
            return Vec::new();
        }

        let segments = G2pEngine::text_to_phonemes(phrase);
        if segments.is_empty() {
            return Vec::new();
        }

        let n_pitch = self.config.pitch_frequencies.len().max(1);
        let n_speed = self.config.speaking_rates.len().max(1);
        let n_scale = self.config.vocal_tract_scales.len().max(1);
        let n_rd = self.config.glottal_rd_factors.len().max(1);
        let n_breath = self.config.breathiness_levels.len().max(1);
        let n_accent = self.config.accents.len().max(1);
        let n_contour = self.config.contours.len().max(1);

        let mut exemplars = Vec::with_capacity(count);

        for i in 0..count {
            let mut synth = KlattSynthesizer::new(self.sample_rate);
            let f0 = self.config.pitch_frequencies[i % n_pitch];
            let rate = self.config.speaking_rates[(i / n_pitch) % n_speed];
            let scale = self.config.vocal_tract_scales[(i / (n_pitch * n_speed)) % n_scale];
            let rd = self.config.glottal_rd_factors[(i / (n_pitch * n_speed * n_scale)) % n_rd];
            let breath = self.config.breathiness_levels
                [(i / (n_pitch * n_speed * n_scale * n_rd)) % n_breath];
            let accent = self.config.accents
                [(i / (n_pitch * n_speed * n_scale * n_rd * n_breath)) % n_accent];
            let contour = self.config.contours
                [(i / (n_pitch * n_speed * n_scale * n_rd * n_breath * n_accent)) % n_contour];

            synth.set_f0(f0);
            synth.set_speaking_rate(rate);
            synth.set_vocal_tract_scale(scale);
            synth.set_glottal_rd(rd);
            synth.set_breathiness(breath);
            synth.set_accent(accent);
            synth.set_intonation_contour(contour);

            let audio = synth.synthesize(&segments);
            if !audio.is_empty() {
                exemplars.push(SyntheticExemplarMetadata {
                    audio,
                    phrase: phrase.to_string(),
                    pitch_f0: f0,
                    speaking_rate: rate,
                    vocal_tract_scale: scale,
                    glottal_rd: rd,
                    breathiness: breath,
                    accent,
                    contour,
                });
            }
        }

        exemplars
    }

    /// Generate `count` diverse synthetic audio exemplars for the given phrase text.
    pub fn generate_exemplars(&self, phrase: &str, count: usize) -> Vec<Vec<f32>> {
        self.generate_exemplars_with_metadata(phrase, count)
            .into_iter()
            .map(|meta| meta.audio)
            .collect()
    }

    /// Generate `count` diverse synthetic audio exemplars directly from a custom sequence of phoneme segments.
    pub fn generate_exemplars_from_segments(
        &self,
        segments: &[PhonemeSegment],
        count: usize,
    ) -> Vec<Vec<f32>> {
        if count == 0 || segments.is_empty() {
            return Vec::new();
        }

        let n_pitch = self.config.pitch_frequencies.len().max(1);
        let n_speed = self.config.speaking_rates.len().max(1);
        let n_scale = self.config.vocal_tract_scales.len().max(1);
        let n_rd = self.config.glottal_rd_factors.len().max(1);
        let n_breath = self.config.breathiness_levels.len().max(1);
        let n_accent = self.config.accents.len().max(1);
        let n_contour = self.config.contours.len().max(1);

        let mut exemplars = Vec::with_capacity(count);

        for i in 0..count {
            let mut synth = KlattSynthesizer::new(self.sample_rate);
            let f0 = self.config.pitch_frequencies[i % n_pitch];
            let rate = self.config.speaking_rates[(i / n_pitch) % n_speed];
            let scale = self.config.vocal_tract_scales[(i / (n_pitch * n_speed)) % n_scale];
            let rd = self.config.glottal_rd_factors[(i / (n_pitch * n_speed * n_scale)) % n_rd];
            let breath = self.config.breathiness_levels
                [(i / (n_pitch * n_speed * n_scale * n_rd)) % n_breath];
            let accent = self.config.accents
                [(i / (n_pitch * n_speed * n_scale * n_rd * n_breath)) % n_accent];
            let contour = self.config.contours
                [(i / (n_pitch * n_speed * n_scale * n_rd * n_breath * n_accent)) % n_contour];

            synth.set_f0(f0);
            synth.set_speaking_rate(rate);
            synth.set_vocal_tract_scale(scale);
            synth.set_glottal_rd(rd);
            synth.set_breathiness(breath);
            synth.set_accent(accent);
            synth.set_intonation_contour(contour);

            let audio = synth.synthesize(segments);
            if !audio.is_empty() {
                exemplars.push(audio);
            }
        }

        exemplars
    }
}

/// Encodes mono floating-point audio samples into a standard 16-bit 1-channel PCM WAV byte stream.
pub fn encode_wav_16bit(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let num_samples = samples.len() as u32;
    let bytes_per_sample = 2u16;
    let num_channels = 1u16;
    let byte_rate = sample_rate * (num_channels as u32) * (bytes_per_sample as u32);
    let block_align = num_channels * bytes_per_sample;
    let subchunk2_size = num_samples * (bytes_per_sample as u32);
    let chunk_size = 36 + subchunk2_size;

    let mut buf = Vec::with_capacity(44 + subchunk2_size as usize);

    // RIFF header
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&chunk_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    // "fmt " subchunk
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // Subchunk1Size (16 for PCM)
    buf.extend_from_slice(&1u16.to_le_bytes());  // AudioFormat (1 = PCM)
    buf.extend_from_slice(&num_channels.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&16u16.to_le_bytes()); // BitsPerSample

    // "data" subchunk
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&subchunk2_size.to_le_bytes());

    // 16-bit signed PCM audio samples
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let sample_i16 = (clamped * 32767.0).round() as i16;
        buf.extend_from_slice(&sample_i16.to_le_bytes());
    }

    buf
}

/// Encodes and writes mono floating-point audio samples to a standard 16-bit PCM WAV file.
pub fn write_wav_file(
    path: impl AsRef<Path>,
    samples: &[f32],
    sample_rate: u32,
) -> std::io::Result<()> {
    let bytes = encode_wav_16bit(samples, sample_rate);
    std::fs::write(path, bytes)
}

