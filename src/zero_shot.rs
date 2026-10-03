//! Zero-shot wake-word enrollment via cross-attention phonetic-acoustic alignment,
//! multi-lingual Grapheme-to-Phoneme (G2P), Phonetic Posteriorgram (PPG) embedding space,
//! and automated minimal-pair phonetic foil discrimination margin calibration.

#![deny(unsafe_code)]

use crate::dtw::{ConfusionMatrix, DtwMatcher};
use crate::engine::SononEngine;
use crate::phonetic::{
    G2pEngine, KlattSynthesizer, Phoneme, PhonemeSegment, SyntheticExemplarGenerator, VocalAccent,
};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Supported international languages for zero-shot wake-word spotting and G2P.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SupportedLanguage {
    English,
    Spanish,
    French,
    German,
    Japanese,
    Mandarin,
}

impl SupportedLanguage {
    /// Canonical ISO 639-1 two-letter language code string.
    pub fn code(&self) -> &'static str {
        match self {
            SupportedLanguage::English => "en",
            SupportedLanguage::Spanish => "es",
            SupportedLanguage::French => "fr",
            SupportedLanguage::German => "de",
            SupportedLanguage::Japanese => "ja",
            SupportedLanguage::Mandarin => "zh",
        }
    }

    /// Human-readable language name.
    pub fn name(&self) -> &'static str {
        match self {
            SupportedLanguage::English => "English",
            SupportedLanguage::Spanish => "Spanish",
            SupportedLanguage::French => "French",
            SupportedLanguage::German => "German",
            SupportedLanguage::Japanese => "Japanese",
            SupportedLanguage::Mandarin => "Mandarin Chinese",
        }
    }
}

/// Category of phonetic negative distractor (foil) for discrimination margin evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FoilType {
    /// Consonant voicing, place, or manner minimal pair (e.g. "take" vs "make", "bake", "shake").
    ConsonantMinimalPair,
    /// Vowel formant shift minimal pair (e.g. "take" vs "tick", "tuck", "took").
    VowelFormantShift,
    /// Lexical suffix or prefix boundary substitution (e.g. "take off" vs "take up", "take out").
    LexicalBoundary,
    /// Operational robotics command distractor from an alternate language.
    CrossLingualDistractor,
}

/// Generated phonetic foil candidate for discrimination margin calibration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhoneticFoil {
    /// Orthographic foil text string.
    pub foil_text: String,
    /// Source language of the foil.
    pub language: SupportedLanguage,
    /// Minimal-pair category.
    pub foil_type: FoilType,
    /// Corresponding phoneme sequence.
    pub phonemes: Vec<Phoneme>,
}

/// Multi-lingual Grapheme-to-Phoneme converter for zero-shot wake-word enrollment.
#[derive(Debug, Clone)]
pub struct MultiLingualG2p;

impl MultiLingualG2p {
    /// Convert phrase text in the specified language to timed phoneme segments.
    pub fn text_to_phonemes(text: &str, language: SupportedLanguage) -> Vec<PhonemeSegment> {
        match language {
            SupportedLanguage::English => G2pEngine::text_to_phonemes(text),
            SupportedLanguage::Spanish => Self::spanish_to_phonemes(text),
            SupportedLanguage::French => Self::french_to_phonemes(text),
            SupportedLanguage::German => Self::german_to_phonemes(text),
            SupportedLanguage::Japanese => Self::japanese_to_phonemes(text),
            SupportedLanguage::Mandarin => Self::mandarin_to_phonemes(text),
        }
    }

    /// Spanish phonetic parser with flight control lexicon and deterministic phonetic rules.
    fn spanish_to_phonemes(text: &str) -> Vec<PhonemeSegment> {
        let mut segments = Vec::new();
        let words: Vec<&str> = text
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphabetic()))
            .filter(|w| !w.is_empty())
            .collect();

        for (w_idx, word) in words.iter().enumerate() {
            let lower = word.to_lowercase();
            let phonemes = match lower.as_str() {
                "despegar" => vec![Phoneme::D, Phoneme::EH, Phoneme::S, Phoneme::P, Phoneme::EH, Phoneme::G, Phoneme::AA, Phoneme::R],
                "aterrizar" => vec![Phoneme::AA, Phoneme::T, Phoneme::EH, Phoneme::R, Phoneme::IY, Phoneme::Z, Phoneme::AA, Phoneme::R],
                "alto" => vec![Phoneme::AA, Phoneme::L, Phoneme::T, Phoneme::OW],
                "parar" => vec![Phoneme::P, Phoneme::AA, Phoneme::R, Phoneme::AA, Phoneme::R],
                "emergencia" => vec![Phoneme::EH, Phoneme::M, Phoneme::EH, Phoneme::R, Phoneme::HH, Phoneme::EH, Phoneme::N, Phoneme::S, Phoneme::IY, Phoneme::AA],
                "regresar" => vec![Phoneme::R, Phoneme::EH, Phoneme::G, Phoneme::R, Phoneme::EH, Phoneme::S, Phoneme::AA, Phoneme::R],
                "subir" => vec![Phoneme::S, Phoneme::UW, Phoneme::B, Phoneme::IY, Phoneme::R],
                "bajar" => vec![Phoneme::B, Phoneme::AA, Phoneme::HH, Phoneme::AA, Phoneme::R],
                "adelante" => vec![Phoneme::AA, Phoneme::D, Phoneme::EH, Phoneme::L, Phoneme::AA, Phoneme::N, Phoneme::T, Phoneme::EH],
                "atras" => vec![Phoneme::AA, Phoneme::T, Phoneme::R, Phoneme::AA, Phoneme::S],
                _ => Self::rule_based_spanish(&lower),
            };

            for (p_idx, &p) in phonemes.iter().enumerate() {
                let targets = p.acoustic_targets();
                let stress = if p_idx == 0 || p == Phoneme::AA || p == Phoneme::EH || p == Phoneme::OW { 1 } else { 0 };
                segments.push(PhonemeSegment {
                    phoneme: p,
                    duration_ms: targets.default_duration_ms,
                    stress,
                });
            }

            if w_idx + 1 < words.len() {
                segments.push(PhonemeSegment {
                    phoneme: Phoneme::SIL,
                    duration_ms: 35.0,
                    stress: 0,
                });
            }
        }

        segments
    }

    fn rule_based_spanish(word: &str) -> Vec<Phoneme> {
        let chars: Vec<char> = word.chars().collect();
        let len = chars.len();
        let mut phonemes = Vec::new();
        let mut i = 0;

        while i < len {
            if i + 2 <= len {
                let pair: String = chars[i..i + 2].iter().collect();
                match pair.as_str() {
                    "ch" => { phonemes.push(Phoneme::CH); i += 2; continue; }
                    "ll" => { phonemes.push(Phoneme::Y); i += 2; continue; }
                    "rr" => { phonemes.push(Phoneme::R); i += 2; continue; }
                    "qu" => { phonemes.push(Phoneme::K); i += 2; continue; }
                    "gu" if i + 2 < len && (chars[i + 2] == 'e' || chars[i + 2] == 'i') => {
                        phonemes.push(Phoneme::G); i += 2; continue;
                    }
                    _ => {}
                }
            }

            let c = chars[i];
            match c {
                'a' | 'á' => phonemes.push(Phoneme::AA),
                'e' | 'é' => phonemes.push(Phoneme::EH),
                'i' | 'í' | 'y' => phonemes.push(Phoneme::IY),
                'o' | 'ó' => phonemes.push(Phoneme::OW),
                'u' | 'ú' | 'ü' => phonemes.push(Phoneme::UW),
                'b' | 'v' => phonemes.push(Phoneme::B),
                'c' => {
                    if i + 1 < len && (chars[i + 1] == 'e' || chars[i + 1] == 'i') {
                        phonemes.push(Phoneme::S);
                    } else {
                        phonemes.push(Phoneme::K);
                    }
                }
                'd' => phonemes.push(Phoneme::D),
                'f' => phonemes.push(Phoneme::F),
                'g' => {
                    if i + 1 < len && (chars[i + 1] == 'e' || chars[i + 1] == 'i') {
                        phonemes.push(Phoneme::HH);
                    } else {
                        phonemes.push(Phoneme::G);
                    }
                }
                'h' => { /* Silent h in Spanish */ }
                'j' => phonemes.push(Phoneme::HH),
                'k' => phonemes.push(Phoneme::K),
                'l' => phonemes.push(Phoneme::L),
                'm' => phonemes.push(Phoneme::M),
                'n' | 'ñ' => phonemes.push(Phoneme::N),
                'p' => phonemes.push(Phoneme::P),
                'r' => phonemes.push(Phoneme::R),
                's' | 'z' => phonemes.push(Phoneme::S),
                't' => phonemes.push(Phoneme::T),
                'w' => phonemes.push(Phoneme::W),
                'x' => { phonemes.push(Phoneme::K); phonemes.push(Phoneme::S); }
                _ => {}
            }
            i += 1;
        }

        if phonemes.is_empty() {
            vec![Phoneme::AA]
        } else {
            phonemes
        }
    }

    /// French phonetic parser with flight control lexicon and rule-based phoneme synthesis.
    fn french_to_phonemes(text: &str) -> Vec<PhonemeSegment> {
        let mut segments = Vec::new();
        let words: Vec<&str> = text
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphabetic()))
            .filter(|w| !w.is_empty())
            .collect();

        for (w_idx, word) in words.iter().enumerate() {
            let lower = word.to_lowercase();
            let phonemes = match lower.as_str() {
                "decoller" => vec![Phoneme::D, Phoneme::EH, Phoneme::K, Phoneme::AO, Phoneme::L, Phoneme::EY],
                "atterrir" => vec![Phoneme::AA, Phoneme::T, Phoneme::EH, Phoneme::R, Phoneme::IY, Phoneme::R],
                "arret" => vec![Phoneme::AA, Phoneme::R, Phoneme::EH],
                "urgence" => vec![Phoneme::UW, Phoneme::R, Phoneme::ZH, Phoneme::AA, Phoneme::N, Phoneme::S],
                "retour" => vec![Phoneme::R, Phoneme::EH, Phoneme::T, Phoneme::UW, Phoneme::R],
                "monter" => vec![Phoneme::M, Phoneme::AO, Phoneme::N, Phoneme::T, Phoneme::EY],
                "descendre" => vec![Phoneme::D, Phoneme::EH, Phoneme::S, Phoneme::AA, Phoneme::N, Phoneme::D, Phoneme::R],
                "gauche" => vec![Phoneme::G, Phoneme::OW, Phoneme::SH],
                "droite" => vec![Phoneme::D, Phoneme::R, Phoneme::W, Phoneme::AA, Phoneme::T],
                _ => Self::rule_based_french(&lower),
            };

            for (p_idx, &p) in phonemes.iter().enumerate() {
                let targets = p.acoustic_targets();
                let stress = if p_idx + 1 == phonemes.len() { 1 } else { 0 }; // French oxytonic stress
                segments.push(PhonemeSegment {
                    phoneme: p,
                    duration_ms: targets.default_duration_ms,
                    stress,
                });
            }

            if w_idx + 1 < words.len() {
                segments.push(PhonemeSegment {
                    phoneme: Phoneme::SIL,
                    duration_ms: 35.0,
                    stress: 0,
                });
            }
        }

        segments
    }

    fn rule_based_french(word: &str) -> Vec<Phoneme> {
        let chars: Vec<char> = word.chars().collect();
        let len = chars.len();
        let mut phonemes = Vec::new();
        let mut i = 0;

        while i < len {
            if i + 3 <= len {
                let triplet: String = chars[i..i + 3].iter().collect();
                if triplet == "eau" {
                    phonemes.push(Phoneme::OW);
                    i += 3;
                    continue;
                }
            }

            if i + 2 <= len {
                let pair: String = chars[i..i + 2].iter().collect();
                match pair.as_str() {
                    "ou" => { phonemes.push(Phoneme::UW); i += 2; continue; }
                    "ch" => { phonemes.push(Phoneme::SH); i += 2; continue; }
                    "ai" | "ei" => { phonemes.push(Phoneme::EH); i += 2; continue; }
                    "au" => { phonemes.push(Phoneme::OW); i += 2; continue; }
                    "qu" => { phonemes.push(Phoneme::K); i += 2; continue; }
                    "on" => { phonemes.push(Phoneme::AO); phonemes.push(Phoneme::N); i += 2; continue; }
                    "an" | "en" => { phonemes.push(Phoneme::AA); phonemes.push(Phoneme::N); i += 2; continue; }
                    _ => {}
                }
            }

            let c = chars[i];
            match c {
                'a' | 'à' | 'â' => phonemes.push(Phoneme::AA),
                'e' | 'é' | 'è' | 'ê' | 'ë' => phonemes.push(Phoneme::EH),
                'i' | 'î' | 'ï' | 'y' => phonemes.push(Phoneme::IY),
                'o' | 'ô' => phonemes.push(Phoneme::OW),
                'u' | 'û' | 'ù' => phonemes.push(Phoneme::UW),
                'b' => phonemes.push(Phoneme::B),
                'c' | 'ç' => {
                    if c == 'ç' || (i + 1 < len && (chars[i + 1] == 'e' || chars[i + 1] == 'i')) {
                        phonemes.push(Phoneme::S);
                    } else {
                        phonemes.push(Phoneme::K);
                    }
                }
                'd' => phonemes.push(Phoneme::D),
                'f' => phonemes.push(Phoneme::F),
                'g' => {
                    if i + 1 < len && (chars[i + 1] == 'e' || chars[i + 1] == 'i') {
                        phonemes.push(Phoneme::ZH);
                    } else {
                        phonemes.push(Phoneme::G);
                    }
                }
                'j' => phonemes.push(Phoneme::ZH),
                'k' => phonemes.push(Phoneme::K),
                'l' => phonemes.push(Phoneme::L),
                'm' => phonemes.push(Phoneme::M),
                'n' => phonemes.push(Phoneme::N),
                'p' => phonemes.push(Phoneme::P),
                'r' => phonemes.push(Phoneme::R),
                's' => {
                    if i > 0 && i + 1 < len && is_vowel(chars[i - 1]) && is_vowel(chars[i + 1]) {
                        phonemes.push(Phoneme::Z);
                    } else {
                        phonemes.push(Phoneme::S);
                    }
                }
                't' => phonemes.push(Phoneme::T),
                'v' => phonemes.push(Phoneme::V),
                'z' => phonemes.push(Phoneme::Z),
                _ => {}
            }
            i += 1;
        }

        if phonemes.is_empty() {
            vec![Phoneme::EH]
        } else {
            phonemes
        }
    }

    /// German phonetic parser with flight control lexicon and rule-based phoneme synthesis.
    fn german_to_phonemes(text: &str) -> Vec<PhonemeSegment> {
        let mut segments = Vec::new();
        let words: Vec<&str> = text
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphabetic()))
            .filter(|w| !w.is_empty())
            .collect();

        for (w_idx, word) in words.iter().enumerate() {
            let lower = word.to_lowercase();
            let phonemes = match lower.as_str() {
                "abheben" => vec![Phoneme::AA, Phoneme::P, Phoneme::HH, Phoneme::EY, Phoneme::B, Phoneme::EH, Phoneme::N],
                "landen" => vec![Phoneme::L, Phoneme::AA, Phoneme::N, Phoneme::D, Phoneme::EH, Phoneme::N],
                "halt" => vec![Phoneme::HH, Phoneme::AA, Phoneme::L, Phoneme::T],
                "notlandung" => vec![Phoneme::N, Phoneme::OW, Phoneme::T, Phoneme::L, Phoneme::AA, Phoneme::N, Phoneme::D, Phoneme::UH, Phoneme::NG],
                "zurueck" => vec![Phoneme::T, Phoneme::S, Phoneme::UW, Phoneme::R, Phoneme::IH, Phoneme::K],
                "steigen" => vec![Phoneme::SH, Phoneme::T, Phoneme::AY, Phoneme::G, Phoneme::EH, Phoneme::N],
                "sinken" => vec![Phoneme::Z, Phoneme::IH, Phoneme::NG, Phoneme::K, Phoneme::EH, Phoneme::N],
                "links" => vec![Phoneme::L, Phoneme::IH, Phoneme::NG, Phoneme::K, Phoneme::S],
                "rechts" => vec![Phoneme::R, Phoneme::EH, Phoneme::HH, Phoneme::T, Phoneme::S],
                _ => Self::rule_based_german(&lower),
            };

            for (p_idx, &p) in phonemes.iter().enumerate() {
                let targets = p.acoustic_targets();
                let stress = if p_idx == 0 || p == Phoneme::AA || p == Phoneme::AY || p == Phoneme::OW { 1 } else { 0 };
                segments.push(PhonemeSegment {
                    phoneme: p,
                    duration_ms: targets.default_duration_ms,
                    stress,
                });
            }

            if w_idx + 1 < words.len() {
                segments.push(PhonemeSegment {
                    phoneme: Phoneme::SIL,
                    duration_ms: 35.0,
                    stress: 0,
                });
            }
        }

        segments
    }

    fn rule_based_german(word: &str) -> Vec<Phoneme> {
        let chars: Vec<char> = word.chars().collect();
        let len = chars.len();
        let mut phonemes = Vec::new();
        let mut i = 0;

        while i < len {
            if i + 3 <= len {
                let triplet: String = chars[i..i + 3].iter().collect();
                if triplet == "sch" {
                    phonemes.push(Phoneme::SH);
                    i += 3;
                    continue;
                }
            }

            if i + 2 <= len {
                let pair: String = chars[i..i + 2].iter().collect();
                match pair.as_str() {
                    "ch" => { phonemes.push(Phoneme::HH); i += 2; continue; }
                    "ei" => { phonemes.push(Phoneme::AY); i += 2; continue; }
                    "ie" => { phonemes.push(Phoneme::IY); i += 2; continue; }
                    "eu" | "äu" => { phonemes.push(Phoneme::OY); i += 2; continue; }
                    "au" => { phonemes.push(Phoneme::AW); i += 2; continue; }
                    "sp" if i == 0 => { phonemes.push(Phoneme::SH); phonemes.push(Phoneme::P); i += 2; continue; }
                    "st" if i == 0 => { phonemes.push(Phoneme::SH); phonemes.push(Phoneme::T); i += 2; continue; }
                    "ck" => { phonemes.push(Phoneme::K); i += 2; continue; }
                    "ng" => { phonemes.push(Phoneme::NG); i += 2; continue; }
                    _ => {}
                }
            }

            let c = chars[i];
            match c {
                'a' => phonemes.push(Phoneme::AA),
                'ä' | 'e' => phonemes.push(Phoneme::EH),
                'i' => phonemes.push(Phoneme::IH),
                'o' => phonemes.push(Phoneme::OW),
                'ö' => phonemes.push(Phoneme::ER),
                'u' => phonemes.push(Phoneme::UW),
                'ü' => phonemes.push(Phoneme::UW),
                'b' => phonemes.push(Phoneme::B),
                'd' => phonemes.push(Phoneme::D),
                'f' | 'v' => phonemes.push(Phoneme::F),
                'g' => phonemes.push(Phoneme::G),
                'h' => phonemes.push(Phoneme::HH),
                'j' => phonemes.push(Phoneme::Y),
                'k' => phonemes.push(Phoneme::K),
                'l' => phonemes.push(Phoneme::L),
                'm' => phonemes.push(Phoneme::M),
                'n' => phonemes.push(Phoneme::N),
                'p' => phonemes.push(Phoneme::P),
                'r' => phonemes.push(Phoneme::R),
                's' => {
                    if i + 1 < len && is_vowel(chars[i + 1]) {
                        phonemes.push(Phoneme::Z);
                    } else {
                        phonemes.push(Phoneme::S);
                    }
                }
                't' => phonemes.push(Phoneme::T),
                'w' => phonemes.push(Phoneme::V),
                'z' => { phonemes.push(Phoneme::T); phonemes.push(Phoneme::S); }
                'ß' => phonemes.push(Phoneme::S),
                _ => {}
            }
            i += 1;
        }

        if phonemes.is_empty() {
            vec![Phoneme::AA]
        } else {
            phonemes
        }
    }

    /// Japanese (Romaji / Hepburn) phonetic parser.
    fn japanese_to_phonemes(text: &str) -> Vec<PhonemeSegment> {
        let mut segments = Vec::new();
        let words: Vec<&str> = text
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphabetic()))
            .filter(|w| !w.is_empty())
            .collect();

        for (w_idx, word) in words.iter().enumerate() {
            let lower = word.to_lowercase();
            let phonemes = match lower.as_str() {
                "ritsuriku" => vec![Phoneme::R, Phoneme::IH, Phoneme::T, Phoneme::S, Phoneme::UW, Phoneme::R, Phoneme::IH, Phoneme::K, Phoneme::UW],
                "chakuriku" => vec![Phoneme::CH, Phoneme::AA, Phoneme::K, Phoneme::UW, Phoneme::R, Phoneme::IH, Phoneme::K, Phoneme::UW],
                "tomare" => vec![Phoneme::T, Phoneme::OW, Phoneme::M, Phoneme::AA, Phoneme::R, Phoneme::EH],
                "kinkyuu" => vec![Phoneme::K, Phoneme::IH, Phoneme::NG, Phoneme::K, Phoneme::Y, Phoneme::UW],
                "modore" => vec![Phoneme::M, Phoneme::OW, Phoneme::D, Phoneme::OW, Phoneme::R, Phoneme::EH],
                "ue" => vec![Phoneme::UW, Phoneme::EH],
                "shita" => vec![Phoneme::SH, Phoneme::IH, Phoneme::T, Phoneme::AA],
                "hidari" => vec![Phoneme::HH, Phoneme::IH, Phoneme::D, Phoneme::AA, Phoneme::R, Phoneme::IY],
                "migi" => vec![Phoneme::M, Phoneme::IY, Phoneme::G, Phoneme::IY],
                _ => Self::rule_based_japanese(&lower),
            };

            for (p_idx, &p) in phonemes.iter().enumerate() {
                let targets = p.acoustic_targets();
                let stress = if p_idx == 0 || (p == Phoneme::AA || p == Phoneme::OW) { 1 } else { 0 };
                segments.push(PhonemeSegment {
                    phoneme: p,
                    duration_ms: targets.default_duration_ms,
                    stress,
                });
            }

            if w_idx + 1 < words.len() {
                segments.push(PhonemeSegment {
                    phoneme: Phoneme::SIL,
                    duration_ms: 35.0,
                    stress: 0,
                });
            }
        }

        segments
    }

    fn rule_based_japanese(word: &str) -> Vec<Phoneme> {
        let chars: Vec<char> = word.chars().collect();
        let len = chars.len();
        let mut phonemes = Vec::new();
        let mut i = 0;

        while i < len {
            if i + 3 <= len {
                let triplet: String = chars[i..i + 3].iter().collect();
                match triplet.as_str() {
                    "chi" => { phonemes.push(Phoneme::CH); phonemes.push(Phoneme::IY); i += 3; continue; }
                    "shi" => { phonemes.push(Phoneme::SH); phonemes.push(Phoneme::IY); i += 3; continue; }
                    "tsu" => { phonemes.push(Phoneme::T); phonemes.push(Phoneme::S); phonemes.push(Phoneme::UW); i += 3; continue; }
                    _ => {}
                }
            }

            if i + 2 <= len {
                let pair: String = chars[i..i + 2].iter().collect();
                match pair.as_str() {
                    "ka" => { phonemes.push(Phoneme::K); phonemes.push(Phoneme::AA); i += 2; continue; }
                    "ki" => { phonemes.push(Phoneme::K); phonemes.push(Phoneme::IY); i += 2; continue; }
                    "ku" => { phonemes.push(Phoneme::K); phonemes.push(Phoneme::UW); i += 2; continue; }
                    "ke" => { phonemes.push(Phoneme::K); phonemes.push(Phoneme::EH); i += 2; continue; }
                    "ko" => { phonemes.push(Phoneme::K); phonemes.push(Phoneme::OW); i += 2; continue; }
                    "sa" => { phonemes.push(Phoneme::S); phonemes.push(Phoneme::AA); i += 2; continue; }
                    "su" => { phonemes.push(Phoneme::S); phonemes.push(Phoneme::UW); i += 2; continue; }
                    "se" => { phonemes.push(Phoneme::S); phonemes.push(Phoneme::EH); i += 2; continue; }
                    "so" => { phonemes.push(Phoneme::S); phonemes.push(Phoneme::OW); i += 2; continue; }
                    "ta" => { phonemes.push(Phoneme::T); phonemes.push(Phoneme::AA); i += 2; continue; }
                    "te" => { phonemes.push(Phoneme::T); phonemes.push(Phoneme::EH); i += 2; continue; }
                    "to" => { phonemes.push(Phoneme::T); phonemes.push(Phoneme::OW); i += 2; continue; }
                    "na" => { phonemes.push(Phoneme::N); phonemes.push(Phoneme::AA); i += 2; continue; }
                    "ni" => { phonemes.push(Phoneme::N); phonemes.push(Phoneme::IY); i += 2; continue; }
                    "nu" => { phonemes.push(Phoneme::N); phonemes.push(Phoneme::UW); i += 2; continue; }
                    "ne" => { phonemes.push(Phoneme::N); phonemes.push(Phoneme::EH); i += 2; continue; }
                    "no" => { phonemes.push(Phoneme::N); phonemes.push(Phoneme::OW); i += 2; continue; }
                    "ha" => { phonemes.push(Phoneme::HH); phonemes.push(Phoneme::AA); i += 2; continue; }
                    "hi" => { phonemes.push(Phoneme::HH); phonemes.push(Phoneme::IY); i += 2; continue; }
                    "fu" => { phonemes.push(Phoneme::F); phonemes.push(Phoneme::UW); i += 2; continue; }
                    "he" => { phonemes.push(Phoneme::HH); phonemes.push(Phoneme::EH); i += 2; continue; }
                    "ho" => { phonemes.push(Phoneme::HH); phonemes.push(Phoneme::OW); i += 2; continue; }
                    "ma" => { phonemes.push(Phoneme::M); phonemes.push(Phoneme::AA); i += 2; continue; }
                    "mi" => { phonemes.push(Phoneme::M); phonemes.push(Phoneme::IY); i += 2; continue; }
                    "mu" => { phonemes.push(Phoneme::M); phonemes.push(Phoneme::UW); i += 2; continue; }
                    "me" => { phonemes.push(Phoneme::M); phonemes.push(Phoneme::EH); i += 2; continue; }
                    "mo" => { phonemes.push(Phoneme::M); phonemes.push(Phoneme::OW); i += 2; continue; }
                    "ra" => { phonemes.push(Phoneme::R); phonemes.push(Phoneme::AA); i += 2; continue; }
                    "ri" => { phonemes.push(Phoneme::R); phonemes.push(Phoneme::IY); i += 2; continue; }
                    "ru" => { phonemes.push(Phoneme::R); phonemes.push(Phoneme::UW); i += 2; continue; }
                    "re" => { phonemes.push(Phoneme::R); phonemes.push(Phoneme::EH); i += 2; continue; }
                    "ro" => { phonemes.push(Phoneme::R); phonemes.push(Phoneme::OW); i += 2; continue; }
                    _ => {}
                }
            }

            let c = chars[i];
            match c {
                'a' => phonemes.push(Phoneme::AA),
                'i' => phonemes.push(Phoneme::IY),
                'u' => phonemes.push(Phoneme::UW),
                'e' => phonemes.push(Phoneme::EH),
                'o' => phonemes.push(Phoneme::OW),
                'n' => phonemes.push(Phoneme::NG),
                _ => {}
            }
            i += 1;
        }

        if phonemes.is_empty() {
            vec![Phoneme::AA]
        } else {
            phonemes
        }
    }

    /// Mandarin Chinese (Pinyin) phonetic parser.
    fn mandarin_to_phonemes(text: &str) -> Vec<PhonemeSegment> {
        let mut segments = Vec::new();
        let words: Vec<&str> = text
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphabetic()))
            .filter(|w| !w.is_empty())
            .collect();

        for (w_idx, word) in words.iter().enumerate() {
            let lower = word.to_lowercase();
            let phonemes = match lower.as_str() {
                "qifei" => vec![Phoneme::CH, Phoneme::IY, Phoneme::F, Phoneme::EY],
                "jiangluo" => vec![Phoneme::JH, Phoneme::IY, Phoneme::AA, Phoneme::NG, Phoneme::L, Phoneme::OW],
                "tingzhi" => vec![Phoneme::T, Phoneme::IH, Phoneme::NG, Phoneme::JH, Phoneme::IY],
                "jinji" => vec![Phoneme::JH, Phoneme::IH, Phoneme::N, Phoneme::JH, Phoneme::IY],
                "fanhang" => vec![Phoneme::F, Phoneme::AA, Phoneme::N, Phoneme::HH, Phoneme::AA, Phoneme::NG],
                "shangsheng" => vec![Phoneme::SH, Phoneme::AA, Phoneme::NG, Phoneme::SH, Phoneme::EH, Phoneme::NG],
                "xiajiang" => vec![Phoneme::SH, Phoneme::IY, Phoneme::AA, Phoneme::JH, Phoneme::IY, Phoneme::AA, Phoneme::NG],
                _ => Self::rule_based_mandarin(&lower),
            };

            for (p_idx, &p) in phonemes.iter().enumerate() {
                let targets = p.acoustic_targets();
                let stress = if p_idx == 0 || (p == Phoneme::AA || p == Phoneme::EY || p == Phoneme::OW) { 1 } else { 0 };
                segments.push(PhonemeSegment {
                    phoneme: p,
                    duration_ms: targets.default_duration_ms,
                    stress,
                });
            }

            if w_idx + 1 < words.len() {
                segments.push(PhonemeSegment {
                    phoneme: Phoneme::SIL,
                    duration_ms: 35.0,
                    stress: 0,
                });
            }
        }

        segments
    }

    fn rule_based_mandarin(word: &str) -> Vec<Phoneme> {
        let chars: Vec<char> = word.chars().collect();
        let len = chars.len();
        let mut phonemes = Vec::new();
        let mut i = 0;

        while i < len {
            if i + 2 <= len {
                let pair: String = chars[i..i + 2].iter().collect();
                match pair.as_str() {
                    "zh" => { phonemes.push(Phoneme::JH); i += 2; continue; }
                    "ch" => { phonemes.push(Phoneme::CH); i += 2; continue; }
                    "sh" => { phonemes.push(Phoneme::SH); i += 2; continue; }
                    "ng" => { phonemes.push(Phoneme::NG); i += 2; continue; }
                    "ai" => { phonemes.push(Phoneme::AY); i += 2; continue; }
                    "ei" => { phonemes.push(Phoneme::EY); i += 2; continue; }
                    "ao" => { phonemes.push(Phoneme::AW); i += 2; continue; }
                    "ou" => { phonemes.push(Phoneme::OW); i += 2; continue; }
                    "an" => { phonemes.push(Phoneme::AA); phonemes.push(Phoneme::N); i += 2; continue; }
                    "en" => { phonemes.push(Phoneme::EH); phonemes.push(Phoneme::N); i += 2; continue; }
                    _ => {}
                }
            }

            let c = chars[i];
            match c {
                'a' => phonemes.push(Phoneme::AA),
                'e' => phonemes.push(Phoneme::EH),
                'i' => phonemes.push(Phoneme::IY),
                'o' => phonemes.push(Phoneme::OW),
                'u' | 'v' => phonemes.push(Phoneme::UW),
                'b' => phonemes.push(Phoneme::B),
                'p' => phonemes.push(Phoneme::P),
                'm' => phonemes.push(Phoneme::M),
                'f' => phonemes.push(Phoneme::F),
                'd' => phonemes.push(Phoneme::D),
                't' => phonemes.push(Phoneme::T),
                'n' => phonemes.push(Phoneme::N),
                'l' => phonemes.push(Phoneme::L),
                'g' => phonemes.push(Phoneme::G),
                'k' => phonemes.push(Phoneme::K),
                'h' => phonemes.push(Phoneme::HH),
                'j' => phonemes.push(Phoneme::JH),
                'q' => phonemes.push(Phoneme::CH),
                'x' => phonemes.push(Phoneme::SH),
                'r' => phonemes.push(Phoneme::R),
                'z' => { phonemes.push(Phoneme::T); phonemes.push(Phoneme::S); }
                'c' => { phonemes.push(Phoneme::T); phonemes.push(Phoneme::S); }
                's' => phonemes.push(Phoneme::S),
                'y' => phonemes.push(Phoneme::Y),
                'w' => phonemes.push(Phoneme::W),
                _ => {}
            }
            i += 1;
        }

        if phonemes.is_empty() {
            vec![Phoneme::AA]
        } else {
            phonemes
        }
    }
}

fn is_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'y' | 'à' | 'é' | 'è' | 'ê' | 'ë' | 'î' | 'ï' | 'ô' | 'û' | 'ù')
}

/// Universal 38-class Phonetic Posteriorgram (PPG) articulatory manifold.
#[derive(Debug, Clone)]
pub struct PhoneticEmbeddingSpace {
    /// Ordered inventory of 38 non-silence phonemes.
    phonemes: [Phoneme; 38],
    /// 13-dimensional canonical acoustic centroids.
    centroids: [[f32; 13]; 38],
    /// Distribution scaling variance.
    variance: f32,
}

impl Default for PhoneticEmbeddingSpace {
    fn default() -> Self {
        Self::new()
    }
}

impl PhoneticEmbeddingSpace {
    /// Construct the universal articulatory embedding space from acoustic targets.
    pub fn new() -> Self {
        let phonemes = [
            Phoneme::AA, Phoneme::AE, Phoneme::AH, Phoneme::AO, Phoneme::AW, Phoneme::AY,
            Phoneme::EH, Phoneme::ER, Phoneme::EY, Phoneme::IH, Phoneme::IY, Phoneme::OW,
            Phoneme::OY, Phoneme::UH, Phoneme::UW, Phoneme::B, Phoneme::D, Phoneme::G,
            Phoneme::P, Phoneme::T, Phoneme::K, Phoneme::DH, Phoneme::F, Phoneme::S,
            Phoneme::SH, Phoneme::TH, Phoneme::V, Phoneme::Z, Phoneme::ZH, Phoneme::HH,
            Phoneme::CH, Phoneme::JH, Phoneme::M, Phoneme::N, Phoneme::NG, Phoneme::L,
            Phoneme::R, Phoneme::W,
        ];

        let mut centroids = [[0.0f32; 13]; 38];

        let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
        let mel_to_hz = |mel: f32| 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0);
        let min_mel = hz_to_mel(150.0);
        let max_mel = hz_to_mel(4000.0);

        let mut center_freqs = [0.0f32; 13];
        for b in 0..13 {
            let m = min_mel + (b as f32) * (max_mel - min_mel) / 12.0;
            center_freqs[b] = mel_to_hz(m);
        }

        for (idx, &p) in phonemes.iter().enumerate() {
            let t = p.acoustic_targets();
            let mut log_spec = [0.0f32; 13];

            for b in 0..13 {
                let fc = center_freqs[b];
                let r1 = 1.0 / (1.0 + ((fc - t.f1) / (t.b1 * 0.5)).powi(2));
                let r2 = 1.0 / (1.0 + ((fc - t.f2) / (t.b2 * 0.5)).powi(2));
                let r3 = 1.0 / (1.0 + ((fc - t.f3) / (t.b3 * 0.5)).powi(2));

                let mut power = 1e-4;
                power += t.voicing_amp * (1.0 * r1 + 0.6 * r2 + 0.3 * r3);

                if t.friction_amp > 0.01 {
                    let high_factor = (fc / 4000.0).powi(2);
                    power += t.friction_amp * high_factor * 1.5;
                }

                if t.aspiration_amp > 0.01 {
                    let mid_factor = (fc / 2500.0).clamp(0.2, 1.2);
                    power += t.aspiration_amp * mid_factor * 0.5;
                }

                log_spec[b] = power.ln();
            }

            // Zero-mean and unit-normalize across 13 bins
            let mean = log_spec.iter().sum::<f32>() / 13.0;
            let mut norm_sq = 0.0f32;
            for val in &mut log_spec {
                *val -= mean;
                norm_sq += (*val) * (*val);
            }
            let norm = norm_sq.sqrt().max(1e-6);
            for val in &mut log_spec {
                *val /= norm;
            }

            centroids[idx] = log_spec;
        }

        Self {
            phonemes,
            centroids,
            variance: 1.25,
        }
    }

    /// Access reference to the 38 phoneme classes.
    pub fn phonemes(&self) -> &[Phoneme; 38] {
        &self.phonemes
    }

    /// Get canonical 13-dimensional centroid for a given phoneme.
    pub fn centroid(&self, phoneme: Phoneme) -> [f32; 13] {
        if phoneme == Phoneme::SIL {
            return [0.0f32; 13];
        }
        for (i, &p) in self.phonemes.iter().enumerate() {
            if p == phoneme {
                return self.centroids[i];
            }
        }
        // Fallback to neutral schwa /AH/
        self.centroids[2]
    }

    /// Compute frame-level Phonetic Posteriorgram (PPG) distribution over 38 phonemes.
    pub fn compute_ppg(&self, frame: &[f32]) -> [f32; 38] {
        let mut ppg = [0.0f32; 38];
        if frame.len() < 13 {
            ppg.fill(1.0 / 38.0);
            return ppg;
        }

        // Frame zero-mean and unit normalization
        let mean = frame[..13].iter().sum::<f32>() / 13.0;
        let mut norm_frame = [0.0f32; 13];
        let mut f_sq = 0.0f32;
        for d in 0..13 {
            let diff = frame[d] - mean;
            norm_frame[d] = diff;
            f_sq += diff * diff;
        }
        let f_norm = f_sq.sqrt().max(1e-6);
        for d in 0..13 {
            norm_frame[d] /= f_norm;
        }

        let mut max_log_prob = f32::NEG_INFINITY;
        let mut logits = [0.0f32; 38];

        for i in 0..38 {
            let mut dist_sq = 0.0f32;
            for d in 0..13 {
                let diff = norm_frame[d] - self.centroids[i][d];
                dist_sq += diff * diff;
            }
            let logit = -dist_sq / (2.0 * self.variance);
            logits[i] = logit;
            if logit > max_log_prob {
                max_log_prob = logit;
            }
        }

        let mut sum_exp = 0.0f32;
        for i in 0..38 {
            let exp_val = (logits[i] - max_log_prob).exp();
            ppg[i] = exp_val;
            sum_exp += exp_val;
        }

        let inv_sum = 1.0 / sum_exp.max(1e-9);
        for p in &mut ppg {
            *p *= inv_sum;
        }

        ppg
    }

    /// Cosine similarity between an acoustic frame and a target phoneme's centroid.
    pub fn frame_similarity(&self, frame: &[f32], phoneme: Phoneme) -> f32 {
        if frame.len() < 13 || phoneme == Phoneme::SIL {
            return 0.0;
        }
        let c = self.centroid(phoneme);
        let mean = frame[..13].iter().sum::<f32>() / 13.0;
        let mut dot = 0.0f32;
        let mut n1 = 0.0f32;
        let mut n2 = 0.0f32;
        for d in 0..13 {
            let x = frame[d] - mean;
            let y = c[d];
            dot += x * y;
            n1 += x * x;
            n2 += y * y;
        }
        dot / (n1.sqrt() * n2.sqrt() + 1e-9)
    }
}

/// Alignment results produced by cross-attention phonetic-acoustic matching.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CrossAttentionResult {
    /// Soft cross-attention matrix alpha (L tokens x T frames). Flattened row-major.
    pub attention_matrix: Vec<f32>,
    /// Number of phonetic tokens L.
    pub num_tokens: usize,
    /// Number of acoustic frames T.
    pub num_frames: usize,
    /// Aligned, smoothed acoustic feature frames (T x 13).
    pub aligned_features: Vec<Vec<f32>>,
    /// Average alignment entropy across frames (bits).
    pub alignment_entropy: f32,
    /// Monotonic progression fidelity in [0.0, 1.0].
    pub monotonicity_score: f32,
    /// Ratio of phoneme tokens successfully bound to acoustic frames in [0.0, 1.0].
    pub phonetic_coverage: f32,
}

/// Soft cross-attention aligner mapping phonetic tokens to acoustic time frames.
#[derive(Debug, Clone)]
pub struct CrossAttentionAligner {
    space: PhoneticEmbeddingSpace,
    monotonic_weight: f32,
    smoothing_factor: f32,
}

impl Default for CrossAttentionAligner {
    fn default() -> Self {
        Self::new()
    }
}

impl CrossAttentionAligner {
    /// Construct aligner with default embedding space.
    pub fn new() -> Self {
        Self {
            space: PhoneticEmbeddingSpace::new(),
            monotonic_weight: 4.0,
            smoothing_factor: 0.08,
        }
    }

    /// Access reference to underlying phonetic embedding space.
    pub fn embedding_space(&self) -> &PhoneticEmbeddingSpace {
        &self.space
    }

    /// Align target phoneme sequence with continuous acoustic feature frames.
    pub fn align(
        &self,
        phonemes: &[Phoneme],
        acoustic_frames: &[Vec<f32>],
    ) -> Result<CrossAttentionResult, String> {
        if phonemes.is_empty() {
            return Err("Cannot align empty phoneme sequence".to_string());
        }
        if acoustic_frames.is_empty() {
            return Err("Cannot align empty acoustic frames".to_string());
        }

        let l = phonemes.len();
        let t = acoustic_frames.len();
        let d = 13;
        let scale = 2.5f32;

        // 1. Prepare queries: Q_i = centroid(p_i) + positional_encoding(i, L)
        let mut queries = Vec::with_capacity(l);
        for i in 0..l {
            let mut q = self.space.centroid(phonemes[i]);
            for dim in 0..d {
                let pe = 0.12 * ((i as f32 + 0.5) / l as f32 * PI * (dim as f32 + 1.0) / d as f32).sin();
                q[dim] += pe;
            }
            queries.push(q);
        }

        // 2. Prepare keys: K_j = norm(x_j) + positional_encoding(j, T)
        let mut keys = Vec::with_capacity(t);
        for j in 0..t {
            let mut k = [0.0f32; 13];
            let frame = &acoustic_frames[j];
            let mean = frame.iter().take(d).sum::<f32>() / (d as f32);
            let mut norm_sq = 0.0f32;
            for dim in 0..d {
                let diff = if dim < frame.len() { frame[dim] - mean } else { 0.0 };
                k[dim] = diff;
                norm_sq += diff * diff;
            }
            let norm = norm_sq.sqrt().max(1e-6);
            for dim in 0..d {
                k[dim] /= norm;
                let pe = 0.12 * ((j as f32 + 0.5) / t as f32 * PI * (dim as f32 + 1.0) / d as f32).sin();
                k[dim] += pe;
            }
            keys.push(k);
        }

        // 3. Compute scaled dot-product attention scores S_{i, j} with monotonic Gaussian prior
        let mut scores = vec![0.0f32; l * t];
        for i in 0..l {
            let q_i = &queries[i];
            let norm_i = (i as f32 + 0.5) / (l as f32);
            for j in 0..t {
                let k_j = &keys[j];
                let norm_j = (j as f32 + 0.5) / (t as f32);

                let mut dot = 0.0f32;
                for dim in 0..d {
                    dot += q_i[dim] * k_j[dim];
                }

                let monotonic_penalty = self.monotonic_weight * (norm_i - norm_j) * (norm_i - norm_j);
                scores[i * t + j] = dot * scale - monotonic_penalty;
            }
        }

        // 4. Softmax across phoneme tokens for each frame j (column-wise softmax)
        let mut attention = vec![0.0f32; l * t];
        let mut frame_entropies = Vec::with_capacity(t);
        let mut argmax_path = Vec::with_capacity(t);

        for j in 0..t {
            let mut max_score = f32::NEG_INFINITY;
            for i in 0..l {
                let s = scores[i * t + j];
                if s > max_score {
                    max_score = s;
                }
            }

            let mut sum_exp = 0.0f32;
            for i in 0..l {
                let exp_val = (scores[i * t + j] - max_score).exp();
                attention[i * t + j] = exp_val;
                sum_exp += exp_val;
            }

            let inv_sum = 1.0 / sum_exp.max(1e-9);
            let mut best_i = 0;
            let mut best_val = -1.0f32;
            let mut entropy = 0.0f32;

            for i in 0..l {
                let alpha = attention[i * t + j] * inv_sum;
                attention[i * t + j] = alpha;
                if alpha > best_val {
                    best_val = alpha;
                    best_i = i;
                }
                if alpha > 1e-6 {
                    entropy -= alpha * alpha.ln();
                }
            }

            argmax_path.push(best_i);
            frame_entropies.push(entropy);
        }

        // 5. Compute alignment diagnostics
        let alignment_entropy = frame_entropies.iter().sum::<f32>() / (t as f32);

        let mut monotonic_steps = 0;
        for j in 1..t {
            if argmax_path[j] >= argmax_path[j - 1] {
                monotonic_steps += 1;
            }
        }
        let monotonicity_score = if t > 1 {
            (monotonic_steps as f32) / ((t - 1) as f32)
        } else {
            1.0
        };

        let mut covered_tokens = 0;
        for i in 0..l {
            let mut max_frame_att = 0.0f32;
            for j in 0..t {
                let att = attention[i * t + j];
                if att > max_frame_att {
                    max_frame_att = att;
                }
            }
            if max_frame_att >= 0.20 {
                covered_tokens += 1;
            }
        }
        let phonetic_coverage = (covered_tokens as f32) / (l as f32);

        // 6. Aligned, smoothed feature synthesis
        let mut aligned_features = Vec::with_capacity(t);
        for j in 0..t {
            let mut blended = vec![0.0f32; d];
            let raw_frame = &acoustic_frames[j];
            let frame_norm = raw_frame.iter().map(|&x| x * x).sum::<f32>().sqrt();

            // Blend raw features with attention-weighted phoneme centroid
            for dim in 0..d {
                let raw_val = if dim < raw_frame.len() { raw_frame[dim] } else { 0.0 };
                let mut phonetic_val = 0.0f32;
                for i in 0..l {
                    let alpha = attention[i * t + j];
                    let c_val = self.space.centroid(phonemes[i])[dim];
                    phonetic_val += alpha * c_val;
                }
                blended[dim] = (1.0 - self.smoothing_factor) * raw_val + self.smoothing_factor * frame_norm * phonetic_val;
            }
            aligned_features.push(blended);
        }

        Ok(CrossAttentionResult {
            attention_matrix: attention,
            num_tokens: l,
            num_frames: t,
            aligned_features,
            alignment_entropy,
            monotonicity_score,
            phonetic_coverage,
        })
    }
}

/// Minimal-pair phonetic foil generator for zero-shot discrimination margin evaluation.
#[derive(Debug, Clone)]
pub struct PhoneticFoilGenerator;

impl PhoneticFoilGenerator {
    /// Generate a structured inventory of minimal-pair phonetic foils for the given keyword phrase.
    pub fn generate_foils(phrase: &str, language: SupportedLanguage) -> Vec<PhoneticFoil> {
        let segments = MultiLingualG2p::text_to_phonemes(phrase, language);
        let phonemes: Vec<Phoneme> = segments
            .iter()
            .map(|s| s.phoneme)
            .filter(|&p| p != Phoneme::SIL)
            .collect();

        let mut foils = Vec::new();
        let words: Vec<&str> = phrase.split_whitespace().collect();

        // 1. Consonant Minimal Pairs: Voice/place/manner switch
        foils.extend(Self::consonant_foils(&words, language, &phonemes));

        // 2. Vowel Formant Shift Pairs: Vowel quadrangle substitution
        foils.extend(Self::vowel_foils(&words, language, &phonemes));

        // 3. Lexical Boundary Subtitle Pairs: Boundary modifier changes
        foils.extend(Self::lexical_boundary_foils(&words, language));

        // 4. Cross-Lingual Negative Distractors: Command words in alternate languages
        foils.extend(Self::cross_lingual_foils(language));

        foils
    }

    fn consonant_foils(words: &[&str], lang: SupportedLanguage, phonemes: &[Phoneme]) -> Vec<PhoneticFoil> {
        let mut list = Vec::new();
        if words.is_empty() || phonemes.is_empty() {
            return list;
        }

        let first = words[0].to_lowercase();
        match first.as_str() {
            "take" => {
                list.push(Self::make_foil("make off", lang, FoilType::ConsonantMinimalPair, vec![Phoneme::M, Phoneme::EY, Phoneme::K, Phoneme::AO, Phoneme::F]));
                list.push(Self::make_foil("bake off", lang, FoilType::ConsonantMinimalPair, vec![Phoneme::B, Phoneme::EY, Phoneme::K, Phoneme::AO, Phoneme::F]));
                list.push(Self::make_foil("shake off", lang, FoilType::ConsonantMinimalPair, vec![Phoneme::SH, Phoneme::EY, Phoneme::K, Phoneme::AO, Phoneme::F]));
            }
            "land" => {
                list.push(Self::make_foil("hand", lang, FoilType::ConsonantMinimalPair, vec![Phoneme::HH, Phoneme::AE, Phoneme::N, Phoneme::D]));
                list.push(Self::make_foil("sand", lang, FoilType::ConsonantMinimalPair, vec![Phoneme::S, Phoneme::AE, Phoneme::N, Phoneme::D]));
                list.push(Self::make_foil("band", lang, FoilType::ConsonantMinimalPair, vec![Phoneme::B, Phoneme::AE, Phoneme::N, Phoneme::D]));
            }
            "abort" => {
                list.push(Self::make_foil("aport", lang, FoilType::ConsonantMinimalPair, vec![Phoneme::AH, Phoneme::P, Phoneme::AO, Phoneme::R, Phoneme::T]));
                list.push(Self::make_foil("report", lang, FoilType::ConsonantMinimalPair, vec![Phoneme::R, Phoneme::IH, Phoneme::P, Phoneme::AO, Phoneme::R, Phoneme::T]));
            }
            "despegar" => {
                list.push(Self::make_foil("despejar", lang, FoilType::ConsonantMinimalPair, vec![Phoneme::D, Phoneme::EH, Phoneme::S, Phoneme::P, Phoneme::EH, Phoneme::HH, Phoneme::AA, Phoneme::R]));
                list.push(Self::make_foil("despesar", lang, FoilType::ConsonantMinimalPair, vec![Phoneme::D, Phoneme::EH, Phoneme::S, Phoneme::P, Phoneme::EH, Phoneme::S, Phoneme::AA, Phoneme::R]));
            }
            "ritsuriku" => {
                list.push(Self::make_foil("mitsuriku", lang, FoilType::ConsonantMinimalPair, vec![Phoneme::M, Phoneme::IH, Phoneme::T, Phoneme::S, Phoneme::UW, Phoneme::R, Phoneme::IH, Phoneme::K, Phoneme::UW]));
                list.push(Self::make_foil("hitsuriku", lang, FoilType::ConsonantMinimalPair, vec![Phoneme::HH, Phoneme::IH, Phoneme::T, Phoneme::S, Phoneme::UW, Phoneme::R, Phoneme::IH, Phoneme::K, Phoneme::UW]));
            }
            _ => {
                // Generic single-consonant substitution
                let mut alt = phonemes.to_vec();
                if let Some(pos) = alt.iter().position(|&p| matches!(p, Phoneme::T | Phoneme::D | Phoneme::P | Phoneme::B | Phoneme::K | Phoneme::G)) {
                    alt[pos] = match alt[pos] {
                        Phoneme::T => Phoneme::D,
                        Phoneme::D => Phoneme::T,
                        Phoneme::P => Phoneme::B,
                        Phoneme::B => Phoneme::P,
                        Phoneme::K => Phoneme::G,
                        _ => Phoneme::M,
                    };
                    list.push(Self::make_foil(&format!("foil_{}", first), lang, FoilType::ConsonantMinimalPair, alt));
                }
            }
        }

        list
    }

    fn vowel_foils(words: &[&str], lang: SupportedLanguage, phonemes: &[Phoneme]) -> Vec<PhoneticFoil> {
        let mut list = Vec::new();
        if words.is_empty() || phonemes.is_empty() {
            return list;
        }

        let first = words[0].to_lowercase();
        match first.as_str() {
            "take" => {
                list.push(Self::make_foil("tick off", lang, FoilType::VowelFormantShift, vec![Phoneme::T, Phoneme::IH, Phoneme::K, Phoneme::AO, Phoneme::F]));
                list.push(Self::make_foil("tuck off", lang, FoilType::VowelFormantShift, vec![Phoneme::T, Phoneme::AH, Phoneme::K, Phoneme::AO, Phoneme::F]));
                list.push(Self::make_foil("took off", lang, FoilType::VowelFormantShift, vec![Phoneme::T, Phoneme::UH, Phoneme::K, Phoneme::AO, Phoneme::F]));
            }
            "land" => {
                list.push(Self::make_foil("lend", lang, FoilType::VowelFormantShift, vec![Phoneme::L, Phoneme::EH, Phoneme::N, Phoneme::D]));
                list.push(Self::make_foil("lind", lang, FoilType::VowelFormantShift, vec![Phoneme::L, Phoneme::IH, Phoneme::N, Phoneme::D]));
            }
            "despegar" => {
                list.push(Self::make_foil("despagar", lang, FoilType::VowelFormantShift, vec![Phoneme::D, Phoneme::EH, Phoneme::S, Phoneme::P, Phoneme::AA, Phoneme::G, Phoneme::AA, Phoneme::R]));
            }
            _ => {
                let mut alt = phonemes.to_vec();
                if let Some(pos) = alt.iter().position(|&p| matches!(p, Phoneme::AA | Phoneme::AE | Phoneme::EH | Phoneme::IY | Phoneme::OW | Phoneme::UW)) {
                    alt[pos] = match alt[pos] {
                        Phoneme::AA => Phoneme::OW,
                        Phoneme::AE => Phoneme::EH,
                        Phoneme::EH => Phoneme::IH,
                        Phoneme::IY => Phoneme::UW,
                        _ => Phoneme::AA,
                    };
                    list.push(Self::make_foil(&format!("vowel_{}", first), lang, FoilType::VowelFormantShift, alt));
                }
            }
        }

        list
    }

    fn lexical_boundary_foils(words: &[&str], lang: SupportedLanguage) -> Vec<PhoneticFoil> {
        let mut list = Vec::new();
        if words.is_empty() {
            return list;
        }

        if words.len() >= 2 && words[0].eq_ignore_ascii_case("take") && words[1].eq_ignore_ascii_case("off") {
            list.push(Self::make_foil("take up", lang, FoilType::LexicalBoundary, vec![Phoneme::T, Phoneme::EY, Phoneme::K, Phoneme::AH, Phoneme::P]));
            list.push(Self::make_foil("take out", lang, FoilType::LexicalBoundary, vec![Phoneme::T, Phoneme::EY, Phoneme::K, Phoneme::AW, Phoneme::T]));
            list.push(Self::make_foil("take on", lang, FoilType::LexicalBoundary, vec![Phoneme::T, Phoneme::EY, Phoneme::K, Phoneme::AA, Phoneme::N]));
        } else if words[0].eq_ignore_ascii_case("hold") {
            list.push(Self::make_foil("hold on", lang, FoilType::LexicalBoundary, vec![Phoneme::HH, Phoneme::OW, Phoneme::L, Phoneme::D, Phoneme::AA, Phoneme::N]));
            list.push(Self::make_foil("cold", lang, FoilType::LexicalBoundary, vec![Phoneme::K, Phoneme::OW, Phoneme::L, Phoneme::D]));
        }

        list
    }

    fn cross_lingual_foils(target_lang: SupportedLanguage) -> Vec<PhoneticFoil> {
        let mut list = Vec::new();
        let distractors = [
            ("despegar", SupportedLanguage::Spanish, vec![Phoneme::D, Phoneme::EH, Phoneme::S, Phoneme::P, Phoneme::EH, Phoneme::G, Phoneme::AA, Phoneme::R]),
            ("abheben", SupportedLanguage::German, vec![Phoneme::AA, Phoneme::P, Phoneme::HH, Phoneme::EY, Phoneme::B, Phoneme::EH, Phoneme::N]),
            ("decoller", SupportedLanguage::French, vec![Phoneme::D, Phoneme::EH, Phoneme::K, Phoneme::AO, Phoneme::L, Phoneme::EY]),
            ("ritsuriku", SupportedLanguage::Japanese, vec![Phoneme::R, Phoneme::IH, Phoneme::T, Phoneme::S, Phoneme::UW, Phoneme::R, Phoneme::IH, Phoneme::K, Phoneme::UW]),
            ("qifei", SupportedLanguage::Mandarin, vec![Phoneme::CH, Phoneme::IY, Phoneme::F, Phoneme::EY]),
        ];

        for (word, lang, phonemes) in distractors {
            if lang != target_lang {
                list.push(Self::make_foil(word, lang, FoilType::CrossLingualDistractor, phonemes));
            }
        }

        list
    }

    fn make_foil(text: &str, lang: SupportedLanguage, foil_type: FoilType, phonemes: Vec<Phoneme>) -> PhoneticFoil {
        PhoneticFoil {
            foil_text: text.to_string(),
            language: lang,
            foil_type,
            phonemes,
        }
    }
}

/// Comprehensive zero-shot calibration report containing discrimination margins and optimal threshold.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ZeroShotCalibrationReport {
    /// Keyword identifier.
    pub keyword: String,
    /// Language profile.
    pub language: SupportedLanguage,
    /// Recognized target phoneme string representation.
    pub target_phonemes: Vec<String>,
    /// Maximum intra-class DTW distance between target synthetic variations.
    pub intra_target_distance: f32,
    /// Minimum DTW distance to the nearest minimal-pair phonetic foil.
    pub min_foil_distance: f32,
    /// Orthographic label of the hardest/closest phonetic foil.
    pub hardest_foil: String,
    /// Net acoustic discrimination margin (min_foil_distance - intra_target_distance).
    pub discrimination_margin: f32,
    /// Mathematically calibrated zero-shot DTW threshold.
    pub calibrated_threshold: f32,
    /// Average cross-attention alignment entropy across time frames.
    pub alignment_entropy: f32,
    /// Monotonicity score of the phonetic-acoustic attention path.
    pub monotonicity_score: f32,
    /// Ratio of target phonemes bound to acoustic energy.
    pub phonetic_coverage: f32,
    /// Relative separation ratio: discrimination_margin / intra_target_distance.
    pub margin_ratio: f32,
    /// Empirical confusion matrix over all target variations and minimal-pair foils.
    pub confusion_matrix: ConfusionMatrix,
}

/// Engine for zero-shot wake-word calibration and threshold solving.
#[derive(Debug, Clone)]
pub struct ZeroShotCalibrator {
    aligner: CrossAttentionAligner,
}

impl Default for ZeroShotCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

impl ZeroShotCalibrator {
    /// Construct a new zero-shot calibrator.
    pub fn new() -> Self {
        Self {
            aligner: CrossAttentionAligner::new(),
        }
    }

    /// Access reference to internal cross-attention aligner.
    pub fn aligner(&self) -> &CrossAttentionAligner {
        &self.aligner
    }

    /// Perform zero-shot calibration for a keyword phrase in the specified language and accent.
    pub fn calibrate(
        &self,
        phrase: &str,
        language: SupportedLanguage,
        accent: VocalAccent,
        engine: &SononEngine,
    ) -> Result<(Vec<Vec<f32>>, ZeroShotCalibrationReport), String> {
        let segments = MultiLingualG2p::text_to_phonemes(phrase, language);
        if segments.is_empty() {
            return Err(format!("Could not extract phonemes for phrase '{}'", phrase));
        }

        let raw_phonemes: Vec<Phoneme> = segments
            .iter()
            .map(|s| s.phoneme)
            .filter(|&p| p != Phoneme::SIL)
            .collect();

        // 1. Synthesize diverse variations of the target phrase (different pitch, speed, accent)
        let sample_rate = engine.sample_rate();
        let generator = SyntheticExemplarGenerator::new(sample_rate);
        let target_exemplars = generator.generate_exemplars_from_segments(&segments, 4);
        if target_exemplars.is_empty() {
            return Err("Failed to synthesize target exemplars".to_string());
        }

        // 2. Extract acoustic feature sequences for target variations
        let mut target_feature_sets = Vec::with_capacity(target_exemplars.len());
        for audio in &target_exemplars {
            let feats = engine.extract_features(audio);
            if !feats.is_empty() {
                target_feature_sets.push(feats);
            }
        }
        if target_feature_sets.is_empty() {
            return Err("Feature extraction produced empty frames for target".to_string());
        }

        // 3. Compute reference template via DTW Barycenter Averaging and cross-attention alignment
        let band_radius = 8;
        let dba_template = crate::dtw::dtw_barycenter_averaging(&target_feature_sets, 5, band_radius);
        let align_res = self.aligner.align(&raw_phonemes, &dba_template)?;
        let reference_template = dba_template;

        // 4. Compute intra-target DTW distance (maximum distance among positive variations)
        let mut max_intra_dist = 0.0f32;
        for feats in &target_feature_sets {
            let d = DtwMatcher::compute_distance_banded(&reference_template, feats, band_radius);
            if d > max_intra_dist {
                max_intra_dist = d;
            }
        }
        max_intra_dist = max_intra_dist.max(1.2);

        // 5. Generate and synthesize minimal-pair phonetic foils
        let foils = PhoneticFoilGenerator::generate_foils(phrase, language);
        let mut min_foil_dist = f32::INFINITY;
        let mut hardest_foil = String::new();
        let mut foil_dists = Vec::with_capacity(foils.len());

        for foil in &foils {
            let g2p_segments = MultiLingualG2p::text_to_phonemes(&foil.foil_text, foil.language);
            let foil_segments: Vec<PhonemeSegment> = if !g2p_segments.is_empty() {
                g2p_segments
            } else {
                foil.phonemes
                    .iter()
                    .map(|&p| {
                        let targets = p.acoustic_targets_with_accent(accent);
                        PhonemeSegment {
                            phoneme: p,
                            duration_ms: targets.default_duration_ms,
                            stress: 1,
                        }
                    })
                    .collect()
            };

            let mut synth = KlattSynthesizer::new(sample_rate);
            synth.set_accent(accent);
            let foil_audio = synth.synthesize(&foil_segments);
            if foil_audio.is_empty() {
                continue;
            }

            let foil_feats = engine.extract_features(&foil_audio);
            if foil_feats.is_empty() {
                continue;
            }

            let d = DtwMatcher::compute_distance_banded(&reference_template, &foil_feats, band_radius);
            foil_dists.push((foil.foil_text.clone(), d));
            if d < min_foil_dist {
                min_foil_dist = d;
                hardest_foil = foil.foil_text.clone();
            }
        }

        if min_foil_dist.is_infinite() {
            min_foil_dist = max_intra_dist * 2.2;
            hardest_foil = "default_distractor".to_string();
        }

        // 6. Calculate discrimination margin and calibrated decision threshold
        let discrimination_margin = min_foil_dist - max_intra_dist;
        let calibrated_threshold = if discrimination_margin > 0.0 {
            (max_intra_dist + 0.04 * discrimination_margin).min(max_intra_dist * 1.05)
        } else {
            max_intra_dist * 1.02
        };

        // 7. Evaluate confusion matrix over positive targets and negative foils
        let mut tp = 0;
        let mut fn_count = 0;
        for feats in &target_feature_sets {
            let d = DtwMatcher::compute_distance_banded(&reference_template, feats, band_radius);
            if d <= calibrated_threshold {
                tp += 1;
            } else {
                fn_count += 1;
            }
        }

        let mut fp = 0;
        let mut tn = 0;
        for (_, d) in &foil_dists {
            if *d <= calibrated_threshold {
                fp += 1;
            } else {
                tn += 1;
            }
        }

        let target_phonemes_str: Vec<String> = raw_phonemes.iter().map(|p| format!("{:?}", p)).collect();
        let margin_ratio = if max_intra_dist > 1e-4 {
            discrimination_margin / max_intra_dist
        } else {
            1.0
        };

        let report = ZeroShotCalibrationReport {
            keyword: phrase.to_string(),
            language,
            target_phonemes: target_phonemes_str,
            intra_target_distance: max_intra_dist,
            min_foil_distance: min_foil_dist,
            hardest_foil,
            discrimination_margin,
            calibrated_threshold,
            alignment_entropy: align_res.alignment_entropy,
            monotonicity_score: align_res.monotonicity_score,
            phonetic_coverage: align_res.phonetic_coverage,
            margin_ratio,
            confusion_matrix: ConfusionMatrix::new(tp, fp, tn, fn_count),
        };

        Ok((reference_template, report))
    }
}
