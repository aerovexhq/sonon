//! Integration test suite verifying multi-speaker synthetic data augmentation,
//! regional vocal accents, prosodic intonation contours, WAV encoding, and on-device confusion matrix evaluation.

#![deny(unsafe_code)]

use sonon::{
    encode_wav_16bit, write_wav_file, ConfusionMatrix, FeatureMode, IntonationContour,
    KlattSynthesizer, Phoneme, SononEngine, SyntheticExemplarGenerator, VocalAccent,
};
use std::f32::consts::PI;

#[test]
fn test_vocal_accent_formant_shifts() {
    // 1. Received Pronunciation (RP) shifts
    let rp_aa = Phoneme::AA.acoustic_targets_with_accent(VocalAccent::ReceivedPronunciation);
    let ga_aa = Phoneme::AA.acoustic_targets_with_accent(VocalAccent::GeneralAmerican);
    assert!(
        rp_aa.f1 < ga_aa.f1,
        "RP /AA/ (father) must have lower F1 than General American"
    );

    let rp_er = Phoneme::ER.acoustic_targets_with_accent(VocalAccent::ReceivedPronunciation);
    let ga_er = Phoneme::ER.acoustic_targets_with_accent(VocalAccent::GeneralAmerican);
    assert!(
        rp_er.f3 > ga_er.f3,
        "RP non-rhotic /ER/ must have higher F3 than rhotic General American"
    );

    // 2. International accent centralization
    let int_ae = Phoneme::AE.acoustic_targets_with_accent(VocalAccent::International);
    let ga_ae = Phoneme::AE.acoustic_targets_with_accent(VocalAccent::GeneralAmerican);
    // GA /AE/ has F1=660, F2=1720; neutral vowel has 500, 1500
    assert!(
        int_ae.f1 < ga_ae.f1,
        "International /AE/ must centralize F1 toward neutral 500 Hz"
    );
    assert!(
        int_ae.f2 < ga_ae.f2,
        "International /AE/ must centralize F2 toward neutral 1500 Hz"
    );
}

#[test]
fn test_intonation_contour_pitch_trajectories() {
    // 1. Declarative contour: peaks mid-utterance, ends slightly below base
    let dec_start = IntonationContour::Declarative.evaluate(0.0, 0.0);
    let dec_mid = IntonationContour::Declarative.evaluate(0.5, 0.5);
    let dec_end = IntonationContour::Declarative.evaluate(1.0, 1.0);
    assert!(
        dec_mid > dec_start,
        "Declarative contour must peak mid-utterance"
    );
    assert!(
        dec_end < dec_start,
        "Declarative contour must terminate with falling cadence"
    );

    // 2. Authoritative Command contour: high initial attack, rapid decisive fall
    let cmd_start = IntonationContour::AuthoritativeCommand.evaluate(0.0, 0.0);
    let cmd_end = IntonationContour::AuthoritativeCommand.evaluate(1.0, 1.0);
    assert!(
        cmd_start >= 1.20,
        "Authoritative command must exhibit high initial pitch attack"
    );
    assert!(
        cmd_end <= 0.88,
        "Authoritative command must fall decisively by utterance completion"
    );
    assert!(
        cmd_start - cmd_end > 0.30,
        "Command dynamic range must exceed 30% drop"
    );

    // 3. Interrogative contour: rising inflection
    let q_start = IntonationContour::Interrogative.evaluate(0.0, 0.0);
    let q_end = IntonationContour::Interrogative.evaluate(1.0, 1.0);
    assert!(
        q_end > q_start + 0.30,
        "Interrogative contour must rise significantly toward the end"
    );
}

#[test]
fn test_wav_16bit_encoder_byte_specification() {
    let sample_rate = 16000u32;
    // Generate 0.1s of 440 Hz test tone (1600 samples)
    let num_samples = 1600;
    let samples: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.75 * (2.0 * PI * 440.0 * t).sin()
        })
        .collect();

    let wav_bytes = encode_wav_16bit(&samples, sample_rate);

    // Verify RIFF header structure
    assert_eq!(&wav_bytes[0..4], b"RIFF");
    assert_eq!(&wav_bytes[8..12], b"WAVE");
    assert_eq!(&wav_bytes[12..16], b"fmt ");

    // Verify subchunk sizes and formats
    let subchunk1_size = u32::from_le_bytes(wav_bytes[16..20].try_into().unwrap());
    assert_eq!(subchunk1_size, 16); // Standard PCM

    let audio_format = u16::from_le_bytes(wav_bytes[20..22].try_into().unwrap());
    assert_eq!(audio_format, 1); // 1 = PCM

    let num_channels = u16::from_le_bytes(wav_bytes[22..24].try_into().unwrap());
    assert_eq!(num_channels, 1); // Mono

    let sr = u32::from_le_bytes(wav_bytes[24..28].try_into().unwrap());
    assert_eq!(sr, sample_rate);

    let bits_per_sample = u16::from_le_bytes(wav_bytes[34..36].try_into().unwrap());
    assert_eq!(bits_per_sample, 16);

    assert_eq!(&wav_bytes[36..40], b"data");
    let subchunk2_size = u32::from_le_bytes(wav_bytes[40..44].try_into().unwrap());
    assert_eq!(subchunk2_size, (num_samples * 2) as u32);
    assert_eq!(wav_bytes.len(), 44 + (num_samples * 2));

    // Test writing to disk
    let tmp_path = std::env::temp_dir().join("sonon_test_tone.wav");
    write_wav_file(&tmp_path, &samples, sample_rate).expect("Must write WAV file");
    assert!(tmp_path.exists());
    let metadata = std::fs::metadata(&tmp_path).expect("Must read file metadata");
    assert_eq!(metadata.len(), (44 + num_samples * 2) as u64);
    let _ = std::fs::remove_file(&tmp_path);
}

#[test]
fn test_synthetic_exemplar_generator_with_metadata() {
    let sample_rate = 16000.0f32;
    let generator = SyntheticExemplarGenerator::new(sample_rate);

    let exemplars = generator.generate_exemplars_with_metadata("abort", 6);
    assert_eq!(exemplars.len(), 6);

    for (idx, ex) in exemplars.iter().enumerate() {
        assert_eq!(ex.phrase, "abort");
        assert!(ex.audio.len() > 1600); // At least 100ms
        assert!(ex.pitch_f0 >= 100.0 && ex.pitch_f0 <= 250.0);
        assert!(ex.speaking_rate >= 0.8 && ex.speaking_rate <= 1.3);
        assert!(ex.vocal_tract_scale >= 0.85 && ex.vocal_tract_scale <= 1.2);

        // Verify peak normalization within [-1.0, 1.0]
        let peak = ex.audio.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
        assert!(peak <= 1.0 && peak > 0.5, "Exemplar {} must be normalized", idx);
    }

    // Verify acoustic waveform diversity across accents and contours
    let d1 = exemplars[0].audio.len();
    let d2 = exemplars[1].audio.len();
    let has_variation = exemplars.iter().any(|ex| ex.audio.len() != d1) || d1 != d2;
    assert!(has_variation, "Generated exemplars must possess varying temporal lengths");
}

#[test]
fn test_confusion_matrix_mathematical_metrics() {
    // 90 True Positives, 10 False Positives, 95 True Negatives, 5 False Negatives
    let matrix = ConfusionMatrix::new(90, 10, 95, 5);

    assert_eq!(matrix.total_samples(), 200);

    // Precision = 90 / (90 + 10) = 0.90
    assert!((matrix.precision() - 0.90).abs() < 1e-4);

    // Recall = 90 / (90 + 5) = 90 / 95 = 0.947368
    assert!((matrix.recall() - 90.0 / 95.0).abs() < 1e-4);

    // F1 Score = 2 * (0.9 * (90/95)) / (0.9 + 90/95) = 0.923076
    assert!((matrix.f1_score() - 0.923076).abs() < 1e-4);

    // FPR = 10 / (10 + 95) = 10 / 105 = 0.095238
    assert!((matrix.false_positive_rate() - 10.0 / 105.0).abs() < 1e-4);

    // FNR = 5 / (5 + 90) = 5 / 95 = 0.052631
    assert!((matrix.false_negative_rate() - 5.0 / 95.0).abs() < 1e-4);

    // Accuracy = (90 + 95) / 200 = 185 / 200 = 0.925
    assert!((matrix.accuracy() - 0.925).abs() < 1e-4);
}

#[test]
fn test_minimal_pair_phonetic_foil_generation() {
    let foils_takeoff = SononEngine::phonetic_foils_for_keyword("take off");
    assert!(foils_takeoff.contains(&"shake off"));
    assert!(foils_takeoff.contains(&"make off"));
    assert!(foils_takeoff.contains(&"lake loft"));

    let foils_land = SononEngine::phonetic_foils_for_keyword("land");
    assert!(foils_land.contains(&"hand"));
    assert!(foils_land.contains(&"sand"));

    let foils_emergency = SononEngine::phonetic_foils_for_keyword("emergency");
    assert!(foils_emergency.contains(&"urgency"));
    assert!(foils_emergency.contains(&"agency"));
}

#[test]
fn test_keyword_discrimination_and_margin() {
    let sample_rate = 16000.0;
    let mut engine = SononEngine::new(sample_rate, 256, 128, 13);
    engine.set_feature_mode(FeatureMode::Pcen);

    // 1. Enroll "hold" using synthetic pipeline
    let keyword = "hold";
    engine.enroll_keyword_synthetic_pipeline(keyword, keyword, 5, 12, 1.40);

    // 2. Synthesize test utterances of true keyword "hold"
    let mut positive_audio = Vec::new();
    let mut male_synth = KlattSynthesizer::new(sample_rate);
    male_synth.set_f0(115.0);
    male_synth.set_speaking_rate(0.95);
    positive_audio.push(male_synth.synthesize(&sonon::G2pEngine::text_to_phonemes(keyword)));

    let mut female_synth = KlattSynthesizer::new(sample_rate);
    female_synth.set_f0(205.0);
    female_synth.set_speaking_rate(1.10);
    female_synth.set_accent(VocalAccent::ReceivedPronunciation);
    positive_audio.push(female_synth.synthesize(&sonon::G2pEngine::text_to_phonemes(keyword)));

    // 3. Synthesize negative phonetic foils ("cold", "bold", "fold", "sold")
    let foils = SononEngine::phonetic_foils_for_keyword(keyword);
    let mut negative_audio = Vec::new();
    for foil in &foils {
        let mut foil_synth = KlattSynthesizer::new(sample_rate);
        foil_synth.set_f0(140.0);
        negative_audio.push(foil_synth.synthesize(&sonon::G2pEngine::text_to_phonemes(foil)));
    }

    // 4. Run discrimination evaluation
    let report = engine.evaluate_keyword_discrimination(keyword, &positive_audio, &negative_audio);

    // Assert that true positive distance is strictly lower than negative foil distance
    assert!(
        report.discrimination_margin > 0.05,
        "Discrimination margin ({:.4}) must be positive and significant",
        report.discrimination_margin
    );
    assert!(
        report.mean_positive_distance < report.mean_negative_distance,
        "Mean positive distance ({:.4}) must be lower than negative foil distance ({:.4})",
        report.mean_positive_distance,
        report.mean_negative_distance
    );
    assert_eq!(
        report.matrix.true_positives,
        positive_audio.len(),
        "All true positive keyword utterances must be detected"
    );
}
