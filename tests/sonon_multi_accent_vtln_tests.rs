use sonon::engine::{FeatureMode, SononEngine};
use sonon::mel::MelFilterbank;
use sonon::phonetic::{KlattSynthesizer, Phoneme, VocalAccent};
use sonon::vtln::{VtlnWarpEstimator, VtlnWarping};
use sonon::zero_shot::{MultiAccentCalibrator, MultiLingualG2p, SupportedLanguage};

#[test]
fn test_vtln_piecewise_warping_properties() {
    let nyquist = 8000.0f32; // for 16 kHz sample rate

    // Test across typical alpha values (0.80 to 1.25)
    for &alpha in &[0.80f32, 0.90, 1.0, 1.10, 1.25] {
        // Boundary preservation
        let w_zero = VtlnWarping::warp_freq(0.0, alpha, nyquist);
        assert!(w_zero.abs() < 1e-4, "g_alpha(0) must be 0, got {}", w_zero);

        let w_nyq = VtlnWarping::warp_freq(nyquist, alpha, nyquist);
        assert!(
            (w_nyq - nyquist).abs() < 1e-3,
            "g_alpha(Nyquist) must equal Nyquist, got {}",
            w_nyq
        );

        // Monotonicity and inverse identity across frequency sweep
        let mut prev_w = -1.0f32;
        let num_steps = 100;
        for i in 0..=num_steps {
            let f = (i as f32 / num_steps as f32) * nyquist;
            let w = VtlnWarping::warp_freq(f, alpha, nyquist);
            assert!(
                w >= prev_w,
                "VTLN mapping must be strictly monotonic at f={}: {} < {}",
                f,
                w,
                prev_w
            );
            prev_w = w;

            let inv_f = VtlnWarping::inv_warp_freq(w, alpha, nyquist);
            assert!(
                (inv_f - f).abs() < 0.5,
                "Inverse mapping identity failed at f={}: got {}, diff={}",
                f,
                inv_f,
                (inv_f - f).abs()
            );
        }
    }
}

#[test]
fn test_vtln_mel_filterbank_and_engine_integration() {
    let sample_rate = 16000.0f32;
    let frame_size = 512;
    let hop_size = 160;
    let num_mfcc = 13;

    let mut engine = SononEngine::new(sample_rate, frame_size, hop_size, num_mfcc);

    assert_eq!(engine.vtln_alpha(), 1.0);
    assert!(!engine.is_vtln_active());

    engine.enable_vtln(1.15);
    assert!(engine.is_vtln_active());
    assert!((engine.vtln_alpha() - 1.15).abs() < 1e-4);

    // Verify filterbank construction
    let mel = MelFilterbank::new_with_vtln(26, frame_size, sample_rate, 80.0, 7600.0, 1.15);
    assert_eq!(mel.num_filters(), 26);
    assert!((mel.warping_factor() - 1.15).abs() < 1e-4);

    // Test feature extraction with explicit VTLN
    let dummy_audio = vec![0.05f32; 1600];
    let feats_canonical = engine.extract_features_with_vtln(&dummy_audio, 1.0);
    let feats_warped = engine.extract_features_with_vtln(&dummy_audio, 1.15);
    assert_eq!(feats_canonical.len(), feats_warped.len());
    assert!(!feats_canonical.is_empty());

    // Disable VTLN and verify state reset
    engine.disable_vtln();
    assert!(!engine.is_vtln_active());
    assert_eq!(engine.vtln_alpha(), 1.0);

    engine.enable_vtln(0.88);
    engine.reset();
    assert!(!engine.is_vtln_active());
    assert_eq!(engine.vtln_alpha(), 1.0);
}

#[test]
fn test_vtln_spectral_centroid_estimator_and_grid_search() {
    let mut estimator = VtlnWarpEstimator::new(1350.0);
    assert!((estimator.current_alpha() - 1.0).abs() < 1e-4);

    let sample_rate = 16000.0f32;
    let num_bins = 257; // 512 FFT
    let bin_width = (sample_rate * 0.5) / (num_bins - 1) as f32;

    // Simulate spectrum with high formant energy around 2200 Hz (shorter vocal tract / child / female)
    let mut high_spectrum = vec![0.001f32; num_bins];
    let high_bin = (2200.0 / bin_width).round() as usize;
    for k in (high_bin - 10)..=(high_bin + 10) {
        high_spectrum[k] = 10.0;
    }
    // Update estimator multiple times to allow smoothing to converge
    for _ in 0..10 {
        estimator.estimate_from_centroid(&high_spectrum, sample_rate);
    }
    let est_alpha_high = estimator.current_alpha();
    assert!(
        est_alpha_high > 1.05,
        "High spectral centroid should yield alpha > 1.05, got {}",
        est_alpha_high
    );

    // Reset and test low formant energy around 950 Hz (longer vocal tract / adult male)
    estimator.reset();
    let mut low_spectrum = vec![0.001f32; num_bins];
    let low_bin = (950.0 / bin_width).round() as usize;
    for k in (low_bin - 10)..=(low_bin + 10) {
        low_spectrum[k] = 10.0;
    }
    for _ in 0..10 {
        estimator.estimate_from_centroid(&low_spectrum, sample_rate);
    }
    let est_alpha_low = estimator.current_alpha();
    assert!(
        est_alpha_low < 0.95,
        "Low spectral centroid should yield alpha < 0.95, got {}",
        est_alpha_low
    );

    // Test grid search for optimal alpha
    let candidate_alphas = [0.85f32, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15];
    let target_optimal = 1.05f32;
    let (best_alpha, min_dist) = VtlnWarpEstimator::grid_search_optimal_alpha(
        &candidate_alphas,
        |a| (a - target_optimal).abs() * 2.5,
    );
    assert!((best_alpha - 1.05).abs() < 1e-4);
    assert!(min_dist < 1e-4);
}

#[test]
fn test_accented_articulatory_targets() {
    // All supported accents should be listed
    let supported = VocalAccent::all_supported();
    assert_eq!(supported.len(), 7);

    // Australian vowel shifts: [AE] raised (lower F1, higher F2)
    let ae_gen = Phoneme::AE.acoustic_targets_with_accent(VocalAccent::GeneralAmerican);
    let ae_aus = Phoneme::AE.acoustic_targets_with_accent(VocalAccent::Australian);
    assert!(
        ae_aus.f1 < ae_gen.f1,
        "Australian AE F1 should be lower than GenAm (vowel raising): {} vs {}",
        ae_aus.f1,
        ae_gen.f1
    );
    assert!(
        ae_aus.f2 > ae_gen.f2,
        "Australian AE F2 should be higher than GenAm (vowel fronting): {} vs {}",
        ae_aus.f2,
        ae_gen.f2
    );

    // Indian English retroflex F3 depression on alveolar consonants
    let t_gen = Phoneme::T.acoustic_targets_with_accent(VocalAccent::GeneralAmerican);
    let t_ind = Phoneme::T.acoustic_targets_with_accent(VocalAccent::IndianEnglish);
    assert!(
        t_ind.f3 < t_gen.f3,
        "Indian retroflex T should depress F3: {} vs {}",
        t_ind.f3,
        t_gen.f3
    );

    // Spanish-accented short VOT / zero aspiration on voiceless plosives
    let p_gen = Phoneme::P.acoustic_targets_with_accent(VocalAccent::GeneralAmerican);
    let p_spa = Phoneme::P.acoustic_targets_with_accent(VocalAccent::SpanishAccented);
    assert_eq!(
        p_spa.aspiration_amp, 0.0,
        "Spanish voiceless stop should have zero aspiration"
    );
    assert!(p_gen.aspiration_amp > 0.0);
}

#[test]
fn test_multi_accent_calibrator_and_zero_shot_corridor() {
    let sample_rate = 16000.0f32;
    let engine = SononEngine::new(sample_rate, 512, 160, 13);

    let accents = [
        VocalAccent::GeneralAmerican,
        VocalAccent::Australian,
        VocalAccent::IndianEnglish,
        VocalAccent::SpanishAccented,
    ];

    let calibrator = MultiAccentCalibrator::new();
    let (template, report) = calibrator
        .calibrate("abort", SupportedLanguage::English, &accents, &engine)
        .expect("MultiAccent calibration should succeed");

    assert!(!template.is_empty(), "Fused DBA template must not be empty");
    assert_eq!(report.keyword, "abort");
    assert_eq!(report.accents_evaluated.len(), 4);
    assert!(
        report.discrimination_margin > 0.0,
        "Discrimination margin must be positive: got {}",
        report.discrimination_margin
    );
    assert!(
        report.calibrated_threshold > report.max_inter_accent_distance,
        "Calibrated threshold ({}) must exceed max inter-accent distance ({})",
        report.calibrated_threshold,
        report.max_inter_accent_distance
    );
    assert!(
        report.calibrated_threshold < report.min_foil_distance,
        "Calibrated threshold ({}) must be strictly below min foil distance ({})",
        report.calibrated_threshold,
        report.min_foil_distance
    );
    assert_eq!(
        report.confusion_matrix.false_negatives, 0,
        "All accented variations must fall within calibrated decision corridor"
    );
    assert!(report.is_universally_separated);
}

#[test]
fn test_engine_enroll_and_streaming_spotting_multi_accent() {
    let sample_rate = 16000.0f32;
    let mut engine = SononEngine::new(sample_rate, 512, 160, 13);
    engine.set_feature_mode(FeatureMode::LogMel);

    let target_accents = [
        VocalAccent::GeneralAmerican,
        VocalAccent::Australian,
        VocalAccent::IndianEnglish,
    ];

    // Enroll keyword "abort" using multi-accent calibration
    let report = engine
        .enroll_keyword_multi_accent(
            "abort",
            "abort",
            SupportedLanguage::English,
            &target_accents,
        )
        .expect("Enroll multi-accent should succeed");

    assert!(report.is_universally_separated);

    // Synthesize audio using Indian English accent
    let mut indian_synth = KlattSynthesizer::new(sample_rate);
    indian_synth.set_accent(VocalAccent::IndianEnglish);
    indian_synth.set_f0(145.0);
    let phonemes = MultiLingualG2p::text_to_phonemes("abort", SupportedLanguage::English);
    let indian_audio = indian_synth.synthesize(&phonemes);
    assert!(!indian_audio.is_empty());

    // Stream silence prefix, accented audio, and trailing silence into engine
    let mut spotted_indian = false;
    let prefix_silence = vec![0.0f32; 1600];
    let suffix_silence = vec![0.0f32; 3200];

    engine.ingest_samples(&prefix_silence);
    for chunk in indian_audio.chunks(160) {
        let events = engine.ingest_samples(chunk);
        for ev in events {
            if ev.keyword == "abort" {
                spotted_indian = true;
            }
        }
    }
    for chunk in suffix_silence.chunks(160) {
        let events = engine.ingest_samples(chunk);
        for ev in events {
            if ev.keyword == "abort" {
                spotted_indian = true;
            }
        }
    }

    assert!(
        spotted_indian,
        "Engine must spot keyword 'abort' spoken with Indian English accent"
    );

    // Reset engine and spot with Australian English accent
    engine.reset();
    // Re-enroll after reset
    engine
        .enroll_keyword_multi_accent(
            "abort",
            "abort",
            SupportedLanguage::English,
            &target_accents,
        )
        .unwrap();

    let mut aus_synth = KlattSynthesizer::new(sample_rate);
    aus_synth.set_accent(VocalAccent::Australian);
    aus_synth.set_f0(160.0);
    let aus_audio = aus_synth.synthesize(&phonemes);

    let mut spotted_aus = false;
    engine.ingest_samples(&prefix_silence);
    for chunk in aus_audio.chunks(160) {
        let events = engine.ingest_samples(chunk);
        for ev in events {
            if ev.keyword == "abort" {
                spotted_aus = true;
            }
        }
    }
    for chunk in suffix_silence.chunks(160) {
        let events = engine.ingest_samples(chunk);
        for ev in events {
            if ev.keyword == "abort" {
                spotted_aus = true;
            }
        }
    }

    assert!(
        spotted_aus,
        "Engine must spot keyword 'abort' spoken with Australian accent"
    );
}
