//! Comprehensive integration tests for continual few-shot domain adaptation,
//! bounded exemplar memory consolidation with Fisher information pruning,
//! Online Dynamic Time Warping Barycenter Averaging (Online DBA),
//! and edge active learning trigger for borderline wake-word speech.

use sonon::adaptation::{
    ActiveLearningConfig, ActiveLearningTrigger, ExemplarMemoryBuffer,
    FisherInformationEvaluator, OnlineDbaUpdater,
};
use sonon::dtw::{DtwMatcher, StreamingDtwConfig};
use sonon::engine::{FeatureMode, SononEngine};
use sonon::phonetic::{SyntheticExemplarGenerator, VocalAccent};
use sonon::zero_shot::SupportedLanguage;

#[test]
fn test_fisher_information_importance_and_acoustic_diversity() {
    let d = 13;
    let len = 20;

    // Base reference template: synthetic stationary vowel trajectory
    let ref_template: Vec<Vec<f32>> = (0..len)
        .map(|t| {
            let mut frame = vec![0.0f32; d];
            frame[0] = 18.0; // log energy
            frame[1] = -2.5 + 0.1 * (t as f32);
            frame[2] = 3.0;
            frame[3] = 1.2;
            frame
        })
        .collect();

    let feature_variances = vec![1.2f32; d];
    let band_radius = 8;

    // 1. Identical exemplar: small Fisher divergence
    let identical_exemplar = ref_template.clone();
    let score_ident = FisherInformationEvaluator::compute_importance(
        &identical_exemplar,
        &ref_template,
        &feature_variances,
        band_radius,
    );
    assert!(
        score_ident < 0.15,
        "Identical exemplar Fisher importance {} should be near zero",
        score_ident
    );

    // 2. Shifted / accented exemplar: distinct acoustic realization
    let shifted_exemplar: Vec<Vec<f32>> = (0..len)
        .map(|t| {
            let mut frame = vec![0.0f32; d];
            frame[0] = 19.5;
            frame[1] = -1.0 + 0.2 * (t as f32);
            frame[2] = 5.5; // Significant F2 shift
            frame[3] = -0.8;
            frame
        })
        .collect();

    let score_shifted = FisherInformationEvaluator::compute_importance(
        &shifted_exemplar,
        &ref_template,
        &feature_variances,
        band_radius,
    );
    assert!(
        score_shifted > score_ident * 5.0,
        "Shifted exemplar importance {} must significantly exceed identical {}",
        score_shifted,
        score_ident
    );

    // 3. Pairwise acoustic diversity distance
    let dist_self = FisherInformationEvaluator::acoustic_distance(&ref_template, &ref_template, band_radius);
    assert!(
        dist_self < 1e-4,
        "Self distance {} must be zero",
        dist_self
    );

    let dist_cross = FisherInformationEvaluator::acoustic_distance(&ref_template, &shifted_exemplar, band_radius);
    assert!(
        dist_cross > 1.0,
        "Cross exemplar acoustic distance {} must be distinctly positive",
        dist_cross
    );
}

#[test]
fn test_exemplar_memory_buffer_bounded_retention_and_anchor_preservation() {
    let d = 13;
    let len = 18;
    let band_radius = 6;
    let capacity = 4;

    let mut buffer = ExemplarMemoryBuffer::new("take off", capacity, band_radius);
    assert_eq!(buffer.len(), 0);
    assert_eq!(buffer.capacity(), capacity);

    // 1. Set immutable enrollment anchor
    let anchor_template: Vec<Vec<f32>> = (0..len)
        .map(|_| vec![1.0f32; d])
        .collect();
    buffer.set_anchor_exemplar(anchor_template.clone(), 24.0);

    assert_eq!(buffer.len(), 1);
    assert!(buffer.anchor().is_some());
    assert!(buffer.anchor().unwrap().is_anchor);

    // 2. Insert candidate exemplars up to capacity
    for k in 1..4 {
        let ex: Vec<Vec<f32>> = (0..len)
            .map(|t| {
                let mut f = vec![1.0f32 + 0.3 * (k as f32); d];
                f[1] += 0.1 * (t as f32);
                f
            })
            .collect();
        let admitted = buffer.insert_exemplar(ex, (k as f32) * 5.0, 18.0, &anchor_template);
        assert!(admitted, "Candidate {} should be admitted", k);
    }
    assert_eq!(buffer.len(), capacity);

    // 3. Overflow buffer with a highly informative exemplar
    let novel_exemplar: Vec<Vec<f32>> = (0..len)
        .map(|_| vec![4.5f32; d]) // Distinct acoustic region
        .collect();

    let admitted_novel = buffer.insert_exemplar(novel_exemplar, 25.0, 20.0, &anchor_template);
    assert!(admitted_novel, "High-diversity novel exemplar must be admitted");

    // Capacity must remain strictly bounded
    assert_eq!(
        buffer.len(),
        capacity,
        "Buffer length {} must not exceed capacity {}",
        buffer.len(),
        capacity
    );

    // Anchor exemplar must NEVER be pruned
    let has_anchor = buffer.exemplars().iter().any(|e| e.is_anchor);
    assert!(
        has_anchor,
        "Immutable enrollment anchor must be strictly preserved in memory buffer"
    );
}

#[test]
fn test_online_dba_incremental_update_and_anchor_drift_bounds() {
    let d = 13;
    let len = 16;
    let band_radius = 6;
    let max_drift_radius = 3.0f32;
    let learning_rate = 0.20f32;

    let updater = OnlineDbaUpdater::new(learning_rate, max_drift_radius, band_radius);

    let anchor: Vec<Vec<f32>> = (0..len).map(|_| vec![2.0f32; d]).collect();
    let mut current_centroid = anchor.clone();

    // Observation shifted in positive direction
    let observation: Vec<Vec<f32>> = (0..len).map(|_| vec![3.5f32; d]).collect();

    // 1. Single incremental update
    let updated_1 = updater
        .update_centroid(&current_centroid, &observation, &anchor)
        .expect("Update failed");

    assert_eq!(updated_1.len(), len);
    // Centroid must have shifted towards observation: (1 - 0.2)*2.0 + 0.2*3.5 = 1.6 + 0.7 = 2.3
    let val_1 = updated_1[0][0];
    assert!(
        (val_1 - 2.3).abs() < 0.15,
        "Updated value {} should be ~2.3",
        val_1
    );

    // 2. Stress test: Apply 50 consecutive extreme observations to test anchor drift bounds
    let extreme_obs: Vec<Vec<f32>> = (0..len).map(|_| vec![15.0f32; d]).collect();
    current_centroid = updated_1;

    for _ in 0..50 {
        current_centroid = updater
            .update_centroid(&current_centroid, &extreme_obs, &anchor)
            .expect("Continuous update failed");
    }

    // Distance to anchor must be strictly bounded by max_drift_radius (plus small tolerance)
    let drift = DtwMatcher::compute_distance_banded(&current_centroid, &anchor, band_radius);
    assert!(
        drift <= max_drift_radius + 0.25,
        "Drift from anchor {} exceeded max allowable drift radius {}",
        drift,
        max_drift_radius
    );
}

#[test]
fn test_edge_active_learning_trigger_uncertainty_band() {
    let config = ActiveLearningConfig {
        uncertainty_band_ratio: 0.20, // +/- 20% around threshold
        min_snr_db: 6.0,
        trigger_uncertainty_threshold: 0.30,
        auto_admit_confidence: 0.90,
        ..Default::default()
    };

    let mut trigger = ActiveLearningTrigger::new(config);

    let threshold = 5.0f32; // Calibrated DTW decision threshold
    let features: Vec<Vec<f32>> = vec![vec![1.0; 13]; 15];
    let audio: Vec<f32> = vec![0.1; 4800];

    // 1. High-confidence positive match: dist = 1.5 << 5.0 (outside uncertainty band)
    let cand_clear = trigger.evaluate_window(
        "take off",
        1.5,
        threshold,
        15.0, // High SNR
        1.0,
        features.clone(),
        audio.clone(),
    );
    assert!(
        cand_clear.is_none(),
        "High confidence match should not trigger active learning"
    );

    // 2. Far out-of-vocabulary negative: dist = 9.0 >> 5.0
    let cand_far = trigger.evaluate_window(
        "take off",
        9.0,
        threshold,
        12.0,
        2.0,
        features.clone(),
        audio.clone(),
    );
    assert!(
        cand_far.is_none(),
        "Far negative non-match should not trigger active learning"
    );

    // 3. Low SNR noise segment: dist = 5.1 (borderline), but SNR = 2.0 dB (too noisy)
    let cand_noisy = trigger.evaluate_window(
        "take off",
        5.1,
        threshold,
        2.0, // Low SNR
        3.0,
        features.clone(),
        audio.clone(),
    );
    assert!(
        cand_noisy.is_none(),
        "Low SNR speech must be rejected from active learning trigger"
    );

    // 4. Borderline ambiguous speech: dist = 5.2 (within +/- 1.0 band) with good SNR
    let cand_borderline = trigger.evaluate_window(
        "take off",
        5.2,
        threshold,
        14.0,
        4.0,
        features.clone(),
        audio.clone(),
    );
    assert!(
        cand_borderline.is_some(),
        "Borderline speech within uncertainty band must trigger active learning"
    );

    let candidate = cand_borderline.unwrap();
    assert_eq!(candidate.keyword, "take off");
    assert!(
        candidate.uncertainty_score > 0.70,
        "Uncertainty score {} must be high near decision boundary",
        candidate.uncertainty_score
    );
    assert!(
        candidate.requires_verification,
        "Borderline candidate must require verification"
    );

    // 5. Verify pending candidates ring buffer
    assert_eq!(trigger.pending_candidates().len(), 1);
    assert_eq!(trigger.pending_candidates()[0].candidate_id, candidate.candidate_id);
}

#[test]
fn test_sonon_engine_streaming_continual_adaptation_and_noise_shift() {
    let sample_rate = 16000.0f32;
    let mut engine = SononEngine::new(sample_rate, 512, 160, 13);
    engine.set_feature_mode(FeatureMode::LogMel);
    let dtw_cfg = StreamingDtwConfig {
        min_scale: 0.85,
        ..Default::default()
    };
    engine.set_streaming_dtw_config(dtw_cfg);

    // 1. Zero-shot enroll keyword "take off"
    let rep = engine
        .enroll_keyword_zero_shot(
            "take off",
            "take off",
            SupportedLanguage::English,
            VocalAccent::GeneralAmerican,
        )
        .expect("Zero-shot enrollment failed");

    assert_eq!(rep.keyword, "take off");
    assert!(rep.calibrated_threshold > 0.0);

    // 2. Enable continual few-shot adaptation with capacity 6
    engine
        .enable_continual_adaptation("take off", 6)
        .expect("Enable continual adaptation failed");

    assert!(engine.continual_adaptation().is_some());
    let init_telemetry = engine.latest_adaptation_telemetry().unwrap();
    assert_eq!(init_telemetry.keyword, "take off");
    assert_eq!(init_telemetry.exemplar_count, 1); // Anchor initialized
    assert_eq!(init_telemetry.total_updates_applied, 0);

    // 3. Synthesize natural speech variation of "take off"
    let synth = SyntheticExemplarGenerator::new(sample_rate);
    let segments = sonon::zero_shot::MultiLingualG2p::text_to_phonemes("take off", SupportedLanguage::English);
    let target_audio = synth.generate_exemplars_from_segments(&segments, 1);
    assert!(!target_audio.is_empty());

    // Stream through engine
    let mut padded_stream = vec![0.0f32; 4800]; // 0.3s silence lead
    padded_stream.extend_from_slice(&target_audio[0]);
    padded_stream.extend_from_slice(&vec![0.0f32; 4800]); // 0.3s silence trail

    let events = engine.ingest_samples(&padded_stream);
    assert!(
        events.iter().any(|e| e.keyword == "take off"),
        "Streaming keyword 'take off' must be spotted"
    );

    // Check that continual adaptation registered observation
    let updated_telemetry = engine.latest_adaptation_telemetry().unwrap();
    assert!(
        updated_telemetry.total_updates_applied >= 1,
        "Continual adaptation must have applied at least 1 online update (got {})",
        updated_telemetry.total_updates_applied
    );
    assert!(
        updated_telemetry.drift_from_anchor <= updated_telemetry.max_drift_radius,
        "Drift from anchor {} must remain bounded",
        updated_telemetry.drift_from_anchor
    );

    // 4. Verify minimal-pair foil rejection remains 100% robust after adaptation
    engine.reset();
    let foil_segments = sonon::zero_shot::MultiLingualG2p::text_to_phonemes("shake off", SupportedLanguage::English);
    let foil_audio = synth.generate_exemplars_from_segments(&foil_segments, 1);
    assert!(!foil_audio.is_empty());

    let mut padded_foil = vec![0.0f32; 4800];
    padded_foil.extend_from_slice(&foil_audio[0]);
    padded_foil.extend_from_slice(&vec![0.0f32; 4800]);

    let foil_events = engine.ingest_samples(&padded_foil);
    assert_eq!(
        foil_events.len(),
        0,
        "Phonetic foil 'shake off' must not trigger false alarms after continual adaptation"
    );
}
