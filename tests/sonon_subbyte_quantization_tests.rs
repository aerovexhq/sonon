//! Comprehensive integration tests for ultra-low-bitrate acoustic quantization,
//! non-uniform Lloyd-Max codebook optimization, sub-byte weight packing (1-bit, 2-bit, 4-bit),
//! Asymmetric Distance Computation (ADC) with precomputed LUTs, and edge KWS dictionary compression.

use sonon::engine::{FeatureMode, SononEngine};
use sonon::phonetic::{SyntheticExemplarGenerator, VocalAccent};
use sonon::subbyte::{
    LloydMaxTrainer, SubByteBitWidth, SubByteCodebook, SubByteDtwMatcher,
    SubBytePackedFrame, SubBytePhraseTemplate,
};
use sonon::zero_shot::SupportedLanguage;

#[test]
fn test_lloyd_max_scalar_quantizer_convergence_and_codebook_properties() {
    // Generate synthetic speech feature distribution (mixture of log-energies and formants)
    let n_samples = 2000;
    let mut samples = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let t = i as f32 * 0.01;
        // Bimodal distribution representative of vowel formant resonances and fricative energy
        let val = if i % 2 == 0 {
            -2.0 + 0.8 * (t * 1.5).sin()
        } else {
            3.5 + 1.2 * (t * 2.3).cos()
        };
        samples.push(val);
    }

    let trainer = LloydMaxTrainer::new(50, 1e-6);

    // 1. Train 1-bit codebook (2 centroids)
    let cb_1bit = trainer.train_1d(&samples, SubByteBitWidth::OneBit);
    assert_eq!(cb_1bit.normalized_centroids.len(), 2);
    assert_eq!(cb_1bit.normalized_boundaries.len(), 1);
    assert!(cb_1bit.normalized_centroids[0] < cb_1bit.normalized_centroids[1]);
    assert!(cb_1bit.normalized_centroids[0] < cb_1bit.normalized_boundaries[0]);
    assert!(cb_1bit.normalized_boundaries[0] < cb_1bit.normalized_centroids[1]);

    // 2. Train 2-bit codebook (4 centroids)
    let cb_2bit = trainer.train_1d(&samples, SubByteBitWidth::TwoBit);
    assert_eq!(cb_2bit.normalized_centroids.len(), 4);
    assert_eq!(cb_2bit.normalized_boundaries.len(), 3);
    for i in 0..3 {
        assert!(cb_2bit.normalized_centroids[i] < cb_2bit.normalized_centroids[i + 1]);
        assert!(cb_2bit.normalized_boundaries[i] > cb_2bit.normalized_centroids[i]);
        assert!(cb_2bit.normalized_boundaries[i] < cb_2bit.normalized_centroids[i + 1]);
    }

    // 3. Train 4-bit codebook (16 centroids)
    let cb_4bit = trainer.train_1d(&samples, SubByteBitWidth::FourBit);
    assert_eq!(cb_4bit.normalized_centroids.len(), 16);
    assert_eq!(cb_4bit.normalized_boundaries.len(), 15);
    for i in 0..15 {
        assert!(cb_4bit.normalized_centroids[i] < cb_4bit.normalized_centroids[i + 1]);
        assert!(cb_4bit.normalized_boundaries[i] > cb_4bit.normalized_centroids[i]);
        assert!(cb_4bit.normalized_boundaries[i] < cb_4bit.normalized_centroids[i + 1]);
    }

    // 4. Verify Mean Squared Error (MSE) strictly decreases with increasing bit-width
    let calc_mse = |cb: &SubByteCodebook| -> f32 {
        let mut err_sum = 0.0f32;
        for &x in &samples {
            let q = cb.quantize_scalar(0, x);
            let recon = cb.dequantize_index(0, q);
            err_sum += (x - recon) * (x - recon);
        }
        err_sum / (samples.len() as f32)
    };

    let mse_1bit = calc_mse(&cb_1bit);
    let mse_2bit = calc_mse(&cb_2bit);
    let mse_4bit = calc_mse(&cb_4bit);

    assert!(
        mse_4bit < mse_2bit,
        "4-bit MSE ({}) must be lower than 2-bit MSE ({})",
        mse_4bit,
        mse_2bit
    );
    assert!(
        mse_2bit < mse_1bit,
        "2-bit MSE ({}) must be lower than 1-bit MSE ({})",
        mse_2bit,
        mse_1bit
    );
    assert!(
        mse_4bit < 0.15,
        "4-bit Lloyd-Max reconstruction MSE must be tight (< 0.15, got {})",
        mse_4bit
    );
}

#[test]
fn test_subbyte_bit_packing_and_unpacking_roundtrip() {
    let dim = 13; // 13-dim MFCC feature vector

    // 1. Test 4-bit packing (2 values per byte)
    let indices_4bit: Vec<u8> = (0..dim).map(|i| (i % 16) as u8).collect();
    let packed_4bit = SubBytePackedFrame::from_indices(&indices_4bit, SubByteBitWidth::FourBit);
    assert_eq!(packed_4bit.packed_bytes.len(), 7); // ceil(13 / 2) = 7 bytes vs 52 bytes (float32)

    let unpacked_4bit = packed_4bit.unpack_indices();
    assert_eq!(unpacked_4bit, indices_4bit);
    for (i, &expected) in indices_4bit.iter().enumerate() {
        assert_eq!(packed_4bit.get_index(i), expected);
    }

    // 2. Test 2-bit packing (4 values per byte)
    let indices_2bit: Vec<u8> = (0..dim).map(|i| (i % 4) as u8).collect();
    let packed_2bit = SubBytePackedFrame::from_indices(&indices_2bit, SubByteBitWidth::TwoBit);
    assert_eq!(packed_2bit.packed_bytes.len(), 4); // ceil(13 / 4) = 4 bytes vs 52 bytes (float32)

    let unpacked_2bit = packed_2bit.unpack_indices();
    assert_eq!(unpacked_2bit, indices_2bit);
    for (i, &expected) in indices_2bit.iter().enumerate() {
        assert_eq!(packed_2bit.get_index(i), expected);
    }

    // 3. Test 1-bit packing (8 values per byte)
    let indices_1bit: Vec<u8> = (0..dim).map(|i| (i % 2) as u8).collect();
    let packed_1bit = SubBytePackedFrame::from_indices(&indices_1bit, SubByteBitWidth::OneBit);
    assert_eq!(packed_1bit.packed_bytes.len(), 2); // ceil(13 / 8) = 2 bytes vs 52 bytes (float32)

    let unpacked_1bit = packed_1bit.unpack_indices();
    assert_eq!(unpacked_1bit, indices_1bit);
    for (i, &expected) in indices_1bit.iter().enumerate() {
        assert_eq!(packed_1bit.get_index(i), expected);
    }
}

#[test]
fn test_subbyte_phrase_template_compression_ratio() {
    let dim = 13;
    let frames_len = 50;

    // Synthetic 50-frame feature template
    let features: Vec<Vec<f32>> = (0..frames_len)
        .map(|t| {
            (0..dim)
                .map(|d| (t as f32 * 0.1 + d as f32 * 0.2).sin() * 5.0)
                .collect()
        })
        .collect();

    let _uncompressed_raw_bytes = frames_len * dim * 4; // 50 * 13 * 4 = 2600 bytes

    // 4-bit template
    let tmpl_4bit = SubBytePhraseTemplate::from_features(
        "take off",
        &features,
        SubByteBitWidth::FourBit,
        3.5,
        8,
    );
    let footprint_4bit = tmpl_4bit.memory_footprint_bytes();
    let ratio_4bit = tmpl_4bit.compression_ratio();
    assert!(
        footprint_4bit < 650,
        "4-bit template footprint {} bytes must be < 650 bytes",
        footprint_4bit
    );
    assert!(
        ratio_4bit > 4.0,
        "4-bit template compression ratio {} must exceed 4.0x",
        ratio_4bit
    );

    // 2-bit template
    let tmpl_2bit = SubBytePhraseTemplate::from_features(
        "take off",
        &features,
        SubByteBitWidth::TwoBit,
        3.5,
        8,
    );
    let footprint_2bit = tmpl_2bit.memory_footprint_bytes();
    let ratio_2bit = tmpl_2bit.compression_ratio();
    assert!(
        footprint_2bit < footprint_4bit,
        "2-bit footprint {} must be smaller than 4-bit footprint {}",
        footprint_2bit,
        footprint_4bit
    );
    assert!(
        ratio_2bit > 7.0,
        "2-bit template compression ratio {} must exceed 7.0x",
        ratio_2bit
    );

    // 1-bit template
    let tmpl_1bit = SubBytePhraseTemplate::from_features(
        "take off",
        &features,
        SubByteBitWidth::OneBit,
        3.5,
        8,
    );
    let footprint_1bit = tmpl_1bit.memory_footprint_bytes();
    let ratio_1bit = tmpl_1bit.compression_ratio();
    assert!(
        footprint_1bit < footprint_2bit,
        "1-bit footprint {} must be smaller than 2-bit footprint {}",
        footprint_1bit,
        footprint_2bit
    );
    assert!(
        ratio_1bit > 10.0,
        "1-bit template compression ratio {} must exceed 10.0x",
        ratio_1bit
    );
}

#[test]
fn test_asymmetric_distance_computation_adc_precision_and_speed() {
    let dim = 13;
    let n_frames = 40;

    let template_features: Vec<Vec<f32>> = (0..n_frames)
        .map(|t| {
            (0..dim)
                .map(|d| (t as f32 * 0.15 + d as f32 * 0.3).sin() * 4.0)
                .collect()
        })
        .collect();

    // Query sequence with minor perturbations
    let query_features: Vec<Vec<f32>> = (0..n_frames)
        .map(|t| {
            (0..dim)
                .map(|d| (t as f32 * 0.15 + d as f32 * 0.3).sin() * 4.0 + 0.2 * (d as f32).cos())
                .collect()
        })
        .collect();

    let tmpl_4bit = SubBytePhraseTemplate::from_features(
        "take off",
        &template_features,
        SubByteBitWidth::FourBit,
        5.0,
        8,
    );

    // Compute exact continuous float DTW distance
    let dist_float = sonon::dtw::DtwMatcher::compute_distance_banded(
        &query_features,
        &template_features,
        8,
    );

    // Compute Asymmetric Distance Computation (ADC) sub-byte DTW distance
    let dist_adc_4bit = SubByteDtwMatcher::compute_distance_adc_banded(&query_features, &tmpl_4bit);

    assert!(dist_float.is_finite());
    assert!(dist_adc_4bit.is_finite());

    // Absolute distance approximation error must be < 0.15 for 4-bit ADC
    let abs_err_4bit = (dist_float - dist_adc_4bit).abs();
    assert!(
        abs_err_4bit < 0.15,
        "4-bit ADC absolute distance error {} must be < 0.15",
        abs_err_4bit
    );

    // 2-bit ADC evaluation
    let tmpl_2bit = SubBytePhraseTemplate::from_features(
        "take off",
        &template_features,
        SubByteBitWidth::TwoBit,
        5.0,
        8,
    );
    let dist_adc_2bit = SubByteDtwMatcher::compute_distance_adc_banded(&query_features, &tmpl_2bit);
    let abs_err_2bit = (dist_float - dist_adc_2bit).abs();
    assert!(
        abs_err_2bit < 0.95,
        "2-bit ADC absolute distance error {} must be < 0.95",
        abs_err_2bit
    );

    // Benchmark throughput: evaluate 500 frame distances
    let start_time = std::time::Instant::now();
    let lut = SubByteDtwMatcher::build_adc_lut(&query_features[0], &tmpl_4bit.codebook);
    let mut dummy_sum = 0.0f32;
    for _ in 0..1000 {
        dummy_sum += SubByteDtwMatcher::frame_distance_adc(&lut, &tmpl_4bit.frames[0], 16);
    }
    let elapsed = start_time.elapsed();
    assert!(dummy_sum > 0.0);
    assert!(
        elapsed.as_millis() < 50,
        "1000 ADC frame distance evaluations must complete in < 50ms (got {:?})",
        elapsed
    );
}

#[test]
fn test_sonon_engine_subbyte_kws_export_and_streaming_spotting() {
    let sample_rate = 16000.0f32;
    let mut engine = SononEngine::new(sample_rate, 512, 160, 13);
    engine.set_feature_mode(FeatureMode::LogMel);

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

    // 2. Export 4-bit sub-byte phrase template and build sub-byte matcher
    let tmpl_4bit = engine
        .export_subbyte_template("take off", SubByteBitWidth::FourBit)
        .expect("Export sub-byte template failed");

    assert_eq!(tmpl_4bit.name, "take off");
    assert!(!tmpl_4bit.is_empty());
    assert!(tmpl_4bit.compression_ratio() > 5.0);

    let mut subbyte_matcher = SubByteDtwMatcher::new();
    subbyte_matcher.add_template(tmpl_4bit.clone());

    // 3. Synthesize speech of target "take off"
    let synth = SyntheticExemplarGenerator::new(sample_rate);
    let target_segments =
        sonon::zero_shot::MultiLingualG2p::text_to_phonemes("take off", SupportedLanguage::English);
    let target_audio = synth.generate_exemplars_from_segments(&target_segments, 1);
    assert!(!target_audio.is_empty());

    let target_features = engine.extract_features(&target_audio[0]);
    assert!(!target_features.is_empty());

    // Evaluate matching using sub-byte ADC matcher
    let dtw_cfg = sonon::dtw::StreamingDtwConfig {
        min_scale: 0.85,
        ..Default::default()
    };
    let match_res = subbyte_matcher.match_streaming_window(
        &target_features,
        0.000001,
        &dtw_cfg,
    );

    assert!(
        match_res.is_some(),
        "Target audio 'take off' must be spotted by sub-byte ADC matcher"
    );
    let res = match_res.unwrap();
    assert_eq!(res.keyword, "take off");
    assert!(res.confidence > 0.0);

    // 4. Test minimal-pair foil rejection: synthesize "shake off"
    let foil_segments =
        sonon::zero_shot::MultiLingualG2p::text_to_phonemes("shake off", SupportedLanguage::English);
    let foil_audio = synth.generate_exemplars_from_segments(&foil_segments, 1);
    assert!(!foil_audio.is_empty());

    let foil_features = engine.extract_features(&foil_audio[0]);
    let foil_match = subbyte_matcher.match_streaming_window(
        &foil_features,
        0.000001,
        &dtw_cfg,
    );

    assert!(
        foil_match.is_none(),
        "Phonetic foil 'shake off' must not trigger false alarms in sub-byte ADC matcher"
    );
}
