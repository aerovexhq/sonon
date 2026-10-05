//! Integration and unit tests for Non-Autoregressive Conditional Flow Matching (CFM)
//! Diffusion Transformer (DiT) in Sonon.

#![deny(unsafe_code)]

use sonon::flow_matching::{
    AdaLayerNormZero, CfmConfig, CfmSpeechSynthesizer, CrossAttention, DeterministicRng,
    DiTBlock, FlowConditioning, FlowMatchingDiT, FlowOdeSolver, FlowSolverScheme,
    MultiHeadAttention, OptimalTransportPath, SpeakerPromptEmbedding,
    TextConditioningEncoder, TimestepEmbedding,
};
use sonon::SononEngine;

#[test]
fn test_optimal_transport_path_interpolation_and_velocity_accuracy() {
    let ot = OptimalTransportPath::new(1e-4);
    assert_eq!(ot.sigma_min, 1e-4);

    let num_frames = 8;
    let num_channels = 80;

    // Construct deterministic synthetic prior noise x0 and target speech latent x1
    let mut x0 = Vec::with_capacity(num_frames);
    let mut x1 = Vec::with_capacity(num_frames);
    for i in 0..num_frames {
        let mut row0 = Vec::with_capacity(num_channels);
        let mut row1 = Vec::with_capacity(num_channels);
        for j in 0..num_channels {
            row0.push(((i * 7 + j * 13) as f32 * 0.1).sin());
            row1.push(((i * 11 + j * 17) as f32 * 0.1).cos() * 2.0);
        }
        x0.push(row0);
        x1.push(row1);
    }

    // 1. Endpoint test: t = 0.0 should exactly equal x0
    let xt_0 = ot.interpolate(&x0, &x1, 0.0);
    assert_eq!(xt_0.len(), num_frames);
    for i in 0..num_frames {
        for j in 0..num_channels {
            let diff = (xt_0[i][j] - x0[i][j]).abs();
            assert!(diff < 1e-6, "At t=0, xt must equal x0");
        }
    }

    // 2. Endpoint test: t = 1.0 should equal sigma_min * x0 + x1
    let xt_1 = ot.interpolate(&x0, &x1, 1.0);
    for i in 0..num_frames {
        for j in 0..num_channels {
            let expected = 1e-4 * x0[i][j] + x1[i][j];
            let diff = (xt_1[i][j] - expected).abs();
            assert!(diff < 1e-5, "At t=1, xt must equal sigma_min * x0 + x1");
        }
    }

    // 3. Midpoint test: t = 0.5
    let xt_mid = ot.interpolate(&x0, &x1, 0.5);
    for i in 0..num_frames {
        for j in 0..num_channels {
            let expected = (1.0 - 0.5 * (1.0 - 1e-4)) * x0[i][j] + 0.5 * x1[i][j];
            let diff = (xt_mid[i][j] - expected).abs();
            assert!(diff < 1e-5, "At t=0.5, interpolation formula must hold");
        }
    }

    // 4. Target velocity accuracy test: compare analytical velocity to numerical finite differences
    let target_vel = ot.target_velocity(&x0, &x1);
    let t_eval = 0.35f32;
    let dt = 1e-3f32;
    let xt_a = ot.interpolate(&x0, &x1, t_eval);
    let xt_b = ot.interpolate(&x0, &x1, t_eval + dt);

    for i in 0..num_frames {
        for j in 0..num_channels {
            let num_deriv = (xt_b[i][j] - xt_a[i][j]) / dt;
            let analytical = target_vel[i][j];
            let diff = (num_deriv - analytical).abs();
            assert!(
                diff < 5e-3,
                "Finite difference derivative must match target velocity: num={}, ana={}",
                num_deriv,
                analytical
            );
        }
    }

    // 5. Flat slice interpolation tests
    let flat_x0 = vec![1.0, 2.0, 3.0];
    let flat_x1 = vec![4.0, 5.0, 6.0];
    let flat_interp = ot.interpolate_flat(&flat_x0, &flat_x1, 0.5);
    assert_eq!(flat_interp.len(), 3);
    let flat_vel = ot.target_velocity_flat(&flat_x0, &flat_x1);
    assert_eq!(flat_vel.len(), 3);
}

#[test]
fn test_sinusoidal_timestep_embedding_and_adaln_zero_modulation() {
    let embedding_dim = 64;
    let hidden_dim = 128;
    let te = TimestepEmbedding::new(embedding_dim, hidden_dim);

    assert_eq!(te.embedding_dim, embedding_dim);
    assert_eq!(te.hidden_dim, hidden_dim);

    // Timestep embeddings for distinct times
    let emb_0 = te.forward(0.0);
    let emb_mid = te.forward(0.5);
    let emb_1 = te.forward(1.0);

    assert_eq!(emb_0.len(), hidden_dim);
    assert_eq!(emb_mid.len(), hidden_dim);
    assert_eq!(emb_1.len(), hidden_dim);

    // Verify embeddings are finite and non-identical
    for &v in &emb_0 {
        assert!(v.is_finite());
    }
    assert_ne!(emb_0, emb_1, "Embeddings at t=0 and t=1 must be distinct");
    assert_ne!(emb_0, emb_mid, "Embeddings at t=0 and t=0.5 must be distinct");

    // Test AdaLayerNormZero
    let adaln = AdaLayerNormZero::new(hidden_dim, 42);
    let x: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.1).sin()).collect();

    let (mod_x, alpha) = adaln.modulate_latent(&x, &emb_0);
    assert_eq!(mod_x.len(), hidden_dim);
    assert_eq!(alpha.len(), hidden_dim);

    for &val in &mod_x {
        assert!(val.is_finite());
    }
    for &val in &alpha {
        assert!(val.is_finite());
    }

    // Test gating: x + alpha * sublayer_output
    let sublayer_out: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.2).cos()).collect();
    let gated = adaln.gate(&x, &sublayer_out, &alpha);
    assert_eq!(gated.len(), hidden_dim);
    for i in 0..hidden_dim {
        let expected = x[i] + alpha[i] * sublayer_out[i];
        assert!((gated[i] - expected).abs() < 1e-6);
    }

    // Test sequence modulation
    let seq = vec![x.clone(), x.clone()];
    let (mod_seq, seq_alpha) = adaln.modulate_sequence(&seq, &emb_0);
    assert_eq!(mod_seq.len(), 2);
    assert_eq!(seq_alpha.len(), hidden_dim);

    // Test zero-initialized AdaLayerNormZero identity gating property
    let zero_adaln = AdaLayerNormZero::new_zero_init(hidden_dim);
    let (_, zero_alpha) = zero_adaln.modulate_latent(&x, &emb_0);
    for &a in &zero_alpha {
        assert_eq!(a, 0.0, "Zero-init alpha must be exactly zero");
    }
    let zero_gated = zero_adaln.gate(&x, &sublayer_out, &zero_alpha);
    for i in 0..hidden_dim {
        assert_eq!(zero_gated[i], x[i], "With zero alpha, gating must be identity");
    }
}

#[test]
fn test_multi_head_self_attention_and_cross_attention() {
    let hidden_dim = 64;
    let num_heads = 4;
    let seq_len = 10;

    // 1. MultiHeadAttention (Self-Attention)
    let mha = MultiHeadAttention::new(hidden_dim, num_heads, 11);
    let mut input = Vec::with_capacity(seq_len);
    for i in 0..seq_len {
        let frame: Vec<f32> = (0..hidden_dim)
            .map(|j| ((i * 7 + j * 13) as f32 * 0.05).sin())
            .collect();
        input.push(frame);
    }

    let (out_mha, attn_weights) = mha.forward_with_weights(&input);
    assert_eq!(out_mha.len(), seq_len);
    for row in &out_mha {
        assert_eq!(row.len(), hidden_dim);
        for &val in row {
            assert!(val.is_finite());
        }
    }

    // Verify self-attention weights sum to 1.0 along the key sequence
    assert_eq!(attn_weights.len(), num_heads);
    for h in 0..num_heads {
        assert_eq!(attn_weights[h].len(), seq_len);
        for i in 0..seq_len {
            let sum: f32 = attn_weights[h][i].iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Self-attention head {} query {} weights sum must be 1.0, got {}",
                h,
                i,
                sum
            );
        }
    }

    // 2. CrossAttention
    let context_dim = 48;
    let ctx_len = 6;
    let ca = CrossAttention::new(hidden_dim, context_dim, num_heads, 22);

    let mut context = Vec::with_capacity(ctx_len);
    for i in 0..ctx_len {
        let frame: Vec<f32> = (0..context_dim)
            .map(|j| ((i * 5 + j * 19) as f32 * 0.07).cos())
            .collect();
        context.push(frame);
    }

    let (out_ca, cross_weights) = ca.forward_with_weights(&input, &context);
    assert_eq!(out_ca.len(), seq_len);
    for row in &out_ca {
        assert_eq!(row.len(), hidden_dim);
        for &val in row {
            assert!(val.is_finite());
        }
    }

    // Verify cross-attention weights sum to 1.0 along the context sequence
    assert_eq!(cross_weights.len(), num_heads);
    for h in 0..num_heads {
        for i in 0..seq_len {
            let sum: f32 = cross_weights[h][i].iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Cross-attention head {} query {} weights sum must be 1.0, got {}",
                h,
                i,
                sum
            );
        }
    }

    // Test graceful handling of empty context
    let out_empty = ca.forward(&input, &[]);
    assert_eq!(out_empty.len(), seq_len);
}

#[test]
fn test_dit_block_forward_pass_shape_and_stability() {
    let hidden_dim = 64;
    let context_dim = 32;
    let num_heads = 4;
    let block = DiTBlock::new(hidden_dim, context_dim, num_heads, 0);

    let seq_len = 12;
    let mut x = Vec::with_capacity(seq_len);
    for i in 0..seq_len {
        let row: Vec<f32> = (0..hidden_dim).map(|j| ((i + j) as f32 * 0.1).sin()).collect();
        x.push(row);
    }

    let t_emb: Vec<f32> = (0..hidden_dim).map(|j| (j as f32 * 0.2).cos()).collect();
    let ctx: Vec<Vec<f32>> = (0..5)
        .map(|i| (0..context_dim).map(|j| ((i * 3 + j) as f32 * 0.1).sin()).collect())
        .collect();

    // Forward with conditioning context
    let out_with_ctx = block.forward(&x, &t_emb, Some(&ctx));
    assert_eq!(out_with_ctx.len(), seq_len);
    for row in &out_with_ctx {
        assert_eq!(row.len(), hidden_dim);
        for &val in row {
            assert!(val.is_finite(), "Block output must be finite");
        }
    }

    // Forward without conditioning context
    let out_no_ctx = block.forward(&x, &t_emb, None);
    assert_eq!(out_no_ctx.len(), seq_len);
    for row in &out_no_ctx {
        assert_eq!(row.len(), hidden_dim);
        for &val in row {
            assert!(val.is_finite(), "Block output must be finite");
        }
    }

    // Verify context influences output
    let mut diff = 0.0f32;
    for i in 0..seq_len {
        for j in 0..hidden_dim {
            diff += (out_with_ctx[i][j] - out_no_ctx[i][j]).abs();
        }
    }
    assert!(
        diff > 1e-4,
        "Context must modulate block output: diff={}",
        diff
    );
}

#[test]
fn test_ode_solvers_trajectory_convergence() {
    let hidden_dim = 32;
    let latent_dim = 16;
    let context_dim = 32;
    let num_heads = 2;
    let num_layers = 2;

    let dit = FlowMatchingDiT::new(hidden_dim, latent_dim, context_dim, num_heads, num_layers);
    let solver = FlowOdeSolver::new();

    let seq_len = 8;
    let mut x0 = Vec::with_capacity(seq_len);
    for i in 0..seq_len {
        let row: Vec<f32> = (0..latent_dim).map(|j| ((i * 5 + j * 9) as f32 * 0.1).sin()).collect();
        x0.push(row);
    }

    let text_encoder = TextConditioningEncoder::new(context_dim);
    let text_tokens = text_encoder.encode_text("hover");
    let cond = FlowConditioning::new(text_tokens, None);

    // 1. Euler solver with 16 steps
    let out_euler = solver.solve(&dit, &x0, Some(&cond), 16, FlowSolverScheme::Euler);
    assert_eq!(out_euler.len(), seq_len);
    assert_eq!(out_euler[0].len(), latent_dim);
    for row in &out_euler {
        for &val in row {
            assert!(val.is_finite());
        }
    }

    // 2. Midpoint solver with 8 steps
    let out_midpoint = solver.solve(&dit, &x0, Some(&cond), 8, FlowSolverScheme::Midpoint);
    assert_eq!(out_midpoint.len(), seq_len);
    assert_eq!(out_midpoint[0].len(), latent_dim);
    for row in &out_midpoint {
        for &val in row {
            assert!(val.is_finite());
        }
    }

    // 3. Runge-Kutta 4 solver with 4 steps
    let out_rk4 = solver.solve(&dit, &x0, Some(&cond), 4, FlowSolverScheme::RungeKutta4);
    assert_eq!(out_rk4.len(), seq_len);
    assert_eq!(out_rk4[0].len(), latent_dim);
    for row in &out_rk4 {
        for &val in row {
            assert!(val.is_finite());
        }
    }

    // 4. Consistency Flow 2-step fast inference solver
    let out_consistency_2 = solver.solve(&dit, &x0, Some(&cond), 2, FlowSolverScheme::ConsistencyFlow);
    assert_eq!(out_consistency_2.len(), seq_len);
    assert_eq!(out_consistency_2[0].len(), latent_dim);
    for row in &out_consistency_2 {
        for &val in row {
            assert!(val.is_finite());
        }
    }

    // 5. Consistency Flow 4-step solver
    let out_consistency_4 = solver.solve(&dit, &x0, Some(&cond), 4, FlowSolverScheme::ConsistencyFlow);
    assert_eq!(out_consistency_4.len(), seq_len);
    assert_eq!(out_consistency_4[0].len(), latent_dim);

    // Trajectory convergence: verify 2-step and 4-step consistency solutions lie in the same basin
    let mut diff = 0.0f32;
    for i in 0..seq_len {
        for j in 0..latent_dim {
            diff += (out_consistency_2[i][j] - out_consistency_4[i][j]).abs();
        }
    }
    let mean_diff = diff / (seq_len * latent_dim) as f32;
    assert!(
        mean_diff < 1.0,
        "Consistency flow 2-step and 4-step trajectories should be convergent: mean_diff={}",
        mean_diff
    );
}

#[test]
fn test_zero_shot_speaker_prompt_conditioning_modulation() {
    let mut config = CfmConfig::default();
    config.hidden_dim = 64;
    config.latent_dim = 80;
    config.num_heads = 4;
    config.num_layers = 2;
    config.context_dim = 64;
    config.seed = 1001;

    let synth = CfmSpeechSynthesizer::new(config);

    // Create distinct speaker reference acoustic spectrograms
    let num_ref_frames = 20;
    let mut ref_a = Vec::with_capacity(num_ref_frames * 80);
    let mut ref_b = Vec::with_capacity(num_ref_frames * 80);

    for f in 0..num_ref_frames {
        for c in 0..80 {
            // Speaker A: low-frequency dominant (fundamental voice)
            let val_a = (-((c as f32 - 10.0) / 15.0).powi(2)).exp() * 2.0;
            // Speaker B: high-frequency dominant (bright voice)
            let val_b = (-((c as f32 - 50.0) / 15.0).powi(2)).exp() * 2.0;
            ref_a.push(val_a + (f as f32 * 0.01));
            ref_b.push(val_b + (f as f32 * 0.01));
        }
    }

    // Synthesize latents under different speaker conditioning contexts
    let latent_a = synth.synthesize_latent(
        "hold position",
        Some(&ref_a),
        2,
        FlowSolverScheme::ConsistencyFlow,
    );
    let latent_b = synth.synthesize_latent(
        "hold position",
        Some(&ref_b),
        2,
        FlowSolverScheme::ConsistencyFlow,
    );
    let latent_unconditioned = synth.synthesize_latent(
        "hold position",
        None,
        2,
        FlowSolverScheme::ConsistencyFlow,
    );

    assert_eq!(latent_a.len(), latent_b.len());
    assert_eq!(latent_a.len(), latent_unconditioned.len());

    // Verify Speaker A and Speaker B produce distinct acoustic latent trajectories
    let mut spk_diff = 0.0f32;
    for i in 0..latent_a.len() {
        for j in 0..latent_a[i].len() {
            spk_diff += (latent_a[i][j] - latent_b[i][j]).abs();
        }
    }
    assert!(
        spk_diff > 0.1,
        "Speaker prompt conditioning must modulate generated latents: spk_diff={}",
        spk_diff
    );

    // Verify unconditioned generation is also distinct from conditioned generation
    let mut uncond_diff = 0.0f32;
    for i in 0..latent_a.len() {
        for j in 0..latent_a[i].len() {
            uncond_diff += (latent_a[i][j] - latent_unconditioned[i][j]).abs();
        }
    }
    assert!(
        uncond_diff > 0.1,
        "Speaker prompt must differentiate from unconditioned generation: uncond_diff={}",
        uncond_diff
    );
}

#[test]
fn test_sonon_engine_synthesize_speech_latent_end_to_end() {
    let mut engine = SononEngine::new(16000.0, 512, 160, 13);

    // Prior to enabling, synthesis should return Err
    let unconfigured_res = engine.synthesize_speech_latent(
        "emergency land",
        None,
        2,
        FlowSolverScheme::ConsistencyFlow,
    );
    assert!(
        unconfigured_res.is_err(),
        "Synthesize without enabled CFM synthesizer must fail"
    );

    // Enable CFM synthesizer
    let mut cfm_cfg = CfmConfig::default();
    cfm_cfg.hidden_dim = 64;
    cfm_cfg.latent_dim = 80;
    cfm_cfg.num_layers = 2;
    cfm_cfg.context_dim = 64;

    engine.enable_cfm_synthesizer(cfm_cfg);
    assert!(engine.cfm_synthesizer().is_some());
    assert!(engine.cfm_synthesizer_mut().is_some());

    // Synthesize command latent using ConsistencyFlow (2-step)
    let command_latent = engine.synthesize_speech_latent(
        "take off",
        None,
        2,
        FlowSolverScheme::ConsistencyFlow,
    );
    assert!(command_latent.is_ok());
    let frames = command_latent.unwrap();
    assert!(frames.len() >= 8, "Expected at least 8 frames for 'take off'");
    assert_eq!(frames[0].len(), 80, "Expected 80 mel latent channels");

    // Synthesize command latent using Midpoint (4-step) with speaker reference
    let dummy_spk_ref = vec![0.25f32; 80 * 8];
    let spk_latent = engine.synthesize_speech_latent(
        "land now",
        Some(&dummy_spk_ref),
        4,
        FlowSolverScheme::Midpoint,
    );
    assert!(spk_latent.is_ok());
    let spk_frames = spk_latent.unwrap();
    assert!(spk_frames.len() >= 8);
    assert_eq!(spk_frames[0].len(), 80);

    // Disable CFM synthesizer
    engine.disable_cfm_synthesizer();
    assert!(engine.cfm_synthesizer().is_none());

    let disabled_res = engine.synthesize_speech_latent(
        "take off",
        None,
        2,
        FlowSolverScheme::ConsistencyFlow,
    );
    assert!(disabled_res.is_err());
}

#[test]
fn test_deterministic_reproducibility_across_identical_noise_seeds() {
    let mut config = CfmConfig::default();
    config.hidden_dim = 64;
    config.latent_dim = 80;
    config.num_heads = 4;
    config.num_layers = 2;
    config.context_dim = 64;
    config.seed = 2026;

    let synth1 = CfmSpeechSynthesizer::new(config.clone());
    let mut synth2 = CfmSpeechSynthesizer::new(config.clone());

    assert_eq!(synth1.seed(), 2026);
    assert_eq!(synth2.seed(), 2026);

    // Generate latents from identical initial noise seeds and text
    let latents1 = synth1.synthesize_latent(
        "abort mission immediately",
        None,
        2,
        FlowSolverScheme::ConsistencyFlow,
    );
    let latents2 = synth2.synthesize_latent(
        "abort mission immediately",
        None,
        2,
        FlowSolverScheme::ConsistencyFlow,
    );

    assert_eq!(latents1.len(), latents2.len());
    for i in 0..latents1.len() {
        assert_eq!(latents1[i].len(), latents2[i].len());
        for j in 0..latents1[i].len() {
            assert_eq!(
                latents1[i][j], latents2[i][j],
                "Identical seeds must produce bit-exact identical latents at [{}, {}]",
                i, j
            );
        }
    }

    // Now change seed on synthesizer 2 and verify latents diverge
    synth2.set_seed(7777);
    assert_eq!(synth2.seed(), 7777);

    let latents3 = synth2.synthesize_latent(
        "abort mission immediately",
        None,
        2,
        FlowSolverScheme::ConsistencyFlow,
    );
    let mut seed_diff = 0.0f32;
    for i in 0..latents1.len() {
        for j in 0..latents1[i].len() {
            seed_diff += (latents1[i][j] - latents3[i][j]).abs();
        }
    }
    assert!(
        seed_diff > 1.0,
        "Divergent seeds must yield divergent latent outputs: seed_diff={}",
        seed_diff
    );
}

#[test]
fn test_deterministic_rng_box_muller_properties() {
    let mut rng = DeterministicRng::new(42);
    let num_samples = 10000;
    let mut sum = 0.0f32;
    let mut sum_sq = 0.0f32;

    for _ in 0..num_samples {
        let (g1, g2) = rng.sample_gaussian();
        assert!(g1.is_finite());
        assert!(g2.is_finite());
        sum += g1 + g2;
        sum_sq += g1 * g1 + g2 * g2;
    }

    let n = (num_samples * 2) as f32;
    let mean = sum / n;
    let variance = (sum_sq / n) - (mean * mean);

    // Standard normal N(0, 1) should have mean ~ 0 and variance ~ 1
    assert!(
        mean.abs() < 0.05,
        "Box-Muller mean must be approximately 0.0: got {}",
        mean
    );
    assert!(
        (variance - 1.0).abs() < 0.1,
        "Box-Muller variance must be approximately 1.0: got {}",
        variance
    );
}

#[test]
fn test_text_and_speaker_embedding_components() {
    let text_enc = TextConditioningEncoder::new(64);
    let tokens1 = text_enc.encode_text("take off");
    let tokens2 = text_enc.encode_text("land");
    assert_eq!(tokens1.len(), 8);
    assert_eq!(tokens2.len(), 4);
    assert_ne!(tokens1[0], tokens2[0]);

    // Test SpeakerPromptEmbedding
    let spk_emb = SpeakerPromptEmbedding::new(80, 64, 8);
    let empty_tokens = spk_emb.embed_speaker(&[]);
    assert!(empty_tokens.is_empty());

    let flat_mels = vec![0.1f32; 80 * 15];
    let pooled = spk_emb.embed_speaker_slice(&flat_mels, 80);
    assert_eq!(pooled.len(), 8, "Expected 8 pooled speaker tokens");
    for tok in &pooled {
        assert_eq!(tok.len(), 64);
        for &val in tok {
            assert!(val.is_finite());
        }
    }
}
