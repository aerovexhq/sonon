//! Non-Autoregressive Conditional Flow Matching (CFM) Diffusion Transformer (DiT)
//! for continuous acoustic speech latent generation in pure safe Rust.
//!
//! Implements Optimal Transport (OT) displacement interpolation paths,
//! continuous Diffusion Transformer vector field estimators with Adaptive
//! Layer Normalization (AdaLN-Zero), multi-head self-attention, cross-attention
//! conditioning on byte text tokens and reference speaker acoustic prompts,
//! and numerical ODE integration schemes (Euler, Midpoint RK2, Runge-Kutta 4,
//! and 2-step / 4-step Consistency Flow).

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Small epsilon constant for numerical stability and zero division prevention.
const EPSILON: f32 = 1e-5;

/// Activation function for neural network layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationType {
    SiLU,
    GELU,
}

/// Compute SiLU (Sigmoid Linear Unit / Swish) activation.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Compute GELU (Gaussian Error Linear Unit) activation using standard approximation.
#[inline]
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0 / PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

/// Apply activation function to a scalar value.
#[inline]
pub fn activate(x: f32, act: ActivationType) -> f32 {
    match act {
        ActivationType::SiLU => silu(x),
        ActivationType::GELU => gelu(x),
    }
}

/// Fully-connected dense linear projection layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Linear {
    pub in_features: usize,
    pub out_features: usize,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
}

impl Linear {
    /// Construct a new linear layer from existing weight and bias tensors.
    pub fn new(in_features: usize, out_features: usize, weights: Vec<f32>, bias: Vec<f32>) -> Self {
        assert_eq!(
            weights.len(),
            in_features * out_features,
            "Linear layer weight matrix length mismatch"
        );
        assert_eq!(
            bias.len(),
            out_features,
            "Linear layer bias vector length mismatch"
        );
        Self {
            in_features,
            out_features,
            weights,
            bias,
        }
    }

    /// Construct a linear layer with all zero weights and biases.
    pub fn new_zeros(in_features: usize, out_features: usize) -> Self {
        Self {
            in_features,
            out_features,
            weights: vec![0.0; in_features * out_features],
            bias: vec![0.0; out_features],
        }
    }

    /// Construct a linear layer with deterministic Xavier/Glorot pseudo-random weights.
    pub fn new_deterministic(in_features: usize, out_features: usize, seed_salt: usize) -> Self {
        let n_elements = in_features * out_features;
        let scale = (2.0 / (in_features + out_features) as f32).sqrt();
        let mut weights = Vec::with_capacity(n_elements);
        for idx in 0..n_elements {
            let angle = (idx + seed_salt * 31337 + 1) as f32 * 0.1731;
            let val = angle.sin() * (angle * 1.6180339).cos();
            weights.push(val * scale);
        }
        let bias = vec![0.0; out_features];
        Self {
            in_features,
            out_features,
            weights,
            bias,
        }
    }

    /// Forward pass for a single 1D feature vector.
    pub fn forward_vector(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(
            input.len(),
            self.in_features,
            "Input vector dimension mismatch in linear forward"
        );
        let mut out = self.bias.clone();
        for i in 0..self.out_features {
            let row_offset = i * self.in_features;
            let mut sum = 0.0f32;
            for j in 0..self.in_features {
                sum += self.weights[row_offset + j] * input[j];
            }
            out[i] += sum;
        }
        out
    }

    /// Forward pass for a 2D sequence matrix (sequence_length x in_features).
    pub fn forward_matrix(&self, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
        input.iter().map(|row| self.forward_vector(row)).collect()
    }
}

/// Standard Layer Normalization over feature dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerNorm {
    pub dim: usize,
    pub eps: f32,
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
}

impl LayerNorm {
    /// Construct a new LayerNorm with unit scaling and zero bias.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            eps: EPSILON,
            gamma: vec![1.0; dim],
            beta: vec![0.0; dim],
        }
    }

    /// Normalize a 1D vector.
    pub fn forward_vector(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.dim, "LayerNorm input dimension mismatch");
        let n = self.dim as f32;
        let mean = x.iter().sum::<f32>() / n;
        let mut var = 0.0f32;
        for &val in x {
            let diff = val - mean;
            var += diff * diff;
        }
        var /= n;
        let inv_std = 1.0 / (var + self.eps).sqrt();

        let mut out = Vec::with_capacity(self.dim);
        for i in 0..self.dim {
            let norm = (x[i] - mean) * inv_std;
            out.push(norm * self.gamma[i] + self.beta[i]);
        }
        out
    }

    /// Normalize a 2D sequence matrix.
    pub fn forward_matrix(&self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        x.iter().map(|row| self.forward_vector(row)).collect()
    }
}

// ============================================================================
// 1. Optimal Transport Vector Field
// ============================================================================

/// Optimal Transport displacement interpolation path and conditional velocity field.
///
/// Given base prior noise x0 ~ N(0, I) and target acoustic latent x1:
/// xt = (1 - (1 - sigma_min) * t) * x0 + t * x1
/// Target velocity: ut(x1 | x0) = x1 - (1 - sigma_min) * x0
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OptimalTransportPath {
    pub sigma_min: f32,
}

impl Default for OptimalTransportPath {
    fn default() -> Self {
        Self::new(1e-4)
    }
}

impl OptimalTransportPath {
    /// Create a new Optimal Transport path with specified minimum noise level sigma_min.
    pub fn new(sigma_min: f32) -> Self {
        Self { sigma_min }
    }

    /// Compute state interpolation xt along the straight OT geodesic at time t in [0, 1].
    pub fn interpolate(&self, x0: &[Vec<f32>], x1: &[Vec<f32>], t: f32) -> Vec<Vec<f32>> {
        assert_eq!(x0.len(), x1.len(), "Sequence length mismatch in OT interpolate");
        let alpha = 1.0 - (1.0 - self.sigma_min) * t;
        let beta = t;

        let mut xt = Vec::with_capacity(x0.len());
        for i in 0..x0.len() {
            assert_eq!(x0[i].len(), x1[i].len(), "Channel dimension mismatch in OT interpolate");
            let mut row = Vec::with_capacity(x0[i].len());
            for j in 0..x0[i].len() {
                row.push(alpha * x0[i][j] + beta * x1[i][j]);
            }
            xt.push(row);
        }
        xt
    }

    /// Compute target conditional velocity field ut(x1 | x0) = x1 - (1 - sigma_min) * x0.
    pub fn target_velocity(&self, x0: &[Vec<f32>], x1: &[Vec<f32>]) -> Vec<Vec<f32>> {
        assert_eq!(x0.len(), x1.len(), "Sequence length mismatch in OT target_velocity");
        let coeff0 = 1.0 - self.sigma_min;

        let mut ut = Vec::with_capacity(x0.len());
        for i in 0..x0.len() {
            assert_eq!(x0[i].len(), x1[i].len(), "Channel dimension mismatch in OT target_velocity");
            let mut row = Vec::with_capacity(x0[i].len());
            for j in 0..x0[i].len() {
                row.push(x1[i][j] - coeff0 * x0[i][j]);
            }
            ut.push(row);
        }
        ut
    }

    /// Compute flat vector interpolation xt for 1D slices.
    pub fn interpolate_flat(&self, x0: &[f32], x1: &[f32], t: f32) -> Vec<f32> {
        assert_eq!(x0.len(), x1.len(), "Dimension mismatch in OT interpolate_flat");
        let alpha = 1.0 - (1.0 - self.sigma_min) * t;
        let beta = t;
        x0.iter()
            .zip(x1.iter())
            .map(|(&a, &b)| alpha * a + beta * b)
            .collect()
    }

    /// Compute flat vector target velocity ut for 1D slices.
    pub fn target_velocity_flat(&self, x0: &[f32], x1: &[f32]) -> Vec<f32> {
        assert_eq!(x0.len(), x1.len(), "Dimension mismatch in OT target_velocity_flat");
        let coeff0 = 1.0 - self.sigma_min;
        x0.iter()
            .zip(x1.iter())
            .map(|(&a, &b)| b - coeff0 * a)
            .collect()
    }
}

// ============================================================================
// 2. Diffusion Transformer (DiT) Neural Vector Field Estimator Components
// ============================================================================

/// Sinusoidal timestep embedding with 2-layer MLP projection and SiLU activation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimestepEmbedding {
    pub embedding_dim: usize,
    pub hidden_dim: usize,
    pub frequencies: Vec<f32>,
    pub mlp1: Linear,
    pub mlp2: Linear,
}

impl TimestepEmbedding {
    /// Construct a new sinusoidal timestep embedding layer.
    pub fn new(embedding_dim: usize, hidden_dim: usize) -> Self {
        assert!(embedding_dim % 2 == 0, "Embedding dimension must be even");
        let half_dim = embedding_dim / 2;
        let mut frequencies = Vec::with_capacity(half_dim);
        for k in 0..half_dim {
            let freq = (-(k as f32 / half_dim as f32) * 10000.0f32.ln()).exp();
            frequencies.push(freq);
        }

        let mlp1 = Linear::new_deterministic(embedding_dim, hidden_dim, 101);
        let mlp2 = Linear::new_deterministic(hidden_dim, hidden_dim, 102);

        Self {
            embedding_dim,
            hidden_dim,
            frequencies,
            mlp1,
            mlp2,
        }
    }

    /// Compute raw sinusoidal embedding for scalar time t in [0, 1].
    pub fn sinusoidal(&self, t: f32) -> Vec<f32> {
        let mut emb = Vec::with_capacity(self.embedding_dim);
        for &freq in &self.frequencies {
            let arg = t * freq;
            emb.push(arg.sin());
            emb.push(arg.cos());
        }
        emb
    }

    /// Forward pass through sinusoidal encoder and 2-layer MLP projection with SiLU.
    pub fn forward(&self, t: f32) -> Vec<f32> {
        let raw = self.sinusoidal(t);
        let h1 = self.mlp1.forward_vector(&raw);
        let act1: Vec<f32> = h1.into_iter().map(silu).collect();
        let h2 = self.mlp2.forward_vector(&act1);
        h2.into_iter().map(silu).collect()
    }
}

/// Adaptive Layer Normalization with zero-initialization gating (adaLN-Zero).
///
/// Modulates normalized acoustic latents:
/// h = (1 + gamma(t)) * LayerNorm(x) + beta(t)
/// with residual gating: x + alpha(t) * sublayer(h).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaLayerNormZero {
    pub hidden_dim: usize,
    pub proj: Linear,
    pub eps: f32,
}

impl AdaLayerNormZero {
    /// Construct a new AdaLayerNormZero layer with deterministic initialization.
    pub fn new(hidden_dim: usize, seed_salt: usize) -> Self {
        let mut proj = Linear::new_deterministic(hidden_dim, 3 * hidden_dim, seed_salt);
        // Initialize modulation weights with small magnitude
        for w in &mut proj.weights {
            *w *= 0.05;
        }
        // Initialize scale gamma bias to 0.0, shift beta bias to 0.0, gate alpha bias to 0.1
        for i in 0..hidden_dim {
            proj.bias[i] = 0.0;                  // gamma
            proj.bias[hidden_dim + i] = 0.0;     // beta
            proj.bias[2 * hidden_dim + i] = 0.1; // alpha
        }

        Self {
            hidden_dim,
            proj,
            eps: EPSILON,
        }
    }

    /// Construct a zero-initialized AdaLayerNormZero layer.
    pub fn new_zero_init(hidden_dim: usize) -> Self {
        Self {
            hidden_dim,
            proj: Linear::new_zeros(hidden_dim, 3 * hidden_dim),
            eps: EPSILON,
        }
    }

    /// Modulate a normalized 1D vector directly given explicit gamma and beta scale/shift vectors.
    pub fn modulate(&self, x: &[f32], gamma: &[f32], beta: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.hidden_dim);
        assert_eq!(gamma.len(), self.hidden_dim);
        assert_eq!(beta.len(), self.hidden_dim);

        let n = self.hidden_dim as f32;
        let mean = x.iter().sum::<f32>() / n;
        let mut var = 0.0f32;
        for &val in x {
            let diff = val - mean;
            var += diff * diff;
        }
        var /= n;
        let inv_std = 1.0 / (var + self.eps).sqrt();

        let mut out = Vec::with_capacity(self.hidden_dim);
        for i in 0..self.hidden_dim {
            let norm = (x[i] - mean) * inv_std;
            out.push((1.0 + gamma[i]) * norm + beta[i]);
        }
        out
    }

    /// Apply residual gating: residual + alpha * sublayer_output.
    pub fn gate(&self, residual: &[f32], sublayer_out: &[f32], alpha: &[f32]) -> Vec<f32> {
        assert_eq!(residual.len(), self.hidden_dim);
        assert_eq!(sublayer_out.len(), self.hidden_dim);
        assert_eq!(alpha.len(), self.hidden_dim);

        let mut out = Vec::with_capacity(self.hidden_dim);
        for i in 0..self.hidden_dim {
            out.push(residual[i] + alpha[i] * sublayer_out[i]);
        }
        out
    }

    /// Regress (gamma, beta, alpha) modulation parameters from timestep embedding.
    pub fn compute_parameters(&self, t_emb: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        assert_eq!(t_emb.len(), self.hidden_dim);
        let params = self.proj.forward_vector(t_emb);
        let h = self.hidden_dim;
        let gamma = params[0..h].to_vec();
        let beta = params[h..2 * h].to_vec();
        let alpha = params[2 * h..3 * h].to_vec();
        (gamma, beta, alpha)
    }

    /// Modulate a 1D latent vector using timestep embedding, returning (modulated, alpha_gate).
    pub fn modulate_latent(&self, x: &[f32], t_emb: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let (gamma, beta, alpha) = self.compute_parameters(t_emb);
        let h = self.modulate(x, &gamma, &beta);
        (h, alpha)
    }

    /// Modulate a 2D sequence of latent frames using timestep embedding, returning (modulated_sequence, alpha_gate).
    pub fn modulate_sequence(&self, sequence: &[Vec<f32>], t_emb: &[f32]) -> (Vec<Vec<f32>>, Vec<f32>) {
        let (gamma, beta, alpha) = self.compute_parameters(t_emb);
        let mut modulated = Vec::with_capacity(sequence.len());
        for frame in sequence {
            modulated.push(self.modulate(frame, &gamma, &beta));
        }
        (modulated, alpha)
    }
}

/// Multi-Head Self-Attention layer with scaled dot-product attention.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiHeadAttention {
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub w_q: Linear,
    pub w_k: Linear,
    pub w_v: Linear,
    pub w_o: Linear,
}

impl MultiHeadAttention {
    /// Construct a new MultiHeadAttention layer.
    pub fn new(hidden_dim: usize, num_heads: usize, seed_salt: usize) -> Self {
        assert!(
            hidden_dim % num_heads == 0,
            "Hidden dimension must be divisible by number of attention heads"
        );
        let head_dim = hidden_dim / num_heads;
        let w_q = Linear::new_deterministic(hidden_dim, hidden_dim, seed_salt * 4 + 1);
        let w_k = Linear::new_deterministic(hidden_dim, hidden_dim, seed_salt * 4 + 2);
        let w_v = Linear::new_deterministic(hidden_dim, hidden_dim, seed_salt * 4 + 3);
        let w_o = Linear::new_deterministic(hidden_dim, hidden_dim, seed_salt * 4 + 4);

        Self {
            hidden_dim,
            num_heads,
            head_dim,
            w_q,
            w_k,
            w_v,
            w_o,
        }
    }

    /// Forward pass on acoustic sequence (seq_len x hidden_dim).
    pub fn forward(&self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let (out, _) = self.forward_with_weights(x);
        out
    }

    /// Forward pass returning attention output and attention weight matrices [head, t, s].
    pub fn forward_with_weights(&self, x: &[Vec<f32>]) -> (Vec<Vec<f32>>, Vec<Vec<Vec<f32>>>) {
        let seq_len = x.len();
        if seq_len == 0 {
            return (Vec::new(), Vec::new());
        }

        let q = self.w_q.forward_matrix(x);
        let k = self.w_k.forward_matrix(x);
        let v = self.w_v.forward_matrix(x);

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mut head_outputs = vec![vec![vec![0.0f32; self.head_dim]; seq_len]; self.num_heads];
        let mut attention_weights = vec![vec![vec![0.0f32; seq_len]; seq_len]; self.num_heads];

        for h in 0..self.num_heads {
            let offset = h * self.head_dim;
            for i in 0..seq_len {
                // Compute dot-product logits for query i against all keys j
                let mut logits = Vec::with_capacity(seq_len);
                let mut max_logit = f32::NEG_INFINITY;
                for j in 0..seq_len {
                    let mut dot = 0.0f32;
                    for d in 0..self.head_dim {
                        dot += q[i][offset + d] * k[j][offset + d];
                    }
                    let score = dot * scale;
                    if score > max_logit {
                        max_logit = score;
                    }
                    logits.push(score);
                }

                // Stable softmax over sequence dimension j
                let mut sum_exp = 0.0f32;
                let mut exp_scores = Vec::with_capacity(seq_len);
                for &score in &logits {
                    let e = (score - max_logit).exp();
                    exp_scores.push(e);
                    sum_exp += e;
                }
                let inv_sum = 1.0 / sum_exp.max(EPSILON);
                for j in 0..seq_len {
                    let weight = exp_scores[j] * inv_sum;
                    attention_weights[h][i][j] = weight;
                    for d in 0..self.head_dim {
                        head_outputs[h][i][d] += weight * v[j][offset + d];
                    }
                }
            }
        }

        // Concatenate all heads for each time step i
        let mut concatenated = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let mut row = Vec::with_capacity(self.hidden_dim);
            for h in 0..self.num_heads {
                row.extend_from_slice(&head_outputs[h][i]);
            }
            concatenated.push(row);
        }

        let out = self.w_o.forward_matrix(&concatenated);
        (out, attention_weights)
    }
}

/// Multi-Head Cross-Attention layer.
///
/// Query is projected from acoustic latent sequence (T x hidden_dim).
/// Key and Value are projected from conditioning context (S x context_dim).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossAttention {
    pub hidden_dim: usize,
    pub context_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub w_q: Linear,
    pub w_k: Linear,
    pub w_v: Linear,
    pub w_o: Linear,
}

impl CrossAttention {
    /// Construct a new CrossAttention layer.
    pub fn new(hidden_dim: usize, context_dim: usize, num_heads: usize, seed_salt: usize) -> Self {
        assert!(
            hidden_dim % num_heads == 0,
            "Hidden dimension must be divisible by number of attention heads"
        );
        let head_dim = hidden_dim / num_heads;
        let w_q = Linear::new_deterministic(hidden_dim, hidden_dim, seed_salt * 4 + 5);
        let w_k = Linear::new_deterministic(context_dim, hidden_dim, seed_salt * 4 + 6);
        let w_v = Linear::new_deterministic(context_dim, hidden_dim, seed_salt * 4 + 7);
        let w_o = Linear::new_deterministic(hidden_dim, hidden_dim, seed_salt * 4 + 8);

        Self {
            hidden_dim,
            context_dim,
            num_heads,
            head_dim,
            w_q,
            w_k,
            w_v,
            w_o,
        }
    }

    /// Forward pass mapping acoustic latent query sequence and conditioning context sequence.
    pub fn forward(&self, x: &[Vec<f32>], context: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let (out, _) = self.forward_with_weights(x, context);
        out
    }

    /// Forward pass returning output and cross-attention weight matrices [head, t, s].
    pub fn forward_with_weights(
        &self,
        x: &[Vec<f32>],
        context: &[Vec<f32>],
    ) -> (Vec<Vec<f32>>, Vec<Vec<Vec<f32>>>) {
        let seq_len = x.len();
        let ctx_len = context.len();

        if seq_len == 0 {
            return (Vec::new(), Vec::new());
        }
        if ctx_len == 0 {
            return (vec![vec![0.0f32; self.hidden_dim]; seq_len], Vec::new());
        }

        let q = self.w_q.forward_matrix(x);
        let k = self.w_k.forward_matrix(context);
        let v = self.w_v.forward_matrix(context);

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mut head_outputs = vec![vec![vec![0.0f32; self.head_dim]; seq_len]; self.num_heads];
        let mut attention_weights = vec![vec![vec![0.0f32; ctx_len]; seq_len]; self.num_heads];

        for h in 0..self.num_heads {
            let offset = h * self.head_dim;
            for i in 0..seq_len {
                let mut logits = Vec::with_capacity(ctx_len);
                let mut max_logit = f32::NEG_INFINITY;
                for j in 0..ctx_len {
                    let mut dot = 0.0f32;
                    for d in 0..self.head_dim {
                        dot += q[i][offset + d] * k[j][offset + d];
                    }
                    let score = dot * scale;
                    if score > max_logit {
                        max_logit = score;
                    }
                    logits.push(score);
                }

                let mut sum_exp = 0.0f32;
                let mut exp_scores = Vec::with_capacity(ctx_len);
                for &score in &logits {
                    let e = (score - max_logit).exp();
                    exp_scores.push(e);
                    sum_exp += e;
                }
                let inv_sum = 1.0 / sum_exp.max(EPSILON);
                for j in 0..ctx_len {
                    let weight = exp_scores[j] * inv_sum;
                    attention_weights[h][i][j] = weight;
                    for d in 0..self.head_dim {
                        head_outputs[h][i][d] += weight * v[j][offset + d];
                    }
                }
            }
        }

        let mut concatenated = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let mut row = Vec::with_capacity(self.hidden_dim);
            for h in 0..self.num_heads {
                row.extend_from_slice(&head_outputs[h][i]);
            }
            concatenated.push(row);
        }

        let out = self.w_o.forward_matrix(&concatenated);
        (out, attention_weights)
    }
}

/// Pointwise Feed-Forward Network: Linear -> Activation -> Linear.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedForwardNetwork {
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
    pub activation: ActivationType,
    pub fc1: Linear,
    pub fc2: Linear,
}

impl FeedForwardNetwork {
    /// Construct a new Feed-Forward Network.
    pub fn new(hidden_dim: usize, intermediate_dim: usize, seed_salt: usize) -> Self {
        let fc1 = Linear::new_deterministic(hidden_dim, intermediate_dim, seed_salt * 2 + 9);
        let fc2 = Linear::new_deterministic(intermediate_dim, hidden_dim, seed_salt * 2 + 10);
        Self {
            hidden_dim,
            intermediate_dim,
            activation: ActivationType::SiLU,
            fc1,
            fc2,
        }
    }

    /// Forward pass on a single feature vector.
    pub fn forward_vector(&self, x: &[f32]) -> Vec<f32> {
        let h1 = self.fc1.forward_vector(x);
        let act: Vec<f32> = h1.into_iter().map(|v| activate(v, self.activation)).collect();
        self.fc2.forward_vector(&act)
    }

    /// Forward pass on a sequence of feature frames.
    pub fn forward(&self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        x.iter().map(|frame| self.forward_vector(frame)).collect()
    }
}

/// Full DiT block combining AdaLN-Zero, Multi-Head Self-Attention, Cross-Attention, and FFN.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiTBlock {
    pub hidden_dim: usize,
    pub context_dim: usize,
    pub ada_ln_self: AdaLayerNormZero,
    pub self_attn: MultiHeadAttention,
    pub ada_ln_cross: AdaLayerNormZero,
    pub cross_attn: CrossAttention,
    pub ada_ln_ffn: AdaLayerNormZero,
    pub ffn: FeedForwardNetwork,
}

impl DiTBlock {
    /// Construct a new DiT block.
    pub fn new(hidden_dim: usize, context_dim: usize, num_heads: usize, layer_idx: usize) -> Self {
        let seed = layer_idx * 17 + 1;
        let ada_ln_self = AdaLayerNormZero::new(hidden_dim, seed + 1);
        let self_attn = MultiHeadAttention::new(hidden_dim, num_heads, seed + 2);
        let ada_ln_cross = AdaLayerNormZero::new(hidden_dim, seed + 3);
        let cross_attn = CrossAttention::new(hidden_dim, context_dim, num_heads, seed + 4);
        let ada_ln_ffn = AdaLayerNormZero::new(hidden_dim, seed + 5);
        let ffn = FeedForwardNetwork::new(hidden_dim, hidden_dim * 4, seed + 6);

        Self {
            hidden_dim,
            context_dim,
            ada_ln_self,
            self_attn,
            ada_ln_cross,
            cross_attn,
            ada_ln_ffn,
            ffn,
        }
    }

    /// Forward pass through the DiT block:
    /// 1. Self-Attention with AdaLN-Zero modulation and residual gating
    /// 2. Cross-Attention on conditioning context (if provided)
    /// 3. Feed-Forward Network with AdaLN-Zero modulation and residual gating
    pub fn forward(
        &self,
        x: &[Vec<f32>],
        t_emb: &[f32],
        context: Option<&[Vec<f32>]>,
    ) -> Vec<Vec<f32>> {
        let seq_len = x.len();
        if seq_len == 0 {
            return Vec::new();
        }

        // Sublayer 1: Multi-Head Self-Attention
        let (mod_self, alpha_self) = self.ada_ln_self.modulate_sequence(x, t_emb);
        let attn_out = self.self_attn.forward(&mod_self);
        let mut h = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            h.push(self.ada_ln_self.gate(&x[i], &attn_out[i], &alpha_self));
        }

        // Sublayer 2: Cross-Attention on context (if present)
        if let Some(ctx) = context {
            if !ctx.is_empty() {
                let (mod_cross, alpha_cross) = self.ada_ln_cross.modulate_sequence(&h, t_emb);
                let cross_out = self.cross_attn.forward(&mod_cross, ctx);
                let mut h_next = Vec::with_capacity(seq_len);
                for i in 0..seq_len {
                    h_next.push(self.ada_ln_cross.gate(&h[i], &cross_out[i], &alpha_cross));
                }
                h = h_next;
            }
        }

        // Sublayer 3: Feed-Forward Network
        let (mod_ffn, alpha_ffn) = self.ada_ln_ffn.modulate_sequence(&h, t_emb);
        let ffn_out = self.ffn.forward(&mod_ffn);
        let mut out = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            out.push(self.ada_ln_ffn.gate(&h[i], &ffn_out[i], &alpha_ffn));
        }

        out
    }
}

/// Continuous Flow Matching Diffusion Transformer (DiT) Neural Vector Field Estimator.
///
/// Computes predicted velocity field v_theta(x_t, t, c) in R^{T x D}.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowMatchingDiT {
    pub hidden_dim: usize,
    pub latent_dim: usize,
    pub context_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub in_proj: Linear,
    pub timestep_embed: TimestepEmbedding,
    pub blocks: Vec<DiTBlock>,
    pub final_ada_ln: AdaLayerNormZero,
    pub out_proj: Linear,
}

impl FlowMatchingDiT {
    /// Construct a new Flow Matching DiT model.
    pub fn new(
        hidden_dim: usize,
        latent_dim: usize,
        context_dim: usize,
        num_heads: usize,
        num_layers: usize,
    ) -> Self {
        let in_proj = Linear::new_deterministic(latent_dim, hidden_dim, 201);
        let timestep_embed = TimestepEmbedding::new(hidden_dim, hidden_dim);
        let mut blocks = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            blocks.push(DiTBlock::new(hidden_dim, context_dim, num_heads, i));
        }
        let final_ada_ln = AdaLayerNormZero::new(hidden_dim, 301);
        let out_proj = Linear::new_deterministic(hidden_dim, latent_dim, 401);

        Self {
            hidden_dim,
            latent_dim,
            context_dim,
            num_heads,
            num_layers,
            in_proj,
            timestep_embed,
            blocks,
            final_ada_ln,
            out_proj,
        }
    }

    /// Add continuous sinusoidal positional encodings to frames in-place.
    fn add_positional_encoding(&self, frames: &mut [Vec<f32>]) {
        let seq_len = frames.len();
        let half_dim = self.hidden_dim / 2;
        for pos in 0..seq_len {
            for k in 0..half_dim {
                let freq = (-(k as f32 / half_dim as f32) * 10000.0f32.ln()).exp();
                let arg = pos as f32 * freq;
                frames[pos][2 * k] += 0.05 * arg.sin();
                frames[pos][2 * k + 1] += 0.05 * arg.cos();
            }
        }
    }

    /// Forward pass predicting velocity vector field v_theta(x_t, t, c).
    pub fn forward(
        &self,
        x_t: &[Vec<f32>],
        t: f32,
        context: Option<&[Vec<f32>]>,
    ) -> Vec<Vec<f32>> {
        let seq_len = x_t.len();
        if seq_len == 0 {
            return Vec::new();
        }

        // 1. Timestep embedding vector
        let t_emb = self.timestep_embed.forward(t);

        // 2. Linear input projection D -> H
        let mut h = self.in_proj.forward_matrix(x_t);

        // 3. Positional encoding injection
        self.add_positional_encoding(&mut h);

        // 4. DiT transformer blocks stack
        for block in &self.blocks {
            h = block.forward(&h, &t_emb, context);
        }

        // 5. Final AdaLN-Zero modulation
        let (mod_final, _) = self.final_ada_ln.modulate_sequence(&h, &t_emb);

        // 6. Linear output projection H -> D
        self.out_proj.forward_matrix(&mod_final)
    }

    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    pub fn latent_dim(&self) -> usize {
        self.latent_dim
    }

    pub fn context_dim(&self) -> usize {
        self.context_dim
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
}

// ============================================================================
// 3. Conditioning & Voice Cloning Context
// ============================================================================

/// Byte-level text conditioning encoder mapping characters/bytes into continuous conditioning tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextConditioningEncoder {
    pub context_dim: usize,
    pub embedding_table: Vec<Vec<f32>>,
    pub proj1: Linear,
    pub proj2: Linear,
}

impl TextConditioningEncoder {
    /// Construct a new TextConditioningEncoder.
    pub fn new(context_dim: usize) -> Self {
        let vocab_size = 256;
        let mut embedding_table = Vec::with_capacity(vocab_size);
        for b in 0..vocab_size {
            let mut emb = Vec::with_capacity(context_dim);
            for d in 0..context_dim {
                let angle = (b * 37 + d * 13 + 7) as f32 * 0.1337;
                emb.push(angle.sin() * 0.5);
            }
            embedding_table.push(emb);
        }

        let proj1 = Linear::new_deterministic(context_dim, context_dim, 501);
        let proj2 = Linear::new_deterministic(context_dim, context_dim, 502);

        Self {
            context_dim,
            embedding_table,
            proj1,
            proj2,
        }
    }

    /// Encode input text string into a sequence of continuous conditioning vectors.
    pub fn encode_text(&self, text: &str) -> Vec<Vec<f32>> {
        let bytes = text.as_bytes();
        let tokens: Vec<u8> = if bytes.is_empty() {
            vec![32] // single space token
        } else {
            bytes.to_vec()
        };

        let mut sequence = Vec::with_capacity(tokens.len());
        let half_dim = self.context_dim / 2;

        for (pos, &b) in tokens.iter().enumerate() {
            let mut vec = self.embedding_table[b as usize].clone();
            // Inject positional encoding
            for k in 0..half_dim {
                let freq = (-(k as f32 / half_dim as f32) * 10000.0f32.ln()).exp();
                let arg = pos as f32 * freq;
                vec[2 * k] += 0.1 * arg.sin();
                vec[2 * k + 1] += 0.1 * arg.cos();
            }

            // 2-layer MLP projection with SiLU
            let h1 = self.proj1.forward_vector(&vec);
            let act1: Vec<f32> = h1.into_iter().map(silu).collect();
            let h2 = self.proj2.forward_vector(&act1);
            sequence.push(h2);
        }

        sequence
    }
}

/// Speaker acoustic prompt embedding layer for zero-shot voice cloning.
///
/// Projects reference mel frames into speaker reference tokens via temporal pooling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerPromptEmbedding {
    pub mel_dim: usize,
    pub context_dim: usize,
    pub num_speaker_tokens: usize,
    pub proj: Linear,
    pub post_proj: Linear,
}

impl SpeakerPromptEmbedding {
    /// Construct a new SpeakerPromptEmbedding.
    pub fn new(mel_dim: usize, context_dim: usize, num_speaker_tokens: usize) -> Self {
        let proj = Linear::new_deterministic(mel_dim, context_dim, 601);
        let post_proj = Linear::new_deterministic(context_dim, context_dim, 602);

        Self {
            mel_dim,
            context_dim,
            num_speaker_tokens,
            proj,
            post_proj,
        }
    }

    /// Project reference acoustic mel frames into speaker reference tokens.
    pub fn embed_speaker(&self, frames: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if frames.is_empty() || self.num_speaker_tokens == 0 {
            return Vec::new();
        }

        // Project all frames to context_dim
        let projected = self.proj.forward_matrix(frames);

        // 1. Global average pooled token (represents global vocal tract timbre)
        let mut global_mean = vec![0.0f32; self.context_dim];
        for frame in &projected {
            for d in 0..self.context_dim {
                global_mean[d] += frame[d];
            }
        }
        let inv_len = 1.0 / (projected.len() as f32);
        for val in &mut global_mean {
            *val *= inv_len;
        }

        let mut speaker_tokens = Vec::with_capacity(self.num_speaker_tokens);
        speaker_tokens.push(global_mean);

        // 2. Segment-level pooled tokens
        let remaining_tokens = self.num_speaker_tokens - 1;
        if remaining_tokens > 0 {
            let chunk_size = (projected.len() as f32 / remaining_tokens as f32).max(1.0);
            for k in 0..remaining_tokens {
                let start_idx = (k as f32 * chunk_size) as usize;
                let end_idx = (((k + 1) as f32 * chunk_size) as usize).min(projected.len());

                let mut seg_mean = vec![0.0f32; self.context_dim];
                if start_idx < end_idx {
                    for i in start_idx..end_idx {
                        for d in 0..self.context_dim {
                            seg_mean[d] += projected[i][d];
                        }
                    }
                    let seg_inv = 1.0 / ((end_idx - start_idx) as f32);
                    for val in &mut seg_mean {
                        *val *= seg_inv;
                    }
                } else if !projected.is_empty() {
                    seg_mean = projected[projected.len() - 1].clone();
                }
                speaker_tokens.push(seg_mean);
            }
        }

        // Post-projection through linear layer with SiLU activation
        speaker_tokens
            .into_iter()
            .map(|tok| {
                let h = self.post_proj.forward_vector(&tok);
                h.into_iter().map(silu).collect()
            })
            .collect()
    }

    /// Embed flat slice of mel spectrogram frames.
    pub fn embed_speaker_slice(&self, flat: &[f32], channels: usize) -> Vec<Vec<f32>> {
        if flat.is_empty() || channels == 0 {
            return Vec::new();
        }
        let num_full = flat.len() / channels;
        let mut frames = Vec::with_capacity(num_full + 1);
        for i in 0..num_full {
            frames.push(flat[i * channels..(i + 1) * channels].to_vec());
        }
        let remainder = flat.len() % channels;
        if remainder > 0 {
            let mut last = flat[num_full * channels..].to_vec();
            last.resize(channels, 0.0);
            frames.push(last);
        }
        self.embed_speaker(&frames)
    }
}

/// Conditioning context combining textual tokens and optional speaker reference prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowConditioning {
    pub text_tokens: Vec<Vec<f32>>,
    pub speaker_tokens: Option<Vec<Vec<f32>>>,
}

impl FlowConditioning {
    /// Construct a new FlowConditioning.
    pub fn new(text_tokens: Vec<Vec<f32>>, speaker_tokens: Option<Vec<Vec<f32>>>) -> Self {
        Self {
            text_tokens,
            speaker_tokens,
        }
    }

    /// Combine speaker tokens and text tokens into a unified key-value context sequence.
    pub fn combined_tokens(&self) -> Vec<Vec<f32>> {
        let mut combined = Vec::new();
        if let Some(spk) = &self.speaker_tokens {
            combined.extend(spk.clone());
        }
        combined.extend(self.text_tokens.clone());
        combined
    }
}

// ============================================================================
// 4. Numerical ODE Solvers
// ============================================================================

/// Integration scheme for numerical Flow Matching ODE solvers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlowSolverScheme {
    Euler,
    Midpoint,
    RungeKutta4,
    ConsistencyFlow,
}

/// Numerical ODE solver integrating dx_t / dt = v_theta(x_t, t, c) from t = 0 to t = 1.
#[derive(Debug, Clone, Copy, Default)]
pub struct FlowOdeSolver;

impl FlowOdeSolver {
    pub fn new() -> Self {
        Self
    }

    /// Integrate flow ODE from t = 0 to t = 1 using the specified numerical scheme.
    pub fn solve(
        &self,
        model: &FlowMatchingDiT,
        x0: &[Vec<f32>],
        condition: Option<&FlowConditioning>,
        num_steps: usize,
        scheme: FlowSolverScheme,
    ) -> Vec<Vec<f32>> {
        let n_steps = num_steps.max(1);
        let h = 1.0f32 / (n_steps as f32);
        let mut x = x0.to_vec();

        let ctx_tokens = condition.map(|c| c.combined_tokens());
        let ctx_ref = ctx_tokens.as_deref();

        match scheme {
            FlowSolverScheme::Euler => {
                for step in 0..n_steps {
                    let t = step as f32 * h;
                    let v = model.forward(&x, t, ctx_ref);
                    for i in 0..x.len() {
                        for j in 0..x[i].len() {
                            x[i][j] += h * v[i][j];
                        }
                    }
                }
            }
            FlowSolverScheme::Midpoint => {
                for step in 0..n_steps {
                    let t = step as f32 * h;
                    let k1 = model.forward(&x, t, ctx_ref);

                    // Midpoint evaluation: x_mid = x + (h / 2) * k1
                    let mut x_mid = Vec::with_capacity(x.len());
                    for i in 0..x.len() {
                        let mut row = Vec::with_capacity(x[i].len());
                        for j in 0..x[i].len() {
                            row.push(x[i][j] + 0.5 * h * k1[i][j]);
                        }
                        x_mid.push(row);
                    }
                    let t_mid = t + 0.5 * h;
                    let k2 = model.forward(&x_mid, t_mid, ctx_ref);

                    // Update: x = x + h * k2
                    for i in 0..x.len() {
                        for j in 0..x[i].len() {
                            x[i][j] += h * k2[i][j];
                        }
                    }
                }
            }
            FlowSolverScheme::RungeKutta4 => {
                for step in 0..n_steps {
                    let t = step as f32 * h;
                    let k1 = model.forward(&x, t, ctx_ref);

                    // k2 = f(x + 0.5 * h * k1, t + 0.5 * h)
                    let mut x2 = Vec::with_capacity(x.len());
                    for i in 0..x.len() {
                        let mut row = Vec::with_capacity(x[i].len());
                        for j in 0..x[i].len() {
                            row.push(x[i][j] + 0.5 * h * k1[i][j]);
                        }
                        x2.push(row);
                    }
                    let k2 = model.forward(&x2, t + 0.5 * h, ctx_ref);

                    // k3 = f(x + 0.5 * h * k2, t + 0.5 * h)
                    let mut x3 = Vec::with_capacity(x.len());
                    for i in 0..x.len() {
                        let mut row = Vec::with_capacity(x[i].len());
                        for j in 0..x[i].len() {
                            row.push(x[i][j] + 0.5 * h * k2[i][j]);
                        }
                        x3.push(row);
                    }
                    let k3 = model.forward(&x3, t + 0.5 * h, ctx_ref);

                    // k4 = f(x + h * k3, t + h)
                    let mut x4 = Vec::with_capacity(x.len());
                    for i in 0..x.len() {
                        let mut row = Vec::with_capacity(x[i].len());
                        for j in 0..x[i].len() {
                            row.push(x[i][j] + h * k3[i][j]);
                        }
                        x4.push(row);
                    }
                    let k4 = model.forward(&x4, t + h, ctx_ref);

                    // x = x + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
                    let h6 = h / 6.0;
                    for i in 0..x.len() {
                        for j in 0..x[i].len() {
                            x[i][j] += h6 * (k1[i][j] + 2.0 * k2[i][j] + 2.0 * k3[i][j] + k4[i][j]);
                        }
                    }
                }
            }
            FlowSolverScheme::ConsistencyFlow => {
                // High-order 2-step / 4-step Heun predictor-corrector consistency solver
                for step in 0..n_steps {
                    let t = step as f32 * h;
                    let t_next = ((step + 1) as f32 * h).min(1.0);
                    let k1 = model.forward(&x, t, ctx_ref);

                    // Predictor step: x_pred = x + h * k1
                    let mut x_pred = Vec::with_capacity(x.len());
                    for i in 0..x.len() {
                        let mut row = Vec::with_capacity(x[i].len());
                        for j in 0..x[i].len() {
                            row.push(x[i][j] + h * k1[i][j]);
                        }
                        x_pred.push(row);
                    }

                    // Corrector step: k2 = f(x_pred, t_next)
                    let k2 = model.forward(&x_pred, t_next, ctx_ref);

                    // Consistency update: x = x + 0.5 * h * (k1 + k2)
                    let h_half = 0.5 * h;
                    for i in 0..x.len() {
                        for j in 0..x[i].len() {
                            x[i][j] += h_half * (k1[i][j] + k2[i][j]);
                        }
                    }
                }
            }
        }

        x
    }
}

// ============================================================================
// 5. Deterministic Pseudo-Random Gaussian Sampler
// ============================================================================

/// Deterministic Pseudo-Random Number Generator with Box-Muller normal sampling.
#[derive(Debug, Clone)]
pub struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    /// Construct a new deterministic PRNG with seed.
    pub fn new(seed: u64) -> Self {
        let state = if seed == 0 { 0x853c49e6748fea9b } else { seed };
        Self { state }
    }

    /// Generate next 64-bit unsigned pseudo-random integer.
    pub fn next_u64(&mut self) -> u64 {
        let mut z = self.state.wrapping_add(0x9e3779b97f4a7c15);
        self.state = z;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// Generate uniform random float in (0, 1].
    pub fn next_f32(&mut self) -> f32 {
        let val = (self.next_u64() >> 40) as f32;
        (val + 1.0) / (16777216.0 + 2.0)
    }

    /// Sample a pair of standard normal variables N(0, 1) using the Box-Muller transform.
    pub fn sample_gaussian(&mut self) -> (f32, f32) {
        let u1 = self.next_f32().max(1e-7);
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        (r * theta.cos(), r * theta.sin())
    }

    /// Sample a 2D Gaussian acoustic latent tensor of shape (frames x channels).
    pub fn sample_latent(&mut self, frames: usize, channels: usize) -> Vec<Vec<f32>> {
        let mut result = Vec::with_capacity(frames);
        for _ in 0..frames {
            let mut row = Vec::with_capacity(channels);
            let mut c = 0;
            while c < channels {
                let (g1, g2) = self.sample_gaussian();
                row.push(g1);
                c += 1;
                if c < channels {
                    row.push(g2);
                    c += 1;
                }
            }
            result.push(row);
        }
        result
    }
}

// ============================================================================
// 6. Latent Speech Synthesizer & Configuration
// ============================================================================

/// Configuration for Conditional Flow Matching Diffusion Transformer synthesizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfmConfig {
    pub hidden_dim: usize,
    pub latent_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub context_dim: usize,
    pub seed: u64,
    pub frames_per_char: usize,
}

impl Default for CfmConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 128,
            latent_dim: 80,
            num_heads: 4,
            num_layers: 4,
            context_dim: 128,
            seed: 42,
            frames_per_char: 5,
        }
    }
}

/// Conditional Flow Matching (CFM) Diffusion Transformer Speech Synthesizer.
#[derive(Debug, Clone)]
pub struct CfmSpeechSynthesizer {
    pub dit: FlowMatchingDiT,
    pub solver: FlowOdeSolver,
    pub text_encoder: TextConditioningEncoder,
    pub speaker_embedder: SpeakerPromptEmbedding,
    pub seed: u64,
    pub latent_dim: usize,
    pub frames_per_char: usize,
}

impl CfmSpeechSynthesizer {
    /// Construct a new CFM speech synthesizer from configuration.
    pub fn new(config: CfmConfig) -> Self {
        let dit = FlowMatchingDiT::new(
            config.hidden_dim,
            config.latent_dim,
            config.context_dim,
            config.num_heads,
            config.num_layers,
        );
        let solver = FlowOdeSolver::new();
        let text_encoder = TextConditioningEncoder::new(config.context_dim);
        let speaker_embedder = SpeakerPromptEmbedding::new(config.latent_dim, config.context_dim, 8);

        Self {
            dit,
            solver,
            text_encoder,
            speaker_embedder,
            seed: config.seed,
            latent_dim: config.latent_dim,
            frames_per_char: config.frames_per_char.max(1),
        }
    }

    /// Set deterministic pseudo-random seed.
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
    }

    /// Get current seed.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Access reference to internal Flow Matching DiT model.
    pub fn dit(&self) -> &FlowMatchingDiT {
        &self.dit
    }

    /// Access mutable reference to internal Flow Matching DiT model.
    pub fn dit_mut(&mut self) -> &mut FlowMatchingDiT {
        &mut self.dit
    }

    /// Synthesize continuous acoustic speech latent frames from text and optional speaker prompt.
    pub fn synthesize_latent(
        &self,
        text: &str,
        speaker_ref: Option<&[f32]>,
        num_steps: usize,
        scheme: FlowSolverScheme,
    ) -> Vec<Vec<f32>> {
        let trimmed = text.trim();
        let num_chars = if trimmed.is_empty() { 2 } else { trimmed.chars().count() };
        let num_frames = (num_chars * self.frames_per_char).max(8);

        // 1. Text token conditioning sequence
        let text_tokens = self.text_encoder.encode_text(text);

        // 2. Speaker acoustic prompt embedding (if provided)
        let speaker_tokens = speaker_ref.and_then(|slice| {
            let tokens = self.speaker_embedder.embed_speaker_slice(slice, self.latent_dim);
            if tokens.is_empty() {
                None
            } else {
                Some(tokens)
            }
        });

        let conditioning = FlowConditioning::new(text_tokens, speaker_tokens);

        // 3. Prior standard Gaussian noise tensor x0 ~ N(0, I)
        let mut rng = DeterministicRng::new(self.seed);
        let x0 = rng.sample_latent(num_frames, self.latent_dim);

        // 4. Numerical ODE integration from t = 0 to t = 1
        self.solver.solve(&self.dit, &x0, Some(&conditioning), num_steps, scheme)
    }
}
