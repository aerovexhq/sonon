#![deny(unsafe_code)]

use std::f32::consts::PI;

/// SincNet Parametric Convolutional Frontend with Dynamic Motor RPM Notching.
///
/// Implements bandpass filter banks parameterized analytically by rectangular
/// frequency cutoffs [f1, f2] in Hz using normalized sinc kernels:
/// g_i[n] = 2 * f2 * sinc(2 * pi * f2 * t) - 2 * f1 * sinc(2 * pi * f1 * t)
///
/// Directly modulates filter gains to notch out Blade Pass Frequencies (BPF)
/// based on real-time autopilot ESC telemetry.
#[derive(Debug, Clone)]
pub struct SincConvFrontend {
    sample_rate: f32,
    kernel_size: usize,
    num_filters: usize,
    band_edges: Vec<(f32, f32)>, // (f1, f2) per filter in Hz
    filter_weights: Vec<Vec<f32>>,
    notched_filters: Vec<bool>,
}

impl SincConvFrontend {
    /// Constructs a new SincConvFrontend with Mel-scale distributed bandpass filters.
    pub fn new(sample_rate: f32, kernel_size: usize, num_filters: usize) -> Self {
        assert!(kernel_size % 2 == 1, "Kernel size must be odd for symmetric linear phase");
        assert!(num_filters > 0);

        let nyquist = sample_rate * 0.5;
        let mut band_edges = Vec::with_capacity(num_filters);

        // Distribute band edges logarithmically across speech band (80 Hz to Nyquist)
        let f_min = 80.0f32;
        let f_max = nyquist - 100.0;
        let mel_min = 2595.0 * (1.0 + f_min / 700.0).log10();
        let mel_max = 2595.0 * (1.0 + f_max / 700.0).log10();

        for i in 0..num_filters {
            let m1 = mel_min + (mel_max - mel_min) * (i as f32) / (num_filters as f32);
            let m2 = mel_min + (mel_max - mel_min) * ((i + 1) as f32) / (num_filters as f32);

            let f1 = 700.0 * (10.0f32.powf(m1 / 2595.0) - 1.0);
            let f2 = 700.0 * (10.0f32.powf(m2 / 2595.0) - 1.0);
            band_edges.push((f1, f2));
        }

        let mut frontend = Self {
            sample_rate,
            kernel_size,
            num_filters,
            band_edges,
            filter_weights: Vec::with_capacity(num_filters),
            notched_filters: vec![false; num_filters],
        };

        frontend.recompute_filters();
        frontend
    }

    /// Recomputes analytical sinc coefficients for all filters.
    fn recompute_filters(&mut self) {
        let half_k = (self.kernel_size / 2) as isize;
        self.filter_weights.clear();

        for &(f1, f2) in &self.band_edges {
            let mut kernel = Vec::with_capacity(self.kernel_size);
            let f1_norm = f1 / self.sample_rate;
            let f2_norm = f2 / self.sample_rate;

            for n in -half_k..=half_k {
                let val = if n == 0 {
                    2.0 * (f2_norm - f1_norm)
                } else {
                    let t = n as f32;
                    let sinc2 = (2.0 * PI * f2_norm * t).sin() / (PI * t);
                    let sinc1 = (2.0 * PI * f1_norm * t).sin() / (PI * t);
                    sinc2 - sinc1
                };

                // Apply Hamming window to suppress side-lobes
                let w = 0.54 - 0.46 * (2.0 * PI * (n + half_k) as f32 / (self.kernel_size - 1) as f32).cos();
                kernel.push(val * w);
            }

            self.filter_weights.push(kernel);
        }
    }

    /// Modulates filter weights according to live motor telemetry RPM.
    /// Filters whose passband overlaps with rotor blade pass frequency f_BPF are notched.
    pub fn update_rotor_telemetry(&mut self, rpm: f32, num_blades: usize) {
        let bpf = (num_blades as f32 * rpm.max(0.0)) / 60.0;
        let bpf_harmonics = [bpf, bpf * 2.0, bpf * 3.0];

        for i in 0..self.num_filters {
            let (f1, f2) = self.band_edges[i];
            let mut overlaps_rotor = false;

            for &harmonic in &bpf_harmonics {
                // If harmonic falls within filter passband or within 25 Hz transition margin
                if harmonic >= (f1 - 25.0) && harmonic <= (f2 + 25.0) {
                    overlaps_rotor = true;
                    break;
                }
            }

            self.notched_filters[i] = overlaps_rotor;
        }
    }

    /// Convolves an input audio slice with the SincNet filterbank into a provided buffer,
    /// eliminating heap allocation in streaming loops.
    pub fn process_frame_into(&self, frame: &[f32], out: &mut [f32]) {
        if frame.len() < self.kernel_size || out.len() < self.num_filters {
            out.fill(0.0);
            return;
        }

        let offset = frame.len() - self.kernel_size;
        let sub = &frame[offset..];

        for i in 0..self.num_filters {
            if self.notched_filters[i] {
                // Notched channel produces 0 energy
                out[i] = 0.0;
                continue;
            }

            let kernel = &self.filter_weights[i];
            let mut sum = 0.0f32;
            for j in 0..self.kernel_size {
                sum += sub[j] * kernel[j];
            }
            out[i] = sum.abs();
        }
    }

    /// Convolves an input audio slice with the SincNet filterbank, producing
    /// an output vector of filter energies of length `num_filters`.
    pub fn process_frame(&self, frame: &[f32]) -> Vec<f32> {
        let mut outputs = vec![0.0f32; self.num_filters];
        self.process_frame_into(frame, &mut outputs);
        outputs
    }

    /// Number of filter channels.
    pub fn num_filters(&self) -> usize {
        self.num_filters
    }

    /// Returns passband edge frequencies for filter channel `i`.
    pub fn filter_band(&self, i: usize) -> (f32, f32) {
        self.band_edges[i]
    }
}

/// Selective Structured State-Space Model Cell (AeroSSM).
///
/// Implements continuous-to-discrete state-space recurrence with input-dependent
/// selection mechanism (Mamba principle):
///
/// continuous:  h'(t) = A * h(t) + B * x(t),  y(t) = C * h(t) + D * x(t)
/// discrete:    h_k   = A_bar_k * h_{k-1} + B_bar_k * x_k
///              y_k   = C_k * h_k + D * x_k
///
/// Zero heap allocations during streaming step processing via pre-allocated scratch buffers.
#[derive(Debug, Clone)]
pub struct AeroSsmCell {
    d_model: usize,
    state_dim: usize,
    // Diagonal continuous transition matrix A (d_model x state_dim), initialized with HiPPO decay
    a_diag: Vec<Vec<f32>>,
    // Input-dependent projections (weights for B, C, Delta)
    w_b: Vec<Vec<f32>>,     // (state_dim x d_model)
    w_c: Vec<Vec<f32>>,     // (state_dim x d_model)
    w_delta: Vec<Vec<f32>>, // (d_model x d_model)
    b_delta: Vec<f32>,      // (d_model)
    d_skip: Vec<f32>,       // Feed-through skip gain (d_model)
    // Streaming hidden state h_k (d_model x state_dim)
    hidden_state: Vec<Vec<f32>>,
    // Preallocated reusable scratch buffers
    delta_scratch: Vec<f32>,
    b_scratch: Vec<f32>,
    c_scratch: Vec<f32>,
    y_scratch: Vec<f32>,
}

impl AeroSsmCell {
    /// Constructs a new AeroSSM cell.
    ///
    /// # Arguments
    /// * `d_model` - Channel feature dimension (e.g., 32 or 64).
    /// * `state_dim` - Latent state space dimension per channel (typically 16).
    pub fn new(d_model: usize, state_dim: usize) -> Self {
        assert!(d_model > 0 && state_dim > 0);

        // Initialize A_diag using HiPPO continuous-time diagonal decay
        let mut a_diag = Vec::with_capacity(d_model);
        for _ in 0..d_model {
            let mut row = Vec::with_capacity(state_dim);
            for n in 0..state_dim {
                let lambda = -(0.5 + n as f32);
                row.push(lambda);
            }
            a_diag.push(row);
        }

        // Initialize pseudo-random deterministic projection weights (normalized)
        let mut w_b = vec![vec![0.0f32; d_model]; state_dim];
        let mut w_c = vec![vec![0.0f32; d_model]; state_dim];
        let scale = (2.0f32 / (d_model as f32)).sqrt();

        for n in 0..state_dim {
            for d in 0..d_model {
                let phase = (n * 31 + d * 17) as f32;
                w_b[n][d] = scale * (phase.sin());
                w_c[n][d] = scale * ((phase + 1.0).cos());
            }
        }

        let mut w_delta = vec![vec![0.0f32; d_model]; d_model];
        for i in 0..d_model {
            w_delta[i][i] = 1.0;
        }

        let b_delta = vec![-1.0f32; d_model];
        let d_skip = vec![1.0f32; d_model];
        let hidden_state = vec![vec![0.0f32; state_dim]; d_model];

        Self {
            d_model,
            state_dim,
            a_diag,
            w_b,
            w_c,
            w_delta,
            b_delta,
            d_skip,
            hidden_state,
            delta_scratch: vec![0.0f32; d_model],
            b_scratch: vec![0.0f32; state_dim],
            c_scratch: vec![0.0f32; state_dim],
            y_scratch: vec![0.0f32; d_model],
        }
    }

    /// Evaluates a single streaming audio feature frame in linear O(1) time.
    /// Updates the internal hidden state h_k and returns the output slice.
    #[inline]
    pub fn step(&mut self, x: &[f32]) -> &[f32] {
        let d_m = self.d_model;
        let s_dim = self.state_dim;
        assert_eq!(x.len(), d_m, "Input length must match d_model");

        // 1. Compute selective step size Delta_k = Softplus(W_delta * x + b_delta)
        for i in 0..d_m {
            let row = &self.w_delta[i];
            let mut sum = self.b_delta[i];
            for j in 0..d_m {
                sum += row[j] * x[j];
            }
            let sp = if sum > 20.0 {
                sum
            } else if sum < -20.0 {
                0.0
            } else {
                (1.0 + sum.exp()).ln()
            };
            self.delta_scratch[i] = sp.max(1e-4);
        }

        // 2. Compute selective B_k and C_k projections
        for n in 0..s_dim {
            let wb = &self.w_b[n];
            let wc = &self.w_c[n];
            let mut sum_b = 0.0f32;
            let mut sum_c = 0.0f32;
            for d in 0..d_m {
                sum_b += wb[d] * x[d];
                sum_c += wc[d] * x[d];
            }
            self.b_scratch[n] = sum_b;
            self.c_scratch[n] = sum_c;
        }

        // 3. State-space recurrence update (in-place)
        for d in 0..d_m {
            let dt = self.delta_scratch[d];
            let mut state_output = 0.0f32;
            let a_row = &self.a_diag[d];
            let h_row = &mut self.hidden_state[d];
            let x_d = x[d];

            for n in 0..s_dim {
                let a_val = a_row[n];
                let a_bar = (dt * a_val).exp();
                let b_bar = if a_val.abs() > 1e-6 {
                    ((a_bar - 1.0) / a_val) * self.b_scratch[n]
                } else {
                    dt * self.b_scratch[n]
                };

                let prev_h = h_row[n];
                let next_h = a_bar * prev_h + b_bar * x_d;
                h_row[n] = next_h;

                state_output += self.c_scratch[n] * next_h;
            }

            self.y_scratch[d] = state_output + self.d_skip[d] * x_d;
        }

        &self.y_scratch
    }

    /// Evaluates an entire sequence of frames, updating state progressively.
    pub fn forward_sequence(&mut self, sequence: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut outputs = Vec::with_capacity(sequence.len());
        for frame in sequence {
            outputs.push(self.step(frame).to_vec());
        }
        outputs
    }

    /// Resets latent state h to zero.
    pub fn reset(&mut self) {
        for row in self.hidden_state.iter_mut() {
            row.fill(0.0);
        }
        self.delta_scratch.fill(0.0);
        self.b_scratch.fill(0.0);
        self.c_scratch.fill(0.0);
        self.y_scratch.fill(0.0);
    }

    /// Model dimension.
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// State space dimension.
    pub fn state_dim(&self) -> usize {
        self.state_dim
    }
}
