#![deny(unsafe_code)]

//! Speech audio sample generator for Sonon generative synthesis engines.
//!
//! Synthesizes speech waveforms across multiple architectural paradigms:
//! 1. Edge & WebAssembly Streaming Runtime (INT8 Quantization + 3-Formant Resonators + LF Glottal Pulse)
//! 2. Biomechanical Port-Hamiltonian Dyadic Wavelet Flow Synthesizer
//! 3. Multi-Speaker Phonetic Articulatory Formant Synthesizer with Prosodic Variations

use sonon::edge_runtime::{EdgeSpeechRuntime, StreamingEdgeConfig};
use sonon::flow_matching::FlowSolverScheme;
use sonon::phonetic::{write_wav_file, SyntheticExemplarGenerator};
use sonon::wavelet_synthesis::{WaveletPhysicalFlowSynthesizer, WaveletSynthesizerConfig};
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = "output/speech_synthesis";
    fs::create_dir_all(output_dir)?;

    println!("============================================================");
    println!("Sonon Generative Speech Synthesis - Local Audio Generation");
    println!("Output directory: {}", output_dir);
    println!("============================================================");

    let sample_rate = 16000u32;

    // -------------------------------------------------------------------------
    // 1. Edge & WebAssembly Streaming Runtime
    // -------------------------------------------------------------------------
    println!("\n[1/3] Synthesizing via Deterministic Edge & WASM Streaming Runtime...");
    let edge_phrases = [
        ("takeoff", "TAKEOFF"),
        ("abort_mission", "ABORT MISSION"),
        ("status_normal", "STATUS NORMAL"),
        ("waypoint_reached", "WAYPOINT REACHED"),
    ];

    let edge_cfg = StreamingEdgeConfig {
        sample_rate: sample_rate as f32,
        frame_size: 256,
        hop_size: 128,
        ring_buffer_capacity: 16384,
        hidden_dim: 64,
        chunk_frames: 4,
        pitch_f0: 135.0,
    };
    let mut edge_runtime = EdgeSpeechRuntime::new(edge_cfg);

    for &(slug, phrase) in &edge_phrases {
        let audio = edge_runtime.synthesize_chunk(phrase)
            .map_err(|e| format!("Edge synthesis error for '{}': {}", phrase, e))?;

        let file_path = format!("{}/{}_edge_runtime.wav", output_dir, slug);
        write_wav_file(&file_path, &audio, sample_rate)?;
        let dur_sec = audio.len() as f32 / sample_rate as f32;
        println!("  - Created: {} ({} samples, {:.2}s)", file_path, audio.len(), dur_sec);
    }

    // -------------------------------------------------------------------------
    // 2. Physical Port-Hamiltonian Dyadic Wavelet Flow Synthesizer
    // -------------------------------------------------------------------------
    println!("\n[2/3] Synthesizing via Dyadic Morlet Wavelet & Port-Hamiltonian Flow Matching...");
    let wavelet_phrases = [
        ("takeoff", "TAKEOFF", 150.0),
        ("hold_position", "HOLD POSITION", 140.0),
        ("return_to_launch", "RETURN TO LAUNCH", 160.0),
    ];

    let wavelet_cfg = WaveletSynthesizerConfig {
        sample_rate: sample_rate as f32,
        octaves: 6,
        voices_per_octave: 10,
        hidden_dim: 64,
        num_layers: 2,
        num_heads: 4,
        default_solver_scheme: FlowSolverScheme::Midpoint,
        default_num_steps: 4,
        glottal_weight: 0.35,
        seed: 42,
        hop_size: 160,
        samples_per_char: 800,
        ..Default::default()
    };
    let wavelet_synth = WaveletPhysicalFlowSynthesizer::new(wavelet_cfg);

    for &(slug, phrase, f0) in &wavelet_phrases {
        let audio = wavelet_synth.synthesize(phrase, f0, 4, FlowSolverScheme::Midpoint);
        let file_path = format!("{}/{}_wavelet_flow.wav", output_dir, slug);
        write_wav_file(&file_path, &audio, sample_rate)?;
        let dur_sec = audio.len() as f32 / sample_rate as f32;
        println!("  - Created: {} ({} samples, {:.2}s, F0: {}Hz)", file_path, audio.len(), dur_sec, f0);
    }

    // -------------------------------------------------------------------------
    // 3. Multi-Speaker Phonetic Articulatory Formant Synthesizer
    // -------------------------------------------------------------------------
    println!("\n[3/3] Synthesizing via Multi-Speaker Articulatory Formant Generator...");
    let formant_phrases = [
        ("take_off", "take off"),
        ("land", "land"),
        ("abort", "abort"),
        ("plank", "plank"),
    ];

    let exemplar_gen = SyntheticExemplarGenerator::new(sample_rate as f32);

    for &(slug, phrase) in &formant_phrases {
        let exemplars = exemplar_gen.generate_exemplars_with_metadata(phrase, 2);
        for (idx, ex) in exemplars.iter().enumerate() {
            let file_path = format!("{}/{}_formant_speaker_{}.wav", output_dir, slug, idx + 1);
            write_wav_file(&file_path, &ex.audio, sample_rate)?;
            let dur_sec = ex.audio.len() as f32 / sample_rate as f32;
            println!(
                "  - Created: {} ({} samples, {:.2}s, F0: {:.0}Hz, Rate: {:.2}x, Accent: {:?})",
                file_path, ex.audio.len(), dur_sec, ex.pitch_f0, ex.speaking_rate, ex.accent
            );
        }
    }

    println!("\nSpeech audio synthesis test completed successfully!");
    println!("All audio files written to: {}", Path::new(output_dir).canonicalize()?.display());
    Ok(())
}
