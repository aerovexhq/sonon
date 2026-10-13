#![deny(unsafe_code)]

//! Arduino Uno R4 / STM32 Micro-DDSP Neural Synthesizer Hardware Simulation.
//!
//! Synthesizes speech using the pure safe Rust MicroDdspSynthesizer running on simulated MCU hardware.
//! Memory ceiling: < 512 bytes SRAM, zero heap allocations during synthesis loop.
//!
//! Outputs:
//! - modules/sonon/output/speech_synthesis/sonon_arduino_b4_mcu_sim.wav
//!
//! Zero emojis. Strictly safe Rust execution.

use serde::Deserialize;
use sonon::edge_runtime::MicroDdspSynthesizer;
use sonon::phonetic::write_wav_file;
use std::fs;
use std::time::Instant;

#[derive(Debug, Deserialize)]
struct FrameData {
    f0: f32,
    harmonic_amps: Vec<f32>,
    noise_bands: Vec<f32>,
    gain: f32,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ReferenceFramesJson {
    sample_rate: u32,
    hop_length: usize,
    num_frames: usize,
    frames: Vec<FrameData>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Aerovex Sonon - Series B.4 Micro-DDSP MCU Simulation Engine");
    println!("============================================================");

    let json_path = "models/micro_ddsp/reference_frames.json";
    let json_str = fs::read_to_string(json_path)
        .expect("Failed to read Micro-DDSP reference frames JSON");
    let ref_data: ReferenceFramesJson = serde_json::from_str(&json_str)?;

    let sample_rate = ref_data.sample_rate;
    let hop_length = ref_data.hop_length;
    let num_frames = ref_data.frames.len();

    println!("Target Hardware: Arduino Uno R4 (Renesas RA4M1, 48 MHz, 32 KB SRAM)");
    println!("Audio Format:    {} Hz, Mono PCM-16", sample_rate);
    println!("Input Sequence:  {} frames (hop={}, {:.2}s duration)", num_frames, hop_length, (num_frames * hop_length) as f32 / sample_rate as f32);

    let mut synth = MicroDdspSynthesizer::new(sample_rate as f32);
    let ram_bytes = synth.ram_state_bytes();
    println!("Synthesizer Static SRAM Footprint: {} bytes ({:.2} KB)", ram_bytes, ram_bytes as f32 / 1024.0);
    assert!(ram_bytes <= 512, "SRAM usage must be <= 512 bytes");

    let total_samples = num_frames * hop_length;
    let mut output_audio = Vec::with_capacity(total_samples);
    let mut hop_buffer = vec![0.0f32; hop_length];

    println!("\nExecuting real-time streaming synthesis simulation...");
    let start_time = Instant::now();

    for frame in &ref_data.frames {
        synth.process_frame(
            frame.f0,
            &frame.harmonic_amps,
            &frame.noise_bands,
            frame.gain,
            hop_length,
            &mut hop_buffer,
        );
        output_audio.extend_from_slice(&hop_buffer);
    }

    let elapsed = start_time.elapsed();
    let duration_sec = output_audio.len() as f32 / sample_rate as f32;
    let rtf = elapsed.as_secs_f32() / duration_sec;
    let speedup = duration_sec / elapsed.as_secs_f32();

    println!("Simulation completed in {:.2} ms", elapsed.as_secs_f32() * 1000.0);
    println!("Real-Time Factor (RTF): {:.5}", rtf);
    println!("Simulation Throughput:  {:.1}x faster than real-time", speedup);

    // Peak normalize to -1.0 dBFS
    let max_abs = output_audio.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
    if max_abs > 1e-4 {
        let target_peak = 0.891f32;
        let scale = target_peak / max_abs;
        for s in &mut output_audio {
            *s *= scale;
        }
    }

    // Clean fade tail (50ms)
    let fade_samples = (0.05 * sample_rate as f32) as usize;
    if output_audio.len() > fade_samples {
        let n = output_audio.len();
        for i in 0..fade_samples {
            let factor = 1.0 - (i as f32 / fade_samples as f32);
            output_audio[n - fade_samples + i] *= factor;
        }
    }

    let out_dir = "output/speech_synthesis";
    fs::create_dir_all(out_dir)?;
    let out_wav = format!("{}/sonon_arduino_b4_mcu_sim.wav", out_dir);
    write_wav_file(&out_wav, &output_audio, sample_rate)?;
    println!("\nExported B.4 Audio Master: {}", out_wav);

    // Adjacent sample delta audit (transient click check)
    let mut max_delta = 0.0f32;
    let mut click_count = 0;
    for i in 1..output_audio.len() {
        let delta = (output_audio[i] - output_audio[i - 1]).abs();
        if delta > max_delta {
            max_delta = delta;
        }
        if delta > 0.45 {
            click_count += 1;
        }
    }

    println!("\nB.4 Microcontroller Signal Quality Audit:");
    println!("  Maximum Adjacent Sample Jump: {:.4}", max_delta);
    println!("  Transient Spikes (>0.45 jump): {}", click_count);
    println!("  SRAM Utilization (32 KB limit): {:.2}%", 100.0 * ram_bytes as f32 / (32.0 * 1024.0));
    println!("============================================================");

    Ok(())
}
