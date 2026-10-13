#![deny(unsafe_code)]

//! STM32 Minimalist Speech Synthesis Example
//! Synthesizes speech using the INT8 quantized STM32 Nano Engine and exports a WAV file.
//! Zero emojis. Strictly safe Rust execution.

use sonon::edge_runtime::{EdgeSpeechRuntime, StreamingEdgeConfig};
use sonon::phonetic::write_wav_file;
use std::fs;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Aerovex Sonon - STM32 Minimalist Speech Synthesizer");
    println!("===================================================");

    let json_path = "models/stm32/sonon_stm32_nano_weights.json";
    let json_content = fs::read_to_string(json_path)
        .expect("Failed to read STM32 nano weights JSON");

    let sample_rate = 16000u32;
    let config = StreamingEdgeConfig {
        sample_rate: sample_rate as f32,
        frame_size: 256,
        hop_size: 128,
        ring_buffer_capacity: 2048,
        hidden_dim: 64,
        chunk_frames: 4,
        pitch_f0: 135.0,
    };

    let mut runtime = EdgeSpeechRuntime::from_stm32_nano_json(&json_content, config)?;
    let metrics = runtime.stm32_metrics();

    println!("Hardware Compliance Metrics:");
    println!("  Flash Neural Weights: {} bytes ({:.2} KB)", metrics.flash_weights_bytes, metrics.flash_weights_bytes as f32 / 1024.0);
    println!("  SRAM Static Footprint: {} bytes ({:.2} KB)", metrics.sram_usage_bytes, metrics.sram_usage_bytes as f32 / 1024.0);
    println!("  Fits STM32F4 Entry:    {}", metrics.fits_stm32f4_entry);
    println!("  Fits STM32F4 Standard: {}", metrics.fits_stm32f4_standard);
    println!("  Fits STM32H7 High-Perf: {}", metrics.fits_stm32h7_high_perf);
    println!("  Estimated Cortex-M4 CPU Load: {:.2}% at 168 MHz", metrics.estimated_cortex_m4_load_percent);

    let phrase = "Waypoint Alpha reached. Maintaining altitude three thousand feet.";
    println!("\nSynthesizing phrase: '{}'", phrase);

    let start = Instant::now();
    let audio = runtime.synthesize_chunk(phrase)?;
    let elapsed = start.elapsed();

    let duration_sec = audio.len() as f32 / sample_rate as f32;
    let rtf = elapsed.as_secs_f32() / duration_sec;
    let speedup = duration_sec / elapsed.as_secs_f32();

    println!("Synthesis completed in {:.2} ms", elapsed.as_secs_f32() * 1000.0);
    println!("Generated Audio Duration: {:.2} s ({} samples)", duration_sec, audio.len());
    println!("Real-Time Factor (RTF):   {:.4}", rtf);
    println!("Throughput:               {:.1}x faster than real-time", speedup);

    let out_dir = "output/speech_synthesis";
    fs::create_dir_all(out_dir)?;
    let out_path = format!("{}/sonon_stm32_minimalist_voice.wav", out_dir);
    write_wav_file(&out_path, &audio, sample_rate)?;
    println!("\nSaved synthesized audio to: {}", out_path);

    Ok(())
}
