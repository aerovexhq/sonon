#![deny(unsafe_code)]

//! Sonon Command-Line Interface (CLI).
//!
//! Provides edge execution, performance benchmarking, shared memory daemon hosting,
//! and acoustic wake-word evaluation in 100% pure safe Rust.

use sonon::engine::{FeatureMode, SononEngine};
use sonon::neuromorphic::{NeuromorphicCochlea, SpikingKwsCell};
use sonon::phonetic::{write_wav_file, SyntheticExemplarGenerator};
use sonon::shm::{ShmAudioChannel, DEFAULT_SHM_PATH};
use std::env;
use std::f32::consts::PI;
use std::fs;
use std::time::Instant;

fn print_usage() {
    println!("Sonon Acoustic Intelligence & Wake-Word Engine (v0.1.0)");
    println!("Usage: sonon <command> [options]");
    println!();
    println!("Commands:");
    println!("  synthesize <phrase>  Generate multi-speaker synthetic acoustic training dataset");
    println!("                       Options: --output <dir>, --count <n>");
    println!("  evaluate <keyword>   Run on-device confusion matrix & phonetic foil discrimination");
    println!("  benchmark            Run comprehensive multi-subsystem DSP throughput benchmarks");
    println!("  shm [path]           Host POSIX shared memory ring buffer daemon (default: /dev/shm/sonon_audio)");
    println!("  demo-spiking         Execute neuromorphic silicon cochlea and spiking KWS demo");
    println!("  help                 Display this usage documentation");
    println!();
    println!("Examples:");
    println!("  cargo run --release --bin sonon -- synthesize \"take off\" --output dataset/take_off -n 5");
    println!("  cargo run --release --bin sonon -- evaluate \"take off\"");
    println!("  cargo run --release --bin sonon -- benchmark");
    println!("  cargo run --release --bin sonon -- shm /dev/shm/sonon_audio");
}

fn run_benchmark() {
    println!("============================================================");
    println!("Sonon Performance & Real-Time Throughput Benchmark");
    println!("============================================================");

    let sample_rate = 16000.0f32;
    let block_size = 512;
    let hop_size = 160;
    let num_mfcc = 26;

    let mut engine = SononEngine::new(sample_rate, block_size, hop_size, num_mfcc);
    engine.set_feature_mode(FeatureMode::Pcen);

    // 1. Classical DSP & PCEN Pipeline Benchmark
    let test_seconds = 2.0;
    let total_samples = (sample_rate * test_seconds) as usize;
    let test_audio: Vec<f32> = (0..total_samples)
        .map(|i| {
            let t = i as f32 / sample_rate;
            0.6 * (2.0 * PI * 440.0 * t).sin() + 0.3 * (2.0 * PI * 1200.0 * t).sin()
        })
        .collect();

    // Enroll synthetic keyword
    let ref_features = engine.extract_features(&test_audio[0..3200]);
    engine.enroll_keyword("test_phrase", ref_features, 3.5);

    let dsp_start = Instant::now();
    let num_passes = 10;
    for _ in 0..num_passes {
        let _ = engine.ingest_samples(&test_audio);
    }
    let dsp_elapsed = dsp_start.elapsed();

    let dsp_samples_total = total_samples * num_passes;
    let dsp_rate = (dsp_samples_total as f64) / dsp_elapsed.as_secs_f64();
    let dsp_rtf = dsp_rate / (sample_rate as f64);

    println!(
        "[DSP + PCEN + DTW Pipeline] Throughput: {:.0} samples/sec ({:.1}x real-time)",
        dsp_rate, dsp_rtf
    );

    // 2. Neuromorphic Silicon Cochlea & LIF Spiking Engine Benchmark
    let mut cochlea = NeuromorphicCochlea::new(sample_rate, 16, 100.0, 7500.0);
    let mut kws = SpikingKwsCell::for_plank(16);

    let mut spikes = Vec::with_capacity(32768);
    let neuro_start = Instant::now();
    for _ in 0..num_passes {
        spikes.clear();
        cochlea.process_buffer(&test_audio, 0, &mut spikes);
        let _ = kws.step_events(&spikes);
    }
    let neuro_elapsed = neuro_start.elapsed();

    let neuro_rate = (dsp_samples_total as f64) / neuro_elapsed.as_secs_f64();
    let neuro_rtf = neuro_rate / (sample_rate as f64);
    let spike_event_rate = (spikes.len() * num_passes) as f64 / neuro_elapsed.as_secs_f64();

    println!(
        "[Neuromorphic Cochlea + SNN] Audio Rate: {:.0} samples/sec ({:.1}x real-time)",
        neuro_rate, neuro_rtf
    );
    println!(
        "[Spike Processing Velocity] Event Rate: {:.0} spikes/sec (Generated {} spikes)",
        spike_event_rate,
        spikes.len()
    );

    println!("============================================================");
    println!("Benchmark completed successfully. All engines operational.");
    println!("============================================================");
}

fn run_shm_daemon(path: &str) {
    println!("Initializing Sonon POSIX Shared Memory Channel at: {path}");
    let mut channel = ShmAudioChannel::create_or_open(path, 16000.0, 65536)
        .expect("Failed to initialize shared memory channel");

    let _ = channel.write_health_status(0.98, 0);
    let _ = channel.write_detection_event("standby", 0.0);

    println!("Channel active. Capacity: 65,536 samples (4.096 seconds at 16 kHz).");
    println!("Ready for zero-copy Python, C++, and Node.js streaming clients.");
}

fn run_demo_spiking() {
    println!("Running Neuromorphic Silicon Cochlea & Spiking KWS Demo...");
    let sample_rate = 16000.0f32;
    let mut cochlea = NeuromorphicCochlea::new(sample_rate, 16, 100.0, 7500.0);
    let mut kws = SpikingKwsCell::for_plank(16);

    // Synthesize "Plank" phoneme progression
    let mut audio = vec![0.0f32; 320]; // 20ms silence
    for i in 0..480 {
        let t = i as f32 / sample_rate;
        audio.push(0.7 * (2.0 * PI * 350.0 * t).sin() + 0.3 * (2.0 * PI * 4200.0 * t).sin());
    }
    for i in 0..960 {
        let t = i as f32 / sample_rate;
        audio.push(0.6 * (2.0 * PI * 750.0 * t).sin() + 0.4 * (2.0 * PI * 1800.0 * t).sin());
    }
    for i in 0..480 {
        let t = i as f32 / sample_rate;
        audio.push(0.7 * (2.0 * PI * 2800.0 * t).sin() + 0.3 * (2.0 * PI * 3500.0 * t).sin());
    }
    audio.extend(vec![0.0f32; 320]);

    let mut spikes = Vec::new();
    cochlea.process_buffer(&audio, 0, &mut spikes);
    println!(
        "Transduced {} acoustic samples into {} asynchronous Address-Event Representation spikes.",
        audio.len(),
        spikes.len()
    );

    let detections = kws.step_events(&spikes);
    println!("Spiking Neural Network Detections: {}", detections.len());
    for det in &detections {
        println!(
            "  -> Triggered Keyword: '{}', Confidence: {:.2}, Trigger Timestamp: {:.3}s",
            det.keyword, det.confidence, det.timestamp_sec
        );
    }
}

fn run_synthesize(args: &[String]) {
    if args.len() < 3 {
        println!("Usage: sonon synthesize <phrase> [--output <dir>] [--count <n>]");
        return;
    }
    let phrase = &args[2];
    let mut out_dir = "synthetic_dataset".to_string();
    let mut count = 5usize;

    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--output" | "-o" => {
                if i + 1 < args.len() {
                    out_dir = args[i + 1].clone();
                    i += 2;
                    continue;
                }
            }
            "--count" | "-n" => {
                if i + 1 < args.len() {
                    if let Ok(c) = args[i + 1].parse() {
                        count = c;
                    }
                    i += 2;
                    continue;
                }
            }
            _ => {
                i += 1;
            }
        }
    }

    println!("============================================================");
    println!("Sonon Synthetic Speech Exemplar Generator");
    println!("Phrase: \"{}\" | Target Count: {} | Output: {}", phrase, count, out_dir);
    println!("============================================================");

    let sample_rate = 16000.0f32;
    let generator = SyntheticExemplarGenerator::new(sample_rate);
    let exemplars = generator.generate_exemplars_with_metadata(phrase, count);

    if exemplars.is_empty() {
        eprintln!("Failed to synthesize exemplars for phrase: {phrase}");
        return;
    }

    let _ = fs::create_dir_all(&out_dir);
    let slug = phrase.to_lowercase().replace(' ', "_");

    println!("Generated {} acoustic training variations:", exemplars.len());
    for (idx, ex) in exemplars.iter().enumerate() {
        let filename = format!("{}/{}_{:03}.wav", out_dir, slug, idx + 1);
        if let Err(e) = write_wav_file(&filename, &ex.audio, sample_rate as u32) {
            eprintln!("Error writing {filename}: {e}");
        } else {
            let dur_sec = ex.audio.len() as f32 / sample_rate;
            println!(
                "  [{:02}] {} -> F0: {:.0} Hz, Rate: {:.2}x, Tract: {:.2}x, Accent: {:?}, Contour: {:?}, Dur: {:.2}s",
                idx + 1, filename, ex.pitch_f0, ex.speaking_rate, ex.vocal_tract_scale, ex.accent, ex.contour, dur_sec
            );
        }
    }

    let manifest_path = format!("{}/manifest.json", out_dir);
    if let Ok(manifest_json) = serde_json::to_string_pretty(&exemplars) {
        let _ = fs::write(&manifest_path, manifest_json);
        println!("Dataset manifest exported to: {manifest_path}");
    }
    println!("Synthesis completed successfully.");
}

fn run_evaluate(args: &[String]) {
    if args.len() < 3 {
        println!("Usage: sonon evaluate <keyword>");
        return;
    }
    let keyword = &args[2];

    println!("============================================================");
    println!("Sonon On-Device Keyword Confusion Matrix & Discrimination Evaluation");
    println!("Target Keyword: \"{}\"", keyword);
    println!("============================================================");

    let sample_rate = 16000.0f32;
    let mut engine = SononEngine::new(sample_rate, 256, 128, 13);
    engine.set_feature_mode(FeatureMode::Pcen);

    // 1. Enroll via synthetic pipeline
    let threshold = engine.enroll_keyword_synthetic_pipeline(keyword, keyword, 5, 12, 1.40);
    println!("Enrolled synthetic reference template. Calibrated threshold: {:.4}", threshold);

    // 2. Generate positive testing variations
    let synth = SyntheticExemplarGenerator::new(sample_rate);
    let positive_audio = synth.generate_exemplars(keyword, 6);

    // 3. Generate negative phonetic foil utterances
    let foils = SononEngine::phonetic_foils_for_keyword(keyword);
    let mut negative_audio = Vec::new();
    for foil in &foils {
        let foil_audio = synth.generate_exemplars(foil, 2);
        negative_audio.extend(foil_audio);
    }

    // 4. Run discrimination evaluation
    let report = engine.evaluate_keyword_discrimination(keyword, &positive_audio, &negative_audio);

    println!("Evaluation Results:");
    println!("  True Positives  (TP) : {}", report.matrix.true_positives);
    println!("  False Positives (FP) : {}", report.matrix.false_positives);
    println!("  True Negatives  (TN) : {}", report.matrix.true_negatives);
    println!("  False Negatives (FN) : {}", report.matrix.false_negatives);
    println!("------------------------------------------------------------");
    println!("  Precision            : {:.1}%", report.matrix.precision() * 100.0);
    println!("  Recall (Sensitivity) : {:.1}%", report.matrix.recall() * 100.0);
    println!("  F1 Score             : {:.4}", report.matrix.f1_score());
    println!("  Accuracy             : {:.1}%", report.matrix.accuracy() * 100.0);
    println!("  False Positive Rate  : {:.1}%", report.matrix.false_positive_rate() * 100.0);
    println!("------------------------------------------------------------");
    println!("  Mean Pos Match Dist  : {:.4}", report.mean_positive_distance);
    println!("  Mean Neg Foil Dist   : {:.4}", report.mean_negative_distance);
    println!("  Discrimination Margin: {:.4}", report.discrimination_margin);
    println!("============================================================");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        return;
    }

    match args[1].as_str() {
        "synthesize" => run_synthesize(&args),
        "evaluate" => run_evaluate(&args),
        "benchmark" => run_benchmark(),
        "shm" => {
            let path = if args.len() > 2 {
                &args[2]
            } else {
                DEFAULT_SHM_PATH
            };
            run_shm_daemon(path);
        }
        "demo-spiking" => run_demo_spiking(),
        "help" | "--help" | "-h" => print_usage(),
        other => {
            eprintln!("Unknown command: {other}");
            print_usage();
        }
    }
}
