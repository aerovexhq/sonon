# Sonon Neural Speech Dataset Training & Fine-Tuning Guide

This directory provides the end-to-end training, curation, and weight export harness for training custom industrial, aerospace, and mission-critical voice models in Sonon.

---

## Workflow Overview

```
[User Audio & Transcripts]
            |
            v
1. `prepare_dataset.py`  --> Curation, WADA-SNR filter (>20 dB), 24 kHz mono resampling, manifests
            |
            v
2. `fine_tune_acoustic.py` --> Transformer & multi-speaker style latent fine-tuning
            |
            v
3. `train_vocoder.py`       --> BigVGAN-v2 universal vocoder training (SnakeBeta + STFT discriminators)
            |
            v
4. `export_sonon_weights.py` --> FP16/INT8 ONNX quantization & Sonon custom voice binary packaging
            |
            v
[Sonon Safe Rust Engine]    --> Direct inference via `SononEngine::synthesize_speech_audio`
```

---

## 1. Organizing Your Datasets

Place your audio files and transcripts in any local folder (e.g., `data/raw_audio/`):

### Supported Audio Formats:
- `.wav`, `.flac`, `.mp3`, `.ogg` (automatically converted to 24,000 Hz, 16-bit mono).

### Transcript Formats:
You can provide transcripts in either:
- **Pipe-delimited text/CSV** (`metadata.csv`):
  ```
  file_001|Waypoint alpha confirmed. Cruising altitude forty-five thousand feet.
  file_002|Tower, this is flight lead. Pre-flight checks are complete.
  ```
- **JSON dictionary** (`transcripts.json`):
  ```json
  {
    "file_001": "Waypoint alpha confirmed. Cruising altitude forty-five thousand feet.",
    "file_002": "Tower, this is flight lead. Pre-flight checks are complete."
  }
  ```

---

## 2. Step 1: Preprocess and Curate Audio

Run the inspection and curation pipeline:

```bash
python modules/sonon/training/prepare_dataset.py \
    --input_dir data/raw_audio/ \
    --output_dir data/curated/ \
    --transcript_file data/raw_audio/metadata.csv \
    --val_split 0.1
```

### What this step does:
- Validates audio quality: Rejects files with SNR < 20 dB or digital clipping (peak > 0.995).
- Resamples to 24 kHz and removes DC offset.
- Trims leading/trailing silence below -45 dBFS.
- Generates `train_manifest.json` and `val_manifest.json`.

---

## 3. Step 2: Fine-Tune the Acoustic Model

Fine-tune the acoustic transformer and speaker style embedding network:

```bash
python modules/sonon/training/fine_tune_acoustic.py \
    --train_manifest data/curated/train_manifest.json \
    --val_manifest data/curated/val_manifest.json \
    --output_dir checkpoints/acoustic \
    --batch_size 16 \
    --epochs 20
```

---

## 4. Step 3: Fine-Tune the BigVGAN-v2 Vocoder

Fine-tune the neural vocoder on your target acoustic environment:

```bash
python modules/sonon/training/train_vocoder.py \
    --train_manifest data/curated/train_manifest.json \
    --output_dir checkpoints/vocoder \
    --sample_rate 24000 \
    --epochs 50
```

---

## 5. Step 4: Export Weights for Sonon

Export the fine-tuned model into quantized ONNX graphs and Sonon voice binary format:

```bash
python modules/sonon/training/export_sonon_weights.py \
    --checkpoint checkpoints/acoustic/best_model.pt \
    --output_dir export/sonon_model/ \
    --quantize_int8
```

This generates:
- `sonon_acoustic_fp16.onnx`: High-precision model for server and GPU cluster inference.
- `sonon_acoustic_int8.onnx`: Quantized model for embedded avionics hardware.
- `sonon_custom_voices.bin`: Extracted speaker style vectors for your trained voices.

---

## 6. Step 5: Native Inference in Sonon

Run high-fidelity speech synthesis using your custom voices directly in Sonon:

```python
python modules/sonon/scripts/sonon_neural_synthesizer.py \
    --text "Waypoint alpha reached. Autopilot online." \
    --voice "conversational_natural" \
    --out "output/custom_sample.wav"
```

Or via the Safe Rust API:

```rust
let engine = SononEngine::new(16000.0, 512, 160, 13);
let normalized_text = engine.normalize_aerospace_text("FL350 cleared RWY28R");
```
