# Sheet Music Transcription — Audio ➜ MIDI ➜ Sheet

**Team:** *Fine-Tuned*  
**Members:** Isaac Bilsel, Christian Schulte

This repository implements a complete, end-to-end pipeline for **automatic piano transcription**:

**Audio recording → Neural model → MIDI file**

The modeled problem is **Audio → MIDI** (polyphonic solo piano), trained on the **MAESTRO dataset**.  
Conversion from MIDI to printable sheet music is intentionally left to external notation tools (MuseScore, LilyPond, etc.), which is standard practice.

---

## What this project can do

###  Audio → MIDI transcription
- Transcribe `.wav`, `.mp3`, or similar audio into a `.mid` file
- Automatically splits long recordings into **30-second chunks**
- Converts audio to **log-mel spectrograms**
- Predicts **framewise 88-pitch piano-rolls**
- Converts predictions into MIDI notes by grouping contiguous active frames

> Note: Sustain/pedal modeling and advanced MIDI cleanup are not implemented yet.

---

###  Train transcription models on MAESTRO
- Uses official MAESTRO train / validation / test splits
- Supports full-file or **chunked training** (memory-efficient)
- Checkpointing, resuming, logging, and live loss plots
- Works with or without cached preprocessing

---

###  Fast training via dataset caching
- One-time preprocessing converts audio+MIDI into cached `.pt` chunks
- Results in **10–50× faster training**
- Supports:
  - **Mel-spectrogram cache** (CNN-RNN models)
  - **Waveform cache** (future Transformer / AST experiments)

---

###  Evaluation + threshold tuning
- Computes **framewise F1 score**
- Optional MIDI generation during evaluation
- Automatic **coarse-to-fine threshold tuning**
- Headless mode for scripts / benchmarks

---

## Model architectures

Two CNN-RNN transcription models are provided through a unified interface.

### Model comparison

| Feature | CNNRNNModel | CNNRNNModelLarge |
|-------|-------------|------------------|
| Parameters | 36M | 89M |
| CNN Depth | 2 blocks | 4 blocks |
| Residual Connections | ✗ | ✓ |
| Multi-Head Attention | ✗ | ✓ (8 heads) |
| Multi-Scale RNN | ✗ | ✓ (dual LSTM) |
| Onset / Offset Heads | ✗ | ✓ |
| Training Speed | Fast | ~2.5× slower |
| Typical Framewise F1 | 0.70–0.78 | 0.85–0.90 |

---

## Repository structure

```
.
├── main.py                          # Inference: audio → MIDI
├── data/
│   ├── dataset.py                   # MAESTRO loader (full or chunked)
│   └── cached_dataset.py            # Cached dataset loader
├── models/
│   ├── cnn_rnn_model.py             # CNNRNNModel + CNNRNNModelLarge
│   ├── transcription_model.py       # Unified model wrapper
│   ├── transformer_model.py         # Placeholder (future work)
│   └── remi_tokenizer.py            # Token scaffolding (future work)
├── scripts/
│   ├── train_cnn.py                 # Training CLI
│   ├── preprocess_dataset.py        # Dataset caching + verification
│   ├── evaluate.py                  # Evaluation + threshold tuning
│   ├── train_ast.py                 # Experimental (future)
│   ├── transcribe_chunk.py          # Transcribe audio chunks
│   ├── transcribe_and_visualize.py  # Transcription with visualization
│   ├── visualize_inference.py       # Visualize model predictions
│   ├── plot_training_comparison.py  # Compare training runs
│   └── find_best_worst_samples.py   # Analyze model performance
├── train/
│   └── train_transcriber.py         # Training loops and utilities
├── example.sh                       # Example end-to-end workflow
├── requirements.txt
└── *.ipynb                          # Optional notebooks
```

---

## Setup

### 1) Create environment (Python 3.12 recommended)
```bash
python3.12 -m venv venv
source venv/bin/activate
```

### 2) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Dataset

Place MAESTRO at the repo root (or pass `--root_dir`):
- `maestro-v3.0.0/` (recommended)
- or `maestro-v2.0.0/`

---

## Quickstart

### Step 1 — (Recommended) Preprocess & cache
```bash
python scripts/preprocess_dataset.py --n_mels 320
```

Background mode:
```bash
python scripts/preprocess_dataset.py --n_mels 320 --background
tail -f preprocess_*.log
```

---

### Step 2 — Train
```bash
python scripts/train_cnn.py \
  --epochs 25 \
  --batch_size 8 \
  --lr 1e-4 \
  --save_every 5
```

Chunked, memory-efficient training:
```bash
python scripts/train_cnn.py \
  --epochs 40 \
  --batch_size 16 \
  --chunk_length 30.0
```

---

### Step 3 — Evaluate
```bash
python scripts/evaluate.py \
  --model outputs/.../checkpoints/model_final.pth
```

Threshold tuning:
```bash
python scripts/evaluate.py \
  --model outputs/.../checkpoints/model_final.pth \
  --tune_threshold \
  --split validation \
  --subset 50
```

---

## Inference (Audio → MIDI)

Basic usage:
```bash
python main.py <audio_file> <model_checkpoint>
```

Examples:
```bash
python main.py my_piano.mp3 model_final.pth
python main.py song.wav model.pth -o output.mid -t 0.45
python main.py recording.wav model.pth -d cuda
```

---

## Example workflow script

[example.sh](example.sh) demonstrates the full pipeline:
```bash
./example.sh preprocess
./example.sh train
./example.sh eval
./example.sh all
```

Edit variables at the top of [example.sh](example.sh) to configure runs.


---

## Acknowledgments
- MAESTRO Dataset (v2.0.0 / v3.0.0) — aligned audio and MIDI for piano transcription
