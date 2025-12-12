# Sheet Music Transcription — Audio ➜ MIDI ➜ Sheet

**Team:** *Fine-Tuned*  
**Members:** Isaac Bilsel, Christian Schulte

This repository implements the core pipeline for **automatic musical transcription**: converting **audio** into **MIDI**, and then (as a later post-process) into **sheet music**. The current codebase focuses on a **CNN + BiLSTM (RNN)** baseline trained on the **MAESTRO v2.0.0** dataset (polyphonic solo piano). A Transformer variant is planned next.

> High-level plan (from the proposal): use MIDI as the bridge format (Audio → **MIDI** → Sheet). The hard part we model is Audio → MIDI; MIDI → Sheet is handled by existing notation tools.

---

## Repository Structure

```
.
├── data/
│   └── dataset.py                 # MaestroDataset: audio→mel + MIDI→piano-roll labels (aligned)
├── models/
│   ├── cnn_rnn_model.py           # CNN (freq-only pooling) + BiLSTM ➜ 88-pitch framewise logits
│   ├── transcription_model.py     # Wrapper: loss, predict(), device mgmt; model factory entry point
│   └── transformer_model.py       # Placeholder for future Transformer-based transcriber
├── train/
│   └── train_transcriber.py       # Training utilities: loaders, collate/padding, loops
├── outputs/
│   └── 2025-11-05_14-32-10/       # timestamped run directory (created automatically)
│       ├── checkpoints/
│       │   ├── model_epoch_10.pth
│       │   ├── model_epoch_20.pth
│       │   └── model_final.pth
│       └── logs/
│           ├── loss_curve.png     # updated after every epoch
│           └── training_log.txt   # appended each epoch with train/val loss
├── main.py                        # Command-line training entry point
└── music_transcription.ipynb      # (Optional) exploratory notebook — not required for training
```

---

## Dataset

- **MAESTRO v3.0.0** (or v2.0.0) expected at the repo root in: `maestro-v3.0.0/`
- The dataset comes with a CSV containing columns:
  `canonical_composer, canonical_title, split, year, midi_filename, audio_filename, duration`.
- Audio files can be `.wav` or `.mp3`; the loader automatically handles both.
- Supports official **train/validation/test splits** from the CSV.

### Chunked Loading (Memory Efficient)

The dataset now supports **chunked loading** to handle GPU memory constraints:
- Split long audio/MIDI files into fixed-length segments (e.g., 30 seconds)
- Optional overlap between chunks for data augmentation
- Dramatically reduces memory usage (10x less than full files)
- Enable with `--chunk_length` and `--chunk_overlap` arguments

### What the `MaestroDataset` returns
- **Input** `mel`: shape `(1, n_mels, T)` — mel-spectrogram computed from audio (`librosa 0.11`).
- **Target** `roll`: shape `(88, T)` — binary piano-roll (A0..C8), derived from MIDI for **supervised labels**.

Time alignment is enforced by trimming both to the same **T**, and batch-time padding is handled by a custom `collate_fn` during training.

---

## Model (current baseline)

**`models/cnn_rnn_model.py`**

- **CNN frontend** with **frequency-only pooling** (`MaxPool2d(2,1)`) so time resolution is preserved for framewise labeling.
- **BiLSTM** models temporal dependencies across frames.
- **Linear head** outputs **88 pitch logits** per frame.  
- Use **`BCEWithLogitsLoss`** (handled in the wrapper class).

**Shapes** (example):  
`(B, 1, 229, T)` → CNN `(B, 64, ~229/4, T)` → reshape to `(B, T, 64*~57)` → BiLSTM `(B, T, 2H)` → Linear → logits `(B, 88, T)`.

**`models/transcription_model.py`** adds:
- Unified `forward`, `compute_loss()`, and `predict(threshold=0.5)`
- Device handling and a future-proof factory (`model_type="cnn_rnn"` for now).  
- Defensive time-axis upsampling if any mismatch sneaks in during experimentation.

---

## Quickstart (Python 3.12)

### 1) Create and activate a virtualenv
```bash
python3.12 -m venv venv
source venv/bin/activate
python -V  # Python 3.12.x
```

### 2) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Or manually:
```bash
pip install torch librosa pretty_midi numpy pandas matplotlib tqdm scipy scikit-learn
```

### 3) Train (command-line)
```bash
python main.py --year 2017 --epochs 25 --batch_size 8 --lr 1e-4 --save_every 5 --subset_size 100
```
This launches training with periodic checkpoints and live-updating loss plots saved under a timestamped run directory in `outputs/` (see next section).

---

## Training with `main.py`

The primary training entry point is **`main.py`**. It supports various options for dataset loading, chunking, caching, and resuming training.

### Command-Line Arguments

```bash
python main.py [OPTIONS]
```

**Required/Common Arguments:**
- `--root_dir` - Path to MAESTRO dataset (default: `maestro-v3.0.0`)
- `--epochs` - Number of training epochs (default: 25)
- `--batch_size` - Batch size (default: 8)
- `--lr` - Learning rate (default: 1e-4)

**Dataset Options:**
- `--year` - Filter by year (e.g., `2017`), or omit for all years
- `--subset_size` - Limit dataset size for debugging (e.g., `100`)
- `--chunk_length` - Chunk length in seconds (e.g., `30.0`). If `None`, loads full files
- `--chunk_overlap` - Overlap ratio 0.0-1.0 (e.g., `0.25` for 25% overlap)

**Training Options:**
- `--save_every` - Save checkpoint every N epochs (default: 10)
- `--resume` - Path to checkpoint to resume from (e.g., `outputs/.../model_epoch_15.pth`)
- `--start_epoch` - Starting epoch (auto-detected when resuming)

### Example: Basic Training
```bash
python main.py \
  --epochs 25 \
  --batch_size 8 \
  --lr 1e-4 \
  --save_every 5
```

### Example: Memory-Efficient Training with Chunking
```bash
python main.py \
  --epochs 40 \
  --batch_size 16 \
  --chunk_length 30.0 \
  --chunk_overlap 0.0 \
  --save_every 5
```

**Benefits of chunking:**
- ~10x less GPU memory per batch
- Enables larger batch sizes
- Faster epochs with no overlap

### Example: Resume from Checkpoint
```bash
python main.py \
  --resume outputs/2025-12-12_05-09-59/checkpoints/model_epoch_15.pth \
  --epochs 40 \
  --batch_size 16 \
  --chunk_length 30.0
```

The script automatically:
- Loads model weights from checkpoint
- Detects epoch number from filename
- Continues training from that epoch

### Example: Quick Debug Run
```bash
python main.py \
  --subset_size 50 \
  --epochs 3 \
  --batch_size 4 \
  --chunk_length 30.0
```

### Example: Background Training with Script
```bash
# Use the provided script for optimal settings
./example.sh

# Or manually with nohup
nohup python3 main.py \
  --epochs 40 \
  --batch_size 16 \
  --chunk_length 30.0 \
  > outputs/train_run_$(date +"%Y-%m-%d_%H-%M-%S").log 2>&1 &
```

### Outputs
Each run creates a unique timestamped directory:
```
outputs/YYYY-MM-DD_HH-MM-SS/
├── checkpoints/
│   ├── model_epoch_5.pth
│   ├── model_epoch_10.pth
│   ├── model_epoch_15.pth
│   └── model_final.pth
└── logs/
    ├── loss_curve.png         # updated after every epoch
    ├── loss_per_step.png      # per-step training loss
    ├── parameters.txt         # all training parameters
    └── training_log.txt       # epoch-by-epoch train/val loss
```

---

## Dataset Preprocessing & Caching (Advanced)

For **10-50x speedup** in training, preprocess the dataset once and cache it:

### Step 1: Preprocess Dataset

```bash
python preprocess_dataset.py --chunk_length 30.0 --overlap 0.0
```

**What this does:**
- Loads all audio/MIDI files
- Computes mel-spectrograms and piano rolls
- Saves preprocessed chunks as `.pt` files in `cached_dataset/`
- Takes ~1-2 hours, requires ~27GB disk space

**Arguments:**
- `--root_dir` - Path to MAESTRO dataset (default: `maestro-v3.0.0`)
- `--cache_dir` - Where to save cache (default: `cached_dataset`)
- `--chunk_length` - Chunk duration in seconds (default: 30.0)
- `--overlap` - Overlap ratio (default: 0.0)

### Step 2: Train with Cache

The training script **automatically detects** the cache:

```bash
python main.py --epochs 40 --batch_size 16 --chunk_length 30.0
```

Output:
```
✓ Using cached dataset (fast mode!)
```

**Performance improvement:**
- **Without cache**: ~2.6s/iteration, ~34 hours for 40 epochs
- **With cache**: ~0.05-0.2s/iteration, ~1-3 hours for 40 epochs

### Cache Details

The cached dataset (`cached_dataset/`):
- **Format**: PyTorch `.pt` files (mel-spectrograms + piano rolls)
- **Size**: ~27GB for full MAESTRO v3.0.0 with 30s chunks
- **Splits**: Separate caches for train/validation/test
- **Total chunks**: 23,902 (19,154 train + 2,344 val + 2,404 test)

### Re-preprocessing

If you change chunk settings, re-run preprocessing:
```bash
# Different chunk length
python preprocess_dataset.py --chunk_length 45.0 --overlap 0.0

# With overlap for augmentation
python preprocess_dataset.py --chunk_length 30.0 --overlap 0.25
```

---

## What's implemented so far

- **Data loading & formatting** (`data/dataset.py`, `data/cached_dataset.py`)
  - Loads MAESTRO CSV; pairs audio with ground-truth MIDI
  - Produces aligned **mel spectrogram** (input) and **binary piano-roll** (target)
  - **Chunked loading** for memory-efficient training with configurable overlap
  - **Dataset caching** for 10-50x faster training
  - Full train/validation/test split support

- **Model baseline** (`models/cnn_rnn_model.py`, `transcription_model.py`)
  - CNN with **freq-only pooling** to preserve time; BiLSTM; 88-pitch framewise logits
  - Wrapper with `BCEWithLogitsLoss`, `predict(threshold)`, device handling, and safe time alignment
  - Mixed precision training (FP16) for faster computation

- **Training utilities & CLI** (`train/train_transcriber.py`, `main.py`)
  - Collate-and-pad for variable-length sequences; train/val loops
  - Multi-worker data loading with prefetching
  - Resume training from checkpoints
  - Timestamped outputs, **periodic checkpoints**, and **live-updating loss curves**
  - Progress tracking with tqdm

- **Preprocessing & optimization** (`preprocess_dataset.py`)
  - One-time dataset preprocessing for massive speedup
  - Automatic cache detection and usage

- **Scripts & utilities**
  - `example.sh` - Optimized training script
  - `resume_training.sh` - Resume from checkpoint
  - `profile_training.py` - Performance profiling

- **(Optional)** `music_transcription.ipynb` for exploratory EDA/visualization

---

## Performance Optimizations

This codebase includes several optimizations for efficient training:

1. **Chunked Loading**: Reduces GPU memory usage by 10x
2. **Dataset Caching**: Speeds up training by 10-50x after one-time preprocessing
3. **Mixed Precision (FP16)**: Faster GPU computation
4. **Multi-worker DataLoading**: Parallel data loading (8 workers)
5. **Pin Memory**: Faster CPU-to-GPU transfer
6. **Persistent Workers**: No restart overhead between epochs

See `IO_BOTTLENECK_SOLUTIONS.md` and `PERFORMANCE_OPTIMIZATION.md` for details.

---

## Roadmap

- Add separate **onset** and **activation** heads + post-processing to clean MIDI (note on/off grouping, sustain)
- Evaluate with **frame-wise F1**, **note onset F1**, and **note with offset** metrics
- Implement **Transformer model** (self-attention over frames)
- Export MIDI via `pretty_midi` and (optionally) MusicXML for notation rendering
- Multi-GPU training support (when needed)

---

## Acknowledgments

- **MAESTRO Dataset (v2.0.0)** — MIDI and Audio Edited for Synchronous TRacks and Organization.
