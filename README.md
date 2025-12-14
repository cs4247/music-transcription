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
├── scripts/
│   ├── train_cnn.py               # Command-line training entry point
│   ├── preprocess_dataset.py      # Dataset preprocessing and caching
│   └── evaluate.py                # Model evaluation and threshold tuning
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

## Model Architectures

We provide two model variants in **`models/cnn_rnn_model.py`**:

### 1. CNNRNNModel (Base Model)

**Lightweight baseline** for quick experimentation and resource-constrained environments.

**Architecture:**
- **CNN frontend** with **frequency-only pooling** (`MaxPool2d(2,1)`) to preserve time resolution
- **BiLSTM** (2-3 layers) models temporal dependencies across frames
- **Linear head** outputs **88 pitch logits** per frame
- Uses **`BCEWithLogitsLoss`** for training

**Model size:** ~36M parameters
**Memory:** ~2-3 GB VRAM (batch_size=4)

**Shapes (example with n_mels=320):**
```
Input:   (B, 1, 320, T)
↓ CNN
(B, 64, 80, T)
↓ Reshape
(B, T, 5120)
↓ BiLSTM
(B, T, 2*hidden_size)
↓ Linear
Output:  (B, 88, T)
```

**Usage:**
```python
model = TranscriptionModel(
    model_type="cnn_rnn",
    n_mels=320,
    hidden_size=256,
    num_layers=2,
    dropout=0.3
)
```

**Performance:**
- **Training speed:** Fast (~0.2s/iteration with cache)
- **F1 Score:** 0.70-0.78 (typical)
- **Best for:** Quick prototyping, limited compute

---

### 2. CNNRNNModelLarge (Enhanced Model)

**Advanced architecture** with 7 major improvements for state-of-the-art performance.

**Key Improvements:**

1. **Deeper CNN** (4 blocks: 1→32→64→128→256 channels)
2. **Residual Connections** - Skip connections for better gradient flow
3. **Multi-Head Attention** (8 heads) - Temporal context modeling
4. **Frequency-Aware Convolutions** - Asymmetric (7×3) kernels for piano harmonics
5. **Advanced Dropout** - Spatial dropout in CNN + variational dropout in LSTM
6. **Multi-Scale Temporal Modeling** - Dual LSTM branches (main + local)
7. **Onset/Offset Detection Heads** - Separate predictions for note boundaries

**Model size:** ~89M parameters (2.5× larger than base)
**Memory:** ~5-7 GB VRAM (batch_size=4)

**Architecture Pipeline:**
```
Input:   (B, 1, 320, T)
↓ Conv Block 1 (1→32)
(B, 32, 160, T)
↓ ResBlock 1 (32→64) + Pool
(B, 64, 80, T) + Dropout2d(0.1)
↓ ResBlock 2 (64→128)
(B, 128, 80, T) + Dropout2d(0.1)
↓ FreqAwareConv (128→256, kernel=7×3) + Pool
(B, 256, 40, T) + Dropout2d(0.15)
↓ Reshape
(B, T, 10240)
↓ Multi-Scale RNN
├─ Main LSTM (512 hidden, 3 layers, bidir)  → (B, T, 1024)
└─ Local LSTM (256 hidden, 1 layer, bidir)  → (B, T, 512)
↓ Concatenate
(B, T, 1536)
↓ Multi-Head Attention (8 heads) + LayerNorm
(B, T, 1536)
↓ Shared FC + Dropout
(B, T, 512)
↓ Three Output Heads
├─ Frame Head  → (B, 88, T)
├─ Onset Head  → (B, 88, T)
└─ Offset Head → (B, 88, T)
```

**Usage:**
```python
model = TranscriptionModel(
    model_type="cnn_rnn_large",
    n_mels=320,
    hidden_size=512,
    num_layers=3,
    dropout=0.2,
    use_attention=True,              # Enable multi-head attention
    use_onset_offset_heads=True      # Enable onset/offset detection
)
```

**Performance:**
- **Training speed:** Moderate (~0.5s/iteration with cache)
- **F1 Score:** 0.85-0.90 (typical)
- **Best for:** Production deployments, maximizing accuracy

**Loss Computation:**
When using onset/offset heads, the total loss is:
```
total_loss = 0.5 × frame_loss + 0.25 × onset_loss + 0.25 × offset_loss
```

**Stability Features:**
- Gradient clipping (max_norm=1.0) to prevent explosion
- Attention logit clipping (±10.0) for numerical stability
- LayerNorm with eps=1e-6
- NaN detection and recovery

---

### Model Comparison

| Feature | CNNRNNModel | CNNRNNModelLarge |
|---------|-------------|------------------|
| Parameters | 36M | 89M |
| CNN Depth | 2 blocks | 4 blocks |
| Residual Connections | ✗ | ✓ |
| Multi-Head Attention | ✗ | ✓ (8 heads) |
| Multi-Scale RNN | ✗ | ✓ (dual LSTM) |
| Onset/Offset Heads | ✗ | ✓ |
| Training Speed | Fast | 2.5× slower |
| F1 Score | 0.70-0.78 | 0.85-0.90 |
| VRAM (batch=4) | 2-3 GB | 5-7 GB |

**Choosing a Model:**
- **Use CNNRNNModel** if you have limited compute or need fast iteration
- **Use CNNRNNModelLarge** for best accuracy and production deployments

See [LARGE_MODEL_USAGE.md](LARGE_MODEL_USAGE.md) for detailed usage guide and [compare_models.py](compare_models.py) for architecture analysis.

---

### TranscriptionModel Wrapper

**`models/transcription_model.py`** provides a unified interface for both models:

**Features:**
- Model selection via `model_type` parameter
- Unified `forward()`, `compute_loss()`, and `predict(threshold=0.5)` interface
- Device handling (CPU/CUDA)
- Automatic multi-head loss computation for large model
- Defensive time-axis alignment

**Example:**
```python
# Load base model
base_model = TranscriptionModel(model_type="cnn_rnn", device="cuda")

# Load large model
large_model = TranscriptionModel(model_type="cnn_rnn_large", device="cuda")

# Both models share the same interface
logits = model(mel_spectrogram)
loss = model.compute_loss(logits, targets, lengths)
predictions = model.predict(mel_spectrogram, threshold=0.5)
```

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
python scripts/train_cnn.py --year 2017 --epochs 25 --batch_size 8 --lr 1e-4 --save_every 5 --subset_size 100
```
This launches training with periodic checkpoints and live-updating loss plots saved under a timestamped run directory in `outputs/` (see next section).

---

## Training with `scripts/train_cnn.py`

The primary training entry point is **`scripts/train_cnn.py`**. It supports various options for dataset loading, chunking, caching, and resuming training.

### Command-Line Arguments

```bash
python scripts/train_cnn.py [OPTIONS]
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
python scripts/train_cnn.py \
  --epochs 25 \
  --batch_size 8 \
  --lr 1e-4 \
  --save_every 5
```

### Example: Memory-Efficient Training with Chunking
```bash
python scripts/train_cnn.py \
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
python scripts/train_cnn.py \
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
python scripts/train_cnn.py \
  --subset_size 50 \
  --epochs 3 \
  --batch_size 4 \
  --chunk_length 30.0
```

### Example: Complete Workflow with example.sh

The `example.sh` script demonstrates the complete pipeline with configurable parameters:

```bash
# Show usage
./example.sh

# Run individual steps
./example.sh preprocess   # Preprocess dataset (~1-2 hours)
./example.sh train        # Train model (~10-20 hours)
./example.sh eval         # Evaluate best model (~30 minutes)

# Or run all steps sequentially
./example.sh all          # Complete pipeline (~12-24 hours)
```

**Configure the workflow** by editing variables at the top of `example.sh`:
- Model architecture: `MODEL_TYPE`, `N_MELS`, `HIDDEN_SIZE`, `NUM_LAYERS`, `DROPOUT`
- Training: `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`
- Data: `CHUNK_LENGTH`, `DATASET_DIR`

### Example: Manual Background Training
```bash
# Run training in background with custom log location
nohup python3 scripts/train_cnn.py \
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

For **10-50x speedup** in training, preprocess the dataset once and cache it to disk. This eliminates the I/O bottleneck by converting raw audio/MIDI files into pre-computed PyTorch tensors.

### Quick Start

```bash
# Preview what will be created (recommended first step)
python scripts/preprocess_dataset.py --n_mels 320 --dry_run

# Preprocess with default settings
python scripts/preprocess_dataset.py --n_mels 320

# Run in background with automatic logging
python scripts/preprocess_dataset.py --n_mels 320 --background
```

### Complete Usage Guide

**View all options:**
```bash
python scripts/preprocess_dataset.py --help
```

**Common Arguments:**
- `--n_mels` - **CRITICAL**: Number of mel bins, MUST match your model (default: 229)
- `--root_dir` - Path to MAESTRO dataset (default: `maestro-v3.0.0`)
- `--cache_dir` - Output cache directory (auto-generated based on n_mels if not specified)
- `--sr` - Sample rate in Hz (default: 16000)
- `--hop_length` - STFT hop length (default: 512)
- `--chunk_length` - Chunk duration in seconds (default: 30.0)
- `--overlap` - Overlap ratio 0.0-1.0 (default: 0.0)

**Utility Arguments:**
- `--dry_run` - Preview what will be created without processing
- `--background` - Run in background with timestamped log file
- `--force` - Overwrite existing cache
- `--verify` - Verify cache integrity after creation
- `--splits` - Process specific splits (e.g., `train,validation`)
- `--show_cache_info <cache_dir>` - Display info about existing cache

### Examples

**1. Preview before preprocessing (recommended):**
```bash
python scripts/preprocess_dataset.py --n_mels 320 --dry_run
```
Output shows:
- Estimated disk space needed
- Number of chunks per split
- Audio parameters
- Compatibility requirements

**2. Basic preprocessing:**
```bash
python scripts/preprocess_dataset.py --n_mels 320
```
Creates: `cached_dataset_mels320/` (~34GB)

**3. Run in background (for long preprocessing):**
```bash
python scripts/preprocess_dataset.py --n_mels 320 --background
# Monitor with: tail -f preprocess_*.log
```

**4. Custom parameters:**
```bash
python scripts/preprocess_dataset.py \
  --n_mels 320 \
  --chunk_length 45.0 \
  --overlap 0.25 \
  --sr 16000 \
  --hop_length 512
```

**5. Process only specific splits:**
```bash
# Only preprocess training data
python scripts/preprocess_dataset.py --n_mels 320 --splits train
```

**6. Inspect existing cache:**
```bash
python scripts/preprocess_dataset.py --show_cache_info cached_dataset_mels320
```
Shows:
- Chunks and size per split
- Audio parameters used
- Total cache size

**7. Force overwrite existing cache:**
```bash
python scripts/preprocess_dataset.py --n_mels 320 --force
```

### Training with Cache

The training script **automatically detects** and uses the cache:

```bash
python scripts/train_cnn.py --cached_dir cached_dataset_mels320 --chunk_length 30.0
```

Output:
```
✓ Using cached dataset (fast mode!)
```

**Performance improvement:**
- **Without cache**: ~2.6s/iteration, ~34 hours for 40 epochs
- **With cache**: ~0.05-0.2s/iteration, ~1-3 hours for 40 epochs

### Cache Details

**Format**: PyTorch `.pt` files containing:
- Pre-computed mel-spectrograms
- Pre-computed piano rolls (labels)

**Typical sizes** (MAESTRO v3.0.0, 30s chunks):
- `n_mels=229`: ~27GB (23,902 chunks)
- `n_mels=320`: ~34GB (23,902 chunks)

**Directory structure:**
```
cached_dataset_mels320/
├── train/
│   ├── chunk_000000.pt
│   ├── chunk_000001.pt
│   └── ...
├── validation/
│   └── ...
├── test/
│   └── ...
├── train_metadata.pkl
├── validation_metadata.pkl
└── test_metadata.pkl
```

### Important: Parameter Compatibility

**The preprocessing parameters MUST match your model configuration!**

If your model uses:
```python
TranscriptionModel(n_mels=320)
```

Then preprocess with:
```bash
python scripts/preprocess_dataset.py --n_mels 320
```

**The script validates this automatically:**
- Cache directories are auto-named based on n_mels to prevent conflicts
- Attempting to use incompatible parameters will trigger a validation error
- Use `--force` only if you intentionally want to overwrite

### Validation & Safety Features

The enhanced preprocessing script includes:

1. **Pre-execution validation**: Checks dataset exists, parameters are valid, disk space is sufficient
2. **Cache conflict detection**: Prevents overwriting caches with incompatible parameters
3. **Dry-run mode**: Preview before committing to hours of processing
4. **Verification**: Optional `--verify` flag to check cache integrity
5. **Resume support**: Automatically skips already-cached chunks

### Migration from Old Script

If you previously used `preprocess_mels320.sh`:

**Old way:**
```bash
./preprocess_mels320.sh
```

**New way (equivalent):**
```bash
python scripts/preprocess_dataset.py --n_mels 320 --background
```

The shell script still works but shows a deprecation notice. The Python version provides better validation, progress tracking, and more features.

---

## Model Evaluation

Evaluate trained models on test/validation data with a **unified Python CLI** that replaces 5 separate scripts. The new interface provides comprehensive evaluation features including threshold tuning, auto-detection, and dry-run previews.

### Quick Start

```bash
# Basic evaluation on test set
python scripts/evaluate.py --model outputs/2025-12-13_20-11-33/checkpoints/model_epoch_20.pth

# Quick validation check (100 samples, headless mode)
python scripts/evaluate.py --model outputs/model.pth \
  --split validation --subset 100 --headless

# Threshold tuning on validation subset
python scripts/evaluate.py --model outputs/model.pth \
  --tune_threshold --split validation --subset 50
```

### Complete Usage Guide

```bash
python scripts/evaluate.py [OPTIONS]
```

**Required Arguments:**
- `--model` - Path to model checkpoint (.pth file)

**Evaluation Options:**
- `--split` - Dataset split: `train`, `validation`, or `test` (default: `test`)
- `--threshold` - Sigmoid threshold for binary piano-roll (default: `0.5`)
- `--subset` - Limit to N samples for quick evaluation
- `--batch_size` - Batch size (default: `1`)

**Data Source:**
- `--data_source` - Data source: `auto`, `cache`, or `full` (default: `auto`)
  - `auto`: Auto-detect (prefers cache if available)
  - `cache`: Use cached dataset chunks
  - `full`: Use full MAESTRO files
- `--root_dir` - Path to MAESTRO dataset (default: `maestro-v3.0.0`)
- `--cache_dir` - Path to cached dataset (auto-detected if not specified)
- `--year` - Filter by year (e.g., `2017`)

**Model Configuration:**
- `--n_mels` - Number of mel bins (auto-detected from cache or default: `320`)
- `--model_type` - Model architecture (default: `cnn_rnn_large`)
- `--hidden_size` - RNN hidden size (default: `512`)
- `--num_layers` - Number of RNN layers (default: `3`)
- `--dropout` - Dropout rate (default: `0.2`)

**Output Options:**
- `--out_dir` - Output directory (default: `eval_outputs`)
- `--headless` - Headless mode: only print `EVAL_MEAN_F1=<value>`
- `--no_midi` - Skip MIDI generation (faster evaluation)

**Threshold Tuning:**
- `--tune_threshold` - Enable threshold tuning mode
- `--tune_rounds` - Number of tuning rounds (default: `6`)
- `--tune_range` - Initial search range as `min,max` (default: `0.05,0.95`)
- `--tune_step` - Initial step size (default: `0.1`)

**Utility Options:**
- `--dry_run` - Preview configuration without running evaluation
- `--show_results` - Display summary of existing evaluation results
- `--verify_compatibility` - Check model/cache compatibility
- `--background` - Run in background with log file
- `--log_file` - Log file path for background mode

### Examples

**1. Basic evaluation:**
```bash
python scripts/evaluate.py --model outputs/2025-12-13_20-11-33/checkpoints/model_epoch_20.pth
```
Evaluates on test set, saves results to `eval_outputs/<timestamp>/`

**2. Quick validation check:**
```bash
python scripts/evaluate.py \
  --model outputs/model.pth \
  --split validation \
  --subset 100 \
  --headless
```
Output: `EVAL_MEAN_F1=0.7891`

**3. Threshold tuning:**
```bash
python scripts/evaluate.py \
  --model outputs/model.pth \
  --tune_threshold \
  --split validation \
  --subset 50
```
Finds optimal threshold using binary search:
```
=== Round 1/6 | range=[0.05, 0.95] step=0.1 ===
  t=0.05  f1=0.6234
  t=0.15  f1=0.6891
  ...
Round best: t=0.45 f1=0.7823

=== FINAL RESULTS ===
Best threshold: 0.4375
Best mean F1:   0.7891
```

**4. Preview before evaluation:**
```bash
python scripts/evaluate.py --model outputs/model.pth --dry_run
```
Shows configuration without running:
```
======================================================================
MODEL EVALUATION - DRY RUN
======================================================================

MODEL CONFIGURATION:
  Checkpoint:    outputs/model.pth
  Model type:    cnn_rnn_large
  n_mels:        320

EVALUATION CONFIGURATION:
  Split:         test
  Threshold:     0.5

DATA SOURCE:
  Mode:          cache
  Path:          cached_dataset_mels320
  Samples:       2404 chunks
======================================================================
```

**5. Show existing results:**
```bash
python scripts/evaluate.py --show_results eval_outputs/2025-12-13_20-11-33
```

**6. Run in background:**
```bash
python scripts/evaluate.py \
  --model outputs/model.pth \
  --background

# Monitor with:
tail -f evaluate_*.log
```

**7. Explicit data source selection:**
```bash
# Use cached chunks (fast)
python scripts/evaluate.py --model outputs/model.pth --data_source cache

# Use full files (slower, more accurate)
python scripts/evaluate.py --model outputs/model.pth --data_source full
```

**8. Custom model configuration:**
```bash
python scripts/evaluate.py \
  --model outputs/model.pth \
  --n_mels 229 \
  --hidden_size 256 \
  --num_layers 2
```

### Auto-Detection Features

The evaluation script automatically detects settings when possible:

1. **Model Configuration**: When using cached data, `n_mels`, `sr`, and `hop_length` are read from cache metadata
2. **Data Source**: With `--data_source auto` (default), prefers cache if available, falls back to full files
3. **Cache Directory**: Auto-detects `cached_dataset_mels<N>` based on model's `n_mels`

**Override auto-detection:**
```bash
# Auto-detected n_mels from cache, but override to 229
python scripts/evaluate.py --model outputs/model.pth --n_mels 229
```

### Outputs

Each evaluation creates a timestamped directory:
```
eval_outputs/YYYY-MM-DD_HH-MM-SS/
├── eval_summary.txt       # Metrics summary
└── midis/                 # Generated MIDI files
    ├── 0000_Composer_Title.mid
    ├── 0001_Composer_Title.mid
    └── ...
```

**eval_summary.txt** contains:
- Configuration details (split, threshold, model path)
- Per-sample F1 scores
- Mean framewise F1
- Best/worst samples

### Validation & Safety Features

The script includes comprehensive validation:

1. **Pre-execution checks**: Model exists, data source valid, parameters compatible
2. **Cache compatibility**: Verifies `n_mels` matches between model and cache
3. **Dry-run mode**: Preview before committing to long evaluations
4. **Verification mode**: Check compatibility without running evaluation

**Example validation error:**
```bash
python scripts/evaluate.py --model nonexistent.pth

ERROR: Model checkpoint not found: nonexistent.pth
Exiting due to validation errors.
```

**Example compatibility check:**
```bash
python scripts/evaluate.py --model outputs/model.pth --verify_compatibility

======================================================================
COMPATIBILITY CHECK
======================================================================

Model Config:
  n_mels:      320

Cache Config (cached_dataset_mels320):
  n_mels:      320  ✓

Status: COMPATIBLE
======================================================================
```

### Migration from Old Scripts

If you previously used the separate evaluation scripts:

**Old:** `evaluate_model.py`
```bash
python evaluate_model.py --model_path outputs/model.pth --split test
```

**New:**
```bash
python scripts/evaluate.py --model outputs/model.pth --split test
```

---

**Old:** `quick_eval.sh`
```bash
./quick_eval.sh
```

**New:**
```bash
python scripts/evaluate.py --model outputs/model.pth \
  --split validation --subset 100
```

---

**Old:** `tune_threshold_cached.sh`
```bash
MODEL=outputs/model.pth ./tune_threshold_cached.sh
```

**New:**
```bash
python scripts/evaluate.py --model outputs/model.pth \
  --tune_threshold --split validation --subset 50
```

---

The old scripts still work but display deprecation warnings. The unified CLI provides better validation, progress tracking, and more features.

---

## What's implemented so far

- **Data loading & formatting** (`data/dataset.py`, `data/cached_dataset.py`)
  - Loads MAESTRO CSV; pairs audio with ground-truth MIDI
  - Produces aligned **mel spectrogram** (input) and **binary piano-roll** (target)
  - **Chunked loading** for memory-efficient training with configurable overlap
  - **Dataset caching** for 10-50x faster training
  - Full train/validation/test split support

- **Model architectures** (`models/cnn_rnn_model.py`, `transcription_model.py`)
  - **CNNRNNModel (base)**: Lightweight CNN+BiLSTM baseline (36M params, F1 ~0.70-0.78)
  - **CNNRNNModelLarge (enhanced)**: Advanced architecture with 7 major improvements (89M params, F1 ~0.85-0.90)
    - Deeper CNN (4 blocks) with residual connections
    - Multi-head attention (8 heads) for temporal modeling
    - Multi-scale dual LSTM (main + local)
    - Onset/offset detection heads for better note boundaries
    - Gradient clipping and numerical stability features
  - Unified wrapper with `BCEWithLogitsLoss`, `predict(threshold)`, device handling
  - Mixed precision training (FP16) for faster computation

- **Training utilities & CLI** (`train/train_transcriber.py`, `scripts/train_cnn.py`)
  - Collate-and-pad for variable-length sequences; train/val loops
  - Multi-worker data loading with prefetching
  - Resume training from checkpoints
  - Timestamped outputs, **periodic checkpoints**, and **live-updating loss curves**
  - Progress tracking with tqdm

- **Preprocessing & optimization** (`scripts/preprocess_dataset.py`)
  - Enhanced CLI with comprehensive validation and safety features
  - Background mode for long-running preprocessing tasks
  - Dry-run mode to preview before processing
  - Cache inspection and verification tools
  - Automatic parameter compatibility checking
  - Configurable audio parameters (n_mels, sr, hop_length)
  - One-time dataset preprocessing for 10-50x training speedup

- **Evaluation** (`scripts/evaluate.py`)
  - Unified CLI for model evaluation and threshold tuning
  - Auto-detection of model/cache parameters
  - MIDI generation from predictions
  - Background mode and dry-run support
  - Comprehensive validation and compatibility checks

- **Scripts & utilities**
  - `example.sh` - Complete workflow demonstration (preprocess → train → eval)
  - Legacy shell scripts with deprecation notices

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
