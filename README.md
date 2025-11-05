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

- **MAESTRO v2.0.0** expected at the repo root in: `maestro-v2.0.0/`
- We currently subset to **year = 2017** for quick iteration.
- The dataset comes with `maestro-v2.0.0.csv` (and `.json`) containing columns:  
  `canonical_composer, canonical_title, split, year, midi_filename, audio_filename, duration`.
- In this project, audio has been converted to **.mp3**; the loader transparently uses the `.mp3` if the CSV points to `.wav`.

### What the `MaestroDataset` returns
- **Input** `mel`: shape `(1, n_mels, T)` — mel-spectrogram computed from audio (`librosa 0.11`, keyword args).  
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
pip install torch librosa==0.11.0 pretty_midi numpy pandas matplotlib tqdm
```

### 3) Train (command-line)
```bash
python main.py --year 2017 --epochs 25 --batch_size 8 --lr 1e-4 --save_every 5 --subset_size 100
```
This launches training with periodic checkpoints and live-updating loss plots saved under a timestamped run directory in `outputs/` (see next section).

---

## Training with `main.py`

The primary training entry point is **`main.py`**. It wires up the dataset, dataloaders with padding, the CNN+RNN model, optimizer, checkpointing, and logging.

### Example (foreground)
```bash
python main.py \
  --year 2017 \
  --epochs 25 \
  --batch_size 8 \
  --lr 1e-4 \
  --save_every 5 \
  --subset_size 100
```

### Example (run in background with `nohup`)
```bash
nohup python3 main.py \
  --year 2017 \
  --epochs 25 \
  --batch_size 8 \
  --lr 1e-4 \
  --save_every 5 \
  --subset_size 100 \
  > outputs/train_run_$(date +"%Y-%m-%d_%H-%M-%S").log 2>&1 &
```

### Outputs
Each run creates a unique timestamped directory:
```
outputs/YYYY-MM-DD_HH-MM-SS/
├── checkpoints/
│   ├── model_epoch_10.pth
│   ├── model_epoch_20.pth
│   └── model_final.pth
└── logs/
    ├── loss_curve.png         # updated after every epoch
    └── training_log.txt       # appended each epoch with train/val loss
```
- `--save_every N` controls periodic checkpointing (default **10**).  
- The final model is always saved as `model_final.pth` at the end of training.

---

## What’s implemented so far

- **Data loading & formatting** (`data/dataset.py`)
  - Loads MAESTRO CSV; pairs audio (`.mp3`) with ground-truth MIDI.
  - Produces aligned **mel spectrogram** (input) and **binary piano-roll** (target).
- **Model baseline** (`models/cnn_rnn_model.py`, `transcription_model.py`)
  - CNN with **freq-only pooling** to preserve time; BiLSTM; 88-pitch framewise logits.
  - Wrapper with `BCEWithLogitsLoss`, `predict(threshold)`, device handling, and safe time alignment.
- **Training utilities & CLI training** (`train/train_transcriber.py`, `main.py`)
  - Collate-and-pad for variable-length sequences; train/val loops; **command-line training method** (`main.py`).
  - Timestamped outputs, **periodic checkpoints**, and **live-updating loss curves** saved to `outputs/.../logs/`.
- **(Optional)** `music_transcription.ipynb` for exploratory EDA/visualization. Training does **not** require the notebook.

---

## Roadmap

- Add separate **onset** and **activation** heads + post-processing to clean MIDI (note on/off grouping, sustain).  
- Evaluate with **frame-wise F1**, **note onset F1**, and **note with offset** metrics.  
- Implement **Transformer model** (self-attention over frames).  
- Caching & faster I/O (precomputed mel / labels to `.npz`).  
- Full train/val/test split support from the CSV.  
- Export MIDI via `pretty_midi` and (optionally) MusicXML for notation rendering.

---

## Acknowledgments

- **MAESTRO Dataset (v2.0.0)** — MIDI and Audio Edited for Synchronous TRacks and Organization.  
- Project motivation and plan summarized from the course proposal (Audio → MIDI → Sheet pipeline; CNN+RNN baseline; Transformer extension).