# Sheet Music Transcription — Audio ➜ MIDI ➜ Sheet

**Team:** *Fine‑Tuned*  
**Members:** Isaac Bilsel, Christian Schulte

This repository implements the core pipeline for **automatic musical transcription**: converting **audio** into **MIDI**, and then (as a later post‑process) into **sheet music**. The current codebase focuses on a **CNN + BiLSTM (RNN)** baseline trained on the **MAESTRO v2.0.0** dataset (polyphonic solo piano). A Transformer variant is planned next.

> High‑level plan (from the proposal): use MIDI as the bridge format (Audio → **MIDI** → Sheet). The hard part we model is Audio → MIDI; MIDI → Sheet is handled by existing notation tools.


---

## Repository Structure

```
.
├── data/
│   └── dataset.py                 # MaestroDataset: audio→mel + MIDI→piano‑roll labels (aligned)
├── models/
│   ├── cnn_rnn_model.py           # CNN (freq-only pooling) + BiLSTM ➜ 88‑pitch framewise logits
│   ├── transcription_model.py     # Wrapper: loss, predict(), device mgmt; model factory entry point
│   └── transformer_model.py       # Placeholder for future Transformer-based transcriber
├── train/
│   └── train_transcriber.py       # Training utilities: loaders, collate/padding, loops, checkpoints
├── outputs/
│   └── checkpoints/               # Saved weights (*.pth)
└── music_transcription.ipynb      # Main notebook: EDA, visualization, tiny training run
```

---

## Dataset

- **MAESTRO v2.0.0** expected at the repo root in: `maestro-v2.0.0/`
- We currently subset to **year = 2017** for quick iteration.
- The dataset comes with `maestro-v2.0.0.csv` (and `.json`) containing columns:  
  `canonical_composer, canonical_title, split, year, midi_filename, audio_filename, duration`.
- In this project, audio has been converted to **.mp3**; the loader transparently uses the `.mp3` if the CSV points to `.wav`.

### What the `MaestroDataset` returns
- **Input** `mel`: shape `(1, n_mels, T)` — mel‑spectrogram computed from audio (`librosa 0.11`, keyword args).  
- **Target** `roll`: shape `(88, T)` — binary piano‑roll (A0..C8), derived from MIDI for **supervised labels**.

Time alignment is enforced by trimming both to the same **T** (and by batch‑time padding in the collate function).

---

## Model (current baseline)

**`models/cnn_rnn_model.py`**

- **CNN frontend** with **frequency‑only pooling** (`MaxPool2d(2,1)`) so time resolution is preserved for framewise labeling.
- **BiLSTM** models temporal dependencies across frames.
- **Linear head** outputs **88 pitch logits** per frame.  
- Use **`BCEWithLogitsLoss`** (handled in the wrapper class).

**Shapes** (example):  
`(B, 1, 229, T)` → CNN `(B, 64, ~229/4, T)` → reshape to `(B, T, 64*~57)` → BiLSTM `(B, T, 2H)` → Linear → logits `(B, 88, T)`.

**`models/transcription_model.py`** adds:
- Unified `forward`, `compute_loss()`, and `predict(threshold=0.5)`
- Device handling and a future‑proof factory (`model_type="cnn_rnn"` for now).  
- Defensive time‑axis upsampling if any mismatch sneaks in during experimentation.

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
pip install torch librosa==0.11.0 pretty_midi numpy pandas matplotlib jupyter tqdm
pip install ipykernel
python -m ipykernel install --user --name=music-transcription
```

> **Note:** With `librosa==0.11.0`, call `librosa.feature.melspectrogram` with **keyword arguments** (e.g., `y=...`, `sr=...`).

### 3) Open the notebook
- Launch `music_transcription.ipynb` and select the `music-transcription` kernel.

---

## Inspect one sample (Notebook)

```python
from data.dataset import MaestroDataset

ds = MaestroDataset(root_dir="maestro-v2.0.0", year="2017", subset_size=5)
mel, roll = ds[0]
mel.shape, roll.shape  # -> (1, 229, T), (88, T)
```

### Visualize: spectrogram + MIDI overlay (log‑frequency)

```python
import numpy as np, matplotlib.pyplot as plt, librosa.display

def plot_transcription_overlay(mel, roll, sr, hop_length, fmax=8000, title="Overlay"):
    mel_db = mel.squeeze(0).numpy()
    roll_np = roll.numpy()

    plt.figure(figsize=(14, 7))
    img = librosa.display.specshow(mel_db, sr=sr, hop_length=hop_length,
                                   x_axis="time", y_axis="mel", fmax=fmax, cmap="magma")
    plt.colorbar(img, format="%+2.0f dB", label="Power (dB)")

    # overlay ground-truth notes in green (convert MIDI pitch -> Hz)
    for p in range(roll_np.shape[0]):
        active = roll_np[p] > 0
        if active.any():
            on = np.where(np.diff(np.r_[0, active, 0]) == 1)[0]
            off = np.where(np.diff(np.r_[0, active, 0]) == -1)[0]
            freq = 440.0 * (2.0 ** ((21 + p - 69) / 12.0))  # A4=440
            for a, b in zip(on, off):
                t0 = a * ds.hop_length / ds.sr
                t1 = b * ds.hop_length / ds.sr
                plt.hlines(freq, t0, t1, color="lime", lw=2, alpha=0.8)

    plt.yscale("log"); plt.ylim(30, fmax)
    plt.title(title); plt.xlabel("Time (s)"); plt.ylabel("Frequency (Hz, log)")
    plt.tight_layout(); plt.show()

plot_transcription_overlay(mel, roll, sr=ds.sr, hop_length=ds.hop_length)
```

---

## Tiny training run (Notebook)

Batch samples in a `DataLoader` with a **collate function** that pads/clips time to a shared length:

```python
import torch, torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models.transcription_model import TranscriptionModel

def collate_fn(batch, max_len=12000):
    mels, rolls = zip(*batch)
    T = min(max(m.shape[-1] for m in mels), max_len)
    pm, pr = [], []
    for m, r in zip(mels, rolls):
        if m.shape[-1] < T:
            pad = T - m.shape[-1]
            m = F.pad(m, (0, pad)); r = F.pad(r, (0, pad))
        elif m.shape[-1] > T:
            m = m[:, :, :T]; r = r[:, :T]
        pm.append(m); pr.append(r)
    return torch.stack(pm), torch.stack(pr)

dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_fn)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TranscriptionModel(model_type="cnn_rnn", device=device)
opt = optim.Adam(model.parameters(), lr=1e-4)

mel_b, roll_b = next(iter(dl))
mel_b, roll_b = mel_b.to(device), roll_b.to(device)
logits = model(mel_b)
loss = model.compute_loss(logits, roll_b)
loss.backward(); opt.step()
print("One step OK — loss:", float(loss))
```

> For a short multi‑step run, loop over `dl` for a couple of epochs and print average train/val loss (see `train/train_transcriber.py`).

---

## What’s implemented so far

- **Data loading & formatting** (`data/dataset.py`)
  - Loads MAESTRO CSV; pairs audio (`.mp3`) with ground‑truth MIDI.
  - Produces aligned **mel spectrogram** (input) and **binary piano‑roll** (target).
- **Visualization utilities** (notebook snippets)
  - Stacked and overlay views with log‑frequency alignment (mel + MIDI lines).
- **Model baseline** (`models/cnn_rnn_model.py`, `transcription_model.py`)
  - CNN with **freq‑only pooling** to preserve time; BiLSTM; 88‑pitch framewise logits.
  - Wrapper with `BCEWithLogitsLoss`, `predict(threshold)`, device handling, and safe time alignment.
- **Training utilities** (`train/train_transcriber.py`)
  - Collate‑and‑pad for variable‑length sequences; train/val loops; checkpoint saving.
- **Notebook** (`music_transcription.ipynb`)
  - Data EDA, visualization, and a tiny test training step.

---

## Roadmap

- Add separate **onset** and **activation** heads + post‑processing to clean MIDI (note on/off grouping, sustain).  
- Evaluate with **frame‑wise F1**, **note onset F1**, and **note with offset** metrics.  
- Implement **Transformer model** (self‑attention over frames).  
- Caching & faster I/O (precomputed mel / labels to `.npz`).  
- Full train/val/test split support from the CSV.  
- Export MIDI via `pretty_midi` and (optionally) MusicXML for notation rendering.

---

## Acknowledgments

- **MAESTRO Dataset (v2.0.0)** — MIDI and Audio Edited for Synchronous TRacks and Organization.  