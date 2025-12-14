#!/usr/bin/env python3
"""
Train a music transcription model from the command line.

Keeps defaults aligned with the original notebook:
- root_dir="maestro-v3.0.0"
- split="train"
- year="2017"
- subset_size=100
- batch_size=2
- num_epochs=5
- lr=1e-4
- model_type="ast"
- freeze_encoder=True
- use_mock_encoder=False
- remi_vocab_size=512
- decoder_layers=4
- decoder_dim=384
- decoder_heads=6
- dropout=0.2
- device="cuda if available else cpu"

Usage examples:
    # Quick test with mock encoder (no download)
    python scripts/train_ast.py --subset-size 10 --num-epochs 2 --use-mock-encoder

    # Train on single year (2017)
    python scripts/train_ast.py --year 2017 --subset-size 100 --num-epochs 5

    # Train on ALL years combined
    python scripts/train_ast.py --year all --subset-size 200 --num-epochs 10

    # Full training on complete dataset (all years, all files)
    python scripts/train_ast.py --year all --subset-size 0 --num-epochs 30 --batch-size 4

    # With custom architecture
    python scripts/train_ast.py --decoder-layers 6 --decoder-dim 512 --decoder-heads 8
"""

import os
import sys
import argparse
from dataclasses import dataclass
from typing import Optional

# Add project root to path so we can import modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import pretty_midi

# Project imports
from data.dataset import MaestroDataset
from models.remi_tokenizer import REMITokenizer
from models.transcription_model import TranscriptionModel
from train.train_transcriber import train_model


def piano_roll_to_midi(piano_roll, fs: float = 31.25, program: int = 0, output_path: str = "output.mid"):
    """
    Convert piano roll to MIDI file.

    Args:
        piano_roll: Array (88, T) with note activations
        fs: Frames per second
        program: MIDI program number (0 = Acoustic Grand Piano)
        output_path: Where to save MIDI file
    """
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    notes_on = {}
    for frame_idx in range(piano_roll.shape[1]):
        time = frame_idx / fs
        for note_num in range(88):
            midi_note = note_num + 21  # A0 = 21
            is_active = piano_roll[note_num, frame_idx] > 0

            if is_active and midi_note not in notes_on:
                notes_on[midi_note] = time
            elif (not is_active) and (midi_note in notes_on):
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=midi_note,
                    start=notes_on[midi_note],
                    end=time,
                )
                instrument.notes.append(note)
                del notes_on[midi_note]

    final_time = piano_roll.shape[1] / fs
    for midi_note, start_time in notes_on.items():
        note = pretty_midi.Note(
            velocity=100,
            pitch=midi_note,
            start=start_time,
            end=final_time,
        )
        instrument.notes.append(note)

    midi.instruments.append(instrument)
    midi.write(output_path)
    print(f"[info] MIDI file saved to {output_path}")


@dataclass
class TrainConfig:
    root_dir: str = "maestro-v3.0.0"
    split: str = "train"
    year: Optional[str] = "2017"  # None = all years, or specific year like "2017"
    subset_size: Optional[int] = 100  # None = full dataset, or int for subset
    batch_size: int = 2
    num_epochs: int = 5
    lr: float = 1e-4
    model_type: str = "ast"
    freeze_encoder: bool = True
    use_mock_encoder: bool = False
    remi_vocab_size: int = 512
    decoder_layers: int = 4
    decoder_dim: int = 384
    decoder_heads: int = 6
    dropout: float = 0.2
    device: str = "auto"  # "auto" => cuda if available else cpu

    # Performance optimizations
    chunk_length: Optional[float] = 30.0  # Chunk audio into 30s segments (None = full files)
    chunk_overlap: float = 0.0  # No overlap between chunks

    # Optional "notebook-like" sanity checks
    run_sanity_checks: bool = True
    sanity_max_len: int = 256


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train transcription model (AST) on MAESTRO.")
    p.add_argument("--root-dir", default="maestro-v3.0.0")
    p.add_argument("--split", default="train")
    p.add_argument("--year", default="2017", help='Specific year (e.g., "2017") or "all" for all years')
    p.add_argument("--subset-size", type=int, default=100, help='Number of files to use, or 0 for full dataset')
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)

    p.add_argument("--model-type", default="ast")
    p.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-mock-encoder", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument("--remi-vocab-size", type=int, default=512)
    p.add_argument("--decoder-layers", type=int, default=4)
    p.add_argument("--decoder-dim", type=int, default=384)
    p.add_argument("--decoder-heads", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.2)

    p.add_argument("--device", default="auto", help='auto | cpu | cuda (default: auto)')

    # Performance options
    p.add_argument("--chunk-length", type=float, default=30.0, help='Chunk length in seconds (0 = full files)')
    p.add_argument("--chunk-overlap", type=float, default=0.0, help='Overlap ratio between chunks (0.0-1.0)')

    p.add_argument("--run-sanity-checks", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sanity-max-len", type=int, default=256)

    args = p.parse_args()

    # Convert special values
    args_dict = vars(args)

    # "all" or empty string -> None for year (means all years)
    if args_dict['year'] in ('all', 'All', 'ALL', ''):
        args_dict['year'] = None

    # 0 -> None for subset_size (means full dataset)
    if args_dict['subset_size'] == 0:
        args_dict['subset_size'] = None

    # 0 -> None for chunk_length (means full files, no chunking)
    if args_dict['chunk_length'] == 0:
        args_dict['chunk_length'] = None

    return TrainConfig(**args_dict)


def resolve_device(device_flag: str) -> str:
    if device_flag == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_flag


def print_env_info():
    print(f"[info] Current directory: {os.getcwd()}")
    print(f"[info] PyTorch version: {torch.__version__}")
    print(f"[info] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[info] CUDA device: {torch.cuda.get_device_name(0)}")


def run_sanity_checks(cfg: TrainConfig):
    """
    Roughly mirrors the notebook “smoke test”:
    - load dataset sample (waveform + roll)
    - encode/decode with REMI tokenizer
    """
    print("[info] Running sanity checks...")

    dataset = MaestroDataset(
        root_dir=cfg.root_dir,
        split=cfg.split,
        subset_size=cfg.subset_size,
        return_waveform=True,
    )

    print(f"[info] Dataset size: {len(dataset)}")
    print(f"[info] Sample rate: {dataset.sr}")
    print(f"[info] Mel bins: {dataset.n_mels}")
    print(f"[info] Hop length: {dataset.hop_length}")

    waveform, roll = dataset[0]
    print(f"[info] Waveform shape: {getattr(waveform, 'shape', None)}")
    print(f"[info] Piano roll shape: {getattr(roll, 'shape', None)}")
    if hasattr(roll, "shape"):
        dur = roll.shape[1] * dataset.hop_length / dataset.sr
        print(f"[info] Duration: {dur:.2f} seconds")

    tokenizer = REMITokenizer()
    print(f"[info] Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"[info] SOS: {tokenizer.sos}, EOS: {tokenizer.eos}, PAD: {tokenizer.pad}")

    tokens = tokenizer.encode_from_pianoroll(roll, max_len=cfg.sanity_max_len)
    print(f"[info] Encoded to {len(tokens)} tokens (truncated to max_len={cfg.sanity_max_len})")
    print(f"[info] First 20 tokens: {tokens[:20]}")

    decoded_roll = tokenizer.decode_to_pianoroll(tokens, max_T=roll.shape[1])
    print(f"[info] Decoded piano roll shape: {decoded_roll.shape}")

    print("[info] Sanity checks complete.")


def main():
    # Matches your notebook environment setting (use newer env var name)
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    cfg = parse_args()
    cfg.device = resolve_device(cfg.device)

    print_env_info()
    print(f"[info] Using device: {cfg.device}")

    if cfg.run_sanity_checks:
        run_sanity_checks(cfg)

    # Optional: instantiate to ensure model downloads / loads cleanly (like your notebook)
    # (train_model will build internally anyway, but this helps catch issues early)
    _ = TranscriptionModel(
        model_type=cfg.model_type,
        device=cfg.device,
        freeze_encoder=cfg.freeze_encoder,
    )

    print("[info] Starting training...")
    print(f"[info] Chunking: {'Enabled (' + str(cfg.chunk_length) + 's chunks)' if cfg.chunk_length else 'Disabled (full files)'}")
    _trained_model = train_model(
        root_dir=cfg.root_dir,
        split=cfg.split,
        year=cfg.year,
        subset_size=cfg.subset_size,
        batch_size=cfg.batch_size,
        num_epochs=cfg.num_epochs,
        lr=cfg.lr,
        model_type=cfg.model_type,
        freeze_encoder=cfg.freeze_encoder,
        use_mock_encoder=cfg.use_mock_encoder,
        remi_vocab_size=cfg.remi_vocab_size,
        decoder_layers=cfg.decoder_layers,
        decoder_dim=cfg.decoder_dim,
        decoder_heads=cfg.decoder_heads,
        dropout=cfg.dropout,
        device=cfg.device,
        chunk_length=cfg.chunk_length,
        chunk_overlap=cfg.chunk_overlap,
    )

    print("[info] Training complete! Model saved and ready for inference.")


if __name__ == "__main__":
    main()