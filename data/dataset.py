import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import librosa
import pretty_midi
import numpy as np

class MaestroDataset(Dataset):
    def __init__(self, root_dir, csv_path=None, year=None, split='train',
                 sr=16000, n_mels=229, hop_length=512, subset_size=None):
        """
        Load MAESTRO dataset with optional split filtering.
        Args:
            root_dir: path to maestro-v2.0.0/
            csv_path: optional path to metadata CSV
            year: optional specific year (e.g. "2017")
            split: one of {"train", "validation", "test"}
            subset_size: limit dataset size for debugging
        """
        self.root_dir = root_dir
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length

        # Load metadata CSV
        if csv_path is None:
            csv_path = os.path.join(root_dir, "maestro-v2.0.0.csv")
        self.df = pd.read_csv(csv_path)

        # Filter by year (optional)
        if year is not None:
            self.df = self.df[self.df["year"] == int(year)]

        # Filter by official MAESTRO split (train/validation/test)
        if split is not None:
            self.df = self.df[self.df["split"] == split]

        # Optionally take a small subset for debugging
        if subset_size:
            self.df = self.df.head(subset_size)

        self.df.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.root_dir, row["audio_filename"])
        midi_path = os.path.join(self.root_dir, row["midi_filename"])

        # Convert .wav references to .mp3 if needed
        if not os.path.exists(audio_path) and audio_path.endswith(".wav"):
            audio_path = audio_path.replace(".wav", ".mp3")

        # Load and process audio ---
        y, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels, hop_length=self.hop_length)
        mel = librosa.power_to_db(mel).astype(np.float32)

        # Load and process MIDI ---
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        fs = self.sr / self.hop_length
        roll = midi_data.get_piano_roll(fs=fs)[21:109]  # 88 keys (A0–C8)
        roll = (roll > 0).astype(np.float32)

        # Align time frames between mel and roll
        min_len = min(mel.shape[1], roll.shape[1])
        mel = mel[:, :min_len]
        roll = roll[:, :min_len]

        # Convert to tensors
        mel_tensor = torch.tensor(mel).unsqueeze(0)  # (1, n_mels, T)
        roll_tensor = torch.tensor(roll)             # (88, T)

        return mel_tensor, roll_tensor