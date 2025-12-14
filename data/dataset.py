import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import librosa
import pretty_midi
import numpy as np

class MaestroDataset(Dataset):
    def __init__(self, root_dir, csv_path=None, year=None, split='train',
                 sr=16000, n_mels=229, hop_length=512, subset_size=None,
                 chunk_length=None, overlap=0.0, return_waveform=False):
        """
        Load MAESTRO dataset with optional split filtering and chunking.
        Args:
            root_dir: path to maestro-v2.0.0/
            csv_path: optional path to metadata CSV
            year: optional specific year (e.g. "2017")
            split: one of {"train", "validation", "test"}
            subset_size: limit dataset size for debugging
            chunk_length: duration in seconds for each chunk. If None, loads full files.
            overlap: overlap ratio between chunks (0.0 to 1.0). E.g., 0.5 = 50% overlap.
            return_waveform: if True, return raw waveform instead of mel spectrogram
        """
        self.root_dir = root_dir
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.overlap = overlap
        self.return_waveform = return_waveform

        # Load metadata CSV
        if csv_path is None:
            csv_path = os.path.join(root_dir, "maestro-v3.0.0.csv")
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

        # Build chunk index if chunking is enabled
        self.chunks = []
        if chunk_length is not None:
            self._build_chunk_index()

    def _build_chunk_index(self):
        """
        Pre-compute chunk positions for each file.
        Each entry: (file_idx, start_time, end_time)
        """
        chunk_samples = int(self.chunk_length * self.sr)
        hop_samples = int(chunk_samples * (1.0 - self.overlap))

        for file_idx, row in self.df.iterrows():
            audio_path = os.path.join(self.root_dir, row["audio_filename"])

            # Convert .wav references to .mp3 if needed
            if not os.path.exists(audio_path) and audio_path.endswith(".wav"):
                audio_path = audio_path.replace(".wav", ".mp3")

            # Get audio duration without loading the full file
            duration = librosa.get_duration(path=audio_path)
            total_samples = int(duration * self.sr)

            # Generate chunk positions
            start_sample = 0
            while start_sample < total_samples:
                end_sample = min(start_sample + chunk_samples, total_samples)

                # Only add chunks that meet minimum length threshold (e.g., 50% of chunk_length)
                if (end_sample - start_sample) >= chunk_samples * 0.5:
                    self.chunks.append({
                        'file_idx': file_idx,
                        'start_sample': start_sample,
                        'end_sample': end_sample,
                        'start_time': start_sample / self.sr,
                        'end_time': end_sample / self.sr
                    })

                start_sample += hop_samples

                # Break if we've reached the end
                if end_sample >= total_samples:
                    break

    def __len__(self):
        if self.chunk_length is not None:
            return len(self.chunks)
        return len(self.df)

    def __getitem__(self, idx):
        if self.chunk_length is not None:
            return self._get_chunk(idx)
        else:
            return self._get_full_file(idx)

    def _get_chunk(self, idx):
        """Load a specific chunk from a file."""
        chunk_info = self.chunks[idx]
        file_idx = chunk_info['file_idx']
        start_sample = chunk_info['start_sample']
        end_sample = chunk_info['end_sample']

        row = self.df.iloc[file_idx]
        audio_path = os.path.join(self.root_dir, row["audio_filename"])
        midi_path = os.path.join(self.root_dir, row["midi_filename"])

        # Convert .wav references to .mp3 if needed
        if not os.path.exists(audio_path) and audio_path.endswith(".wav"):
            audio_path = audio_path.replace(".wav", ".mp3")

        # Load audio chunk
        y, _ = librosa.load(
            audio_path,
            sr=self.sr,
            mono=True,
            offset=chunk_info['start_time'],
            duration=(end_sample - start_sample) / self.sr
        )

        # Load MIDI and extract corresponding time range
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        fs = self.sr / self.hop_length

        # Only compute piano roll for the needed time range
        start_time = chunk_info['start_time']
        end_time = chunk_info['end_time']

        # Get piano roll only for this chunk's time range
        full_roll = midi_data.get_piano_roll(
            fs=fs,
            times=np.linspace(start_time, end_time, int((end_time - start_time) * fs))
        )[21:109]  # 88 keys (A0–C8)

        roll = (full_roll > 0).astype(np.float32)

        if self.return_waveform:
            # Return raw waveform and piano roll (for AST model)
            waveform_tensor = torch.tensor(y, dtype=torch.float32)
            roll_tensor = torch.tensor(roll)  # (88, T)
            return waveform_tensor, roll_tensor
        else:
            # Return mel spectrogram and piano roll (for CNN-RNN model)
            mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels, hop_length=self.hop_length)
            mel = librosa.power_to_db(mel).astype(np.float32)

            # Align time frames between mel and roll
            min_len = min(mel.shape[1], roll.shape[1])
            mel = mel[:, :min_len]
            roll = roll[:, :min_len]

            # Convert to tensors
            mel_tensor = torch.tensor(mel).unsqueeze(0)  # (1, n_mels, T)
            roll_tensor = torch.tensor(roll)             # (88, T)

            return mel_tensor, roll_tensor

    def _get_full_file(self, idx):
        """Load a complete file (original behavior)."""
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.root_dir, row["audio_filename"])
        midi_path = os.path.join(self.root_dir, row["midi_filename"])

        # Convert .wav references to .mp3 if needed
        if not os.path.exists(audio_path) and audio_path.endswith(".wav"):
            audio_path = audio_path.replace(".wav", ".mp3")

        # Load and process audio ---
        y, _ = librosa.load(audio_path, sr=self.sr, mono=True)

        # Load and process MIDI ---
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        fs = self.sr / self.hop_length
        roll = midi_data.get_piano_roll(fs=fs)[21:109]  # 88 keys (A0–C8)
        roll = (roll > 0).astype(np.float32)

        if self.return_waveform:
            # Return raw waveform and piano roll (for AST model)
            waveform_tensor = torch.tensor(y, dtype=torch.float32)
            roll_tensor = torch.tensor(roll)  # (88, T)
            return waveform_tensor, roll_tensor
        else:
            # Return mel spectrogram and piano roll (for CNN-RNN model)
            mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels, hop_length=self.hop_length)
            mel = librosa.power_to_db(mel).astype(np.float32)

            # Align time frames between mel and roll
            min_len = min(mel.shape[1], roll.shape[1])
            mel = mel[:, :min_len]
            roll = roll[:, :min_len]

            # Convert to tensors
            mel_tensor = torch.tensor(mel).unsqueeze(0)  # (1, n_mels, T)
            roll_tensor = torch.tensor(roll)             # (88, T)

            return mel_tensor, roll_tensor