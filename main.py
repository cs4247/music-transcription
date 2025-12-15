import os
import sys
import argparse
import torch
import librosa
import numpy as np
import pretty_midi
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.transcription_model import TranscriptionModel

# Default configuration
MODEL_TYPE = "cnn_rnn_large"        # Model type: cnn_rnn or cnn_rnn_large
N_MELS = 320                         # Number of mel frequency bins
HIDDEN_SIZE = 512                    # RNN hidden layer size
NUM_LAYERS = 3                       # Number of RNN layers
DROPOUT = 0.2                        # Dropout rate
SR = 16000                           # Sample rate
HOP_LENGTH = 512                     # Hop length for mel spectrogram
CHUNK_LENGTH = 30.0                  # Chunk length in seconds
THRESHOLD = 0.5                      # Prediction threshold


def load_model(model_path, device="cpu"):
    """
    Load the trained transcription model from a checkpoint.

    Args:
        model_path: Path to the .pth checkpoint file
        device: Device to load model on ('cpu' or 'cuda')

    Returns:
        Loaded model in eval mode
    """
    print(f"Loading model from {model_path}...")

    # Initialize model
    model = TranscriptionModel(
        model_type=MODEL_TYPE,
        n_mels=N_MELS,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        device=device
    )

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    print(f"Model loaded successfully on {device}")
    return model


def split_audio_into_chunks(audio_path, chunk_length=CHUNK_LENGTH, sr=SR):
    """
    Load audio file and split it into fixed-length chunks.

    Args:
        audio_path: Path to audio file (mp3, wav, etc.)
        chunk_length: Length of each chunk in seconds
        sr: Sample rate to load audio at

    Returns:
        List of audio chunks (numpy arrays)
        Original audio duration in seconds
    """
    print(f"Loading audio from {audio_path}...")

    # Load full audio file
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    duration = len(y) / sr

    print(f"Audio duration: {duration:.2f} seconds")

    # Calculate chunk size in samples
    chunk_samples = int(chunk_length * sr)

    # Split into chunks
    chunks = []
    num_chunks = int(np.ceil(len(y) / chunk_samples))

    for i in range(num_chunks):
        start_idx = i * chunk_samples
        end_idx = min((i + 1) * chunk_samples, len(y))
        chunk = y[start_idx:end_idx]

        # Pad last chunk if needed
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')

        chunks.append(chunk)

    print(f"Split audio into {len(chunks)} chunks of {chunk_length}s each")
    return chunks, duration


def audio_to_mel(audio_chunk, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH):
    """
    Convert audio chunk to mel spectrogram.

    Args:
        audio_chunk: Audio samples (numpy array)
        sr: Sample rate
        n_mels: Number of mel frequency bins
        hop_length: Hop length for STFT

    Returns:
        Mel spectrogram tensor of shape (1, 1, n_mels, T)
    """
    # Compute mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio_chunk,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length
    )

    # Convert to dB scale
    mel = librosa.power_to_db(mel).astype(np.float32)

    # Add batch and channel dimensions: (n_mels, T) -> (1, 1, n_mels, T)
    mel_tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0)

    return mel_tensor


def predict_chunk(model, mel_tensor, device, threshold=THRESHOLD):
    """
    Run model prediction on a single mel spectrogram chunk.

    Args:
        model: Loaded transcription model
        mel_tensor: Mel spectrogram tensor (1, 1, n_mels, T)
        device: Device to run on
        threshold: Threshold for binarizing predictions

    Returns:
        Binary piano roll of shape (88, T)
    """
    mel_tensor = mel_tensor.to(device)

    with torch.no_grad():
        # Forward pass - model returns logits
        logits = model(mel_tensor)  # (1, 88, T)

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Threshold to get binary piano roll
        piano_roll = (probs > threshold).float()

    # Remove batch dimension and move to CPU: (1, 88, T) -> (88, T)
    piano_roll = piano_roll[0].cpu().numpy()

    return piano_roll


def combine_piano_rolls(piano_rolls, chunk_length=CHUNK_LENGTH, sr=SR, hop_length=HOP_LENGTH):
    """
    Combine piano rolls from multiple chunks into a single piano roll.

    Args:
        piano_rolls: List of piano roll arrays, each of shape (88, T)
        chunk_length: Length of each chunk in seconds
        sr: Sample rate
        hop_length: Hop length used for mel spectrogram

    Returns:
        Combined piano roll of shape (88, T_total)
    """
    if len(piano_rolls) == 1:
        return piano_rolls[0]

    # Calculate frames per chunk
    frames_per_chunk = int((chunk_length * sr) / hop_length)

    # Concatenate all piano rolls
    combined = np.concatenate(piano_rolls, axis=1)

    return combined


def pianoroll_to_midi(pianoroll, fs, min_midi=21):
    """
    Convert a (88, T) binary piano roll to a pretty_midi.PrettyMIDI object.

    Args:
        pianoroll: np.ndarray of shape (88, T), values {0,1}
        fs: frames per second (sr / hop_length)
        min_midi: MIDI note number for index 0 in the roll (default 21 = A0)

    Returns:
        pretty_midi.PrettyMIDI object
    """
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

    for pitch_idx in range(pianoroll.shape[0]):
        pitch = min_midi + pitch_idx
        active = pianoroll[pitch_idx] > 0

        # Find on/off transitions
        changes = np.diff(np.concatenate([[0], active.astype(int), [0]]))
        onsets = np.where(changes == 1)[0]
        offsets = np.where(changes == -1)[0]

        for start_idx, end_idx in zip(onsets, offsets):
            start_time = start_idx / fs
            end_time = end_idx / fs
            if end_time > start_time:
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=pitch,
                    start=start_time,
                    end=end_time,
                )
                instrument.notes.append(note)

    midi.instruments.append(instrument)
    return midi


def transcribe_audio(audio_path, model_path, output_path=None, device=None):
    """
    Full pipeline: transcribe an audio file to MIDI.

    Args:
        audio_path: Path to input audio file
        model_path: Path to model checkpoint (.pth file)
        output_path: Path to save output MIDI file (optional)
        device: Device to use ('cpu' or 'cuda', auto-detect if None)

    Returns:
        Path to output MIDI file
    """
    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # Load model
    model = load_model(model_path, device)

    # Split audio file into 30s chunks
    audio_chunks, duration = split_audio_into_chunks(audio_path, CHUNK_LENGTH, SR)

    # Process chunks into mels and run prediction
    print("Processing chunks and running predictions...")
    piano_rolls = []

    for i, chunk in enumerate(audio_chunks):
        print(f"Processing chunk {i+1}/{len(audio_chunks)}...")

        # Convert to mel spectrogram
        mel_tensor = audio_to_mel(chunk, SR, N_MELS, HOP_LENGTH)

        # Run prediction
        piano_roll = predict_chunk(model, mel_tensor, device, THRESHOLD)
        piano_rolls.append(piano_roll)

    # Combine the piano rolls from all chunks
    print("Combining predictions from all chunks...")
    combined_piano_roll = combine_piano_rolls(piano_rolls, CHUNK_LENGTH, SR, HOP_LENGTH)

    # Convert to MIDI
    print("Converting to MIDI...")
    fs = SR / HOP_LENGTH  # Frame rate: 31.25 fps
    midi = pianoroll_to_midi(combined_piano_roll, fs, min_midi=21)

    # Determine output path
    if output_path is None:
        # Create output path based on input audio path
        audio_file = Path(audio_path)
        output_path = audio_file.parent / f"{audio_file.stem}_transcription.mid"

    # Save MIDI file
    midi.write(str(output_path))
    print(f"MIDI file saved to: {output_path}")

    return output_path


def main():
    """Main entry point for the transcription script."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio files to MIDI using trained music transcription model"
    )
    parser.add_argument(
        "audio_file",
        type=str,
        help="Path to input audio file (mp3, wav, etc.)"
    )
    parser.add_argument(
        "model_file",
        type=str,
        help="Path to model checkpoint file (.pth)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to output MIDI file (default: <audio_name>_transcription.mid)"
    )
    parser.add_argument(
        "-d", "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Device to use for inference (default: auto-detect)"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.5,
        help="Threshold for note predictions (default: 0.5)"
    )

    args = parser.parse_args()

    # Update global threshold if specified
    global THRESHOLD
    THRESHOLD = args.threshold

    # Validate input files exist
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        sys.exit(1)

    if not os.path.exists(args.model_file):
        print(f"Error: Model file not found: {args.model_file}")
        sys.exit(1)

    # Run transcription
    print("="*60)
    print("Music Transcription Pipeline")
    print("="*60)

    try:
        output_path = transcribe_audio(
            args.audio_file,
            args.model_file,
            args.output,
            args.device
        )

        print("="*60)
        print("Transcription completed successfully!")
        print(f"Output: {output_path}")
        print("="*60)

    except Exception as e:
        print(f"Error during transcription: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
