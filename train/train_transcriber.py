import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os

from data.dataset import MaestroDataset
from models.transcription_model import TranscriptionModel

import torch.nn.functional as F

from torch.amp import autocast

# Conditional import for AST-specific features
try:
    from models.remi_tokenizer import REMITokenizer
    _REMI_AVAILABLE = True
except ImportError:
    REMITokenizer = None
    _REMI_AVAILABLE = False

def collate_fn(batch):
    """
    Pad variable-length spectrograms and rolls to the same temporal length.
    Returns mel_batch, roll_batch, and valid lengths per sample.
    """
    mels, rolls = zip(*batch)
    lengths = [m.shape[-1] for m in mels]  # number of valid time frames
    max_T = max(lengths)

    mel_padded = [torch.nn.functional.pad(m, (0, max_T - m.shape[-1])) for m in mels]
    roll_padded = [torch.nn.functional.pad(r, (0, max_T - r.shape[-1])) for r in rolls]

    mel_batch = torch.stack(mel_padded)
    roll_batch = torch.stack(roll_padded)
    lengths = torch.tensor(lengths, dtype=torch.long)

    return mel_batch, roll_batch, lengths

def collate_ast(batch, max_token_len: int = 256):
    """
    Collate function for AST model: converts waveforms and piano rolls to tokens.

    Args:
        batch: List of (waveform, roll) tuples from dataset
        max_token_len: Maximum token sequence length

    Returns:
        waveforms_list: List of waveform tensors
        token_tensor: Tensor of token sequences (B, L)
    """
    if not _REMI_AVAILABLE:
        raise ImportError("collate_ast requires REMITokenizer")

    tokenizer = REMITokenizer()
    waveforms, rolls = zip(*batch)

    token_seqs = []
    for roll in rolls:
        # Convert piano roll to tokens
        seq = tokenizer.encode_from_pianoroll(roll, max_len=max_token_len)
        token_seqs.append(seq)

    token_tensor = torch.tensor(token_seqs, dtype=torch.long)
    waveforms_list = list(waveforms)

    return waveforms_list, token_tensor

def collate_ast_tokenized(batch):
    """
    Fast collate function for pre-tokenized AST data.
    Used when dataset already contains tokens (no CPU tokenization needed).

    Args:
        batch: List of (waveform, tokens) tuples from cached dataset

    Returns:
        waveforms_list: List of waveform tensors
        token_tensor: Tensor of token sequences (B, L)
    """
    waveforms, token_tensors = zip(*batch)

    # Stack tokens into batch
    token_tensor = torch.stack(token_tensors)
    waveforms_list = list(waveforms)

    return waveforms_list, token_tensor

def train_one_epoch(model, dataloader, optimizer, device, max_grad_norm=1.0):
    model.train()
    scaler = GradScaler()

    total_loss = 0.0
    step_losses = []
    nan_count = 0

    # Initialize tqdm progress bar
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    # Detect model type
    is_ast = model.model_type in ["ast", "transformer", "audio_transformer"]

    for batch in progress_bar:
        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward
        with autocast('cuda'):
            if is_ast:
                # AST model: batch is (waveforms_list, token_tensor)
                waveforms, token_targets = batch
                token_targets = token_targets.to(device)
                logits = model(waveforms, sampling_rate=16000, targets=token_targets)
                loss = model.compute_loss(logits, token_targets)
            else:
                # CNN-RNN model: batch is (mel, roll, lengths)
                mel, roll, lengths = batch
                mel, roll = mel.to(device), roll.to(device)
                logits = model(mel)
                loss = model.compute_loss(logits, roll, lengths)

        # Check for NaN loss before backward
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            print(f"\n⚠ Warning: NaN/Inf loss detected (count: {nan_count}), skipping batch")
            if nan_count > 10:
                raise RuntimeError("Too many NaN losses - training unstable!")
            continue

        scaler.scale(loss).backward()

        # Gradient clipping (critical for large models with attention)
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Check for NaN gradients
        if torch.isnan(grad_norm):
            nan_count += 1
            print(f"\n⚠ Warning: NaN gradients detected (count: {nan_count}), skipping batch")
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            continue

        scaler.step(optimizer)
        scaler.update()

        # Update running totals
        step_loss = loss.item()
        total_loss += step_loss
        step_losses.append(step_loss)

        # Update tqdm bar dynamically with gradient norm
        progress_bar.set_postfix({
            "step_loss": f"{step_loss:.4f}",
            "grad_norm": f"{grad_norm:.2f}"
        })

    avg_loss = total_loss / len(dataloader) if len(step_losses) > 0 else float('nan')
    progress_bar.close()
    return avg_loss, step_losses

# Validation helper
@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0

    # Detect model type
    is_ast = model.model_type in ["ast", "transformer", "audio_transformer"]

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            # Use mixed precision for inference
            with autocast('cuda'):
                if is_ast:
                    # AST model: batch is (waveforms_list, token_tensor)
                    waveforms, token_targets = batch
                    token_targets = token_targets.to(device)
                    logits = model(waveforms, sampling_rate=16000, targets=token_targets)
                    loss = model.compute_loss(logits, token_targets)
                else:
                    # CNN-RNN model: batch is (mel, roll, lengths)
                    mel, roll, lengths = batch
                    mel, roll = mel.to(device), roll.to(device)
                    logits = model(mel)
                    loss = model.compute_loss(logits, roll, lengths)

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

# Main training function (can be imported in notebook)
def train_model(
    root_dir="maestro-v3.0.0",
    year="2017",
    batch_size=4,
    num_epochs=5,
    lr=1e-4,
    subset_size=None,
    device=None,
    save_path="checkpoints/cnn_rnn.pth",
    chunk_length=None,
    chunk_overlap=0.0,
    model_type="cnn_rnn",
    cached_dir=None,
    **model_kwargs
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine if we need waveforms (for AST) or mel spectrograms (for CNN-RNN)
    return_waveform = model_type in ["ast", "transformer", "audio_transformer"]

    # --- Dataset ---
    # Try to use cached dataset if available
    use_tokenized_cache = False
    if cached_dir:
        try:
            from data.cached_dataset import CachedMaestroDataset
            print(f"Loading from cache: {cached_dir}")

            train_set = CachedMaestroDataset(
                cache_dir=cached_dir,
                split='train'
            )
            val_set = CachedMaestroDataset(
                cache_dir=cached_dir,
                split='validation'
            )

            # Check if this is a tokenized cache
            import pickle
            metadata_path = os.path.join(cached_dir, 'train_metadata.pkl')
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            use_tokenized_cache = metadata.get('tokenize', False)

            if use_tokenized_cache:
                print("✓ Using pre-tokenized cache (MAXIMUM SPEED!)")
            else:
                print("Successfully loaded cached dataset!")
        except Exception as e:
            print(f"Warning: Could not load cached dataset: {e}")
            print("Falling back to raw dataset loading...")
            cached_dir = None

    # Fallback to raw dataset
    if not cached_dir:
        train_set = MaestroDataset(
            root_dir=root_dir,
            year=year,
            split='train',
            subset_size=subset_size,
            chunk_length=chunk_length,
            overlap=chunk_overlap,
            return_waveform=return_waveform
        )
        val_set = MaestroDataset(
            root_dir=root_dir,
            year=year,
            split='validation',
            subset_size=subset_size // 5 if subset_size else None,
            chunk_length=chunk_length,
            overlap=0.0,  # No overlap for validation
            return_waveform=return_waveform
        )

    print(f"Train set size: {len(train_set)} {'chunks' if chunk_length else 'files'}")
    print(f"Validation set size: {len(val_set)} {'chunks' if chunk_length else 'files'}")

    # Select appropriate collate function
    if use_tokenized_cache:
        # Pre-tokenized cache: use fast collate (no CPU tokenization!)
        collate_used = collate_ast_tokenized
    elif return_waveform:
        # Regular AST: tokenize during collate (slow!)
        collate_used = collate_ast
    else:
        # CNN-RNN: pad mels
        collate_used = collate_fn

    # Use multiple workers for parallel data loading
    # More workers for tokenized cache (pure I/O), fewer for regular cache (CPU tokenization)
    num_workers = 8 if use_tokenized_cache else (4 if cached_dir else 0)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_used,
        num_workers=num_workers,
        pin_memory=True,  # Faster CPU->GPU transfer
        prefetch_factor=2 if num_workers > 0 else None,  # Pre-load 2 batches per worker
        persistent_workers=True if num_workers > 0 else False  # Keep workers alive between epochs
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        collate_fn=collate_used,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None
    )

    # Model
    model = TranscriptionModel(model_type=model_type, device=device, **model_kwargs)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Create checkpoint directory if it doesn't exist
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss, step_losses = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model to {save_path}")

    return model

# CLI entrypoint
if __name__ == "__main__":
    train_model(
        root_dir="maestro-v3.0.0",
        year="2017",
        batch_size=2,
        num_epochs=3,
        lr=1e-4,
        subset_size=20,
        chunk_length=30.0,  # 30-second chunks
        chunk_overlap=0.25,  # 25% overlap for data augmentation
    )