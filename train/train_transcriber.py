import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from data.dataset import MaestroDataset
from models.transcription_model import TranscriptionModel

import torch.nn.functional as F

from torch.amp import autocast

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

def train_one_epoch(model, dataloader, optimizer, device, max_grad_norm=1.0):
    model.train()
    scaler = GradScaler()

    total_loss = 0.0
    step_losses = []
    nan_count = 0

    # Initialize tqdm progress bar
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for mel, roll, lengths in progress_bar:
        mel, roll = mel.to(device), roll.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward
        with autocast('cuda'):
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

    with torch.no_grad():
        for mel, roll, lengths in tqdm(dataloader, desc="Validation", leave=False):
            mel, roll = mel.to(device), roll.to(device)

            # Use mixed precision for inference
            with autocast('cuda'):
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
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Dataset ---
    train_set = MaestroDataset(
        root_dir=root_dir,
        year=year,
        split='train',
        subset_size=subset_size,
        chunk_length=chunk_length,
        overlap=chunk_overlap
    )
    val_set = MaestroDataset(
        root_dir=root_dir,
        year=year,
        split='validation',
        subset_size=subset_size // 5 if subset_size else None,
        chunk_length=chunk_length,
        overlap=0.0  # No overlap for validation
    )

    print(f"Train set size: {len(train_set)} {'chunks' if chunk_length else 'files'}")
    print(f"Validation set size: {len(val_set)} {'chunks' if chunk_length else 'files'}")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        collate_fn=collate_fn
)

    # Model
    model = TranscriptionModel(model_type="cnn_rnn", device=device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
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