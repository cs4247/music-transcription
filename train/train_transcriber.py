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

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    scaler = GradScaler()

    total_loss = 0.0
    step_losses = []

    # Initialize tqdm progress bar
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for mel, roll, lengths in progress_bar:
        mel, roll = mel.to(device), roll.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward
        with autocast('cuda'):
            logits = model(mel)
            loss = model.compute_loss(logits, roll, lengths)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update running totals
        step_loss = loss.item()
        total_loss += step_loss
        step_losses.append(step_loss)

        # Update tqdm bar dynamically
        progress_bar.set_postfix({"step_loss": f"{step_loss:.4f}"})

    avg_loss = total_loss / len(dataloader)
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
    root_dir="maestro-v2.0.0",
    year="2017",
    batch_size=4,
    num_epochs=5,
    lr=1e-4,
    subset_size=None,
    device=None,
    save_path="checkpoints/cnn_rnn.pth",
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Dataset ---
    train_set = MaestroDataset(root_dir=root_dir, year=year, subset_size=subset_size)
    val_set = MaestroDataset(root_dir=root_dir, year=year, subset_size=subset_size // 5 if subset_size else 10)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, max_len=12000)  # or None for dynamic padding
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        collate_fn=lambda b: collate_fn(b, max_len=12000)
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
        root_dir="maestro-v2.0.0",
        year="2017",
        batch_size=2,
        num_epochs=3,
        lr=1e-4,
        subset_size=20,
    )