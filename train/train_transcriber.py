import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from data.dataset import MaestroDataset
from models.transcription_model import TranscriptionModel

import torch.nn.functional as F

def collate_fn(batch, max_len=None):
    """
    Pads (mel, roll) pairs in a batch to the same time dimension.
    Args:
        batch: list of (mel_tensor, roll_tensor)
        max_len: optional int, truncate or pad to this max length
    """
    mels, rolls = zip(*batch)

    # Find the longest sample in the batch (or use provided max_len)
    max_T = max([mel.shape[-1] for mel in mels])
    if max_len is not None:
        max_T = min(max_T, max_len)

    padded_mels, padded_rolls = [], []

    for mel, roll in zip(mels, rolls):
        T = mel.shape[-1]
        pad_T = max_T - T
        if pad_T > 0:
            mel = F.pad(mel, (0, pad_T))
            roll = F.pad(roll, (0, pad_T))
        elif pad_T < 0:
            mel = mel[:, :, :max_T]
            roll = roll[:, :max_T]
        padded_mels.append(mel)
        padded_rolls.append(roll)

    batch_mels = torch.stack(padded_mels)
    batch_rolls = torch.stack(padded_rolls)
    return batch_mels, batch_rolls


# Training helper
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for mel, roll in tqdm(dataloader, desc="Training", leave=False):
        mel, roll = mel.to(device), roll.to(device)

        optimizer.zero_grad()
        logits = model(mel)
        loss = model.compute_loss(logits, roll)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

# Validation helper
@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0

    for mel, roll in tqdm(dataloader, desc="Validation", leave=False):
        mel, roll = mel.to(device), roll.to(device)
        logits = model(mel)
        loss = model.compute_loss(logits, roll)
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