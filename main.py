import argparse
import os
from datetime import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from data.dataset import MaestroDataset
from train.train_transcriber import train_one_epoch, evaluate, collate_fn
from models.transcription_model import TranscriptionModel


# Plot: per-epoch averaged losses
def plot_training_curves(train_losses, val_losses, save_path):
    """Plot and save loss curves."""
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", color="royalblue", linewidth=2)
    plt.plot(val_losses, label="Val Loss", color="tomato", linewidth=2)
    plt.title("Training and Validation Loss", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Plot: per-step losses with epoch boundaries
def plot_step_losses(global_step_losses, num_epochs, save_path):
    """
    Plot per-step training loss across all epochs.

    Args:
        global_step_losses: list of lists, where each sublist = losses for that epoch
        num_epochs: total number of epochs (for x-axis scaling)
        save_path: file path for saving the figure
    """
    plt.figure(figsize=(10, 5))
    flat_losses = np.concatenate(global_step_losses)
    plt.plot(flat_losses, color="mediumseagreen", linewidth=1.2)
    plt.title("Training Loss per Step", fontsize=14)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)

    # Add vertical lines to mark epoch boundaries
    step = 0
    for i, epoch_losses in enumerate(global_step_losses, 1):
        step += len(epoch_losses)
        plt.axvline(x=step, color="gray", linestyle="--", alpha=0.3)
        plt.text(step, plt.ylim()[1]*0.95, f"Epoch {i}", rotation=90,
                 fontsize=8, color="gray", va="top", ha="right")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train music transcription model")
    parser.add_argument("--root_dir", type=str, default="maestro-v2.0.0", help="Path to MAESTRO dataset root")
    parser.add_argument("--year", type=str, default=None, help="Subset year (e.g. 2017)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--subset_size", type=int, default=None, help="Limit dataset size for debugging")
    parser.add_argument("--save_every", type=int, default=10, help="Save model checkpoint every N epochs")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # print(torch.cuda.memory_summary(device=device, abbreviated=True)) GPU memory debugging

    # Output structure ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("outputs", timestamp)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    model_path = os.path.join(checkpoint_dir, "model_final.pth")
    loss_plot_path = os.path.join(logs_dir, "loss_curve.png")
    step_plot_path = os.path.join(logs_dir, "loss_per_step.png")

    # Data
    train_ds = MaestroDataset(args.root_dir, split="train", year=args.year, subset_size=args.subset_size)
    val_ds   = MaestroDataset(args.root_dir, split="validation", year=args.year, subset_size=args.subset_size)
    # test_ds  = MaestroDataset(args.root_dir, split="test", year=args.year)

    # Nicely formatted parameters.txt
    with open(os.path.join(logs_dir, "parameters.txt"), "w") as file:
        file.write("=== Training Parameters ===\n")
        file.write(f"Timestamp: {timestamp}\n")
        file.write(f"Device: {device}\n\n")
        for k, v in vars(args).items():
            file.write(f"{k:>15}: {v}\n")
        file.write(f"Training dataset size: {len(train_ds)}\n")
        file.write(f"Validation dataset size: {len(val_ds)}\n")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Model & optimizer
    model = TranscriptionModel(model_type="cnn_rnn", device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Logging
    train_losses, val_losses = [], []
    global_losses = []  # list of per-step lists
    log_txt_path = os.path.join(logs_dir, "training_log.txt")

    with open(log_txt_path, "w") as log_file:
        log_file.write(f"Training started: {timestamp}\n")
        log_file.write(f"Device: {device}\n")
        log_file.write(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}\n\n")

        # Training loop
        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")
            # print(torch.cuda.memory_summary(device=device, abbreviated=True)) #GPU memory debugging

            train_loss, step_losses = train_one_epoch(model, train_loader, optimizer, device)
            val_loss = evaluate(model, val_loader, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            global_losses.append(step_losses)

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            log_file.write(f"Epoch {epoch:02d}: Train={train_loss:.4f}, Val={val_loss:.4f}\n")
            log_file.flush()

            # Update training curve
            plot_training_curves(train_losses, val_losses, loss_plot_path)
            plot_step_losses(global_losses, args.epochs, step_plot_path)

            # Save checkpoint periodically
            if epoch % args.save_every == 0 or epoch == args.epochs:
                ckpt_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
                torch.save(model.state_dict(), ckpt_path)
                print(f"Checkpoint saved to {ckpt_path}")

        # Final save
        torch.save(model.state_dict(), model_path)
        print(f"\nFinal model saved to {model_path}")
        log_file.write("\nTraining complete.\n")


if __name__ == "__main__":
    main()