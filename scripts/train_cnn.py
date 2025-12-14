import argparse
import os
import sys
from datetime import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    parser = argparse.ArgumentParser(
        description="Train music transcription model",
        epilog="""
Examples:
  # Basic training with defaults
  python main.py --epochs 40 --batch_size 16

  # Training with custom model configuration
  python main.py --epochs 40 --n_mels 320 --hidden_size 512 --num_layers 3

  # Background training (no shell wrapper needed)
  python main.py --epochs 100 --batch_size 24 --background

  # Resume from checkpoint
  python main.py --resume outputs/2025-12-13_20-11-33/checkpoints/model_epoch_15.pth \\
    --epochs 100 --batch_size 24

  # Quick debug run
  python main.py --subset_size 50 --epochs 3 --chunk_length 30.0
""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Dataset arguments
    parser.add_argument("--root_dir", type=str, default="maestro-v3.0.0", help="Path to MAESTRO dataset root")
    parser.add_argument("--year", type=str, default=None, help="Subset year (e.g. 2017)")
    parser.add_argument("--cached_dir", type=str, default="cached_dataset", help="Cached dataset directory")
    parser.add_argument("--subset_size", type=int, default=None, help="Limit dataset size for debugging")
    parser.add_argument("--chunk_length", type=float, default=None, help="Chunk length in seconds (e.g. 30.0). If None, loads full files")
    parser.add_argument("--chunk_overlap", type=float, default=0.0, help="Overlap ratio between chunks (0.0-1.0, e.g. 0.25 for 25%% overlap)")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_every", type=int, default=10, help="Save model checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from (e.g. outputs/.../checkpoints/model_epoch_15.pth)")
    parser.add_argument("--start_epoch", type=int, default=1, help="Starting epoch number (auto-detected if resuming)")

    # Model configuration arguments
    parser.add_argument("--model", type=str, default="cnn_rnn", help="Type of model to use (cnn_rnn or cnn_rnn_large)")
    parser.add_argument("--n_mels", type=int, default=320, help="Number of mel bins (must match cached dataset if using cache)")
    parser.add_argument("--hidden_size", type=int, default=512, help="RNN hidden size (larger = more capacity)")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of RNN layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate for regularization")
    parser.add_argument("--use_attention", action="store_true", default=True, help="Use attention mechanism (only for cnn_rnn_large)")
    parser.add_argument("--no_attention", action="store_false", dest="use_attention", help="Disable attention mechanism")
    parser.add_argument("--use_onset_offset_heads", action="store_true", default=True, help="Use onset/offset detection heads (only for cnn_rnn_large)")
    parser.add_argument("--no_onset_offset_heads", action="store_false", dest="use_onset_offset_heads", help="Disable onset/offset detection heads")

    # Execution mode
    parser.add_argument("--background", action="store_true", help="Run training in background mode with log file")
    parser.add_argument("--log_file", type=str, default=None, help="Log file path for background mode (auto-generated if not specified)")
    parser.add_argument("--run_dir", type=str, default=None, help="Output directory for this training run (auto-generated if not specified)")

    args = parser.parse_args()

    # Handle background mode FIRST (before any other setup)
    if args.background:
        import sys
        import subprocess

        # Create output directory structure for this run
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join("outputs", timestamp)
        logs_dir = os.path.join(run_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        # Generate log filename in the run's logs directory
        log_file = args.log_file or os.path.join(logs_dir, "train.log")

        # Build command without --background flag but with --run_dir
        cmd_args = [sys.executable, "scripts/train_cnn.py"] + [
            arg for arg in sys.argv[1:] if arg != '--background'
        ] + ["--run_dir", run_dir]

        # Launch subprocess with output redirected to log file
        with open(log_file, 'w') as log:
            process = subprocess.Popen(
                cmd_args,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True
            )

        print(f"Training started in background (PID: {process.pid})")
        print(f"Output directory: {run_dir}")
        print(f"Log file: {log_file}")
        print(f"Monitor with: tail -f {log_file}")
        sys.exit(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Print chunking configuration
    if args.chunk_length:
        print(f"Chunking enabled: {args.chunk_length}s chunks with {args.chunk_overlap*100:.0f}% overlap")
    else:
        print("Chunking disabled: loading full files")

    # print(torch.cuda.memory_summary(device=device, abbreviated=True)) GPU memory debugging

    # Output structure ---
    # Use provided run_dir or create a new one
    if args.run_dir:
        run_dir = args.run_dir
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join("outputs", timestamp)

    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    model_path = os.path.join(checkpoint_dir, "model_final.pth")
    loss_plot_path = os.path.join(logs_dir, "loss_curve.png")
    step_plot_path = os.path.join(logs_dir, "loss_per_step.png")

    # Data - use cached dataset if available, otherwise fall back to raw
    from data.cached_dataset import HybridMaestroDataset

    train_ds = HybridMaestroDataset(
        root_dir=args.root_dir,
        cache_dir=args.cached_dir,
        split="train",
        year=args.year,
        subset_size=args.subset_size,
        chunk_length=args.chunk_length,
        overlap=args.chunk_overlap
    )
    val_ds = HybridMaestroDataset(
        root_dir=args.root_dir,
        cache_dir=args.cached_dir,
        split="validation",
        year=args.year,
        subset_size=args.subset_size,
        chunk_length=args.chunk_length,
        overlap=0.0  # No overlap for validation
    )
    # test_ds  = MaestroDataset(args.root_dir, split="test", year=args.year)

    # Validate n_mels matches cache if using cached data
    if train_ds.use_cache:
        import pickle
        metadata_path = os.path.join(args.cached_dir, 'train_metadata.pkl')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            cache_n_mels = metadata.get('n_mels')
            if cache_n_mels and cache_n_mels != args.n_mels:
                print("=" * 70)
                print("ERROR: Model/Cache n_mels Mismatch")
                print("=" * 70)
                print(f"Model n_mels:     {args.n_mels}")
                print(f"Cache n_mels:     {cache_n_mels}")
                print(f"\nThis will cause tensor shape errors during training.")
                print(f"Fix: Use --n_mels {cache_n_mels} or recreate cache with --n_mels {args.n_mels}")
                print("=" * 70)
                import sys
                sys.exit(1)

    # Nicely formatted parameters.txt
    with open(os.path.join(logs_dir, "parameters.txt"), "w") as file:
        file.write("=== Training Parameters ===\n")
        file.write(f"Timestamp: {timestamp}\n")
        file.write(f"Device: {device}\n\n")
        file.write(f"{'root_dir':>25}: {args.root_dir}\n")
        file.write(f"{'year':>25}: {args.year}\n")
        file.write(f"{'cached_dir':>25}: {args.cached_dir}\n")
        file.write(f"{'subset_size':>25}: {args.subset_size}\n")
        file.write(f"{'chunk_length':>25}: {args.chunk_length}\n")
        file.write(f"{'chunk_overlap':>25}: {args.chunk_overlap}\n")
        file.write(f"{'batch_size':>25}: {args.batch_size}\n")
        file.write(f"{'epochs':>25}: {args.epochs}\n")
        file.write(f"{'lr':>25}: {args.lr}\n")
        file.write(f"{'save_every':>25}: {args.save_every}\n")
        file.write(f"{'resume':>25}: {args.resume}\n")
        file.write(f"{'start_epoch':>25}: {args.start_epoch}\n")
        file.write(f"\n=== Model Configuration ===\n")
        file.write(f"{'Model type':>25}: {args.model}\n")
        file.write(f"{'n_mels':>25}: {args.n_mels}\n")
        file.write(f"{'Hidden size':>25}: {args.hidden_size}\n")
        file.write(f"{'Num layers':>25}: {args.num_layers}\n")
        file.write(f"{'Dropout':>25}: {args.dropout}\n")
        file.write(f"{'Use attention':>25}: {args.use_attention}\n")
        file.write(f"{'Use onset/offset heads':>25}: {args.use_onset_offset_heads}\n")
        file.write(f"\n=== Dataset Info ===\n")
        file.write(f"Training dataset size: {len(train_ds)} {'chunks' if args.chunk_length else 'files'}\n")
        file.write(f"Validation dataset size: {len(val_ds)} {'chunks' if args.chunk_length else 'files'}\n")
        file.write(f"Using cache: {train_ds.use_cache}\n")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,  # Balanced worker count
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2  # Prefetch 2 batches per worker
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # Model & optimizer
    model = TranscriptionModel(
        model_type=args.model,
        device=device,
        n_mels=args.n_mels,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_attention=args.use_attention,
        use_onset_offset_heads=args.use_onset_offset_heads
        )
    # Adam with epsilon for stability (especially important for large models)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=1e-5)

    # Resume from checkpoint if specified
    start_epoch = args.start_epoch
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming from checkpoint: {args.resume}")
            model.load_state_dict(torch.load(args.resume, map_location=device))

            # Auto-detect epoch from filename if not specified
            if args.start_epoch == 1:  # User didn't specify start_epoch
                import re
                match = re.search(r'epoch[_\-](\d+)', args.resume)
                if match:
                    start_epoch = int(match.group(1)) + 1
                    print(f"Auto-detected: resuming from epoch {start_epoch}")
            print(f"Loaded model weights from {args.resume}")
        else:
            print(f"Warning: Checkpoint not found at {args.resume}, starting from scratch")

    # Logging
    train_losses, val_losses = [], []
    global_losses = []  # list of per-step lists
    best_val_loss = float('inf')  # Track best validation loss
    log_txt_path = os.path.join(logs_dir, "training_log.txt")

    with open(log_txt_path, "w") as log_file:
        log_file.write(f"Training started: {timestamp}\n")
        log_file.write(f"Device: {device}\n")
        if args.resume:
            log_file.write(f"Resumed from: {args.resume}\n")
            log_file.write(f"Starting at epoch: {start_epoch}\n")
        log_file.write(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}\n\n")

        # Training loop
        for epoch in range(start_epoch, args.epochs + 1):
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

            # Save best model (based on lowest validation loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = os.path.join(checkpoint_dir, f"model_best.pth")
                torch.save(model.state_dict(), ckpt_path)
                print(f"✓ New best model saved! Val loss: {val_loss:.4f}")

        # Final save
        torch.save(model.state_dict(), model_path)
        print(f"\nFinal model saved to {model_path}")
        log_file.write("\nTraining complete.\n")


if __name__ == "__main__":
    main()