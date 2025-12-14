"""
Pre-process and cache the entire dataset to disk.
This eliminates I/O bottleneck during training.

Enhanced CLI with validation, background mode, and comprehensive options.
Run: python scripts/preprocess_dataset.py --help
"""

import os
import sys

# Add parent directory to path so we can import from data/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm
from data.dataset import MaestroDataset
import pickle
import subprocess
from datetime import datetime
import shutil
from multiprocessing import Pool


def _process_single_chunk(args):
    """
    Process a single chunk for parallel execution.

    Args:
        args: tuple of (idx, dataset_params, cache_path, force, return_waveform)

    Returns:
        tuple: (success: bool, skipped: bool)
    """
    idx, dataset_params, cache_path, force, return_waveform = args

    # Skip if already cached (unless force mode)
    if os.path.exists(cache_path) and not force:
        return (True, True)  # success, skipped

    try:
        # Create dataset instance (each worker needs its own)
        dataset = MaestroDataset(**dataset_params)

        # Load and process chunk
        data_tensor, roll_tensor = dataset[idx]

        # Save to disk with appropriate key
        if return_waveform:
            torch.save({
                'waveform': data_tensor,
                'roll': roll_tensor
            }, cache_path)
        else:
            torch.save({
                'mel': data_tensor,
                'roll': roll_tensor
            }, cache_path)

        return (True, False)  # success, not skipped
    except Exception as e:
        print(f"\nError processing chunk {idx}: {e}")
        return (False, False)  # failed


def preprocess_and_cache(
    root_dir="maestro-v3.0.0",
    cache_dir="cached_dataset",
    chunk_length=30.0,
    overlap=0.0,
    n_mels=229,
    sr=16000,
    hop_length=512,
    split='train',
    force=False,
    num_workers=1,
    return_waveform=False
):
    """
    Pre-process entire dataset and save to disk.

    This converts:
    - Raw audio files (slow to load/decode)
    - Raw MIDI files (slow to parse)
    - On-the-fly mel-spectrogram computation (CPU intensive)
    - On-the-fly piano roll computation (CPU intensive)

    Into:
    - Pre-computed tensors saved as .pt files (fast to load)

    Expected speedup: 10-50x during training!

    Args:
        num_workers: Number of parallel workers (1 = sequential, >1 = parallel)
    """

    print(f"Loading dataset metadata for {split} split...")
    data_type = "waveform" if return_waveform else "mel"
    print(f"Data type: {data_type}")

    dataset = MaestroDataset(
        root_dir=root_dir,
        split=split,
        chunk_length=chunk_length,
        overlap=overlap,
        n_mels=n_mels,
        sr=sr,
        hop_length=hop_length,
        return_waveform=return_waveform
    )

    print(f"Found {len(dataset)} chunks to preprocess")
    if num_workers > 1:
        print(f"Using {num_workers} parallel workers")

    # Create cache directory
    split_cache_dir = os.path.join(cache_dir, split)
    os.makedirs(split_cache_dir, exist_ok=True)

    # Save dataset metadata
    metadata = {
        'root_dir': root_dir,
        'chunk_length': chunk_length,
        'overlap': overlap,
        'split': split,
        'num_chunks': len(dataset),
        'chunks': dataset.chunks if hasattr(dataset, 'chunks') else None,
        'sr': sr,
        'n_mels': n_mels,
        'hop_length': hop_length,
        'return_waveform': return_waveform,
        'data_type': data_type
    }

    with open(os.path.join(cache_dir, f'{split}_metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    print(f"Preprocessing {len(dataset)} chunks...")
    print(f"Cache directory: {split_cache_dir}")

    # Prepare arguments for parallel processing
    dataset_params = {
        'root_dir': root_dir,
        'split': split,
        'chunk_length': chunk_length,
        'overlap': overlap,
        'n_mels': n_mels,
        'sr': sr,
        'hop_length': hop_length,
        'return_waveform': return_waveform
    }

    chunk_args = [
        (idx, dataset_params, os.path.join(split_cache_dir, f'chunk_{idx:06d}.pt'), force, return_waveform)
        for idx in range(len(dataset))
    ]

    # Process chunks
    skipped = 0
    cached = 0
    failed = 0

    if num_workers > 1:
        # Parallel processing
        with Pool(processes=num_workers) as pool:
            with tqdm(total=len(dataset),
                      desc=f"Caching {split}",
                      unit="chunk",
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:

                for success, was_skipped in pool.imap_unordered(_process_single_chunk, chunk_args):
                    if success:
                        if was_skipped:
                            skipped += 1
                        else:
                            cached += 1
                    else:
                        failed += 1

                    pbar.update(1)
                    pbar.set_postfix({'cached': cached, 'skipped': skipped, 'failed': failed})
    else:
        # Sequential processing (original behavior)
        with tqdm(total=len(dataset),
                  desc=f"Caching {split}",
                  unit="chunk",
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:

            for idx in range(len(dataset)):
                cache_path = os.path.join(split_cache_dir, f'chunk_{idx:06d}.pt')

                # Skip if already cached (unless force mode)
                if os.path.exists(cache_path) and not force:
                    skipped += 1
                    pbar.update(1)
                    pbar.set_postfix({'cached': cached, 'skipped': skipped})
                    continue

                try:
                    # Load and process chunk (this is the slow part)
                    data_tensor, roll_tensor = dataset[idx]

                    # Save to disk with appropriate key
                    if return_waveform:
                        torch.save({
                            'waveform': data_tensor,
                            'roll': roll_tensor
                        }, cache_path)
                    else:
                        torch.save({
                            'mel': data_tensor,
                            'roll': roll_tensor
                        }, cache_path)

                    cached += 1
                except Exception as e:
                    print(f"\nError processing chunk {idx}: {e}")
                    failed += 1

                pbar.update(1)
                pbar.set_postfix({'cached': cached, 'skipped': skipped, 'failed': failed})

    print(f"\nPreprocessing complete for {split} split!")
    print(f"  Cached: {cached} chunks")
    print(f"  Skipped: {skipped} chunks (already existed)")
    if failed > 0:
        print(f"  Failed: {failed} chunks (errors occurred)")
    print(f"  Total size: ~{estimate_cache_size(split_cache_dir):.1f} GB")


def estimate_cache_size(cache_dir):
    """Estimate total cache size in GB."""
    if not os.path.exists(cache_dir):
        return 0.0
    total_size = 0
    for f in os.listdir(cache_dir):
        if f.endswith('.pt'):
            total_size += os.path.getsize(os.path.join(cache_dir, f))
    return total_size / (1024**3)


def validate_arguments(args):
    """Validate all arguments before starting preprocessing."""
    errors = []
    warnings = []

    # 1. Check dataset directory exists
    if not os.path.exists(args.root_dir):
        errors.append(f"Dataset directory not found: {args.root_dir}")
    else:
        csv_path = os.path.join(args.root_dir, "maestro-v3.0.0.csv")
        if not os.path.exists(csv_path):
            errors.append(f"Dataset CSV not found: {csv_path}")
            errors.append("  Make sure you've downloaded the complete MAESTRO dataset")

    # 2. Validate numeric ranges
    if args.n_mels < 1 or args.n_mels > 512:
        errors.append(f"Invalid n_mels: {args.n_mels} (must be 1-512)")

    if args.sr <= 0:
        errors.append(f"Invalid sample rate: {args.sr} (must be positive)")
    elif args.sr not in [8000, 16000, 22050, 44100, 48000]:
        warnings.append(f"Unusual sample rate: {args.sr} (common values: 16000, 22050, 44100)")

    if args.hop_length < 1:
        errors.append(f"Invalid hop_length: {args.hop_length} (must be positive)")
    elif args.hop_length > args.sr:
        warnings.append(f"hop_length ({args.hop_length}) is larger than sample rate ({args.sr})")

    if args.chunk_length <= 0:
        errors.append(f"Chunk length must be positive: {args.chunk_length}")

    if args.overlap < 0 or args.overlap >= 1.0:
        errors.append(f"Overlap must be in [0.0, 1.0): {args.overlap}")

    # 3. Check cache directory conflicts
    if os.path.exists(args.cache_dir) and not args.force:
        metadata_path = os.path.join(args.cache_dir, 'train_metadata.pkl')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'rb') as f:
                    existing = pickle.load(f)

                # Check compatibility
                conflicts = []
                if existing.get('n_mels') != args.n_mels:
                    conflicts.append(f"n_mels: cache={existing.get('n_mels')}, requested={args.n_mels}")
                if existing.get('sr') != args.sr:
                    conflicts.append(f"sr: cache={existing.get('sr')}, requested={args.sr}")
                if existing.get('hop_length') != args.hop_length:
                    conflicts.append(f"hop_length: cache={existing.get('hop_length')}, requested={args.hop_length}")
                if existing.get('chunk_length') != args.chunk_length:
                    conflicts.append(f"chunk_length: cache={existing.get('chunk_length')}, requested={args.chunk_length}")
                if existing.get('overlap') != args.overlap:
                    conflicts.append(f"overlap: cache={existing.get('overlap')}, requested={args.overlap}")

                if conflicts:
                    errors.append(
                        f"Cache directory exists with incompatible settings:\n" +
                        "\n".join(f"  - {c}" for c in conflicts) +
                        "\n\nOptions:" +
                        "\n  1. Use --force to overwrite" +
                        "\n  2. Use a different --cache_dir" +
                        "\n  3. Match existing cache parameters"
                    )
            except Exception as e:
                warnings.append(f"Could not read existing cache metadata: {e}")

    # 4. Disk space check
    if not errors:
        try:
            stat = shutil.disk_usage(os.path.dirname(args.cache_dir) or '.')
            free_gb = stat.free / (1024**3)

            # Rough estimate: ~1.5GB per second of audio across all splits
            estimated_gb = args.chunk_length * 1.5 * 50  # ~50 chunks estimate

            if free_gb < estimated_gb:
                warnings.append(
                    f"Low disk space: {free_gb:.1f}GB free, estimated need: ~{estimated_gb:.1f}GB"
                )
        except Exception:
            pass

    return errors, warnings


def print_cache_info(cache_dir):
    """Display detailed information about an existing cache."""
    if not os.path.exists(cache_dir):
        print(f"Error: Cache directory not found: {cache_dir}")
        return

    print("=" * 70)
    print(f"CACHE INFO: {cache_dir}")
    print("=" * 70)

    total_size = 0
    total_chunks = 0
    split_info = []

    for split in ['train', 'validation', 'test']:
        metadata_path = os.path.join(cache_dir, f'{split}_metadata.pkl')
        if not os.path.exists(metadata_path):
            continue

        try:
            with open(metadata_path, 'rb') as f:
                meta = pickle.load(f)

            split_dir = os.path.join(cache_dir, split)
            split_size = estimate_cache_size(split_dir)
            total_size += split_size
            total_chunks += meta['num_chunks']

            split_info.append({
                'split': split,
                'chunks': meta['num_chunks'],
                'size': split_size,
                'meta': meta
            })
        except Exception as e:
            print(f"Warning: Could not read {split} metadata: {e}")

    if not split_info:
        print("No cache metadata found.")
        print("=" * 70)
        return

    # Display split information
    for info in split_info:
        print(f"\n{info['split'].upper()}:")
        print(f"  Chunks:       {info['chunks']}")
        print(f"  Size:         {info['size']:.2f} GB")

    # Display parameters from first available metadata
    first_meta = split_info[0]['meta']
    print("\nPARAMETERS:")
    print(f"  n_mels:       {first_meta.get('n_mels', 'N/A')}")
    print(f"  sr:           {first_meta.get('sr', 'N/A')}")
    print(f"  hop_length:   {first_meta.get('hop_length', 'N/A')}")
    print(f"  chunk_length: {first_meta.get('chunk_length', 'N/A')}s")
    print(f"  overlap:      {first_meta.get('overlap', 0) * 100:.1f}%")

    print("\nTOTAL:")
    print(f"  Chunks:       {total_chunks}")
    print(f"  Size:         {total_size:.2f} GB")
    print("=" * 70)


def verify_cache(cache_dir):
    """Verify cache integrity after creation."""
    print("\nVerifying cache integrity...")
    issues = []

    for split in ['train', 'validation', 'test']:
        metadata_path = os.path.join(cache_dir, f'{split}_metadata.pkl')
        if not os.path.exists(metadata_path):
            continue

        try:
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)

            split_dir = os.path.join(cache_dir, split)
            if not os.path.exists(split_dir):
                issues.append(f"{split}: split directory not found")
                continue

            expected = metadata['num_chunks']

            # Count actual cache files
            actual = len([f for f in os.listdir(split_dir) if f.endswith('.pt')])

            if actual != expected:
                issues.append(f"{split}: expected {expected} chunks, found {actual}")

            # Spot check first chunk
            first_chunk = os.path.join(split_dir, 'chunk_000000.pt')
            if os.path.exists(first_chunk):
                try:
                    data = torch.load(first_chunk, map_location='cpu')
                    if 'mel' not in data or 'roll' not in data:
                        issues.append(f"{split}: invalid chunk format in chunk_000000.pt")
                except Exception as e:
                    issues.append(f"{split}: cannot load chunk_000000.pt: {e}")
        except Exception as e:
            issues.append(f"{split}: cannot read metadata: {e}")

    if issues:
        print("Verification FAILED - Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("Verification PASSED - Cache is valid!")
        return True


def show_dry_run_preview(args, splits):
    """Display what would be created without actually processing."""
    print("=" * 70)
    print("MAESTRO Dataset Preprocessing - DRY RUN")
    print("=" * 70)

    print("\nSOURCE CONFIGURATION:")
    print(f"  Dataset directory: {args.root_dir}")
    print(f"  Splits to process: {', '.join(splits)}")

    print("\nAUDIO PARAMETERS:")
    print(f"  Sample rate:       {args.sr} Hz")
    print(f"  Mel bins:          {args.n_mels} (n_mels)")
    print(f"  Hop length:        {args.hop_length} samples")
    print(f"  Chunk length:      {args.chunk_length} seconds")
    print(f"  Overlap:           {args.overlap * 100:.1f}% ({args.overlap})")

    print("\nOUTPUT CONFIGURATION:")
    print(f"  Cache directory:   {args.cache_dir}")
    print(f"  Force overwrite:   {'Yes' if args.force else 'No'}")

    # Try to estimate output
    print("\nESTIMATED OUTPUT:")
    try:
        for split in splits:
            try:
                dataset = MaestroDataset(
                    root_dir=args.root_dir,
                    split=split,
                    chunk_length=args.chunk_length,
                    overlap=args.overlap,
                    n_mels=args.n_mels,
                    sr=args.sr,
                    hop_length=args.hop_length
                )
                num_chunks = len(dataset)
                # Rough estimate: each chunk ~30MB
                estimated_gb = (num_chunks * 30) / 1024
                print(f"  {split:12s}: ~{num_chunks:5d} chunks (~{estimated_gb:5.1f} GB)")
            except Exception as e:
                print(f"  {split:12s}: Could not estimate ({e})")
    except Exception:
        print("  Could not estimate (dataset not accessible)")

    print("\nCOMPATIBILITY:")
    print(f"  Models must use: n_mels={args.n_mels}, sr={args.sr}, hop_length={args.hop_length}")

    print("\n" + "=" * 70)
    print("This is a DRY RUN - no files will be created.")
    print("Run without --dry_run to proceed with preprocessing.")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess and cache MAESTRO dataset for faster training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Cache mel spectrograms for CNN-RNN (default)
  python preprocess_dataset.py --n_mels 229 -j 16

  # Cache waveforms for AST/transformer models
  python preprocess_dataset.py --waveform -j 16

  # Custom mel bins (IMPORTANT: must match your model!)
  python preprocess_dataset.py --n_mels 320 -j 16

  # Run in background with automatic logging and parallel processing
  python preprocess_dataset.py --waveform -j 16 --background

  # Preview what will be created
  python preprocess_dataset.py --waveform --dry_run

  # Process only specific splits
  python preprocess_dataset.py --waveform --splits train,validation

  # Show existing cache info
  python preprocess_dataset.py --show_cache_info cached_dataset_waveform

  # Force overwrite existing cache
  python preprocess_dataset.py --waveform --force -j 16

IMPORTANT:
  The n_mels parameter MUST match your model configuration!
  For example, if your model uses TranscriptionModel(n_mels=320),
  you must preprocess with --n_mels 320.

  Cache directory is auto-named based on n_mels to prevent conflicts:
    n_mels=229  -> cached_dataset/
    n_mels=320  -> cached_dataset_mels320/
"""
    )

    # Dataset paths
    parser.add_argument(
        "--root_dir", type=str, default="maestro-v3.0.0",
        help="Path to MAESTRO dataset root directory (default: maestro-v3.0.0)"
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None,
        help="Output cache directory (auto-generated based on n_mels if not specified)"
    )

    # Audio parameters (CRITICAL - must match model)
    parser.add_argument(
        "--n_mels", type=int, default=229,
        help="Number of mel frequency bins - MUST match your model! (default: 229)"
    )
    parser.add_argument(
        "--sr", type=int, default=16000,
        help="Audio sample rate in Hz - MUST match your model! (default: 16000)"
    )
    parser.add_argument(
        "--hop_length", type=int, default=512,
        help="STFT hop length in samples - MUST match your model! (default: 512)"
    )

    # Chunking parameters
    parser.add_argument(
        "--chunk_length", type=float, default=30.0,
        help="Duration of each chunk in seconds (default: 30.0)"
    )
    parser.add_argument(
        "--overlap", type=float, default=0.0,
        help="Overlap ratio between chunks [0.0, 1.0), e.g., 0.25 = 25%% overlap (default: 0.0)"
    )

    # Data type selection
    data_type_group = parser.add_mutually_exclusive_group()
    data_type_group.add_argument(
        "--waveform", action="store_true",
        help="Cache raw waveforms for AST/transformer models (default: False, cache mel spectrograms)"
    )
    data_type_group.add_argument(
        "--mel", action="store_true",
        help="Cache mel spectrograms for CNN-RNN models (default behavior)"
    )

    # Processing options
    parser.add_argument(
        "--splits", type=str, default="train,validation,test",
        help="Comma-separated list of splits to process (default: train,validation,test)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing cache files (default: False)"
    )
    parser.add_argument(
        "-j", "--num_workers", type=int, default=1,
        help="Number of parallel workers for preprocessing (default: 1). Use -j 16 for 16 workers. "
             "Recommended: use number of CPU cores or less."
    )

    # Background execution
    parser.add_argument(
        "--background", action="store_true",
        help="Run preprocessing in background with automatic logging (default: False)"
    )
    parser.add_argument(
        "--log_file", type=str, default=None,
        help="Custom log file path (only with --background)"
    )

    # Validation and info
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Show what would be created without actually processing (default: False)"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify cache integrity after creation (default: False)"
    )
    parser.add_argument(
        "--show_cache_info", type=str, default=None, metavar="CACHE_DIR",
        help="Display information about an existing cache and exit"
    )

    args = parser.parse_args()

    # Handle special read-only mode: show cache info
    if args.show_cache_info:
        print_cache_info(args.show_cache_info)
        sys.exit(0)

    # Handle background mode - re-launch process
    if args.background:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = args.log_file or f"preprocess_{timestamp}.log"

        # Re-run same command without --background flag
        cmd_args = [sys.executable] + [arg for arg in sys.argv if arg != '--background']

        print("=" * 70)
        print("Starting preprocessing in background...")
        print("=" * 70)

        with open(log_file, 'w') as log:
            process = subprocess.Popen(
                cmd_args,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True
            )

        print(f"Preprocessing started in background")
        print(f"PID: {process.pid}")
        print(f"Log file: {log_file}")
        print(f"\nMonitor progress with:")
        print(f"  tail -f {log_file}")
        print(f"\nCheck if running:")
        print(f"  ps aux | grep {process.pid}")
        print("=" * 70)
        sys.exit(0)

    # Determine data type
    return_waveform = args.waveform  # Default is False (mel spectrograms)

    # Auto-generate cache directory name based on data type and n_mels
    if args.cache_dir is None:
        if return_waveform:
            args.cache_dir = "cached_dataset_waveform"
        elif args.n_mels == 229:
            args.cache_dir = "cached_dataset"  # Default for backward compatibility
        else:
            args.cache_dir = f"cached_dataset_mels{args.n_mels}"

    # Parse splits
    splits = [s.strip() for s in args.splits.split(',')]
    valid_splits = ['train', 'validation', 'test']
    for split in splits:
        if split not in valid_splits:
            print(f"Error: Invalid split '{split}'. Must be one of: {', '.join(valid_splits)}")
            sys.exit(1)

    # Validate arguments
    errors, warnings = validate_arguments(args)

    if errors:
        print("=" * 70)
        print("VALIDATION ERRORS")
        print("=" * 70)
        for error in errors:
            print(f"\n{error}")
        print("\n" + "=" * 70)
        print("Please fix the errors above and try again.")
        sys.exit(1)

    if warnings:
        print("=" * 70)
        print("VALIDATION WARNINGS")
        print("=" * 70)
        for warning in warnings:
            print(f"  - {warning}")
        print("=" * 70)
        print()

    # Handle dry-run mode
    if args.dry_run:
        show_dry_run_preview(args, splits)
        sys.exit(0)

    # Display header
    print("=" * 70)
    print("MAESTRO Dataset Preprocessing")
    print("=" * 70)
    print(f"Source:        {args.root_dir}")
    print(f"Cache:         {args.cache_dir}")
    print(f"Splits:        {', '.join(splits)}")
    print(f"Data type:     {'Waveform (AST)' if return_waveform else f'Mel spectrogram (CNN-RNN, n_mels={args.n_mels})'}")
    print(f"Chunk length:  {args.chunk_length}s")
    print(f"Overlap:       {args.overlap * 100:.0f}%")
    print(f"Sample rate:   {args.sr}")
    print(f"Hop length:    {args.hop_length}")
    print(f"Workers:       {args.num_workers} {'(parallel)' if args.num_workers > 1 else '(sequential)'}")
    print(f"Force:         {'Yes' if args.force else 'No'}")
    print("=" * 70)
    print()

    # Process each split
    total_cached = 0
    total_skipped = 0
    split_sizes = {}

    for split in splits:
        print(f"\n{'='*70}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*70}\n")

        try:
            preprocess_and_cache(
                root_dir=args.root_dir,
                cache_dir=args.cache_dir,
                chunk_length=args.chunk_length,
                overlap=args.overlap,
                n_mels=args.n_mels,
                sr=args.sr,
                hop_length=args.hop_length,
                split=split,
                force=args.force,
                num_workers=args.num_workers,
                return_waveform=return_waveform
            )

            # Track size for summary
            split_dir = os.path.join(args.cache_dir, split)
            split_sizes[split] = estimate_cache_size(split_dir)

        except Exception as e:
            print(f"Error: Could not process {split} split: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE!")
    print("=" * 70)

    print(f"\nCACHE SUMMARY:")
    print(f"  Location:      {args.cache_dir}")

    total_size = sum(split_sizes.values())
    print(f"\nSPLIT BREAKDOWN:")
    for split, size in split_sizes.items():
        # Count chunks
        split_dir = os.path.join(args.cache_dir, split)
        if os.path.exists(split_dir):
            num_chunks = len([f for f in os.listdir(split_dir) if f.endswith('.pt')])
            print(f"  {split:12s}: {num_chunks:5d} chunks ({size:5.1f} GB)")

    print(f"\nTOTAL SIZE:    {total_size:.1f} GB")

    print(f"\nCACHE PARAMETERS:")
    print(f"  n_mels:        {args.n_mels}")
    print(f"  sr:            {args.sr}")
    print(f"  hop_length:    {args.hop_length}")
    print(f"  chunk_length:  {args.chunk_length}s")
    print(f"  overlap:       {args.overlap}")

    print(f"\nUSAGE:")
    print(f"  Train with:")
    print(f"    python main.py --cached_dir {args.cache_dir}")
    print(f"\n  Verify cache:")
    print(f"    python preprocess_dataset.py --show_cache_info {args.cache_dir}")

    print(f"\nIMPORTANT:")
    print(f"  Your model MUST use these parameters:")
    print(f"    TranscriptionModel(n_mels={args.n_mels})")
    print(f"  These parameters are validated by HybridMaestroDataset.")

    print("=" * 70)

    # Verify cache if requested
    if args.verify:
        verify_cache(args.cache_dir)
