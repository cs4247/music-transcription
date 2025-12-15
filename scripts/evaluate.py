"""
Unified evaluation CLI for music transcription models.
Consolidates evaluate_model.py and threshold tuning scripts into a single interface.

Enhanced features:
- Auto-detection of model config from cache metadata
- Comprehensive validation and safety checks
- Dry-run preview mode
- Threshold tuning with binary search
- Background execution support
- Results inspection tools

Examples:
  # Basic evaluation
  python scripts/evaluate.py --model outputs/model.pth

  # Quick validation check
  python scripts/evaluate.py --model outputs/model.pth \\
    --split validation --subset 100 --headless

  # Threshold tuning
  python scripts/evaluate.py --model outputs/model.pth \\
    --tune_threshold --split validation --subset 50

  # Dry run preview
  python scripts/evaluate.py --model outputs/model.pth --dry_run

  # Show existing results
  python scripts/evaluate.py --show_results eval_outputs/2025-12-13_20-11-33

  # Background mode
  python scripts/evaluate.py --model outputs/model.pth --background
"""

import os
import sys
import copy
import argparse
from datetime import datetime
import subprocess
import pickle

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Defer heavy imports until after argparse to make --help fast
# These will be imported in main() after argument parsing


# ============================================================================
# Utility Functions
# ============================================================================

def pianoroll_to_midi(pianoroll, fs, min_midi=21):
    """
    Convert a (88, T) binary piano roll to a pretty_midi.PrettyMIDI object.

    Args:
        pianoroll: np.ndarray of shape (88, T), values {0,1}
        fs: frames per second (sr / hop_length)
        min_midi: MIDI note number for index 0 in the roll (default 21 = A0)
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


def clean_filename(s):
    """Make a safe filename from title/composer."""
    bad_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for c in bad_chars:
        s = s.replace(c, '_')
    return s


# ============================================================================
# Auto-detection Functions
# ============================================================================

def detect_data_source(args):
    """
    Detect whether to use cache or full files.
    Returns: ('cache', cache_dir) or ('full', root_dir)
    """
    if args.data_source == 'cache':
        return 'cache', args.cache_dir
    elif args.data_source == 'full':
        return 'full', args.root_dir
    else:  # auto
        # Check if cache exists
        metadata_path = os.path.join(args.cache_dir, f'{args.split}_metadata.pkl')
        cache_exists = os.path.exists(metadata_path)

        if cache_exists:
            if not args.headless:
                print(f"Auto-detected cache: {args.cache_dir}")
            return 'cache', args.cache_dir
        elif os.path.exists(args.root_dir):
            if not args.headless:
                print(f"Cache not found, using full files: {args.root_dir}")
            return 'full', args.root_dir
        else:
            return None, None


def extract_model_config(args, data_source=None):
    """
    Extract model configuration from cache metadata or use args/defaults.
    """
    config = {
        'n_mels': args.n_mels,
        'model_type': args.model_type,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'sr': 16000,
        'hop_length': 512
    }

    # Try to auto-detect from cache metadata
    if data_source == 'cache' or (data_source == 'auto' and args.data_source == 'auto'):
        metadata_path = os.path.join(args.cache_dir, f'{args.split}_metadata.pkl')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)

                # Auto-detect n_mels if not explicitly set
                if args.n_mels is None:
                    config['n_mels'] = metadata.get('n_mels', 320)
                    if not args.headless:
                        print(f"Auto-detected n_mels={config['n_mels']} from cache metadata")

                # Always read sr and hop_length from cache
                config['sr'] = metadata.get('sr', 16000)
                config['hop_length'] = metadata.get('hop_length', 512)
            except Exception as e:
                if not args.headless:
                    print(f"Warning: Could not read cache metadata: {e}")

    # Apply defaults if still None
    if config['n_mels'] is None:
        config['n_mels'] = 320

    return config


# ============================================================================
# Validation Function
# ============================================================================

def validate_arguments(args):
    """Validate all arguments before execution."""
    errors = []
    warnings = []

    # 1. Model file exists
    if not args.show_results and not os.path.exists(args.model):
        errors.append(f"Model checkpoint not found: {args.model}")

    # 2. Threshold range
    if args.threshold < 0 or args.threshold > 1:
        errors.append(f"Threshold must be in [0, 1]: {args.threshold}")

    # 3. Data source validation (skip if showing results)
    if not args.show_results:
        data_source, path = detect_data_source(args)

        if data_source is None:
            errors.append(
                f"Neither cache ({args.cache_dir}) nor dataset ({args.root_dir}) found.\n"
                "  Run: python scripts/preprocess_dataset.py\n"
                "  Or ensure MAESTRO dataset is downloaded"
            )
        elif data_source == 'cache':
            metadata_path = os.path.join(args.cache_dir, f'{args.split}_metadata.pkl')
            if os.path.exists(metadata_path) and args.n_mels is not None:
                try:
                    with open(metadata_path, 'rb') as f:
                        meta = pickle.load(f)

                    if meta.get('n_mels') != args.n_mels:
                        errors.append(
                            f"Model n_mels ({args.n_mels}) does not match cache n_mels ({meta.get('n_mels')})\n"
                            "  Options:\n"
                            f"    1. Use --n_mels {meta.get('n_mels')} (match cache)\n"
                            "    2. Use --data_source full (use raw files)\n"
                            "    3. Regenerate cache with correct n_mels"
                        )
                except Exception:
                    pass

    # 4. Threshold tuning validation
    if args.tune_threshold:
        if args.tune_rounds < 1 or args.tune_rounds > 20:
            errors.append(f"tune_rounds must be 1-20: {args.tune_rounds}")

        tune_min, tune_max = args.tune_range
        if tune_min < 0 or tune_max > 1 or tune_min >= tune_max:
            errors.append(f"Invalid tune_range: {args.tune_range} (must be 0 <= min < max <= 1)")

        if not args.subset:
            warnings.append(
                "Threshold tuning without --subset may be slow.\n"
                "  Recommend: --subset 50 (for faster tuning)"
            )

    # 5. Show results validation
    if args.show_results:
        if not os.path.exists(args.show_results):
            errors.append(f"Results directory not found: {args.show_results}")

    return errors, warnings


# ============================================================================
# Model and Data Loading
# ============================================================================

def load_model_and_data(args, config):
    """
    Load model and create dataset/dataloader.
    Returns: (model, dataloader, dataset_info dict)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.headless:
        print(f"Using device: {device}")

    # Create model
    model = TranscriptionModel(
        model_type=config['model_type'],
        device=device,
        n_mels=config['n_mels'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )

    # Load checkpoint
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Determine data source
    data_source, path = detect_data_source(args)

    # Load dataset
    if data_source == 'cache':
        if not args.headless:
            print(f"Loading cached dataset from: {args.cache_dir}")

        # Suppress dataset print statements in headless mode
        from io import StringIO
        if args.headless:
            old_stdout = sys.stdout
            sys.stdout = StringIO()

        test_ds = CachedMaestroDataset(
            cache_dir=args.cache_dir,
            split=args.split,
        )

        if args.headless:
            sys.stdout = old_stdout

        # Get sr and hop_length from metadata
        fs = config['sr'] / config['hop_length']

        # Apply subset if requested
        if args.subset is not None:
            indices = list(range(min(args.subset, len(test_ds))))
            test_ds = Subset(test_ds, indices)
            if not args.headless:
                print(f"Using subset of {len(test_ds)} chunks")
    else:
        if not args.headless:
            print(f"Loading full-file dataset from: {args.root_dir}")
        test_ds = MaestroDataset(
            root_dir=args.root_dir,
            split=args.split,
            year=args.year,
            subset_size=args.subset,
            n_mels=config['n_mels'],
            sr=config['sr'],
            hop_length=config['hop_length']
        )
        fs = config['sr'] / config['hop_length']

    # Create dataloader
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    dataset_info = {
        'dataset': test_ds,
        'fs': fs,
        'data_source': data_source,
        'num_samples': len(test_ds),
        'device': device
    }

    return model, test_loader, dataset_info


# ============================================================================
# Evaluation Functions
# ============================================================================

def run_evaluation(args, model, dataloader, dataset_info):
    """
    Run evaluation loop.
    Returns: results dict with mean_f1, per_sample_f1, etc.
    """
    device = dataset_info['device']
    test_ds = dataset_info['dataset']
    fs = dataset_info['fs']

    all_f1 = []
    all_preds = []  # Collect all predictions for confusion matrix
    all_targets = []  # Collect all targets for confusion matrix

    if args.headless:
        # Headless mode: minimal output, no file writing
        with torch.no_grad():
            for idx, (mel, roll, lengths) in enumerate(dataloader):
                mel = mel.to(device)
                roll = roll.to(device)
                lengths = lengths.to(device)

                # Forward pass
                logits = model(mel)
                probs = torch.sigmoid(logits)
                preds = (probs > args.threshold).float()

                B, P, T = preds.shape
                assert B == 1, "batch_size=1 required for per-sample evaluation"

                L = lengths[0].item()
                preds_valid = preds[0, :, :L].detach().cpu().numpy()
                target_valid = roll[0, :, :L].detach().cpu().numpy()

                # Flatten for framewise F1
                y_pred = preds_valid.flatten()
                y_true = target_valid.flatten()

                f1 = f1_score(y_true, y_pred, zero_division=0)
                all_f1.append(f1)

                all_preds.append(y_pred)
                all_targets.append(y_true)

        mean_f1 = float(np.mean(all_f1)) if all_f1 else 0.0
        print(f"EVAL_MEAN_F1={mean_f1:.6f}")

        return {'mean_f1': mean_f1, 'all_f1': all_f1}

    else:
        # Normal mode: full output with files
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join(args.out_dir, timestamp)
        midi_dir = os.path.join(run_dir, "midis") if not args.no_midi else None

        if not args.no_midi:
            os.makedirs(midi_dir, exist_ok=True)
        else:
            os.makedirs(run_dir, exist_ok=True)

        summary_path = os.path.join(run_dir, "eval_summary.txt")
        print(f"Saving evaluation outputs to: {run_dir}")

        song_infos = []  # (idx, display_name, f1, midi_path)
        has_metadata = hasattr(test_ds, 'df') or (hasattr(test_ds, 'dataset') and hasattr(test_ds.dataset, 'df'))

        with torch.no_grad(), open(summary_path, "w") as summary_file:
            summary_file.write("=== Evaluation Summary ===\n")
            summary_file.write(f"Timestamp: {timestamp}\n")
            summary_file.write(f"Split: {args.split}\n")
            summary_file.write(f"Year filter: {args.year}\n")
            summary_file.write(f"Model: {args.model}\n")
            summary_file.write(f"Threshold: {args.threshold}\n")
            summary_file.write(f"Mode: {dataset_info['data_source']}\n")
            summary_file.write(f"Num examples: {dataset_info['num_samples']}\n\n")

            # Progress bar for normal mode
            pbar = tqdm(dataloader, desc="Evaluating", unit="sample")
            for idx, (mel, roll, lengths) in enumerate(pbar):
                mel = mel.to(device)
                roll = roll.to(device)
                lengths = lengths.to(device)

                # Forward pass
                logits = model(mel)
                probs = torch.sigmoid(logits)
                preds = (probs > args.threshold).float()

                B, P, T = preds.shape
                assert B == 1, "batch_size=1 required for per-sample evaluation"

                L = lengths[0].item()
                preds_valid = preds[0, :, :L].detach().cpu().numpy()
                target_valid = roll[0, :, :L].detach().cpu().numpy()

                # Flatten for framewise F1
                y_pred = preds_valid.flatten()
                y_true = target_valid.flatten()

                f1 = f1_score(y_true, y_pred, zero_division=0)
                all_f1.append(f1)

                all_preds.append(y_pred)
                all_targets.append(y_true)

                # Get metadata for naming (if available)
                if has_metadata:
                    ds = test_ds.dataset if hasattr(test_ds, 'dataset') else test_ds
                    row = ds.df.iloc[idx]
                    title = str(row.get("canonical_title", row.get("audio_filename", f"piece_{idx}")))
                    composer = str(row.get("canonical_composer", "Unknown"))
                    display_name = f"{composer} - {title}"
                else:
                    display_name = f"chunk_{idx:06d}"
                    title = display_name

                # Save MIDI (unless --no_midi)
                midi_path = None
                if not args.no_midi:
                    midi = pianoroll_to_midi(preds_valid, fs=fs, min_midi=21)
                    safe_name = clean_filename(f"{idx:04d}_{title}.mid")
                    midi_path = os.path.join(midi_dir, safe_name)
                    midi.write(midi_path)

                song_infos.append((idx, display_name, f1, midi_path))

                pbar.set_postfix({'F1': f'{f1:.4f}'})
                summary_file.write(f"{idx:04d}  F1={f1:.4f}  {display_name}\n")

            # Aggregate stats
            mean_f1 = float(np.mean(all_f1)) if all_f1 else 0.0
            summary_file.write("\n=== Aggregate ===\n")
            summary_file.write(f"Mean framewise F1: {mean_f1:.4f}\n")

            if song_infos:
                best = max(song_infos, key=lambda x: x[2])
                worst = min(song_infos, key=lambda x: x[2])

                summary_file.write("\n=== Best ===\n")
                summary_file.write(f"F1={best[2]:.4f}  {best[1]}\n")
                if best[3]:
                    summary_file.write(f"MIDI: {best[3]}\n")

                summary_file.write("\n=== Worst ===\n")
                summary_file.write(f"F1={worst[2]:.4f}  {worst[1]}\n")
                if worst[3]:
                    summary_file.write(f"MIDI: {worst[3]}\n")

            print("\n=== Evaluation complete ===")
            print(f"Mean framewise F1: {mean_f1:.4f}")
            if song_infos:
                print(f"Best:  F1={best[2]:.4f}  {best[1]}")
                print(f"Worst: F1={worst[2]:.4f}  {worst[1]}")
            print(f"Summary written to: {summary_path}")
            if not args.no_midi:
                print(f"MIDIs written to:   {midi_dir}")

        # Generate and save confusion matrix
        if all_preds and all_targets:
            print("Generating confusion matrix...")
            all_preds_flat = np.concatenate(all_preds)
            all_targets_flat = np.concatenate(all_targets)

            cm = confusion_matrix(all_targets_flat, all_preds_flat, labels=[0, 1])

            # Create figure
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Predicted: 0 (Off)', 'Predicted: 1 (On)'],
                       yticklabels=['Actual: 0 (Off)', 'Actual: 1 (On)'])
            plt.title(f'Confusion Matrix (Threshold={args.threshold})\nMean F1: {mean_f1:.4f}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()

            # Save figure
            cm_path = os.path.join(run_dir, "confusion_matrix.png")
            plt.savefig(cm_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Confusion matrix saved to: {cm_path}")

        return {
            'mean_f1': mean_f1,
            'all_f1': all_f1,
            'song_infos': song_infos,
            'run_dir': run_dir
        }


def evaluate_at_threshold(model, dataloader, dataset_info, threshold):
    """
    Helper for threshold tuning: evaluate at specific threshold.
    Returns: mean F1 score
    """
    device = dataset_info['device']
    all_f1 = []

    with torch.no_grad():
        for idx, (mel, roll, lengths) in enumerate(dataloader):
            mel = mel.to(device)
            roll = roll.to(device)
            lengths = lengths.to(device)

            logits = model(mel)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            B, P, T = preds.shape
            L = lengths[0].item()
            preds_valid = preds[0, :, :L].detach().cpu().numpy()
            target_valid = roll[0, :, :L].detach().cpu().numpy()

            y_pred = preds_valid.flatten()
            y_true = target_valid.flatten()

            f1 = f1_score(y_true, y_pred, zero_division=0)
            all_f1.append(f1)

    return float(np.mean(all_f1)) if all_f1 else 0.0


def run_threshold_tuning(args, model, dataloader, dataset_info):
    """
    Binary search for optimal threshold.
    Returns: (best_threshold, best_f1)
    """
    tune_min, tune_max = args.tune_range
    step = args.tune_step
    min_step = args.tune_min_step
    rounds = args.tune_rounds

    best_threshold = 0.5
    best_f1 = -1.0

    print("=" * 70)
    print("THRESHOLD TUNING")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Split: {args.split}")
    print(f"Subset: {args.subset or 'all samples'}")
    print(f"Data source: {dataset_info['data_source']}")
    print(f"Rounds: {rounds}")
    print()

    for round_num in range(1, rounds + 1):
        print(f"=== Round {round_num}/{rounds} | range=[{tune_min:.4f}, {tune_max:.4f}] step={step:.4f} ===")

        round_best_t = best_threshold
        round_best_f1 = best_f1

        # Test thresholds in current range
        thresholds = np.arange(tune_min, tune_max + step/2, step)

        for t in tqdm(thresholds, desc=f"Round {round_num}", leave=False):
            f1 = evaluate_at_threshold(model, dataloader, dataset_info, t)

            tqdm.write(f"  t={t:.4f}  f1={f1:.6f}")

            if f1 > round_best_f1:
                round_best_f1 = f1
                round_best_t = t

        best_threshold = round_best_t
        best_f1 = round_best_f1

        print(f"Round best: t={best_threshold:.4f} f1={best_f1:.6f}\n")

        # Narrow search window
        tune_min = max(0.01, best_threshold - 2*step)
        tune_max = min(0.99, best_threshold + 2*step)
        step = step / 2

        # Stop if step too small
        if step < min_step:
            break

    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Best threshold: {best_threshold:.6f}")
    print(f"Best mean F1:   {best_f1:.6f}")
    print("=" * 70)

    return best_threshold, best_f1


# ============================================================================
# Utility Display Functions
# ============================================================================

def show_dry_run_preview(args, config):
    """Display what would be evaluated without actually running."""
    print("=" * 70)
    print("MODEL EVALUATION - DRY RUN")
    print("=" * 70)

    print("\nMODEL CONFIGURATION:")
    print(f"  Checkpoint:    {args.model}")
    print(f"  Model type:    {config['model_type']}")
    print(f"  n_mels:        {config['n_mels']}")
    print(f"  Hidden size:   {config['hidden_size']}")
    print(f"  Num layers:    {config['num_layers']}")
    print(f"  Dropout:       {config['dropout']}")
    print(f"  SR:            {config['sr']}")
    print(f"  Hop length:    {config['hop_length']}")

    print("\nEVALUATION CONFIGURATION:")
    print(f"  Split:         {args.split}")
    print(f"  Threshold:     {args.threshold}")
    print(f"  Subset:        {args.subset or 'all samples'}")
    print(f"  Batch size:    {args.batch_size}")

    print("\nDATA SOURCE:")
    data_source, path = detect_data_source(args)
    print(f"  Mode:          {data_source}")
    print(f"  Path:          {path}")

    # Estimate samples
    try:
        if data_source == 'cache':
            metadata_path = os.path.join(args.cache_dir, f'{args.split}_metadata.pkl')
            with open(metadata_path, 'rb') as f:
                meta = pickle.load(f)
            num_samples = meta['num_chunks']
        else:
            num_samples = "unknown (requires dataset load)"

        if args.subset:
            print(f"  Samples:       {args.subset} (limited from {num_samples})")
        else:
            print(f"  Samples:       {num_samples}")
    except:
        print(f"  Samples:       unknown")

    print("\nOUTPUT:")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(args.out_dir, timestamp)
    print(f"  Directory:     {run_dir}")
    print(f"  Summary:       {run_dir}/eval_summary.txt")
    if not args.no_midi and not args.headless:
        print(f"  MIDIs:         {run_dir}/midis/")
    else:
        print(f"  MIDIs:         (disabled)")

    if args.tune_threshold:
        print("\nTHRESHOLD TUNING:")
        print(f"  Rounds:        {args.tune_rounds}")
        print(f"  Range:         [{args.tune_range[0]}, {args.tune_range[1]}]")
        print(f"  Initial step:  {args.tune_step}")
        print(f"  Min step:      {args.tune_min_step}")

    print("\n" + "=" * 70)
    print("This is a DRY RUN - no evaluation will be performed.")
    print("Run without --dry_run to proceed.")
    print("=" * 70)


def show_results_summary(results_dir):
    """Display summary of existing evaluation results."""
    summary_path = os.path.join(results_dir, "eval_summary.txt")

    if not os.path.exists(summary_path):
        print(f"Error: Summary file not found: {summary_path}")
        return

    print("=" * 70)
    print(f"EVALUATION RESULTS: {results_dir}")
    print("=" * 70)

    with open(summary_path, 'r') as f:
        content = f.read()
        print(content)

    # Check for MIDI files
    midi_dir = os.path.join(results_dir, "midis")
    if os.path.exists(midi_dir):
        midi_files = [f for f in os.listdir(midi_dir) if f.endswith('.mid')]
        print(f"\nMIDI FILES: {len(midi_files)} files in {midi_dir}")
    else:
        print("\nMIDI FILES: None generated")

    print("=" * 70)


def verify_model_cache_compatibility(args, config):
    """Check if model config is compatible with cache."""
    metadata_path = os.path.join(args.cache_dir, f'{args.split}_metadata.pkl')

    if not os.path.exists(metadata_path):
        print(f"Error: Cache metadata not found: {metadata_path}")
        return

    with open(metadata_path, 'rb') as f:
        cache_meta = pickle.load(f)

    print("=" * 70)
    print("COMPATIBILITY CHECK")
    print("=" * 70)

    print("\nModel Config:")
    print(f"  n_mels:      {config['n_mels']}")
    print(f"  sr:          {config['sr']}")
    print(f"  hop_length:  {config['hop_length']}")

    print(f"\nCache Config ({args.cache_dir}):")
    cache_n_mels = cache_meta.get('n_mels')
    cache_sr = cache_meta.get('sr')
    cache_hop = cache_meta.get('hop_length')

    n_mels_match = cache_n_mels == config['n_mels']
    sr_match = cache_sr == config['sr']
    hop_match = cache_hop == config['hop_length']

    print(f"  n_mels:      {cache_n_mels}  {'✓' if n_mels_match else '✗'}")
    print(f"  sr:          {cache_sr}  {'✓' if sr_match else '✗'}")
    print(f"  hop_length:  {cache_hop}  {'✓' if hop_match else '✗'}")

    all_compatible = n_mels_match and sr_match and hop_match

    print(f"\nStatus: {'COMPATIBLE' if all_compatible else 'INCOMPATIBLE'}")
    print("=" * 70)


# ============================================================================
# Main Function
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified evaluation CLI for music transcription models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python scripts/evaluate.py --model outputs/model.pth

  # Quick validation check
  python scripts/evaluate.py --model outputs/model.pth \\
    --split validation --subset 100 --headless

  # Threshold tuning
  python scripts/evaluate.py --model outputs/model.pth \\
    --tune_threshold --split validation --subset 50

  # Dry run preview
  python scripts/evaluate.py --model outputs/model.pth --dry_run

  # Show existing results
  python scripts/evaluate.py --show_results eval_outputs/2025-12-13_20-11-33

  # Background mode
  python scripts/evaluate.py --model outputs/model.pth --background

  # Verify compatibility
  python scripts/evaluate.py --model outputs/model.pth --verify_compatibility

IMPORTANT:
  Model parameters (n_mels, sr, hop_length) are auto-detected from cache metadata
  when using cached datasets. You can override with CLI arguments if needed.
"""
    )

    # Required
    parser.add_argument(
        "--model",
        type=str,
        help="Path to model checkpoint (.pth file)"
    )

    # Evaluation options
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Which split to evaluate on (default: test)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold for binary prediction (default: 0.5)"
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Limit number of samples (for quick eval)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (default: 1, recommended for per-sample eval)"
    )

    # Data source
    parser.add_argument(
        "--data_source",
        type=str,
        default="auto",
        choices=["auto", "cache", "full"],
        help="Data source: auto-detect, cached chunks, or full files (default: auto)"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="maestro-v3.0.0",
        help="Path to MAESTRO dataset (for full files, default: maestro-v3.0.0)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cached_dataset_mels320",
        help="Cache directory (default: cached_dataset_mels320)"
    )
    parser.add_argument(
        "--year",
        type=str,
        default=None,
        help="Year filter (full files only, e.g., 2017)"
    )

    # Model config (auto-detected from cache, or use these)
    parser.add_argument(
        "--n_mels",
        type=int,
        default=None,
        help="Number of mel bins (auto-detected from cache if not specified)"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="cnn_rnn_large",
        help="Model architecture type (default: cnn_rnn_large)"
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=512,
        help="Hidden size (default: 512)"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="Number of layers (default: 3)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate (default: 0.2)"
    )

    # Output options
    parser.add_argument(
        "--out_dir",
        type=str,
        default="eval_outputs",
        help="Output directory (default: eval_outputs)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Headless mode: only print EVAL_MEAN_F1=<value>"
    )
    parser.add_argument(
        "--no_midi",
        action="store_true",
        help="Skip MIDI generation (faster evaluation)"
    )

    # Threshold tuning
    parser.add_argument(
        "--tune_threshold",
        action="store_true",
        help="Run threshold tuning (binary search)"
    )
    parser.add_argument(
        "--tune_rounds",
        type=int,
        default=6,
        help="Number of tuning rounds (default: 6)"
    )
    parser.add_argument(
        "--tune_range",
        type=float,
        nargs=2,
        default=[0.05, 0.95],
        help="Initial threshold range (default: 0.05 0.95)"
    )
    parser.add_argument(
        "--tune_step",
        type=float,
        default=0.1,
        help="Initial step size (default: 0.1)"
    )
    parser.add_argument(
        "--tune_min_step",
        type=float,
        default=0.01,
        help="Minimum step size (default: 0.01)"
    )

    # Utility modes
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview what will be evaluated without running"
    )
    parser.add_argument(
        "--show_results",
        type=str,
        default=None,
        metavar="RESULTS_DIR",
        help="Display existing evaluation results and exit"
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="Run in background with automatic logging"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Custom log file path (only with --background)"
    )
    parser.add_argument(
        "--verify_compatibility",
        action="store_true",
        help="Check model/cache compatibility and exit"
    )

    args = parser.parse_args()

    # Handle special read-only mode: show results (doesn't need heavy imports)
    if args.show_results:
        show_results_summary(args.show_results)
        sys.exit(0)

    # Import heavy dependencies after argparse (makes --help fast)
    # This happens after show_results since that doesn't need these imports
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from sklearn.metrics import f1_score, confusion_matrix
    import pretty_midi
    from tqdm import tqdm
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns

    from data.dataset import MaestroDataset
    from data.cached_dataset import CachedMaestroDataset
    from train.train_transcriber import collate_fn
    from models.transcription_model import TranscriptionModel

    # Model is required for all other modes
    if not args.model:
        parser.error("--model is required")

    # Handle background mode
    if args.background:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = args.log_file or f"evaluate_{timestamp}.log"

        # Re-run without --background
        cmd_args = [sys.executable] + [arg for arg in sys.argv if arg != '--background']

        print("=" * 70)
        print("Starting evaluation in background...")
        print("=" * 70)

        with open(log_file, 'w') as log:
            process = subprocess.Popen(
                cmd_args,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True
            )

        print(f"Evaluation started in background")
        print(f"PID: {process.pid}")
        print(f"Log file: {log_file}")
        print(f"\nMonitor progress with:")
        print(f"  tail -f {log_file}")
        print(f"\nCheck if running:")
        print(f"  ps aux | grep {process.pid}")
        print("=" * 70)
        sys.exit(0)

    # Extract model config (with auto-detection)
    config = extract_model_config(args)

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
        show_dry_run_preview(args, config)
        sys.exit(0)

    # Handle verify compatibility mode
    if args.verify_compatibility:
        verify_model_cache_compatibility(args, config)
        sys.exit(0)

    # Load model and data
    model, dataloader, dataset_info = load_model_and_data(args, config)

    # Run threshold tuning or standard evaluation
    if args.tune_threshold:
        best_threshold, best_f1 = run_threshold_tuning(args, model, dataloader, dataset_info)
    else:
        results = run_evaluation(args, model, dataloader, dataset_info)
