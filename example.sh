#!/bin/bash

# ==============================================================================
# Music Transcription Pipeline - Example Workflow Script
# ==============================================================================
# This script demonstrates the complete workflow for training a music
# transcription model on the MAESTRO dataset.
#
# Usage:
#   ./example.sh preprocess   # Preprocess dataset and create cache
#   ./example.sh train        # Train the model with cached data
#   ./example.sh eval         # Evaluate the best trained model
#   ./example.sh all          # Run all steps sequentially (WARNING: Takes hours!)
# ==============================================================================

# ------------------------------------------------------------------------------
# Configuration Parameters
# ------------------------------------------------------------------------------
# Feel free to modify these parameters to experiment with different configurations

# Model architecture
MODEL_TYPE="cnn_rnn_large"        # Model type: cnn_rnn or cnn_rnn_large
N_MELS=320                         # Number of mel frequency bins
HIDDEN_SIZE=512                    # RNN hidden layer size
NUM_LAYERS=3                       # Number of RNN layers
DROPOUT=0.2                        # Dropout rate

# Training parameters
EPOCHS=100                         # Number of training epochs
BATCH_SIZE=24                      # Batch size (sweet spot for most GPUs)
LEARNING_RATE=1e-4                 # Learning rate
SAVE_EVERY=5                       # Save checkpoint every N epochs

# Data parameters
CHUNK_LENGTH=30.0                  # Audio chunk length in seconds
CHUNK_OVERLAP=0.0                  # Overlap between chunks (0.0-1.0)
DATASET_DIR="maestro-v3.0.0"       # MAESTRO dataset directory
CACHE_DIR="cached_dataset_mels${N_MELS}"  # Auto-generated cache directory name

# Output directories
OUTPUTS_DIR="outputs"              # Training outputs directory
EVAL_DIR="eval_outputs"            # Evaluation outputs directory

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

print_header() {
    echo ""
    echo "=============================================================================="
    echo "$1"
    echo "=============================================================================="
    echo ""
}

print_config() {
    echo "Configuration:"
    echo "  Model:        $MODEL_TYPE"
    echo "  n_mels:       $N_MELS"
    echo "  Hidden size:  $HIDDEN_SIZE"
    echo "  Num layers:   $NUM_LAYERS"
    echo "  Dropout:      $DROPOUT"
    echo "  Epochs:       $EPOCHS"
    echo "  Batch size:   $BATCH_SIZE"
    echo "  Chunk length: ${CHUNK_LENGTH}s"
    echo "  Cache dir:    $CACHE_DIR"
    echo ""
}

show_usage() {
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  preprocess    Preprocess dataset and create cache (~1-2 hours)"
    echo "  train         Train the model with cached data (~10-20 hours)"
    echo "  eval          Evaluate the best trained model (~30 minutes)"
    echo "  all           Run all steps sequentially (WARNING: Takes 12+ hours!)"
    echo ""
    echo "Examples:"
    echo "  ./example.sh preprocess   # Only preprocess the dataset"
    echo "  ./example.sh train        # Only train (requires preprocessed cache)"
    echo "  ./example.sh eval         # Only evaluate (requires trained model)"
    echo ""
}

# ------------------------------------------------------------------------------
# Task Functions
# ------------------------------------------------------------------------------

run_preprocess() {
    print_header "STEP 1: Preprocessing Dataset"
    print_config

    echo "This will create a cached dataset at: $CACHE_DIR"
    echo "Estimated time: 1-2 hours"
    echo "Estimated disk space: ~34GB"
    echo ""

    # Check if cache already exists
    if [ -d "$CACHE_DIR" ]; then
        echo "WARNING: Cache directory already exists: $CACHE_DIR"
        echo "Existing cache will be used. To recreate, delete it first or use --force"
        echo ""
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 1
        fi
    fi

    echo "Starting preprocessing in background..."
    python scripts/preprocess_dataset.py \
        --n_mels $N_MELS \
        --chunk_length $CHUNK_LENGTH \
        --overlap $CHUNK_OVERLAP \
        --root_dir $DATASET_DIR \
        --cache_dir $CACHE_DIR \
        --background

    echo ""
    echo "Preprocessing started in background!"
    echo "Monitor progress with: tail -f preprocess_*.log"
    echo ""
    echo "When complete, run: ./example.sh train"
}

run_train() {
    print_header "STEP 2: Training Model"
    print_config

    # Check if cache exists
    if [ ! -d "$CACHE_DIR" ]; then
        echo "ERROR: Cache directory not found: $CACHE_DIR"
        echo "Please run preprocessing first: ./example.sh preprocess"
        exit 1
    fi

    echo "Using cached dataset: $CACHE_DIR"
    echo "Estimated time: 10-20 hours for $EPOCHS epochs"
    echo ""

    # Create outputs directory
    mkdir -p $OUTPUTS_DIR

    echo "Starting training in background..."
    python scripts/train_cnn.py \
        --model $MODEL_TYPE \
        --n_mels $N_MELS \
        --hidden_size $HIDDEN_SIZE \
        --num_layers $NUM_LAYERS \
        --dropout $DROPOUT \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LEARNING_RATE \
        --save_every $SAVE_EVERY \
        --chunk_length $CHUNK_LENGTH \
        --cached_dir $CACHE_DIR \
        --background

    echo ""
    echo "Training started in background!"
    echo "Monitor progress with: tail -f train_*.log"
    echo ""
    echo "When complete, run: ./example.sh eval"
}

run_eval() {
    print_header "STEP 3: Evaluating Model"

    # Find the most recent training run
    LATEST_RUN=$(ls -td $OUTPUTS_DIR/*/ 2>/dev/null | head -1)

    if [ -z "$LATEST_RUN" ]; then
        echo "ERROR: No training runs found in $OUTPUTS_DIR"
        echo "Please train a model first: ./example.sh train"
        exit 1
    fi

    # Look for best model checkpoint
    BEST_MODEL="${LATEST_RUN}checkpoints/model_best.pth"

    if [ ! -f "$BEST_MODEL" ]; then
        echo "WARNING: model_best.pth not found, looking for final model..."
        BEST_MODEL="${LATEST_RUN}checkpoints/model_final.pth"
    fi

    if [ ! -f "$BEST_MODEL" ]; then
        echo "ERROR: No model checkpoint found in $LATEST_RUN"
        echo "Available checkpoints:"
        ls -lh "${LATEST_RUN}checkpoints/" 2>/dev/null || echo "  (none)"
        exit 1
    fi

    echo "Using model: $BEST_MODEL"
    echo "Using cache:  $CACHE_DIR"
    echo "Estimated time: 30 minutes on test set"
    echo ""

    echo "Starting evaluation..."
    python scripts/evaluate.py \
        --model "$BEST_MODEL" \
        --model_type $MODEL_TYPE \
        --n_mels $N_MELS \
        --hidden_size $HIDDEN_SIZE \
        --num_layers $NUM_LAYERS \
        --dropout $DROPOUT \
        --split test \
        --data_source cache \
        --cache_dir $CACHE_DIR

    echo ""
    echo "Evaluation complete!"
    echo "Results saved to: $EVAL_DIR/<timestamp>/"
}

run_all() {
    print_header "Running Complete Pipeline"
    echo "WARNING: This will run all three steps sequentially:"
    echo "  1. Preprocess dataset (~1-2 hours)"
    echo "  2. Train model (~10-20 hours)"
    echo "  3. Evaluate model (~30 minutes)"
    echo ""
    echo "Total estimated time: 12-24 hours"
    echo ""

    read -p "Are you sure you want to continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi

    # Run all steps
    run_preprocess
    echo ""
    echo "Waiting for preprocessing to complete..."
    echo "Please monitor the preprocessing log and press ENTER when complete."
    read -p "Press ENTER to continue with training..."

    run_train
    echo ""
    echo "Waiting for training to complete..."
    echo "Please monitor the training log and press ENTER when complete."
    read -p "Press ENTER to continue with evaluation..."

    run_eval

    print_header "Pipeline Complete!"
    echo "All steps completed successfully."
}

# ------------------------------------------------------------------------------
# Main Script
# ------------------------------------------------------------------------------

# Check for command argument
if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

COMMAND=$1

case "$COMMAND" in
    preprocess)
        run_preprocess
        ;;
    train)
        run_train
        ;;
    eval)
        run_eval
        ;;
    all)
        run_all
        ;;
    *)
        echo "ERROR: Unknown command: $COMMAND"
        echo ""
        show_usage
        exit 1
        ;;
esac