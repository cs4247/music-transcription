#!/bin/bash

# Resume training from epoch 15 checkpoint with cached dataset

# Create outputs directory if it doesn't exist
mkdir -p outputs

# Create log filename with timestamp
LOG_FILE="outputs/resume_run_$(date +"%Y-%m-%d_%H-%M-%S").log"

echo "Resuming training from epoch 15..."
echo "Using cached dataset (10-50x faster!)"
echo "Log file: $LOG_FILE"
echo ""

# Resume training with cached dataset
nohup python3 main.py \
  --resume outputs/2025-12-12_05-09-59/checkpoints/model_epoch_15.pth \
  --epochs 40 \
  --batch_size 16 \
  --lr 1e-4 \
  --save_every 5 \
  --chunk_length 30.0 \
  --chunk_overlap 0.0 \
  > "$LOG_FILE" 2>&1 &

# Get the process ID
PID=$!
echo "Training resumed with PID: $PID"
echo "Log file: $LOG_FILE"
echo ""
echo "The cached dataset will be used automatically (much faster!)"
echo "Starting from epoch 16, running until epoch 40 (25 more epochs)"
echo ""
echo "Monitor with: tail -f $LOG_FILE"

# Wait a moment for the file to be created
sleep 2

# Tail the log file
tail -f "$LOG_FILE"
