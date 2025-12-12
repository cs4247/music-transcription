#!/bin/bash

# Create outputs directory if it doesn't exist
mkdir -p outputs

# Create log filename with timestamp
LOG_FILE="outputs/train_run_$(date +"%Y-%m-%d_%H-%M-%S").log"

# Run training in background with chunking enabled
# Sweet spot: batch_size=24 for balanced performance
nohup python3 main.py \
  --epochs 40 \
  --batch_size 24 \
  --lr 1e-4 \
  --save_every 5 \
  --chunk_length 30.0 \
  --chunk_overlap 0.0 \
  > "$LOG_FILE" 2>&1 &

# Get the process ID
PID=$!
echo "Training started with PID: $PID"
echo "Log file: $LOG_FILE"
echo "Configuration: 30s chunks, no overlap, batch_size=24, 8 workers (SWEET SPOT)"

# Wait a moment for the file to be created
sleep 2

# Tail the log file
tail -f "$LOG_FILE"