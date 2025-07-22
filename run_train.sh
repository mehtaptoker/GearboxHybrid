#!/bin/bash

# Training script for gear generation model using train_torch.py

# Set default values from config.py
LEARNING_RATE=0.0003
BATCH_SIZE=64
TOTAL_TIMESTEPS=1000000
GAMMA=0.99
EPSILON=0.2
MAX_GEARS=15
MAX_STEPS_PER_EPISODE=2000
MIN_TEETH=8
MAX_TEETH=40
VERBOSE=0
MODEL_PATH=""
DATA_DIR=""
GPU=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --total-timesteps)
            TOTAL_TIMESTEPS="$2"
            shift 2
            ;;
        --gamma)
            GAMMA="$2"
            shift 2
            ;;
        --epsilon)
            EPSILON="$2"
            shift 2
            ;;
        --max-gears)
            MAX_GEARS="$2"
            shift 2
            ;;
        --max-steps)
            MAX_STEPS_PER_EPISODE="$2"
            shift 2
            ;;
        --min-teeth)
            MIN_TEETH="$2"
            shift 2
            ;;
        --max-teeth)
            MAX_TEETH="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build command with arguments
CMD="python train_torch.py"
[ -n "$GPU" ] && CMD="$CMD --gpu $GPU"
[ -n "$DATA_DIR" ] && CMD="$CMD --data-dir $DATA_DIR"
[ -n "$MODEL_PATH" ] && CMD="$CMD --model-path $MODEL_PATH"
CMD="$CMD --learning-rate $LEARNING_RATE"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --total-timesteps $TOTAL_TIMESTEPS"
CMD="$CMD --gamma $GAMMA"
CMD="$CMD --epsilon $EPSILON"
CMD="$CMD --max-gears $MAX_GEARS"
CMD="$CMD --max-steps $MAX_STEPS_PER_EPISODE"
CMD="$CMD --min-teeth $MIN_TEETH"
CMD="$CMD --max-teeth $MAX_TEETH"
CMD="$CMD --verbose $VERBOSE"

# Execute training command
echo "Starting training with command:"
echo "$CMD"
eval "$CMD"

# Check exit status
if [ $? -eq 0 ]; then
    echo "Training completed successfully"
else
    echo "Training failed with exit code $?"
    exit 1
fi
