#!/bin/bash

# Configuration
VERBOSE=0  # Default verbosity level
DATA_DIR="data/intermediate"
GPU_ID=0  # Set to -1 for CPU
REPORT_DIR="reports/$(date +%Y%m%d_%H%M%S)"
MODEL_PATH="models/best_policy.pth"

# Parse command line arguments
USE_GENERATED_DATA=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --verbose=*)
      VERBOSE="${1#*=}"
      shift
      ;;
    --data-dir=*)
      DATA_DIR="${1#*=}"
      shift
      ;;
    --total-timesteps=*)
      TOTAL_TIMESTEPS="${1#*=}"
      shift
      ;;
    --use-generated-data)
      USE_GENERATED_DATA=1
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Hyperparameters
LEARNING_RATE=0.001
BATCH_SIZE=64
TOTAL_TIMESTEPS=10
GAMMA=0.99
EPSILON=0.2
MAX_GEARS=10
MAX_STEPS=50
MIN_TEETH=8
MAX_TEETH=60

# Create directories
mkdir -p "$REPORT_DIR"
mkdir -p models
mkdir -p logs

# Log file
LOG_FILE="logs/experiment_$(date +%Y%m%d_%H%M%S).log"
exec &> >(tee -a "$LOG_FILE")

# Train the model
echo "Starting training with GPU $GPU_ID..."
echo "Using hyperparameters:"
echo "  LEARNING_RATE: $LEARNING_RATE"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  TOTAL_TIMESTEPS: $TOTAL_TIMESTEPS"
echo "  GAMMA: $GAMMA"
echo "  EPSILON: $EPSILON"
echo "  MAX_GEARS: $MAX_GEARS"
echo "  MAX_STEPS: $MAX_STEPS"
echo "  MIN_TEETH: $MIN_TEETH"
echo "  MAX_TEETH: $MAX_TEETH"

# Check if a pre-trained model exists
if [ -f "$MODEL_PATH" ]; then
    echo "Found pre-trained model at $MODEL_PATH, continuing training."
    MODEL_ARG="--model-path $MODEL_PATH"
else
    echo "No pre-trained model found, starting from scratch."
    MODEL_ARG=""
fi

# Build training command
TRAIN_CMD="python train_torch.py \
    --gpu $GPU_ID \
    --learning-rate $LEARNING_RATE \
    --batch-size $BATCH_SIZE \
    --total-timesteps $TOTAL_TIMESTEPS \
    --gamma $GAMMA \
    --epsilon $EPSILON \
    --max-gears $MAX_GEARS \
    --max-steps $MAX_STEPS \
    --min-teeth $MIN_TEETH \
    --max-teeth $MAX_TEETH \
    --verbose $VERBOSE \
    $MODEL_ARG"

if [ "$USE_GENERATED_DATA" -eq 0 ]; then
    TRAIN_CMD="$TRAIN_CMD --data-dir $DATA_DIR"
fi

eval $TRAIN_CMD

# Move best model
mv gear_generator_policy.pth "$MODEL_PATH"

# Generate evaluation reports
echo "Generating evaluation reports..."
python evaluation.py \
    --model "$MODEL_PATH" \
    --data-dir "$DATA_DIR" \
    --output-dir "$REPORT_DIR" \
    --num-episodes 5 \
    --max-gears $MAX_GEARS

echo "Experiment complete!"
