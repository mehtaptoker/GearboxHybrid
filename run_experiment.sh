#!/bin/bash

# Configuration
DATA_DIR="data/intermediate"
GPU_ID=0  # Set to -1 for CPU
REPORT_DIR="reports/$(date +%Y%m%d_%H%M%S)"
MODEL_PATH="models/best_policy.pth"

# Hyperparameters
LEARNING_RATE=0.001
BATCH_SIZE=64
TOTAL_TIMESTEPS=100000
GAMMA=0.99
EPSILON=0.2
MAX_GEARS=10
MAX_STEPS=50
MIN_TEETH=8
MAX_TEETH=60

# Create directories
mkdir -p "$REPORT_DIR"
mkdir -p models

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

python train_torch.py \
    --data-dir "$DATA_DIR" \
    --gpu $GPU_ID \
    --learning-rate $LEARNING_RATE \
    --batch-size $BATCH_SIZE \
    --total-timesteps $TOTAL_TIMESTEPS \
    --gamma $GAMMA \
    --epsilon $EPSILON \
    --max-gears $MAX_GEARS \
    --max-steps $MAX_STEPS \
    --min-teeth $MIN_TEETH \
    --max-teeth $MAX_TEETH

# Move best model
mv gear_generator_policy.pth "$MODEL_PATH"

# Generate evaluation reports
echo "Generating evaluation reports..."
python evaluation.py \
    --model "$MODEL_PATH" \
    --data-dir "$DATA_DIR" \
    --output-dir "$REPORT_DIR" \
    --num-episodes 5

echo "Experiment complete!"
