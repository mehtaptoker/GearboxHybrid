#!/bin/bash

# Configuration
VERBOSE=0  # Default verbosity level
DATA_DIR="data/intermediate"
REPORT_DIR="reports/$(date +%Y%m%d_%H%M%S)"
MODEL_PATH="models/best_policy.pth"
USE_GENERATED_DATA=0
DEMO_MODE=0  # Default to training mode

# Hyperparameters
LEARNING_RATE=0.001
BATCH_SIZE=64
TOTAL_TIMESTEPS=1000000
GAMMA=0.99
EPSILON=0.2
MAX_GEARS=10
MAX_STEPS=1000
MIN_TEETH=8
MAX_TEETH=60

# Parse command line arguments
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
    --demo)
      DEMO_MODE=1
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Create directories
mkdir -p "$REPORT_DIR"
mkdir -p models
mkdir -p logs

# Log file
LOG_FILE="logs/experiment_$(date +%Y%m%d_%H%M%S).log"
exec &> >(tee -a "$LOG_FILE")

# Demo mode: run end-to-end demonstration
if [ "$DEMO_MODE" -eq 1 ]; then
    echo "Running end-to-end demonstration..."
    python -c "
import sys
sys.path.append('.')
from components import Gear, Vector2D
from pathfinding import generate_collision_free_path
from refinement import generate_gear_train

# Predefined inputs
start_gear = Gear(id=1, center=Vector2D(0, 0), num_teeth=20, module=1.0, is_driver=True)
end_gear = Gear(id=2, center=Vector2D(30, 0), num_teeth=30, module=1.0)

# Obstacle layout: two rectangles
obstacles = [
    [Vector2D(10, -5), Vector2D(10, 5), Vector2D(20, 5), Vector2D(20, -5)]
]

# Generate collision-free path
path = generate_collision_free_path(start_gear.center, end_gear.center, obstacles)
if path is None:
    print('No collision-free path found!')
    exit(1)

print(f'Generated path with {len(path)} waypoints:')
for i, point in enumerate(path):
    print(f'  Point {i}: ({point.x:.2f}, {point.y:.2f})')

# Generate intermediate gears (exclude start and end points)
existing_gears = [start_gear, end_gear]
intermediate_gears = generate_gear_train(
    start_gear=start_gear,
    end_gear=end_gear,
    path=path[1:-1],  # Intermediate joints
    existing_gears=existing_gears,
    boundary_poly=[
        Vector2D(-10, -10),
        Vector2D(-10, 40),
        Vector2D(40, 40),
        Vector2D(40, -10)
    ],
    obstacles=[],
    module=1.0,
    max_iterations=50,
    tolerance=1e-3
)

if intermediate_gears is None:
    print('Failed to generate gear train!')
    exit(1)

print(f'Generated {len(intermediate_gears)} intermediate gears:')
for i, gear in enumerate(intermediate_gears):
    print(f'  Gear {i+1}: center=({gear.center.x:.2f}, {gear.center.y:.2f}), teeth={gear.num_teeth}, radius={gear.radius:.2f}')

print('Demonstration completed successfully!')
"
    exit 0
fi

# Train the model
echo "Starting training..."
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
