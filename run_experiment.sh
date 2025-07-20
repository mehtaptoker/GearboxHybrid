#!/bin/bash

# Configuration
VERBOSE=0  # Default verbosity level
DATA_DIR="data/intermediate"
REPORT_DIR="reports/$(date +%Y%m%d_%H%M%S)"
MODEL_PATH="models/best_policy.pth"
USE_GENERATED_DATA=0
DEMO_MODE=1  # Default to demonstration mode

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
# Always run end-to-end demonstration
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
    max_iterations=500,
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
