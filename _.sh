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
echo "Running end-to-end demonstration with obstacle-aware pathfinding and iterative refinement..."
python -c "
import sys
sys.path.append('.')
from components import Gear, Vector2D
from pathfinding import generate_collision_free_path
from refinement import generate_gear_train
import visualization
import json
import os

# Predefined inputs
start_gear = Gear(id=1, center=Vector2D(0, 0), num_teeth=20, module=1.0, is_driver=True)
end_gear = Gear(id=2, center=Vector2D(100, 0), num_teeth=30, module=1.0)

# More complex obstacle layout
obstacles = [
    [Vector2D(30, -20), Vector2D(30, 20), Vector2D(70, 20), Vector2D(70, -20)],
    [Vector2D(40, 10), Vector2D(40, 30), Vector2D(60, 30), Vector2D(60, 10)]
]

print('Generating collision-free path...')
path = generate_collision_free_path(start_gear.center, end_gear.center, obstacles)
if path is None:
    print('No collision-free path found!')
    exit(1)

print(f'Generated path with {len(path)} waypoints:')
for i, point in enumerate(path):
    print(f'  Point {i}: ({point.x:.2f}, {point.y:.2f})')

# Generate intermediate gears
existing_gears = [start_gear, end_gear]
print('Generating gear train with iterative refinement...')
intermediate_gears = generate_gear_train(
    start_gear=start_gear,
    end_gear=end_gear,
    path=path[1:-1],  # Intermediate joints
    existing_gears=existing_gears,
    boundary_poly=[
        Vector2D(-10, -40),
        Vector2D(-10, 50),
        Vector2D(110, 50),
        Vector2D(110, -40)
    ],
    obstacles=obstacles,
    module=1.0,
    max_iterations=1000,
    tolerance=0.1
)

if intermediate_gears is None:
    print('Failed to generate gear train!')
    exit(1)

print(f'Generated {len(intermediate_gears)} intermediate gears:')
for i, gear in enumerate(intermediate_gears):
    print(f'  Gear {i+1}: center=({gear.center.x:.2f}, {gear.center.y:.2f}), teeth={gear.num_teeth}, radius={gear.radius:.2f}')

# Save results
os.makedirs('data/demo', exist_ok=True)
system_state = {
    'gears': [g.to_dict() for g in [start_gear, end_gear] + intermediate_gears],
    'obstacles': [[p.to_dict() for p in poly] for poly in obstacles]
}
with open('data/demo/demo_system.json', 'w') as f:
    json.dump(system_state, f, indent=2)

# Generate visualization
print('Generating visualization...')
visualization.visualize_system(
    gears=system_state['gears'],
    obstacles=system_state['obstacles'],
    boundary=[[-10, -40], [-10, 50], [110, 50], [110, -40]],
    output_path='data/demo/demo_system.png'
)

print('Demonstration completed successfully!')
"
