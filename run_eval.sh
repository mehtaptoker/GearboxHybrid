#!/bin/bash

# Evaluation script for trained gear generation models

# Set default values
MODEL="gear_generator_policy.pth"
DATA_DIR="data/intermediate"
OUTPUT_DIR="reports/evaluation"
NUM_EPISODES=10
MAX_GEARS=15

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num-episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        --max-gears)
            MAX_GEARS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FULL_OUTPUT_DIR="${OUTPUT_DIR}/${TIMESTAMP}"
mkdir -p "$FULL_OUTPUT_DIR"

# Build command
CMD="python evaluation.py"
CMD="$CMD --model $MODEL"
CMD="$CMD --data-dir $DATA_DIR"
CMD="$CMD --output-dir $FULL_OUTPUT_DIR"
CMD="$CMD --num-episodes $NUM_EPISODES"
CMD="$CMD --max-gears $MAX_GEARS"

# Execute evaluation command
echo "Starting evaluation with command:"
echo "$CMD"
eval "$CMD"

# Check exit status
if [ $? -eq 0 ]; then
    echo -e "\nEvaluation completed successfully"
    echo "Reports saved to: $FULL_OUTPUT_DIR"
    
    # Show summary
    echo -e "\nEvaluation Summary:"
    echo "==================="
    python -c "import json; \
        data=json.load(open('$FULL_OUTPUT_DIR/evaluation_results.json')); \
        rewards=[r['reward'] for r in data]; \
        ratios=[r['achieved_ratio'] for r in data]; \
        gears=[r['gear_count'] for r in data]; \
        print(f'Episodes: {len(data)}'); \
        print(f'Average Reward: {sum(rewards)/len(rewards):.2f}'); \
        print(f'Average Gear Count: {sum(gears)/len(gears):.2f}'); \
        print(f'Average Achieved Ratio: {sum(ratios)/len(ratios):.2f}');"
else
    echo "Evaluation failed with exit code $?"
    exit 1
fi
