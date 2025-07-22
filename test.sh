#!/bin/bash

# Test script for gear generation system with demo cases

# Set max iterations for gear generation
MAX_ITERATIONS=5000

# Create timestamped report directory
REPORT_DIR="reports/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$REPORT_DIR"

# Define test cases
declare -A TEST_CASES=(
    ["Example1"]="data/Example1_constraints.json"
    ["Example2"]="data/Example2_constraints.json"
    ["Example3"]="data/Example3_constraints.json"
    ["Example4"]="data/Example4_constraints.json"
)

# Run each test case
for name in "${!TEST_CASES[@]}"; do
    input_file="${TEST_CASES[$name]}"
    output_dir="$REPORT_DIR/$name"
    mkdir -p "$output_dir"
    
    echo "Running test case: $name with max iterations: $MAX_ITERATIONS"
    echo "Input file: $input_file"
    
    # Run the demo with max iterations parameter and capture output
    python demo.py "$input_file" --max-iterations $MAX_ITERATIONS > "$output_dir/output.log" 2>&1
    
    # Check if demo ran successfully
    if [ $? -eq 0 ]; then
        echo "Success: $name"
        
        # Generate visualization
        python visualization.py --input "data/demo/demo_system.json" --output "$output_dir/system.png"
        
        # Copy input constraints
        cp "$input_file" "$output_dir/input_constraints.json"
        
        # Generate report
        echo "{\"test_case\":\"$name\",\"status\":\"success\",\"input_file\":\"$input_file\",\"max_iterations\":$MAX_ITERATIONS,\"output_dir\":\"$output_dir\"}" > "$output_dir/report.json"
    else
        echo "Failed: $name after $MAX_ITERATIONS iterations"
        echo "{\"test_case\":\"$name\",\"status\":\"failed\",\"input_file\":\"$input_file\",\"max_iterations\":$MAX_ITERATIONS,\"output_dir\":\"$output_dir\"}" > "$output_dir/report.json"
    fi
done

# Generate summary report using Python for JSON parsing
echo "Test Summary:" > "$REPORT_DIR/summary.txt"
echo "=============" >> "$REPORT_DIR/summary.txt"

for name in "${!TEST_CASES[@]}"; do
    # Use Python to extract status from report.json
    status=$(python -c "import json; f=open('$REPORT_DIR/$name/report.json'); data=json.load(f); print(data['status'])")
    echo "- $name: $status" >> "$REPORT_DIR/summary.txt"
done

echo "Test completed. Reports saved to: $REPORT_DIR"
