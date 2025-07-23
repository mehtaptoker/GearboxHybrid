import os
import config
from data_generator import generate_scenario
from components import SystemState, Vector2D
from visualization import generate_report

def create_test_scenarios(num_samples=5):
    """Generate and report on test scenarios."""
    for i in range(num_samples):
        # Generate a scenario
        scenario = generate_scenario()
        
        # Create system state with no gears
        state = SystemState(
            boundary_poly=scenario["boundary_poly"],
            gears=[],
            input_shaft=scenario["input_shaft"],
            output_shaft=scenario["output_shaft"],
            target_ratio=scenario["target_ratio"]
        )
        
        # Generate report
        report_path = generate_report(state)
        print(f"Generated report {i+1}/{num_samples} at {report_path}")

if __name__ == "__main__":
    create_test_scenarios(20)
