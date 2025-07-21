from components import Gear, Vector2D, SystemState
from refinement import generate_gear_train
import json
import sys
import config
import argparse

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run gear generation demo')
    parser.add_argument('input_file', nargs='?', default="data/demo/demo_input.json", 
                        help='Input JSON file')
    parser.add_argument('--max-iterations', type=int, default=config.MAX_GAP_FILLING_ITERATIONS,
                        help=f'Maximum iterations for gear generation (default: {config.MAX_GAP_FILLING_ITERATIONS})')
    args = parser.parse_args()
    
    input_file = args.input_file
    max_iterations = args.max_iterations
    
    # Load input configuration
    with open(input_file) as f:
        config_data = json.load(f)
    
    # Create gears and boundary
    # Convert center dictionary to Vector2D for start and end gears
    start_gear_data = config_data['start_gear']
    start_gear_center = Vector2D(
        start_gear_data['center']['x'], 
        start_gear_data['center']['y']
    )
    start_gear = Gear(
        id=start_gear_data['id'],
        center=start_gear_center,
        num_teeth=start_gear_data['num_teeth'],
        module=start_gear_data['module']
    )
    
    end_gear_data = config_data['end_gear']
    end_gear_center = Vector2D(
        end_gear_data['center']['x'], 
        end_gear_data['center']['y']
    )
    end_gear = Gear(
        id=end_gear_data['id'],
        center=end_gear_center,
        num_teeth=end_gear_data['num_teeth'],
        module=end_gear_data['module']
    )
    
    # Convert obstacle definitions to proper Gear objects
    obstacles = []
    for obs in config_data['obstacles']:
        # Calculate teeth count based on radius and standard module
        radius = obs['radius']
        module = config_data.get('module', 2.0)  # Default module size
        num_teeth = int(radius * 2 / module)
        obstacles.append(Gear(
            id=obs.get('id', 0),
            center=Vector2D(obs['center']['x'], obs['center']['y']),
            num_teeth=num_teeth,
            module=module
        ))
    
    boundary = [Vector2D(p['x'], p['y']) for p in config_data['boundary']]

    # Create initial path if provided in input
    path = []
    if 'initial_path' in config_data:
        path = [Vector2D(p['x'], p['y']) for p in config_data['initial_path']]
    else:
        # Default path (straight line from start to end with one intermediate point)
        start_point = start_gear.center
        end_point = end_gear.center
        midpoint = Vector2D(
            (start_point.x + end_point.x) / 2, 
            (start_point.y + end_point.y) / 2
        )
        path = [midpoint]

    # Generate intermediate gears
    print("Generating gear train...")
    intermediate_gears = generate_gear_train(
        start_gear=start_gear,
        end_gear=end_gear,
        path=path,
        existing_gears=[],
        boundary_poly=boundary,
        obstacles=obstacles,
        module=config.GEAR_MODULE,
        max_iterations=max_iterations,
        tolerance=1e-3
    )

    if intermediate_gears is not None:
        success = True
        gear_train = intermediate_gears
    else:
        success = False
        gear_train = []

    # Output results
    if success:
        print("\nSuccessfully generated gear train!")
        print(f"Generated {len(gear_train)} intermediate gears:")
        for i, gear in enumerate(gear_train):
            print(f"  Gear {i+1}: center=({gear.center.x:.1f}, {gear.center.y:.1f}), teeth={gear.num_teeth}")
        
        # Create a system state with all gears
        all_gears = [start_gear, end_gear] + gear_train
        system_state = SystemState(
            boundary_poly=boundary,
            gears=all_gears,
            input_shaft=start_gear.center,
            output_shaft=end_gear.center,
            target_ratio=2.0,
            obstacles=obstacles
        )
        
        # Save final system state
        output_file = "data/demo/demo_system.json"
        # Convert system state to JSON serializable format
        system_dict = {
            "boundary_poly": [{"x": p.x, "y": p.y} for p in system_state.boundary_poly],
            "gears": [gear.to_dict() for gear in system_state.gears],
            "input_shaft": {"x": system_state.input_shaft.x, "y": system_state.input_shaft.y},
            "output_shaft": {"x": system_state.output_shaft.x, "y": system_state.output_shaft.y},
            "target_ratio": system_state.target_ratio,
            "obstacles": [gear.to_dict() for gear in system_state.obstacles]
        }
        
        with open(output_file, 'w') as f:
            json.dump(system_dict, f, indent=2)
        print(f"\nSaved final system state to {output_file}")
    else:
        print(f"\nFailed to generate gear train after {max_iterations} iterations")
        sys.exit(1)

if __name__ == "__main__":
    main()
