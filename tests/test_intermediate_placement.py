import config
from components import Gear, Vector2D
from environment import GearEnv
import math

def test_intermediate_placement():
    # Create a test scenario with known positions
    input_gear = Gear(
        id=1,
        center=Vector2D(0, 0),
        num_teeth=20,
        module=config.GEAR_MODULE,
        z_layer=0,
        is_driver=True
    )
    output_gear = Gear(
        id=2,
        center=Vector2D(80, 0),  # Reduced distance to 80 units
        num_teeth=30,
        module=config.GEAR_MODULE,
        z_layer=0,
        is_driver=False
    )
    
    env = GearEnv()
    env.state = None  # Bypass reset for this test
    
    # Test with different intermediate teeth counts
    for teeth in [45, 50, 55, 60]:
        print(f"\nTesting with intermediate teeth: {teeth}")
        positions = env._calculate_intermediate_positions(input_gear, output_gear, teeth)
        
        if not positions:
            print("No positions found")
            continue
            
        for i, pos in enumerate(positions):
            print(f"Position {i+1}: ({pos.x:.2f}, {pos.y:.2f})")
            
            # Create intermediate gear
            intermediate = Gear(
                id=3,
                center=pos,
                num_teeth=teeth,
                module=config.GEAR_MODULE,
                z_layer=0,
                is_driver=False
            )
            
            # Check distances
            dist_to_input = math.sqrt((pos.x - input_gear.center.x)**2 + (pos.y - input_gear.center.y)**2)
            expected_input_dist = input_gear.radius + intermediate.radius
            print(f"  Distance to input: {dist_to_input:.2f} (expected: {expected_input_dist:.2f})")
            
            dist_to_output = math.sqrt((pos.x - output_gear.center.x)**2 + (pos.y - output_gear.center.y)**2)
            expected_output_dist = output_gear.radius + intermediate.radius
            print(f"  Distance to output: {dist_to_output:.2f} (expected: {expected_output_dist:.2f})")
            
            # Check if meshing would be valid
            meshing_ok = env._check_gear_meshing(input_gear, intermediate) and \
                         env._check_gear_meshing(intermediate, output_gear)
            print(f"  Meshing valid: {meshing_ok}")

if __name__ == "__main__":
    test_intermediate_placement()
