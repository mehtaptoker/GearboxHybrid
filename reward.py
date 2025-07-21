from typing import Dict, Any
from components import SystemState, Gear
import physics
import config
import math

def calculate_reward(state: SystemState) -> float:
    """Calculate the composite reward for the current system state.
    
    Args:
        state: Current system state
        
    Returns:
        Total reward value based on:
        1. Torque ratio accuracy (R_ratio)
        2. Connectivity gaps (R_connectivity)
        3. Collisions (R_collision)
    """
    # Initialize reward components
    R_ratio = 0.0       # Torque ratio reward
    R_connectivity = 0.0  # Connectivity penalty
    R_collision = 0.0    # Collision penalty
    
    # Get input and output gears
    input_gear = next((g for g in state.gears if g.is_driver), None)
    output_gear = next((g for g in state.gears if not g.is_driver), None)
    
    # Calculate actual gear train ratio if possible
    if input_gear and output_gear:
        current_ratio = physics.calculate_gear_train_ratio(state.gears, input_gear.id)
        ratio_error = abs(current_ratio - state.target_ratio)
        # Use exact formula: R_ratio = exp(-alpha * (T_actual - T_target)^2)
        R_ratio = math.exp(-config.ALPHA * (ratio_error ** 2))
    
    # Calculate connectivity penalty (sum of absolute gaps between adjacent gears)
    total_gap = 0.0
    for connection in state.connections:
        gear1 = state.get_gear_by_id(connection.gear_id1)
        gear2 = state.get_gear_by_id(connection.gear_id2)
        if gear1 and gear2:
            dx = gear1.center.x - gear2.center.x
            dy = gear1.center.y - gear2.center.y
            actual_distance = math.sqrt(dx*dx + dy*dy)
            expected_distance = gear1.radius + gear2.radius
            total_gap += abs(actual_distance - expected_distance)
    
    # Apply connectivity penalty proportional to total gap
    R_connectivity = -config.BETA * total_gap
    
    # Check for collisions (with obstacles or between non-adjacent gears)
    collision_detected = False
    # Check gear-obstacle collisions
    for gear in state.gears:
        if physics.check_collision(gear, state.obstacles):
            collision_detected = True
            break
    
    # Check gear-gear collisions between non-connected gears
    if not collision_detected:
        for i in range(len(state.gears)):
            for j in range(i+1, len(state.gears)):
                if not state.are_gears_connected(state.gears[i], state.gears[j]):
                    if physics.check_collision(state.gears[i], state.gears[j]):
                        collision_detected = True
                        break
            if collision_detected:
                break
    
    # Apply large constant penalty for any collision
    if collision_detected:
        R_collision = config.COLLISION_PENALTY
    
    # Combine rewards according to specification
    total_reward = R_ratio + R_connectivity + R_collision
        
    return total_reward
