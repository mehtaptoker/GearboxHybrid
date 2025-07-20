from typing import Dict, Any
from components import SystemState, Gear
import physics  # Added missing physics import
import config
import math

def calculate_reward(state: SystemState, success: bool) -> float:
    """Calculate the composite reward for the current system state.
    
    Args:
        state: Current system state
        success: Whether the gear train successfully connects input to output
        
    Returns:
        Total reward value
    """
    # Initialize reward components
    R_ratio = 0.0      # Torque ratio reward
    R_connectivity = 0.0  # Connectivity penalty
    R_collision = 0.0   # Collision penalty
    P_efficiency = 0.0  # Efficiency penalty
    R_connection = 0.0  # Connection success bonus
    
    # Get input and output gears
    input_gear = next((g for g in state.gears if g.is_driver), None)
    output_gear = next((g for g in state.gears if not g.is_driver), None)
    
    if input_gear and output_gear:
        # 1. Torque ratio reward (Gaussian function)
        current_ratio = input_gear.num_teeth / output_gear.num_teeth
        ratio_error = abs(current_ratio - state.target_ratio)
        R_ratio = math.exp(-config.ALPHA * (ratio_error ** 2))
        
        # 2. Connectivity penalty (gaps/overlaps between adjacent gears)
        total_gap = 0.0
        connection_count = 0
        for i in range(len(state.gears)):
            for j in range(i+1, len(state.gears)):
                if state.is_connected(state.gears[i], state.gears[j]):
                    # Calculate actual distance between gear centers
                    dx = state.gears[i].center.x - state.gears[j].center.x
                    dy = state.gears[i].center.y - state.gears[j].center.y
                    actual_distance = math.sqrt(dx*dx + dy*dy)
                    
                    # Calculate expected distance for perfect meshing
                    expected_distance = state.gears[i].radius + state.gears[j].radius
                    
                    # Accumulate gap/overlap error
                    total_gap += abs(actual_distance - expected_distance)
                    connection_count += 1
        
        # Normalize by number of connections to avoid excessive penalties
        if connection_count > 0:
            R_connectivity = -config.BETA * (total_gap / connection_count)
        else:
            R_connectivity = 0.0
    
    # 3. Collision penalty
    collision_detected = False
    collision_reason = ""
    
    # Check collisions with obstacles
    for gear in state.gears:
        inside, reason = physics.is_gear_inside_boundary(gear, state.boundary_poly, True)
        if not inside:
            collision_detected = True
            collision_reason = reason
            break
    
    # Check collisions between non-adjacent gears
    if not collision_detected:
        for i in range(len(state.gears)):
            for j in range(i+1, len(state.gears)):
                if not state.is_connected(state.gears[i], state.gears[j]):
                    dx = state.gears[i].center.x - state.gears[j].center.x
                    dy = state.gears[i].center.y - state.gears[j].center.y
                    distance = math.sqrt(dx*dx + dy*dy)
                    min_distance = state.gears[i].radius + state.gears[j].radius
                    
                    if distance < min_distance - config.MESHING_TOLERANCE:
                        collision_detected = True
                        collision_reason = f"Collision between gear {state.gears[i].id} and {state.gears[j].id}"
                        break
            if collision_detected:
                break
    
    if collision_detected:
        R_collision = config.COLLISION_PENALTY
    
    # 4. Connection success bonus (existing functionality)
    if success:
        R_connection = 500.0
        # Only apply efficiency penalty if connection is successful
        if len(state.gears) > 2:
            P_efficiency = (len(state.gears) - 2) * config.P_GEAR_COUNT_PENALTY
    else:
        # Apply efficiency penalty even if connection fails
        if len(state.gears) > 2:
            P_efficiency = (len(state.gears) - 2) * config.P_GEAR_COUNT_PENALTY
    
    # Total reward calculation
    total_reward = R_ratio + R_connectivity + R_collision + P_efficiency + R_connection
    
    return total_reward
