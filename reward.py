from typing import Dict, Any
from components import SystemState, Gear
import config
import math

def calculate_reward(state: SystemState, success: bool) -> float:
    """Calculate the reward for the current system state.
    
    Args:
        state: Current system state
        success: Whether the gear train successfully connects input to output
        
    Returns:
        Total reward value
    """
    # Initialize reward components
    R_ratio = 0.0
    P_efficiency = 0.0
    R_connection = 0.0

    # Calculate the current gear ratio
    input_gear = next((g for g in state.gears if g.is_driver), None)
    output_gear = next((g for g in state.gears if not g.is_driver), None)

    if input_gear and output_gear:
        current_ratio = input_gear.num_teeth / output_gear.num_teeth
        # Reward for being close to the target ratio
        ratio_error = abs(current_ratio - state.target_ratio) / state.target_ratio
        R_ratio = 100.0 * math.exp(-ratio_error * 10)  # Exponential reward
    
    # Penalty for the number of gears to encourage efficiency
    P_efficiency = len(state.gears) * config.P_GEAR_COUNT_PENALTY
    
    # Intermediate gear bonus - reward for gears that connect input to output
    R_intermediate = 0.0
    if input_gear and output_gear:
        # Count gears that are connected to both input and output
        intermediate_gears = [g for g in state.gears if g != input_gear and g != output_gear]
        for gear in intermediate_gears:
            if (state.is_connected(input_gear, gear) and 
                state.is_connected(gear, output_gear)):
                R_intermediate += 50.0  # Bonus for each intermediate gear that connects input to output

    # Bonus for a successful connection
    if success:
        R_connection = 500.0  # Increased bonus for successful connection
        
    # Additional reward for each meshing gear pair
    meshing_pairs = 0
    for i in range(len(state.gears)):
        for j in range(i+1, len(state.gears)):
            if state.is_connected(state.gears[i], state.gears[j]):
                meshing_pairs += 1
                
    R_meshing = meshing_pairs * 30.0  # Reward for each meshing pair
    
    # Total reward calculation
    total_reward = R_ratio + P_efficiency + R_connection + R_meshing + R_intermediate
    
    return total_reward
