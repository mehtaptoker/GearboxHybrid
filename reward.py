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
        R_ratio = math.exp(-ratio_error * 10)  # Exponential reward
    
    # Penalty for the number of gears to encourage efficiency
    P_efficiency = len(state.gears) * config.P_GEAR_COUNT_PENALTY

    # Bonus for a successful connection
    if success:
        R_connection = 10.0
    
    # Total reward calculation
    total_reward = R_ratio + P_efficiency + R_connection
    
    return total_reward
