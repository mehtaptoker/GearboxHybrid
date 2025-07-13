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
    R_torque = 0.0  # Not implemented in initial version
    P_collision = 0.0
    P_out_of_bounds = 0.0
    P_no_connection = 0.0
    P_efficiency = 0.0
    P_weight = 0.0
    R_space_utilization = 0.0
    
    # Check if gear train is successful
    if success:
        R_ratio = config.W_RATIO_SUCCESS
    
    # Penalty if no connection
    if not success:
        P_no_connection = config.P_NO_CONNECTION
    
    # Efficiency penalty (based on number of gears)
    P_efficiency = len(state.gears) * config.P_EFFICIENCY
    
    # Calculate total mass and weight penalty
    total_mass = 0.0
    for gear in state.gears:
        total_mass += gear.mass(config.GEAR_THICKNESS, config.GEAR_DENSITY)
    P_weight = total_mass * config.WEIGHT_PENALTY
    
    # Space utilization reward (percentage of workspace covered)
    total_gear_area = 0.0
    for gear in state.gears:
        total_gear_area += math.pi * (gear.radius ** 2)
    workspace_area = config.WORKSPACE_SIZE ** 2
    space_utilization = total_gear_area / workspace_area
    R_space_utilization = space_utilization * 50.0  # Reward for efficient space usage
    
    # Total reward calculation
    total_reward = (
        R_ratio +
        R_torque +
        P_collision +
        P_out_of_bounds +
        P_no_connection +
        P_efficiency +
        P_weight +
        R_space_utilization
    )
    
    return total_reward
