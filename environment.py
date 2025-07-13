import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import math
import config
from components import SystemState, Gear, Vector2D
from data_generator import generate_scenario
from physics import check_collision, is_inside_boundary, calculate_gear_train
from reward import calculate_reward

class GearEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(GearEnv, self).__init__()
        # Define Action Space: [action_type, x, y, num_teeth, z_layer], normalized.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

        # Define Observation Space
        obs_size = 5 + (config.MAX_GEARS * 4)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        
        # Initialize state and step counter
        self.state = None
        self.step_count = 0

    def _get_observation(self) -> np.ndarray:
        """Flatten system state into observation vector."""
        # Start with system parameters: [target_ratio, input_x, input_y, output_x, output_y]
        obs = [
            self.state.target_ratio,
            self.state.input_shaft.x,
            self.state.input_shaft.y,
            self.state.output_shaft.x,
            self.state.output_shaft.y
        ]
        
        # Add gear parameters: [x, y, num_teeth, z_layer] for each gear
        for gear in self.state.gears:
            obs.extend([gear.center.x, gear.center.y, gear.num_teeth, gear.z_layer])
        
        # Pad with zeros for remaining gear slots
        remaining_slots = config.MAX_GEARS - len(self.state.gears)
        obs.extend([0] * (remaining_slots * 4))
        
        return np.array(obs, dtype=np.float32)

    def _decode_action(self, action: np.ndarray) -> Gear:
        """Convert normalized action to physical gear parameters."""
        # Action type: 0 = new gear, 1 = modify input gear, 2 = modify output gear
        action_type = int((action[0] + 1) * 1.5)  # Map [-1,1] to [0,2]
        
        if action_type == 0:  # Place new gear
            x = (action[1] + 1) / 2 * config.WORKSPACE_SIZE - config.WORKSPACE_SIZE/2
            y = (action[2] + 1) / 2 * config.WORKSPACE_SIZE - config.WORKSPACE_SIZE/2
            num_teeth = int((action[3] + 1) / 2 * (config.MAX_TEETH - config.MIN_TEETH) + config.MIN_TEETH)
            z_layer = int((action[4] + 1) / 2)  # 0 or 1
            
            return Gear(
                id=len(self.state.gears) + 1,
                center=Vector2D(x, y),
                num_teeth=num_teeth,
                module=config.GEAR_MODULE,
                z_layer=z_layer
            )
        elif action_type == 1:  # Modify input gear
            # Only modify teeth count
            input_gear = self.state.gears[0]
            new_teeth = int((action[3] + 1) / 2 * (config.MAX_TEETH - config.MIN_TEETH) + config.MIN_TEETH)
            return Gear(
                id=input_gear.id,
                center=input_gear.center,
                num_teeth=new_teeth,
                module=input_gear.module,
                z_layer=input_gear.z_layer,
                is_driver=True
            )
        else:  # Modify output gear
            output_gear = self.state.gears[1]
            new_teeth = int((action[3] + 1) / 2 * (config.MAX_TEETH - config.MIN_TEETH) + config.MIN_TEETH)
            return Gear(
                id=output_gear.id,
                center=output_gear.center,
                num_teeth=new_teeth,
                module=output_gear.module,
                z_layer=output_gear.z_layer,
                is_driver=False
            )

    def reset(self, seed=None, options=None) -> tuple:
        """Reset environment with new scenario."""
        super().reset(seed=seed)
        # Generate new scenario
        scenario = generate_scenario()
        
        # Create initial state
        self.state = SystemState(
            boundary_poly=scenario["boundary_poly"],
            gears=[],
            input_shaft=scenario["input_shaft"],
            output_shaft=scenario["output_shaft"],
            target_ratio=scenario["target_ratio"]
        )
        
        # Create and add input gear
        input_gear = Gear(
            id=1,
            center=Vector2D(self.state.input_shaft.x, self.state.input_shaft.y),
            num_teeth=random.randint(config.MIN_TEETH, config.MAX_TEETH),
            module=config.GEAR_MODULE,
            z_layer=0,
            is_driver=True
        )
        self.state.gears.append(input_gear)
        
        # Create and add output gear
        output_gear = Gear(
            id=2,
            center=Vector2D(self.state.output_shaft.x, self.state.output_shaft.y),
            num_teeth=random.randint(config.MIN_TEETH, config.MAX_TEETH),
            module=config.GEAR_MODULE,
            z_layer=0,
            is_driver=False
        )
        self.state.gears.append(output_gear)
        
        # Set initial gear train ratio
        self.initial_ratio = input_gear.num_teeth / output_gear.num_teeth
        
        # Reset step counter
        self.step_count = 0
        
        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> tuple:
        """Process agent action and update environment state."""
        self.step_count += 1
        terminated = False
        truncated = False
        info = {}
        
        # Decode action into gear
        new_gear = self._decode_action(action)
        
        # Check for immediate failure conditions
        if check_collision(new_gear, self.state.gears):
            reward = config.P_COLLISION
            terminated = True
        elif not is_inside_boundary(new_gear.center, self.state.boundary_poly):
            reward = config.P_OUT_OF_BOUNDS
            terminated = True
        else:
            # Valid placement - add gear to state
            self.state.gears.append(new_gear)
            
            # Calculate distance to output shaft for the new gear
            dist_to_output = math.sqrt(
                (new_gear.center.x - self.state.output_shaft.x)**2 +
                (new_gear.center.y - self.state.output_shaft.y)**2
            )
            
            # Normalized distance reward (closer is better)
            max_dist = math.sqrt(2) * config.WORKSPACE_SIZE
            dist_reward = (1 - dist_to_output / max_dist) * 0.5
            
            # Check truncation conditions
            if len(self.state.gears) >= config.MAX_GEARS or self.step_count >= config.MAX_STEPS_PER_EPISODE:
                truncated = True
                
            # Calculate final reward if episode ends
            success = False
            if terminated or truncated:
                # Find gear closest to output shaft
                output_gear_id = min(
                    self.state.gears,
                    key=lambda g: math.sqrt((g.center.x - self.state.output_shaft.x)**2 + 
                                          (g.center.y - self.state.output_shaft.y)**2)
                ).id
                
                # Calculate gear train ratio
                input_gear_id = next(g.id for g in self.state.gears if g.is_driver)
                ratio = calculate_gear_train(self.state.gears, input_gear_id, output_gear_id)
                
                # Check if successful connection
                success = ratio is not None
                final_reward = 10.0 if success else -1.0
            else:
                final_reward = 0.0
            
            # Combine rewards
            reward = 0.1 + dist_reward + final_reward
                
        # Ensure episode doesn't get stuck
        if self.step_count > config.MAX_STEPS_PER_EPISODE // 2 and reward < 0.5:
            truncated = True
            reward = -0.5  # Penalty for getting stuck
        
        return self._get_observation(), reward, terminated, truncated, info
