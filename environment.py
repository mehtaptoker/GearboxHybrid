import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import math
import time  # Added for timing
import os
import json
import config
from components import SystemState, Gear, Vector2D
from data_generator import generate_scenario
from physics import check_collision, is_inside_boundary, calculate_gear_train_ratio, is_gear_inside_boundary
from reward import calculate_reward

class GearEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data_dir=None, verbose=0):
        super(GearEnv, self).__init__()
        self.data_dir = data_dir if data_dir else 'data/intermediate'
        self.verbose = verbose  # 0: no debug, 1: basic debug, 2: detailed debug
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
        
        # Ensure consistent observation size
        expected_size = 5 + (config.MAX_GEARS * 4)
        if len(obs) != expected_size:
            # Pad or truncate to expected size
            if len(obs) < expected_size:
                obs.extend([0] * (expected_size - len(obs)))
            else:
                obs = obs[:expected_size]
        
        return np.array(obs, dtype=np.float32)

    def _decode_action(self, action: np.ndarray) -> Gear:
        """Convert normalized action to physical gear parameters."""
        # Action type: 0 = new gear, 1 = modify input gear, 2 = modify output gear
        action_type = int((action[0] + 1) * 1.5)  # Map [-1,1] to [0,2]
        
        if action_type == 0:  # Place new gear
            # Clamp coordinates to workspace boundaries
            x_val = (action[1] + 1) / 2 * config.WORKSPACE_SIZE - config.WORKSPACE_SIZE/2
            y_val = (action[2] + 1) / 2 * config.WORKSPACE_SIZE - config.WORKSPACE_SIZE/2
            x = max(-config.WORKSPACE_SIZE/2, min(config.WORKSPACE_SIZE/2, x_val))
            y = max(-config.WORKSPACE_SIZE/2, min(config.WORKSPACE_SIZE/2, y_val))
            
            # Clamp teeth count to valid range
            teeth_val = int((action[3] + 1) / 2 * (config.MAX_TEETH - config.MIN_TEETH) + config.MIN_TEETH)
            num_teeth = max(config.MIN_TEETH, min(config.MAX_TEETH, teeth_val))
            
            # Ensure teeth count is valid
            if num_teeth < config.MIN_TEETH or num_teeth > config.MAX_TEETH:
                num_teeth = random.randint(config.MIN_TEETH, config.MAX_TEETH)
            
            # Clamp z-layer to 0 or 1
            z_val = int((action[4] + 1) / 2)
            z_layer = max(0, min(1, z_val))
            
            # Create new gear
            new_gear = Gear(
                id=len(self.state.gears) + 1,
                center=Vector2D(x, y),
                num_teeth=num_teeth,
                module=config.GEAR_MODULE,
                z_layer=z_layer
            )
            
            # The validation of the gear placement (collision, boundary)
            # is handled in the step function. _decode_action should just
            # convert the action to a gear without any correction logic.
            
            return new_gear
        elif action_type == 1:  # Modify input gear
            # Only modify teeth count
            input_gear = self.state.gears[0]
            new_teeth = int((action[3] + 1) / 2 * (config.MAX_TEETH - config.MIN_TEETH) + config.MIN_TEETH)
            # Clamp teeth count to valid range
            new_teeth = max(config.MIN_TEETH, min(config.MAX_TEETH, new_teeth))
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
            # Clamp teeth count to valid range
            new_teeth = max(config.MIN_TEETH, min(config.MAX_TEETH, new_teeth))
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
        # Load scenario from data_dir
        if self.data_dir and os.path.exists(self.data_dir):
            if os.path.isdir(self.data_dir):
                scenario_files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
                if scenario_files:
                    scenario_file = random.choice(scenario_files)
                    with open(os.path.join(self.data_dir, scenario_file), 'r') as f:
                        scenario = json.load(f)
                else:
                    scenario = generate_scenario()
            elif os.path.isfile(self.data_dir):
                with open(self.data_dir, 'r') as f:
                    scenario = json.load(f)
            else:
                scenario = generate_scenario()
        else:
            scenario = generate_scenario()

        # Normalize coordinates to fit into the agent's workspace
        boundary_points = [Vector2D(p['x'], p['y']) for p in scenario["boundary_poly"]]
        
        # Find bounding box of the original polygon
        min_x = min(p.x for p in boundary_points)
        max_x = max(p.x for p in boundary_points)
        min_y = min(p.y for p in boundary_points)
        max_y = max(p.y for p in boundary_points)

        original_width = max_x - min_x
        original_height = max_y - min_y

        # Determine scale factor
        scale = config.WORKSPACE_SIZE / max(original_width, original_height)

        # Calculate offsets to center the polygon
        offset_x = -min_x * scale + (config.WORKSPACE_SIZE - original_width * scale) / 2 - config.WORKSPACE_SIZE / 2
        offset_y = -min_y * scale + (config.WORKSPACE_SIZE - original_height * scale) / 2 - config.WORKSPACE_SIZE / 2

        def transform_point(p):
            new_x = p.x * scale + offset_x
            new_y = p.y * scale + offset_y
            return Vector2D(new_x, new_y)

        # Transform all coordinates
        normalized_boundary = [transform_point(p) for p in boundary_points]
        normalized_input_shaft = transform_point(Vector2D(scenario["input_shaft"]['x'], scenario["input_shaft"]['y']))
        normalized_output_shaft = transform_point(Vector2D(scenario["output_shaft"]['x'], scenario["output_shaft"]['y']))

        # Create initial state with normalized coordinates
        self.state = SystemState(
            boundary_poly=normalized_boundary,
            gears=[],
            input_shaft=normalized_input_shaft,
            output_shaft=normalized_output_shaft,
            target_ratio=eval(scenario["constraints"]["torque_ratio"].replace(":", "/"))
        )
        
        # Create and add input gear
        input_gear = Gear(
            id=1,
            center=self.state.input_shaft,
            num_teeth=random.randint(config.MIN_TEETH, config.MAX_TEETH),
            module=config.GEAR_MODULE,
            z_layer=0,
            is_driver=True
        )
        
        # Create and add output gear
        output_gear = Gear(
            id=2,
            center=self.state.output_shaft,
            num_teeth=random.randint(config.MIN_TEETH, config.MAX_TEETH),
            module=config.GEAR_MODULE,
            z_layer=0,
            is_driver=False
        )

        # Calculate distance between input and output shafts
        dist_center = math.sqrt(
            (output_gear.center.x - input_gear.center.x)**2 +
            (output_gear.center.y - input_gear.center.y)**2
        )
        
        # Try to add intermediate gear with timeout
        intermediate_placed = False
        start_time = time.time()
        timeout = 0.5  # 0.5 second timeout
        
        # Calculate possible intermediate teeth range
        min_inter_teeth = max(config.MIN_TEETH, 
                             int((dist_center - input_gear.radius - output_gear.radius) / config.GEAR_MODULE))
        max_inter_teeth = min(config.MAX_TEETH, 
                             int(dist_center / config.GEAR_MODULE))
        
        if min_inter_teeth <= max_inter_teeth:
            # Try teeth counts from largest to smallest (more likely to mesh)
            teeth_options = list(range(max_inter_teeth, min_inter_teeth-1, -1))
            for num_teeth_intermediate in teeth_options[:10]:  # Limit to 10 attempts
                # Check timeout
                if time.time() - start_time > timeout:
                    if self.verbose:
                        print("TIMEOUT: Intermediate gear placement took too long")
                    break
                
                # Calculate possible positions
                positions = self._calculate_intermediate_positions(input_gear, output_gear, num_teeth_intermediate)
                
                for position in positions:
                    intermediate_gear = Gear(
                        id=3,
                        center=position,
                        num_teeth=num_teeth_intermediate,
                        module=config.GEAR_MODULE,
                        z_layer=0,
                        is_driver=False
                    )
                    
                    # Check placement validity
                    boundary_ok = is_gear_inside_boundary(intermediate_gear, self.state.boundary_poly)
                    collision_ok = not check_collision(intermediate_gear, [input_gear, output_gear])
                    meshing_ok = self._check_gear_meshing(input_gear, intermediate_gear) and \
                                 self._check_gear_meshing(intermediate_gear, output_gear)
                    
                    if boundary_ok and collision_ok and meshing_ok:
                        self.state.gears.extend([input_gear, output_gear, intermediate_gear])
                        intermediate_placed = True
                        if self.verbose:
                            print(f"SUCCESS: Intermediate gear placed with {num_teeth_intermediate} teeth")
                        break  # Break out of position loop
                if intermediate_placed:
                    break  # Break out of teeth loop
        
        if not intermediate_placed:
            self.state.gears.extend([input_gear, output_gear])
            if self.verbose:
                print("WARNING: Intermediate gear placement failed after all attempts")
        
        # Calculate actual gear train ratio
        if len(self.state.gears) == 3:
            # With intermediate: ratio = (driver/driven) = (input/intermediate) * (intermediate/output)
            input_gear = self.state.gears[0]
            intermediate_gear = self.state.gears[1]
            output_gear = self.state.gears[2]
            self.initial_ratio = (input_gear.num_teeth / intermediate_gear.num_teeth) * \
                                 (intermediate_gear.num_teeth / output_gear.num_teeth)
        else:
            # Direct connection
            self.initial_ratio = input_gear.num_teeth / output_gear.num_teeth
        
        # Reset step counter
        self.step_count = 0
        
        return self._get_observation(), {}
    
    def _calculate_intermediate_positions(self, gear1, gear2, intermediate_teeth):
        """Calculate possible positions for intermediate gear between two gears using circle-circle intersection."""
        P0 = gear1.center
        P1 = gear2.center
        d = math.sqrt((P1.x - P0.x)**2 + (P1.y - P0.y)**2)
        
        # Radii for meshing circles: distance from gear1 to intermediate
        r0 = gear1.radius + (intermediate_teeth * config.GEAR_MODULE / 2)
        # Distance from gear2 to intermediate
        r1 = gear2.radius + (intermediate_teeth * config.GEAR_MODULE / 2)
        
        # But note: the distance between the two existing gears is d
        # We actually need the distance from gear1 to intermediate (r0) and from gear2 to intermediate (r1)
        # The circles we're intersecting are centered at gear1 and gear2 with radii r0 and r1 respectively
        
        # Check if solution is possible with tolerance for floating point precision
        tolerance = 1e-5
        if d > (r0 + r1) + tolerance:
            return []  # Too far apart
        if d < abs(r0 - r1) - tolerance:
            return []  # One circle inside the other
        
        # Calculate intermediate point
        a = (r0**2 - r1**2 + d**2) / (2 * d)
        
        # Prevent math domain error
        h_sq = r0**2 - a**2
        if h_sq < 0:
            return []
        
        h = math.sqrt(h_sq)
        
        # Calculate P2 point (base point on the line between P0 and P1)
        P2 = Vector2D(
            P0.x + a * (P1.x - P0.x) / d,
            P0.y + a * (P1.y - P0.y) / d
        )
        
        # Calculate possible positions (perpendicular to the line)
        positions = [
            Vector2D(
                P2.x + h * (P1.y - P0.y) / d,
                P2.y - h * (P1.x - P0.x) / d
            ),
            Vector2D(
                P2.x - h * (P1.y - P0.y) / d,
                P2.y + h * (P1.x - P0.x) / d
            )
        ]
        
        return positions

    def step(self, action: np.ndarray) -> tuple:
        """Process agent action and update environment state."""
        start_time = time.time()
        self.step_count += 1
        terminated = False
        truncated = False
        info = {}
        
        if self.verbose >= 1:
            print(f"\n--- Step {self.step_count} ---")
            print(f"Action: {action}")
        
        # Decode action into gear
        new_gear = self._decode_action(action)
        
        # Determine action type (same as in _decode_action)
        action_type = int((action[0] + 1) * 1.5)  # Map [-1,1] to [0,2]
        
        if self.verbose >= 1:
            print(f"Decoded gear: center=({new_gear.center.x:.2f}, {new_gear.center.y:.2f}), "
                  f"teeth={new_gear.num_teeth}, z_layer={new_gear.z_layer}, action_type={action_type}")
        
        # Check for failure conditions
        if action_type == 0:  # Only check for new gears
            # Time boundary check with detailed output
            boundary_start = time.time()
            boundary_inside, boundary_reason = is_gear_inside_boundary(new_gear, self.state.boundary_poly, return_reason=True)
            boundary_time = time.time() - boundary_start
            
            # Time collision check with detailed output
            collision_start = time.time()
            collision_detected, collision_reason = check_collision(new_gear, self.state.gears, return_reason=True)
            collision_time = time.time() - collision_start
            
            if self.verbose >= 1:
                print(f"Boundary check took {boundary_time:.6f} seconds: {boundary_reason}")
                print(f"Collision check took {collision_time:.6f} seconds: {collision_reason}")
            
            if collision_detected:
                if self.verbose >= 1:
                    print("FAIL: Gear collision detected")
                reward = config.P_COLLISION
                terminated = True
            elif not boundary_inside:
                if self.verbose >= 1:
                    print("FAIL: Gear outside boundary")
                reward = config.P_OUT_OF_BOUNDS
                terminated = True
            else:
                # Valid placement - add gear to state
                self.state.gears.append(new_gear)
                # Don't terminate on success
                terminated = False
                reward = 1.0  # Base reward for successful placement
                
                if self.verbose >= 1:
                    print("SUCCESS: Gear placed successfully")
                    print(f"Current gear count: {len(self.state.gears)}")
                    
                if self.verbose >= 2:
                    print("Current gears:")
                    for gear in self.state.gears:
                        print(f"  Gear {gear.id}: center=({gear.center.x:.2f}, {gear.center.y:.2f}), "
                              f"teeth={gear.num_teeth}, z_layer={gear.z_layer}")
                
                # Calculate distance to output shaft for the new gear
                dist_to_output = math.sqrt(
                    (new_gear.center.x - self.state.output_shaft.x)**2 +
                    (new_gear.center.y - self.state.output_shaft.y)**2
                )
                
                # Normalized distance reward (closer is better)
                max_dist = math.sqrt(2) * config.WORKSPACE_SIZE
                dist_reward = (1 - dist_to_output / max_dist) * 0.5
                reward += dist_reward
        else:  # Modification action
            # Find and update existing gear
            found = False
            for i, gear in enumerate(self.state.gears):
                if gear.id == new_gear.id:
                    self.state.gears[i] = new_gear
                    found = True
                    break
            
            if not found:
                if self.verbose >= 1:
                    print(f"FAIL: Gear with id {new_gear.id} not found")
                reward = config.P_INVALID_ACTION
                terminated = True
            else:
                if self.verbose >= 1:
                    print(f"SUCCESS: Modified gear {new_gear.id}")
                terminated = False
                reward = 0.1  # Small reward for successful modification
                
        # Only calculate reward if not terminated
        if not terminated:
            # Check truncation conditions
            if len(self.state.gears) >= config.MAX_GEARS or self.step_count >= config.MAX_STEPS_PER_EPISODE:
                truncated = True
                    
                if self.verbose >= 2:
                    print("Current gears:")
                    for gear in self.state.gears:
                        print(f"  Gear {gear.id}: center=({gear.center.x:.2f}, {gear.center.y:.2f}), "
                              f"teeth={gear.num_teeth}, z_layer={gear.z_layer}")
            
            # Calculate reward only if not terminated
            success = self._is_successful_connection()
            reward = calculate_reward(self.state, success)
        
        end_time = time.time()
        step_duration = end_time - start_time
        if self.verbose >= 1:
            print(f"Step took {step_duration:.4f} seconds")
        
        return self._get_observation(), reward, terminated, truncated, info

    def _check_gear_meshing(self, gear1: Gear, gear2: Gear) -> bool:
        """Check if two gears are properly meshing."""
        dist = math.sqrt((gear1.center.x - gear2.center.x)**2 + 
                         (gear1.center.y - gear2.center.y)**2)
        expected_dist = gear1.radius + gear2.radius
        return abs(dist - expected_dist) < config.GEAR_MODULE * 0.2  # Allow 20% tolerance
    
    def _is_successful_connection(self) -> bool:
        """Check if the gear train connects the input and output shafts."""
        if len(self.state.gears) < 2:
            return False
            
        # Check that all adjacent gears are meshing
        gears = self.state.gears
        for i in range(len(gears) - 1):
            if not self._check_gear_meshing(gears[i], gears[i+1]):
                return False
                
        # Check that input is connected to output through the chain
        return True
