import unittest
import numpy as np
import config
from environment import GearEnv
from components import Gear, Vector2D, SystemState

class TestGearEnvironment(unittest.TestCase):
    def setUp(self):
        self.env = GearEnv(verbose=1)  # Enable verbose mode to see debug messages
        # Create a simple square boundary matching config.WORKSPACE_SIZE
        half_size = config.WORKSPACE_SIZE / 2
        self.boundary = [
            Vector2D(-half_size, -half_size),
            Vector2D(half_size, -half_size),
            Vector2D(half_size, half_size),
            Vector2D(-half_size, half_size)
        ]
        self.env.state = SystemState(
            boundary_poly=self.boundary,
            gears=[],
            input_shaft=Vector2D(0, 0),
            output_shaft=Vector2D(10, 10),
            target_ratio=2.0
        )
        # Only define input and output shafts - no gears initially
        self.env.state.gears = []

    def test_boundary_checks(self):
        """Test gear placement boundary validation"""
        # Valid placement with smaller gear that fits boundaries
        valid_gear = Gear(
            id=1,
            center=Vector2D(5, 15),  # Position away from shafts
            num_teeth=10,  # Smaller gear to fit boundaries
            module=0.5,  # Smaller module to reduce gear size
            z_layer=0
        )
        print(f"Valid gear center: ({valid_gear.center.x}, {valid_gear.center.y})")
        print(f"Gear radius: {valid_gear.radius}")
        print(f"Boundary: {self.boundary}")
        print(f"Boundary: {self.boundary}")
        obs, reward, terminated, truncated, info = self.env.step(
            self._create_action(valid_gear))
        print(f"Terminated: {terminated}, Reward: {reward}, Info: {info}")
        if terminated:
            # Print the reason for termination from info if available
            print(f"Termination reason: {info.get('reason', 'No reason provided')}")
        self.assertFalse(terminated, "Valid gear should be placed successfully")
        self.assertEqual(len(self.env.state.gears), 1, "Gear should be added to state")

        # Out of bounds placement
        out_of_bounds = Gear(
            id=4,
            center=Vector2D(60, 60),  # Outside the workspace
            num_teeth=30,
            module=config.GEAR_MODULE,
            z_layer=0
        )
        obs, reward, terminated, truncated, info = self.env.step(
            self._create_action(out_of_bounds))
        self.assertTrue(terminated, "Out of bounds gear should be rejected")
        self.assertEqual(reward, config.P_OUT_OF_BOUNDS, "Should return boundary penalty")

        # Partially out of bounds
        edge_gear = Gear(
            id=5,
            center=Vector2D(48, 48),  # Near the edge
            num_teeth=10,  # Smaller gear for consistent testing
            module=0.5,  # Smaller module to reduce gear size
            z_layer=0
        )
        obs, reward, terminated, truncated, info = self.env.step(
            self._create_action(edge_gear))
        self.assertTrue(terminated, "Gear exceeding boundary should be rejected")
        self.assertEqual(reward, config.P_OUT_OF_BOUNDS, "Should return boundary penalty")

    def test_collision_checks(self):
        """Test gear collision detection"""
        # First place a gear at (0,0)
        first_gear = Gear(
            id=1,
            center=Vector2D(0, 0),
            num_teeth=10,
            module=0.5,
            z_layer=0
        )
        obs, reward, terminated, truncated, info = self.env.step(
            self._create_action(first_gear))
        self.assertFalse(terminated, "First gear should be placed successfully")
        
        # Now try to place a colliding gear
        collision_gear = Gear(
            id=2,
            center=Vector2D(0.5, 0.5),
            num_teeth=10,
            module=0.5,
            z_layer=0
        )
        obs, reward, terminated, truncated, info = self.env.step(
            self._create_action(collision_gear))
        self.assertTrue(terminated, "Colliding gear should be rejected")
        self.assertEqual(reward, config.P_COLLISION, "Should return collision penalty")

        # Close but not colliding
        close_gear = Gear(
            id=3,
            center=Vector2D(0, 15),  # Move away from shafts
            num_teeth=10,
            module=0.5,  # Smaller module to reduce gear size
            z_layer=0
        )
        print(f"Close gear center: ({close_gear.center.x}, {close_gear.center.y})")
        obs, reward, terminated, truncated, info = self.env.step(
            self._create_action(close_gear))
        print(f"Terminated: {terminated}, Reward: {reward}, Info: {info}")
        self.assertFalse(terminated, "Non-colliding gear should be placed")
        self.assertEqual(len(self.env.state.gears), 2, "Gear should be added to state")

    def test_gear_train_calculation(self):
        """Test valid gear train connections"""
        # Set max steps to a small number
        original_max_steps = config.MAX_STEPS_PER_EPISODE
        config.MAX_STEPS_PER_EPISODE = 3
        
        # Add input gear
        input_gear = Gear(
            id=1,
            center=self.env.state.input_shaft,
            num_teeth=20,
            module=config.GEAR_MODULE,
            z_layer=0,
            is_driver=True
        )
        self.env.state.gears.append(input_gear)
        
        # Add output gear
        output_gear = Gear(
            id=2,
            center=self.env.state.output_shaft,
            num_teeth=40,
            module=config.GEAR_MODULE,
            z_layer=0,
            is_driver=False
        )
        self.env.state.gears.append(output_gear)
        
        # Add a valid intermediate gear that connects input to output
        valid_gear = Gear(
            id=3,
            center=Vector2D(7.5, 0),  # Positioned at correct distance from input
            num_teeth=10,
            module=0.5,
            z_layer=0
        )
        # Set connections
        valid_gear.connected_gears = [1, 2]
        input_gear.connected_gears = [3]
        output_gear.connected_gears = [3]
        self.env.state.gears.append(valid_gear)
        
        # Simulate max steps reached with a valid action
        self.env.step_count = config.MAX_STEPS_PER_EPISODE - 1
        # Use a valid action that doesn't add a new gear (modify input gear)
        # Use action[0] = -0.3 to get action_type=1 (modify input gear)
        action = np.array([-0.3, 0, 0, 0, 0])  # Modify input gear with valid parameters
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.assertTrue(truncated, "Episode should end when max steps reached")
        # We always give reward now with the simplified ratio calculation
        self.assertGreater(reward, 0, "Should get positive reward for connection")
        
        # Restore original max steps
        config.MAX_STEPS_PER_EPISODE = original_max_steps

    def test_invalid_gear_parameters(self):
        """Test handling of invalid gear parameters"""
        # Create and add input gear
        input_gear = Gear(
            id=1,
            center=Vector2D(0, 0),
            num_teeth=20,
            module=config.GEAR_MODULE,
            z_layer=0,
            is_driver=True
        )
        self.env.state.gears.append(input_gear)
        
        # Create and add output gear
        output_gear = Gear(
            id=2,
            center=Vector2D(70, 70),  # Move even further away to avoid collisions
            num_teeth=40,
            module=config.GEAR_MODULE,
            z_layer=0,
            is_driver=False
        )
        self.env.state.gears.append(output_gear)
        
        # Test modifying input gear teeth count
        action = np.array([-0.3, 0, 0, -1, 0])  # Modify input gear (action_type=1)
        obs, reward, terminated, truncated, info = self.env.step(action)
        # With simplified ratio, we don't terminate on invalid teeth count
        self.assertFalse(terminated, "Should handle invalid teeth count")
        # Should modify the existing input gear
        self.assertEqual(self.env.state.gears[0].num_teeth, config.MIN_TEETH, "Teeth should be clamped to min")

        # Reset environment and re-add gears
        self.setUp()
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
            center=Vector2D(70, 70),
            num_teeth=40,
            module=config.GEAR_MODULE,
            z_layer=0,
            is_driver=False
        )
        self.env.state.gears = [input_gear, output_gear]

        # Test modifying output gear teeth count
        action = np.array([0.5, 0, 0, 2, 0])  # Modify output gear (action_type=2)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.assertFalse(terminated, "Should handle teeth count above max")
        self.assertEqual(self.env.state.gears[1].num_teeth, config.MAX_TEETH, "Teeth should be clamped to max")
        
        # Test placing new gear with invalid teeth count
        action = np.array([-1, 0.5, 0.5, 2, 0])  # Place new gear with teeth above max
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.assertFalse(terminated, "Should handle teeth count above max for new gear")
        self.assertEqual(self.env.state.gears[2].num_teeth, config.MAX_TEETH, "Teeth should be clamped to max")
        
        # Position jitter for overlapping gears
        overlap_action = np.array([-1, 0, 0, 0, 0])  # Place at (0,0) - overlaps with input
        obs, reward, terminated, truncated, info = self.env.step(overlap_action)
        self.assertTrue(terminated, "Should terminate on overlapping positions")
        self.assertEqual(reward, config.P_COLLISION, "Should return collision penalty")

    def test_boundary_edge_cases(self):
        """Test edge cases for boundary checks"""
        # Gear within the boundary (safe distance)
        safe_gear = Gear(
            id=6,
            center=Vector2D(46, 46),  # 50 - 4 = 46 (gear radius is 2, so 46+2=48 < 50)
            num_teeth=8,
            module=0.5,
            z_layer=0
        )
        obs, reward, terminated, truncated, info = self.env.step(
            self._create_action(safe_gear))
        self.assertFalse(terminated, "Gear within boundary should be accepted")
        
        # Gear exactly at boundary edge (touching boundary) at a different location
        edge_gear = Gear(
            id=7,
            center=Vector2D(46, -46),  # Different location to avoid collision
            num_teeth=8,
            module=0.5,
            z_layer=0
        )
        obs, reward, terminated, truncated, info = self.env.step(
            self._create_action(edge_gear))
        self.assertFalse(terminated, "Gear at boundary edge should be accepted")
        
        # Gear partially overlapping boundary
        overlap_gear = Gear(
            id=8,
            center=Vector2D(49.9, 49.9),
            num_teeth=15,
            module=1.0,
            z_layer=0
        )
        obs, reward, terminated, truncated, info = self.env.step(
            self._create_action(overlap_gear))
        self.assertTrue(terminated, "Gear overlapping boundary should be rejected")
        
        # Large gear near boundary
        large_gear = Gear(
            id=9,
            center=Vector2D(45, 45),
            num_teeth=30,
            module=1.0,
            z_layer=0
        )
        obs, reward, terminated, truncated, info = self.env.step(
            self._create_action(large_gear))
        self.assertTrue(terminated, "Large gear near boundary should be rejected")

    def _create_action(self, gear):
        """Create action array from gear parameters"""
        # Action type for placing a new gear is -1 (less than -0.5)
        return np.array([
            -1,  # New gear action type
            (gear.center.x + config.WORKSPACE_SIZE/2) / config.WORKSPACE_SIZE * 2 - 1,
            (gear.center.y + config.WORKSPACE_SIZE/2) / config.WORKSPACE_SIZE * 2 - 1,
            (gear.num_teeth - config.MIN_TEETH) / (config.MAX_TEETH - config.MIN_TEETH) * 2 - 1,
            gear.z_layer * 2 - 1
        ])

if __name__ == '__main__':
    unittest.main()
