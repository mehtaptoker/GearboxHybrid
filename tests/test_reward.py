import unittest
from unittest.mock import MagicMock, patch
from components import SystemState, Gear, Point
import physics
import config
import reward

class TestRewardFunction(unittest.TestCase):
    def setUp(self):
        # Create mock gears
        self.input_gear = Gear(center=Point(0, 0), radius=10, num_teeth=20, is_driver=True)
        self.output_gear = Gear(center=Point(30, 0), radius=10, num_teeth=10)
        self.intermediate_gear = Gear(center=Point(15, 0), radius=5, num_teeth=10)
        
        # Create system state
        self.state = SystemState(
            gears=[self.input_gear, self.output_gear],
            boundary_poly=[],
            connections={(0, 1): True},
            target_ratio=2.0
        )
        
        # Mock physics functions
        physics.calculate_gear_train_ratio = MagicMock(return_value=2.0)
        physics.is_gear_inside_boundary = MagicMock(return_value=True)
        physics.check_collision = MagicMock(return_value=False)
        
        # Reset config values
        config.ALPHA = 1.0
        config.BETA = 1.0
        config.COLLISION_PENALTY = -100.0
        config.P_GEAR_COUNT_PENALTY = -10.0
        config.W_RATIO_SUCCESS = 500.0
        config.MESHING_TOLERANCE = 0.1

    def test_perfect_ratio_success(self):
        """Test reward when ratio is perfect and connection is successful"""
        reward_val = reward.calculate_reward(self.state, True)
        
        # Should get ratio reward (1.0) + success bonus (500) - no penalties
        self.assertAlmostEqual(reward_val, 1.0 + 500.0)
        
    def test_ratio_error(self):
        """Test reward with ratio error"""
        physics.calculate_gear_train_ratio.return_value = 2.5  # Target is 2.0
        reward_val = reward.calculate_reward(self.state, True)
        
        # Ratio error penalty: exp(-1*(0.5^2)) = exp(-0.25) ≈ 0.7788
        expected_ratio = 0.7788
        self.assertAlmostEqual(reward_val, expected_ratio + 500.0, places=4)
        
    def test_connectivity_penalty(self):
        """Test connectivity penalty for meshing gaps"""
        # Add intermediate gear with connection issues
        self.state.gears.append(self.intermediate_gear)
        self.state.connections = {(0, 2): True, (2, 1): True}
        
        # Calculate reward - the connectivity penalty will be applied
        reward_val = reward.calculate_reward(self.state, True)
        
        # Should have connectivity penalty in addition to other rewards
        self.assertLess(reward_val, 1.0 + 500.0 - 10.0)  # Gear count penalty also applies
        
    def test_collision_penalty(self):
        """Test collision penalty"""
        physics.is_gear_inside_boundary.return_value = False
        reward_val = reward.calculate_reward(self.state, True)
        
        # Should have collision penalty
        self.assertAlmostEqual(reward_val, 1.0 + 500.0 + config.COLLISION_PENALTY)
        
    def test_gear_count_penalty(self):
        """Test penalty for additional gears"""
        self.state.gears.append(self.intermediate_gear)
        reward_val = reward.calculate_reward(self.state, True)
        
        # Should have gear count penalty
        self.assertAlmostEqual(reward_val, 1.0 + 500.0 + config.P_GEAR_COUNT_PENALTY)
        
    def test_no_success_bonus(self):
        """Test case where connection fails"""
        reward_val = reward.calculate_reward(self.state, False)
        
        # Should not have success bonus
        self.assertAlmostEqual(reward_val, 1.0)

if __name__ == '__main__':
    unittest.main()
