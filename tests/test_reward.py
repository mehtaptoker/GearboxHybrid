import unittest
from unittest.mock import MagicMock, patch
import math
from components import SystemState, Gear, Vector2D
import physics
import config
from reward import calculate_reward

class TestRewardFunction(unittest.TestCase):
    def setUp(self):
        # Create a minimal valid system state
        self.state = SystemState(
            boundary_poly=[Vector2D(0,0), Vector2D(100,0), Vector2D(100,100), Vector2D(0,100)],
            gears=[],
            connections=[],
            input_shaft=Vector2D(10,10),
            output_shaft=Vector2D(90,90),
            target_ratio=2.0,
            obstacles=[]
        )
        
        # Create mock gears
        self.input_gear = Gear(
            id=1,
            center=Vector2D(10,10),
            num_teeth=20,
            module=1.0,
            is_driver=True
        )
        
        self.output_gear = Gear(
            id=2,
            center=Vector2D(30,10),
            num_teeth=40,
            module=1.0
        )
        
        self.intermediate_gear = Gear(
            id=3,
            center=Vector2D(20,10),
            num_teeth=30,
            module=1.0
        )
        
        # Create connections
        from components import Connection
        self.connection1 = Connection(gear1=self.input_gear, gear2=self.intermediate_gear)
        self.connection2 = Connection(gear1=self.intermediate_gear, gear2=self.output_gear)
        
        # Mock get_gear_by_id method
        self.state.get_gear_by_id = MagicMock(side_effect=lambda id: 
            next((g for g in self.state.gears if g.id == id), None))
    
    @patch('physics.calculate_gear_train_ratio')
    def test_torque_ratio_reward(self, mock_ratio):
        """Test torque ratio reward component"""
        self.state.gears = [self.input_gear, self.output_gear]
        self.state.connections = []
        
        # Test perfect match
        mock_ratio.return_value = 2.0
        reward = calculate_reward(self.state)
        self.assertAlmostEqual(reward, 1.0, delta=0.01)
        
        # Test close match
        mock_ratio.return_value = 2.1
        reward = calculate_reward(self.state)
        self.assertGreater(reward, 0.5)
        self.assertLess(reward, 1.0)
        
        # Test poor match
        mock_ratio.return_value = 3.0
        reward = calculate_reward(self.state)
        self.assertLess(reward, 0.1)
    
    def test_connectivity_penalty(self):
        """Test connectivity penalty component"""
        # Create gears with perfect meshing
        self.input_gear.center = Vector2D(10,10)
        self.intermediate_gear.center = Vector2D(30,10)  # 20 units away (10+20=30)
        self.output_gear.center = Vector2D(50,10)  # 20 units away (30+20=50)
        
        self.state.gears = [self.input_gear, self.intermediate_gear, self.output_gear]
        self.state.connections = [self.connection1, self.connection2]
        
        # Calculate reward with perfect meshing
        reward_perfect = calculate_reward(self.state)
        
        # Introduce meshing errors
        self.intermediate_gear.center = Vector2D(31,10)  # 1mm gap
        reward_gap = calculate_reward(self.state)
        
        # Verify penalty is proportional to total gap
        self.assertLess(reward_gap, reward_perfect)
        self.assertAlmostEqual(reward_perfect - reward_gap, config.BETA * 1.0, delta=0.01)
        
        # Test with no connections
        self.state.connections = []
        reward_no_connections = calculate_reward(self.state)
        self.assertAlmostEqual(reward_no_connections, config.COLLISION_PENALTY, delta=0.01)
    
    def test_collision_penalty(self):
        """Test collision penalty component"""
        self.state.gears = [self.input_gear, self.output_gear]
        self.state.connections = []
        
        # Test without collisions
        with patch('physics.check_collision', return_value=False):
            reward_no_collision = calculate_reward(self.state)
        
        # Test with collisions
        with patch('physics.check_collision', return_value=True):
            reward_collision = calculate_reward(self.state)
        
        # Verify large constant penalty applied
        self.assertAlmostEqual(reward_collision, reward_no_collision + config.COLLISION_PENALTY, delta=0.01)
    
    def test_composite_reward(self):
        """Test the composite reward calculation"""
        self.state.gears = [self.input_gear, self.output_gear]
        self.state.connections = []
        
        # Mock dependencies
        with patch('physics.calculate_gear_train_ratio', return_value=2.0), \
             patch('physics.check_collision', return_value=False):
            
            # Create perfect scenario
            self.input_gear.center = Vector2D(10,10)
            self.output_gear.center = Vector2D(30,10)  # Perfect distance (20mm)
            reward_perfect = calculate_reward(self.state)
            
            # Create scenario with issues
            self.output_gear.center = Vector2D(31,10)  # 1mm gap
            with patch('physics.check_collision', return_value=True):  # Collision
                reward_imperfect = calculate_reward(self.state)
        
        # Verify composite reward includes all components
        self.assertAlmostEqual(reward_perfect, 1.0, delta=0.01)
        self.assertAlmostEqual(reward_imperfect, 
                               1.0 - config.BETA * 1.0 + config.COLLISION_PENALTY, 
                               delta=0.01)

if __name__ == '__main__':
    unittest.main()
