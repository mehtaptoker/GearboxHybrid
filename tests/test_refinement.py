import unittest
import math
import sys
import os

# Add parent directory to path to allow module imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from components import Gear, Vector2D
from refinement import propagate_radii, validate_gear_train, adjust_joints
import config

class TestRefinement(unittest.TestCase):
    def setUp(self):
        self.start_gear = Gear(id=1, center=Vector2D(0, 0), num_teeth=20, module=1)
        self.end_gear = Gear(id=2, center=Vector2D(100, 0), num_teeth=30, module=1)
        self.path = [
            Vector2D(30, 0),
            Vector2D(50, 0),
            Vector2D(70, 0)
        ]
        self.existing_gears = []
        self.module = 1
        
    def test_propagate_radii(self):
        """Test radius propagation along a path"""
        radii, endpoint_error = propagate_radii(
            self.start_gear,
            self.end_gear,
            self.path,
            self.module
        )
        
        # Validate results
        self.assertEqual(len(radii), 3)
        self.assertAlmostEqual(radii[0], 10.0, delta=0.1)
        self.assertAlmostEqual(radii[1], 10.0, delta=0.1)
        self.assertAlmostEqual(radii[2], 10.0, delta=0.1)
        self.assertLess(endpoint_error, 1.0)
        
    def test_validate_gear_train_success(self):
        """Test successful gear train validation"""
        radii = [10, 10, 10]
        valid, reason = validate_gear_train(
            self.start_gear,
            self.end_gear,
            self.path,
            radii,
            self.existing_gears,
            self.module
        )
        self.assertTrue(valid)
        self.assertIsNone(reason)
        
    def test_validate_gear_train_failure(self):
        """Test gear train validation failure"""
        invalid_radii = [10, -5, 10]  # Negative radius
        valid, reason = validate_gear_train(
            self.start_gear,
            self.end_gear,
            self.path,
            invalid_radii,
            self.existing_gears,
            self.module
        )
        self.assertFalse(valid)
        self.assertEqual(reason, "Non-positive radius detected in intermediate gears")
        
    def test_adjust_joints(self):
        """Test joint position adjustment"""
        # Create a path with incorrect distances
        malformed_path = [
            Vector2D(20, 0),  # Too close to start
            Vector2D(50, 0),
            Vector2D(80, 0)   # Too close to end
        ]
        radii = [10, 10, 10]
        
        # Calculate endpoint error
        _, endpoint_error = propagate_radii(
            self.start_gear,
            self.end_gear,
            malformed_path,
            self.module
        )
        self.assertGreater(endpoint_error, 5.0)
        
        # Adjust joints
        adjusted_path = adjust_joints(
            malformed_path,
            self.start_gear,
            self.end_gear,
            radii,
            endpoint_error
        )
        
        # Verify adjustment
        self.assertAlmostEqual(adjusted_path[0].x, 30, delta=0.1)
        self.assertAlmostEqual(adjusted_path[-1].x, 70, delta=0.1)

if __name__ == '__main__':
    unittest.main()
