import unittest
from refinement import propagate_radii, validate_gear_train, adjust_joints, generate_gear_train
from components import Gear, Vector2D
from physics import check_meshing
import config

class TestRefinement(unittest.TestCase):
    def setUp(self):
        # Create fixed gears
        self.start_gear = Gear(
            id=1,
            center=Vector2D(0, 0),
            num_teeth=20,
            module=config.GEAR_MODULE,
            is_driver=True
        )
        self.end_gear = Gear(
            id=2,
            center=Vector2D(30, 0),
            num_teeth=30,
            module=config.GEAR_MODULE
        )
        
        # Create intermediate path
        self.path = [
            Vector2D(10, 5),
            Vector2D(20, 5)
        ]
        
        # Existing gears (none for simplicity)
        self.existing_gears = []
        
    def test_propagate_radii(self):
        # Test radius propagation
        radii, endpoint_error = propagate_radii(
            self.start_gear,
            self.end_gear,
            self.path,
            config.GEAR_MODULE
        )
        
        # Check intermediate radii
        self.assertAlmostEqual(radii[0], 9.472, places=2)
        self.assertAlmostEqual(radii[1], 10.0, places=2)
        
        # Check endpoint error
        self.assertGreater(endpoint_error, 0)
        
    def test_validate_gear_train(self):
        # Test validation with valid gear train
        radii = [10.0, 10.0]
        valid, reason = validate_gear_train(
            self.start_gear,
            self.end_gear,
            self.path,
            radii,
            self.existing_gears,
            config.GEAR_MODULE
        )
        self.assertTrue(valid)
        self.assertIsNone(reason)
        
        # Test with non-positive radius
        radii_invalid = [10.0, -5.0]
        valid, reason = validate_gear_train(
            self.start_gear,
            self.end_gear,
            self.path,
            radii_invalid,
            self.existing_gears,
            config.GEAR_MODULE
        )
        self.assertFalse(valid)
        self.assertEqual(reason, "Non-positive radius detected in intermediate gears")
        
    def test_adjust_joints(self):
        # Create initial path with endpoint error
        path = [Vector2D(10, 0), Vector2D(20, 0)]
        _, endpoint_error = propagate_radii(
            self.start_gear,
            self.end_gear,
            path,
            config.GEAR_MODULE
        )
        
        # Adjust joints
        adjusted_path = adjust_joints(
            path,
            self.start_gear,
            self.end_gear,
            [10.0, 10.0],
            endpoint_error
        )
        
        # Last joint should be closer to end gear
        self.assertGreater(adjusted_path[-1].x, path[-1].x)
        
    def test_generate_gear_train(self):
        # Test successful gear train generation
        intermediate_gears = generate_gear_train(
            self.start_gear,
            self.end_gear,
            self.path,
            self.existing_gears,
            config.GEAR_MODULE
        )
        
        self.assertIsNotNone(intermediate_gears)
        self.assertEqual(len(intermediate_gears), 2)
        
        # Check meshing
        self.assertTrue(check_meshing(self.start_gear, intermediate_gears[0]))
        self.assertTrue(check_meshing(intermediate_gears[0], intermediate_gears[1]))
        self.assertTrue(check_meshing(intermediate_gears[1], self.end_gear))

if __name__ == '__main__':
    unittest.main()
