import unittest
from components import Gear, Vector2D
from physics import check_meshing, check_collision, is_inside_boundary, calculate_gear_train

class TestPhysics(unittest.TestCase):
    def test_check_meshing(self):
        # Create two gears that should mesh
        gear1 = Gear(id=1, center=Vector2D(0,0), num_teeth=20, module=1.0)
        gear2 = Gear(id=2, center=Vector2D(30,0), num_teeth=40, module=1.0)  # Distance = 30, sum of radii = (20+40)/2 = 30
        
        self.assertTrue(check_meshing(gear1, gear2))
        
        # Test with different modules
        gear3 = Gear(id=3, center=Vector2D(0,0), num_teeth=20, module=1.5)
        self.assertFalse(check_meshing(gear1, gear3))
        
    def test_check_collision(self):
        gear1 = Gear(id=1, center=Vector2D(0,0), num_teeth=20, module=1.0, is_driver=True)
        gear2 = Gear(id=2, center=Vector2D(19,0), num_teeth=20, module=1.0)  # Distance=19 < sum of radii (20)

        self.assertTrue(check_collision(gear2, [gear1]))
        
    def test_is_inside_boundary(self):
        # Create a square boundary
        boundary = [
            Vector2D(0,0),
            Vector2D(10,0),
            Vector2D(10,10),
            Vector2D(0,10)
        ]
        
        # Point inside
        self.assertTrue(is_inside_boundary(Vector2D(5,5), boundary))
        
        # Point outside
        self.assertFalse(is_inside_boundary(Vector2D(15,5), boundary))
        
    def test_calculate_gear_train(self):
        # Create a simple gear train: gear1 -> gear2 -> gear3
        gear1 = Gear(id=1, center=Vector2D(0,0), num_teeth=20, module=1.0, is_driver=True)
        gear2 = Gear(id=2, center=Vector2D(30,0), num_teeth=40, module=1.0)
        gear3 = Gear(id=3, center=Vector2D(60,0), num_teeth=20, module=1.0)  # Distance from gear2: 30 = (40+20)/2

        gears = [gear1, gear2, gear3]
        ratio = calculate_gear_train(gears, 1, 3)

        # Expected ratio: (20/40) * (40/20) = 1.0
        self.assertAlmostEqual(ratio, 1.0, places=5)

if __name__ == '__main__':
    unittest.main()
