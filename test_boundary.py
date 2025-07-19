import unittest
from components import Vector2D, Gear
from physics import is_gear_inside_boundary

class TestBoundaryChecks(unittest.TestCase):
    def test_square_boundary(self):
        """Test boundary checks with a square boundary"""
        boundary = [
            Vector2D(x=-50, y=-50),
            Vector2D(x=50, y=-50),
            Vector2D(x=50, y=50),
            Vector2D(x=-50, y=50)
        ]
        
        # Test center inside with small gear
        gear = Gear(id=1, center=Vector2D(0, 0), num_teeth=20, module=0.5, z_layer=0)
        self.assertTrue(is_gear_inside_boundary(gear, boundary))
        
        # Test gear touching boundary
        gear = Gear(id=2, center=Vector2D(45, 0), num_teeth=10, module=0.5, z_layer=0)
        self.assertTrue(is_gear_inside_boundary(gear, boundary))
        
        # Test partially outside
        gear = Gear(id=3, center=Vector2D(47, 0), num_teeth=20, module=0.5, z_layer=0)
        self.assertFalse(is_gear_inside_boundary(gear, boundary))
        
        # Test completely outside
        gear = Gear(id=4, center=Vector2D(60, 0), num_teeth=20, module=0.5, z_layer=0)
        self.assertFalse(is_gear_inside_boundary(gear, boundary))
        
        # Test corner case
        gear = Gear(id=5, center=Vector2D(45, 45), num_teeth=10, module=0.5, z_layer=0)
        self.assertTrue(is_gear_inside_boundary(gear, boundary))
        
        gear = Gear(id=6, center=Vector2D(47, 47), num_teeth=20, module=0.5, z_layer=0)
        self.assertFalse(is_gear_inside_boundary(gear, boundary))

    def test_triangle_boundary(self):
        """Test boundary checks with a triangle boundary"""
        boundary = [
            Vector2D(x=0, y=0),
            Vector2D(x=100, y=0),
            Vector2D(x=50, y=100)
        ]
        
        # Test center inside
        gear = Gear(id=1, center=Vector2D(50, 50), num_teeth=20, module=0.5, z_layer=0)
        self.assertTrue(is_gear_inside_boundary(gear, boundary))
        
        # Test edge case
        gear = Gear(id=2, center=Vector2D(25, 25), num_teeth=10, module=0.5, z_layer=0)
        self.assertTrue(is_gear_inside_boundary(gear, boundary))
        
        # Test small gear inside near vertex
        gear = Gear(id=3, center=Vector2D(5, 5), num_teeth=8, module=0.5, z_layer=0)  # radius=2
        self.assertTrue(is_gear_inside_boundary(gear, boundary))
        
        # Test gear extending beyond boundary
        gear = Gear(id=4, center=Vector2D(5, 5), num_teeth=15, module=0.5, z_layer=0)  # radius=3.75
        self.assertFalse(is_gear_inside_boundary(gear, boundary))
        
        # Test clearly outside
        gear = Gear(id=4, center=Vector2D(0, 1), num_teeth=15, module=0.5, z_layer=0)
        self.assertFalse(is_gear_inside_boundary(gear, boundary))
        
        # Test gear extending beyond boundary (should be outside)
        gear = Gear(id=5, center=Vector2D(0, 1), num_teeth=30, module=0.5, z_layer=0)
        result, reason = is_gear_inside_boundary(gear, boundary, return_reason=True)
        if result:
            # If it's incorrectly inside, check each point manually
            from shapely.geometry import Point, Polygon
            from shapely.geometry.polygon import orient
            poly_coords = [(v.x, v.y) for v in boundary]
            polygon = orient(Polygon(poly_coords))
            angles = [0, 45, 90, 135, 180, 225, 270, 315]
            for angle in angles:
                rad = math.radians(angle)
                x = gear.center.x + gear.radius * math.cos(rad)
                y = gear.center.y + gear.radius * math.sin(rad)
                p = Point(x, y)
                if not polygon.contains(p):
                    print(f"Point at ({x:.2f}, {y:.2f}) is outside boundary")
            self.fail(f"Gear at ({gear.center.x}, {gear.center.y}) should be outside boundary")
        else:
            self.assertFalse(result, reason if reason else "Gear should be outside boundary")
        
        # Test gear extending beyond boundary (should be outside)
        gear = Gear(id=6, center=Vector2D(60, 10), num_teeth=15, module=0.5, z_layer=0)
        result, reason = is_gear_inside_boundary(gear, boundary, return_reason=True)
        self.assertFalse(result, reason if reason else "Gear should be outside boundary")
        
        # Test gear extending beyond boundary (should be outside)
        gear = Gear(id=7, center=Vector2D(0, -5), num_teeth=15, module=0.5, z_layer=0)
        result, reason = is_gear_inside_boundary(gear, boundary, return_reason=True)
        self.assertFalse(result, reason if reason else "Gear should be outside boundary")
        
        # Test gear extending beyond boundary (should be outside)
        gear = Gear(id=8, center=Vector2D(10, 10), num_teeth=30, module=0.5, z_layer=0)
        result, reason = is_gear_inside_boundary(gear, boundary, return_reason=True)
        self.assertFalse(result, reason if reason else "Gear should be outside boundary")
        
        # Test gear extending beyond boundary (should be outside)
        gear = Gear(id=6, center=Vector2D(60, 10), num_teeth=15, module=0.5, z_layer=0)
        self.assertFalse(is_gear_inside_boundary(gear, boundary))
        
        # Test gear extending beyond boundary (should be outside)
        gear = Gear(id=7, center=Vector2D(0, -5), num_teeth=15, module=0.5, z_layer=0)
        self.assertFalse(is_gear_inside_boundary(gear, boundary))
        
        # Test gear extending beyond boundary (should be outside)
        gear = Gear(id=8, center=Vector2D(10, 10), num_teeth=30, module=0.5, z_layer=0)
        self.assertFalse(is_gear_inside_boundary(gear, boundary))
        
        # Test larger gear inside
        gear = Gear(id=6, center=Vector2D(30, 30), num_teeth=20, module=0.5, z_layer=0)
        self.assertTrue(is_gear_inside_boundary(gear, boundary))
        
        # Test clearly outside
        gear = Gear(id=6, center=Vector2D(0, -5), num_teeth=15, module=0.5, z_layer=0)
        self.assertFalse(is_gear_inside_boundary(gear, boundary))
        
        # Test clearly outside
        gear = Gear(id=4, center=Vector2D(60, 10), num_teeth=15, module=0.5, z_layer=0)
        self.assertFalse(is_gear_inside_boundary(gear, boundary))
        
        # Test gear extending beyond the boundary
        gear = Gear(id=5, center=Vector2D(10, 10), num_teeth=30, module=0.5, z_layer=0)
        self.assertFalse(is_gear_inside_boundary(gear, boundary))
        
        # Test outside
        gear = Gear(id=4, center=Vector2D(60, 10), num_teeth=15, module=0.5, z_layer=0)
        self.assertFalse(is_gear_inside_boundary(gear, boundary))

    def test_complex_boundary(self):
        """Test boundary checks with a complex polygon"""
        boundary = [
            Vector2D(x=0, y=0),
            Vector2D(x=100, y=0),
            Vector2D(x=100, y=50),
            Vector2D(x=50, y=100),
            Vector2D(x=0, y=50)
        ]
        
        # Test inside
        gear = Gear(id=1, center=Vector2D(50, 50), num_teeth=20, module=0.5, z_layer=0)
        self.assertTrue(is_gear_inside_boundary(gear, boundary))
        
        # Test edge case
        gear = Gear(id=2, center=Vector2D(75, 25), num_teeth=15, module=0.5, z_layer=0)
        self.assertTrue(is_gear_inside_boundary(gear, boundary))
        
        # Test partially outside (in the notch)
        gear = Gear(id=3, center=Vector2D(25, 75), num_teeth=15, module=0.5, z_layer=0)
        self.assertFalse(is_gear_inside_boundary(gear, boundary))
        
        # Test outside
        gear = Gear(id=4, center=Vector2D(75, 75), num_teeth=15, module=0.5, z_layer=0)
        self.assertFalse(is_gear_inside_boundary(gear, boundary))

if __name__ == '__main__':
    unittest.main()
