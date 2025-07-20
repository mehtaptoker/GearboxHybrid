import unittest
from pathfinding import VisibilityGraphPathfinder, generate_collision_free_path
from components import Vector2D

class TestVisibilityGraphPathfinder(unittest.TestCase):
    def setUp(self):
        # Define obstacles: two squares
        self.obstacles = [
            [
                Vector2D(0, 0),
                Vector2D(0, 10),
                Vector2D(10, 10),
                Vector2D(10, 0)
            ],
            [
                Vector2D(20, 20),
                Vector2D(20, 30),
                Vector2D(30, 30),
                Vector2D(30, 20)
            ]
        ]
        self.pathfinder = VisibilityGraphPathfinder(self.obstacles)
        
    def test_line_of_sight(self):
        # Test unobstructed line of sight
        self.assertTrue(self.pathfinder._line_of_sight(Vector2D(5, 5), Vector2D(15, 15)))
        
        # Test obstructed line of sight
        self.assertFalse(self.pathfinder._line_of_sight(Vector2D(5, 5), Vector2D(25, 25)))
        
    def test_build_visibility_graph(self):
        start = Vector2D(-5, -5)
        end = Vector2D(35, 35)
        self.pathfinder.build_visibility_graph(start, end)
        
        # Check if graph has expected connections
        self.assertIn(Vector2D(0, 0), self.pathfinder.graph)
        self.assertIn(Vector2D(35, 35), self.pathfinder.graph[Vector2D(0, 0)])
        
    def test_find_shortest_path(self):
        start = Vector2D(-5, -5)
        end = Vector2D(35, 35)
        path = self.pathfinder.find_shortest_path(start, end)
        
        # Check if path is valid
        self.assertIsNotNone(path)
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], end)
        self.assertGreater(len(path), 2)

class TestGenerateCollisionFreePath(unittest.TestCase):
    def test_generate_path(self):
        obstacles = [
            [
                Vector2D(0, 0),
                Vector2D(0, 10),
                Vector2D(10, 10),
                Vector2D(10, 0)
            ]
        ]
        start = Vector2D(-5, -5)
        end = Vector2D(15, 15)
        path = generate_collision_free_path(start, end, obstacles)
        
        self.assertIsNotNone(path)
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], end)
        
    def test_no_path(self):
        # Create a wall that completely blocks the path
        obstacles = [
            [
                Vector2D(-10, 0),
                Vector2D(20, 0),
                Vector2D(20, 1),
                Vector2D(-10, 1)
            ]
        ]
        start = Vector2D(0, -5)
        end = Vector2D(0, 5)
        path = generate_collision_free_path(start, end, obstacles)
        
        self.assertIsNone(path)

if __name__ == '__main__':
    unittest.main()
