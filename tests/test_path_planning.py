import unittest
import sys
sys.path.append('../')
import os
import matplotlib.pyplot as plt

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from path_planning import generate_gear_path
from components import Vector2D
from physics import line_segment_intersects_polygon

class TestPathPlanning(unittest.TestCase):

    def test_direct_path(self):
        """Test direct path with no obstacles"""
        start = Vector2D(-50, 0)
        end = Vector2D(50, 0)
        boundary = [
            Vector2D(-100, -100), Vector2D(100, -100),
            Vector2D(100, 100), Vector2D(-100, 100)
        ]
        obstacles = []

        path, _ = generate_gear_path(start, end, boundary, obstacles)

        self.assertGreaterEqual(len(path), 3)
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], end)

    def test_path_with_obstacle(self):
        """Test path with an obstacle between start and end"""
        start = Vector2D(-50, 0)
        end = Vector2D(50, 0)
        boundary = [
            Vector2D(-100, -100), Vector2D(100, -100),
            Vector2D(100, 100), Vector2D(-100, 100)
        ]
        obstacles = [[
            Vector2D(-20, -20), Vector2D(20, -20),
            Vector2D(20, 20), Vector2D(-20, 20)
        ]]

        path, graph_edges = generate_gear_path(start, end, boundary, obstacles)

        self.assertGreater(len(path), 1)

        # Verify path doesn't intersect obstacles
        for i in range(len(path) - 1):
            for poly in obstacles:
                self.assertFalse(line_segment_intersects_polygon(path[i], path[i + 1], poly))

        # Visualize for debugging
        self.visualize_path(start, end, boundary, obstacles, path, graph_edges)

    def visualize_path(self, start, end, boundary, obstacles, path, graph_edges=None):
        """Visualize the path and visibility graph for debugging."""
        plt.figure(figsize=(10, 10))

        # Plot visibility graph edges if they exist
        if graph_edges:
            for p1, p2 in graph_edges:
                plt.plot([p1.x, p2.x], [p1.y, p2.y], 'g--', alpha=0.3, linewidth=1)

        # Plot boundary
        bx = [p.x for p in boundary] + [boundary[0].x]
        by = [p.y for p in boundary] + [boundary[0].y]
        plt.plot(bx, by, 'k-', linewidth=2)

        # Plot obstacles
        for poly in obstacles:
            px = [p.x for p in poly] + [poly[0].x]
            py = [p.y for p in poly] + [poly[0].y]
            plt.fill(px, py, 'r', alpha=0.3)

        # Plot final path
        if path:
            px = [p.x for p in path]
            py = [p.y for p in path]
            plt.plot(px, py, 'bo-', linewidth=2, markersize=8)

        # Plot start and end points
        plt.plot(start.x, start.y, 'go', markersize=12, label='Start')
        plt.plot(end.x, end.y, 'ro', markersize=12, label='End')

        plt.title('Gear Path Planning')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        plt.savefig('test_path_planning_result.png')
        plt.close()


if __name__ == '__main__':
    unittest.main()
if __name__ == '__main__':
    unittest.main()
