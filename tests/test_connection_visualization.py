import unittest
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import math

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from visualization import generate_connection_line, render_system
from components import SystemState, Vector2D
from physics import distance_to_line_segment

class TestConnectionVisualization(unittest.TestCase):
    def test_connection_line_generation(self):
        # Create a system state with input and output shafts
        input_shaft = Vector2D(-50, 0)
        output_shaft = Vector2D(50, 0)
        state = SystemState(
            boundary_poly=[],
            gears=[],
            input_shaft=input_shaft,
            output_shaft=output_shaft,
            target_ratio=2.0
        )
        
        # Generate connection line
        line_start, line_end = generate_connection_line(input_shaft, output_shaft)
        
        # Verify points match input/output shafts
        self.assertEqual(line_start.x, input_shaft.x)
        self.assertEqual(line_start.y, input_shaft.y)
        self.assertEqual(line_end.x, output_shaft.x)
        self.assertEqual(line_end.y, output_shaft.y)
        
        # Test point on line
        distance = distance_to_line_segment(Vector2D(0, 0), line_start, line_end)
        self.assertAlmostEqual(distance, 0, delta=0.1)
        
        # Test point off line
        distance = distance_to_line_segment(Vector2D(0, 10), line_start, line_end)
        self.assertAlmostEqual(distance, 10, delta=0.1)
    
    def test_connection_line_rendering(self):
        # Create a system state with input and output shafts
        input_shaft = Vector2D(-50, 0)
        output_shaft = Vector2D(50, 0)
        state = SystemState(
            boundary_poly=[],
            gears=[],
            input_shaft=input_shaft,
            output_shaft=output_shaft,
            target_ratio=2.0
        )
        
        # Render system visualization
        render_system(state)
        
        # Save the plot to a file
        plt.savefig("test_connection_visualization.png")
        plt.close()
        
        # Verify file was created
        self.assertTrue(os.path.exists("test_connection_visualization.png"))

if __name__ == '__main__':
    unittest.main()
