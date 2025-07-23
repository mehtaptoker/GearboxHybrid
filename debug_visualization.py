import math
import matplotlib.pyplot as plt
from typing import List, Tuple

# --- You may need to adjust these imports based on your project structure ---
from components import Vector2D
from physics import point_in_polygon, line_segment_intersects_polygon


# --- This is a direct copy of the logic we are testing ---
def build_visibility_graph(start: Vector2D, end: Vector2D, obstacles: List[List[Vector2D]]) -> List[
    Tuple[Vector2D, Vector2D]]:
    """
    Creates a visibility graph, with verbose printing for debugging.
    """

    def is_segment_valid(p1: Vector2D, p2: Vector2D, obstacles_to_check: List[List[Vector2D]]) -> bool:
        """Helper to check if a line segment is valid."""
        p1_shrunk = p1.interpolate(p2, 0.001)
        p2_shrunk = p2.interpolate(p1, 0.001)

        for obs_poly in obstacles_to_check:
            midpoint = Vector2D((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)
            if point_in_polygon(midpoint, obs_poly):
                print(f"      -> INVALID: Midpoint of {p1} to {p2} is inside an obstacle.")
                return False
            if line_segment_intersects_polygon(p1_shrunk, p2_shrunk, obs_poly):
                print(f"      -> INVALID: Segment from {p1} to {p2} intersects an obstacle edge.")
                return False
        return True

    print("\nBuilding visibility graph...")
    points_of_interest = [start, end]
    for obs_poly in obstacles:
        points_of_interest.extend(obs_poly)

    graph_edges = []
    for i in range(len(points_of_interest)):
        for j in range(i + 1, len(points_of_interest)):
            p1 = points_of_interest[i]
            p2 = points_of_interest[j]

            # Don't check edges that are part of the obstacle itself
            is_obstacle_edge = False
            for obs_poly in obstacles:
                if p1 in obs_poly and p2 in obs_poly:
                    # A more robust check would see if they are adjacent vertices
                    is_obstacle_edge = True
                    break
            if is_obstacle_edge:
                continue

            print(f"  Checking line of sight from {p1} to {p2}...")
            if is_segment_valid(p1, p2, obstacles):
                print("      -> VALID: Edge added.")
                graph_edges.append((p1, p2))
            else:
                # This else block is for clarity; the reason is printed inside is_segment_valid
                pass

    return graph_edges


def visualize_debug(start, end, obstacles, edges):
    """A dedicated visualizer for this script."""
    plt.figure(figsize=(10, 10))
    for p1, p2 in edges:
        plt.plot([p1.x, p2.x], [p1.y, p2.y], 'g--', alpha=0.6)
    for poly in obstacles:
        px = [p.x for p in poly] + [poly[0].x]
        py = [p.y for p in poly] + [poly[0].y]
        plt.fill(px, py, 'r', alpha=0.3)
    plt.plot(start.x, start.y, 'go', markersize=12, label='Start')
    plt.plot(end.x, end.y, 'ro', markersize=12, label='End')
    plt.title("Visibility Graph Debug")
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.savefig('debug_visibility_graph.png')
    print("\nSaved visualization to debug_visibility_graph.png")


# --- Main script execution ---
if __name__ == '__main__':
    # Define the exact geometry from your failing test
    start_point = Vector2D(-50, 0)
    end_point = Vector2D(50, 0)
    obstacle_poly = [[
        Vector2D(-20, -20), Vector2D(20, -20),
        Vector2D(20, 20), Vector2D(-20, 20)
    ]]

    # Run the function
    found_edges = build_visibility_graph(start_point, end_point, obstacle_poly)

    print(f"\n--- DEBUG COMPLETE ---")
    print(f"Total valid edges found: {len(found_edges)}")
    if len(found_edges) == 0:
        print("RESULT: The visibility graph is empty. This is why A* is failing.")
    else:
        print("RESULT: The visibility graph has edges, but A* may still be failing for other reasons.")

    # Create the plot
    visualize_debug(start_point, end_point, obstacle_poly, found_edges)