import numpy as np
import math
from typing import List, Tuple
from components import Vector2D
from physics import line_segment_intersects_polygon, point_in_polygon, is_gear_inside_boundary
import config

import numpy as np
import math
from typing import List, Tuple
from components import Vector2D, Gear  # Assuming Gear is in components
import config


# =================================================================================
# STEP 1: Corrected iterative_path_refinement function
# =================================================================================
# This version now ACCEPTS an initial path instead of creating its own.
def a_star_path(start: Vector2D, end: Vector2D, graph_edges: List[Tuple[Vector2D, Vector2D]]) -> List[Vector2D]:
    """
    Finds the shortest path from start to end using A* algorithm on a visibility graph.
    This version is robust and handles disconnected nodes correctly.
    """
    graph = {}
    all_nodes = {start, end}
    for p1, p2 in graph_edges:
        all_nodes.add(p1)
        all_nodes.add(p2)
        dist = math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)
        graph.setdefault(p1, []).append((p2, dist))
        graph.setdefault(p2, []).append((p1, dist))

    open_set = {start}
    came_from = {}
    g_score = {node: float('inf') for node in all_nodes}
    g_score[start] = 0
    f_score = {node: float('inf') for node in all_nodes}
    f_score[start] = g_score[start] + math.sqrt((end.x - start.x) ** 2 + (end.y - start.y) ** 2)

    while open_set:
        current = min(open_set, key=lambda node: f_score[node])
        if current == end:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return list(reversed(path))

        open_set.remove(current)

        if current in graph:
            for neighbor, dist in graph[current]:
                tentative_g_score = g_score[current] + dist
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + math.sqrt(
                        (end.x - neighbor.x) ** 2 + (end.y - neighbor.y) ** 2)
                    if neighbor not in open_set:
                        open_set.add(neighbor)
    return []


# In your path_planning.py file

def build_visibility_graph(start: Vector2D, end: Vector2D, obstacles: List[List[Vector2D]], boundary: List[Vector2D]) -> \
List[Tuple[Vector2D, Vector2D]]:
    """
    Creates a visibility graph for A* pathfinding.
    This version uses a robust validity check and correctly handles obstacle edges.
    """

    def is_segment_valid(p1: Vector2D, p2: Vector2D, obstacles_to_check: List[List[Vector2D]]) -> bool:
        """Helper to check if a line segment is valid by avoiding obstacle interiors."""
        p1_shrunk = p1.interpolate(p2, 0.001)
        p2_shrunk = p2.interpolate(p1, 0.001)

        for obs_poly in obstacles_to_check:
            midpoint = Vector2D((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)
            if point_in_polygon(midpoint, obs_poly):
                return False
            if line_segment_intersects_polygon(p1_shrunk, p2_shrunk, obs_poly):
                return False
        return True

    def are_adjacent(p1: Vector2D, p2: Vector2D, polygon: List[Vector2D]) -> bool:
        """Checks if two points are adjacent vertices in a polygon."""
        try:
            idx1 = polygon.index(p1)
            idx2 = polygon.index(p2)
            n = len(polygon)
            # Check if indices are consecutive, or wrap around for the last and first points
            return abs(idx1 - idx2) == 1 or (idx1 == 0 and idx2 == n - 1) or (idx2 == 0 and idx1 == n - 1)
        except ValueError:
            return False

    points = [start, end]
    for obs_poly in obstacles:
        points.extend(obs_poly)

    unique_points = []
    for p in points:
        if p not in unique_points:
            unique_points.append(p)

    graph_edges = []
    for i in range(len(unique_points)):
        for j in range(i + 1, len(unique_points)):
            p1 = unique_points[i]
            p2 = unique_points[j]

            # --- START of CORRECTED LOGIC ---
            # Check if the segment is an edge of an obstacle polygon.
            # If so, we should not create a visibility edge for it.
            is_an_obstacle_edge = False
            for obs_poly in obstacles:
                if are_adjacent(p1, p2, obs_poly):
                    is_an_obstacle_edge = True
                    break

            if is_an_obstacle_edge:
                continue  # Skip this pair, as it's a solid wall of an obstacle
            # --- END of CORRECTED LOGIC ---

            if is_segment_valid(p1, p2, obstacles):
                graph_edges.append((p1, p2))

    return graph_edges


def iterative_path_refinement(start_gear, end_gear, boundary, obstacles, initial_path, max_iterations=500,
                              tolerance=0.1) -> Tuple[List[Vector2D], List[float]]:
    """
    Refines a given gear path to satisfy meshing constraints while avoiding obstacles.
    """
    path = list(initial_path)
    r_start, r_end = start_gear.radius, end_gear.radius
    learning_rate = 0.05
    half_workspace = config.WORKSPACE_SIZE / 2.0

    for iteration in range(max_iterations):
        # Propagate radii
        radii = [r_start]
        for i in range(len(path) - 1):
            dist = math.sqrt((path[i + 1].x - path[i].x) ** 2 + (path[i + 1].y - path[i].y) ** 2)
            next_r = dist - radii[-1]
            if next_r <= 0:  # Path is invalid
                radii = None
                break
            radii.append(next_r)
        if not radii:
            # Handle invalid path, e.g., by moving points closer together.
            # This part can be improved with better recovery logic.
            continue

        # Check for convergence
        last_dist = math.sqrt((path[-1].x - end_gear.center.x) ** 2 + (path[-1].y - end_gear.center.y) ** 2)
        endpoint_error = (radii[-1] + r_end) - last_dist
        if abs(endpoint_error) < tolerance:
            break

        # Adjust intermediate points
        if len(path) > 2:
            adjustment = endpoint_error / (len(path) - 1)
            for i in range(1, len(path) - 1):
                vec = Vector2D(path[i + 1].x - path[i - 1].x, path[i + 1].y - path[i - 1].y).normalized()
                proposed_pos = Vector2D(path[i].x - vec.x * adjustment, path[i].y - vec.y * adjustment)

                # Check for collisions before moving
                is_move_valid = True
                for obs in obstacles:
                    if line_segment_intersects_polygon(path[i - 1], proposed_pos, obs) or \
                            line_segment_intersects_polygon(proposed_pos, path[i + 1], obs):
                        is_move_valid = False
                        break
                if is_move_valid:
                    path[i] = proposed_pos

    if iteration == max_iterations - 1:
        print(f"Warning: Path refinement failed to converge after {max_iterations} iterations.")

    final_radii = [r_start] + (radii if 'radii' in locals() and radii is not None else [])
    return path, final_radii


def generate_gear_path(start: Vector2D, end: Vector2D, boundary: List[Vector2D], obstacles: List[List[Vector2D]]) -> \
Tuple[List[Vector2D], List[Tuple[Vector2D, Vector2D]]]:
    """
    Generates a viable gear path, using A* and falling back to a detour if necessary.
    Returns the final path and the visibility graph edges for debugging.
    """
    graph_edges = []
    for obs_poly in obstacles:
        if point_in_polygon(start, obs_poly) or point_in_polygon(end, obs_poly):
            print(f"Error: Start or end point is inside an obstacle.")
            return [], []

    start_gear = Gear(id=-1, center=start, num_teeth=config.MIN_TEETH, module=config.GEAR_MODULE)
    end_gear = Gear(id=-2, center=end, num_teeth=config.MIN_TEETH, module=config.GEAR_MODULE)

    initial_path = []
    if not obstacles:
        print("No obstacles found. Generating direct path.")
        midpoint = Vector2D((start.x + end.x) / 2, (start.y + end.y) / 2)
        initial_path = [start, midpoint, end]
    else:
        print("Obstacles found. Building visibility graph and running A*.")
        try:
            graph_edges = build_visibility_graph(start, end, obstacles, boundary)
            initial_path = a_star_path(start, end, graph_edges)
        except Exception as e:
            print(f"A* Pathfinding failed with an error: {e}")
            initial_path = []

    if not initial_path:
        print("Warning: A* pathfinding failed. Creating a simple detour path.")
        midpoint_y_detour = max(abs(start.y), abs(end.y)) + 40
        midpoint = Vector2D((start.x + end.x) / 2, midpoint_y_detour)
        initial_path = [start, midpoint, end]

    if not initial_path:
        print("Critical Error: Could not generate any path.")
        return [], graph_edges

    path, _ = iterative_path_refinement(start_gear, end_gear, boundary, obstacles, initial_path)
    return path, graph_edges