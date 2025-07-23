import numpy as np
import math
from typing import List, Tuple
from components import Vector2D
from physics import line_segment_intersects_polygon, point_in_polygon, line_segments_intersect, is_gear_inside_boundary,line_segments_intersect
import config

import numpy as np
import math
from typing import List, Tuple
from components import Vector2D, Gear  # Assuming Gear is in components
import config

import math
from typing import List, Tuple, Optional

# --- Assume these are your component and physics modules ---
from components import Vector2D, Gear
from physics import point_in_polygon, line_segment_intersects_polygon, is_gear_inside_boundary
import config


# ---------------------------------------------------------


def a_star_path(start: Vector2D, end: Vector2D, graph_edges: List[Tuple[Vector2D, Vector2D]]) -> List[Vector2D]:
    """
    Finds the shortest path from start to end using A* algorithm on a visibility graph.
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


def are_adjacent(p1: Vector2D, p2: Vector2D, polygon: List[Vector2D]) -> bool:
    """Checks if two points are adjacent vertices in a polygon."""
    try:
        idx1 = polygon.index(p1)
        idx2 = polygon.index(p2)
        n = len(polygon)
        return abs(idx1 - idx2) == 1 or (idx1 == 0 and idx2 == n - 1) or (idx2 == 0 and idx1 == n - 1)
    except ValueError:
        return False


def build_visibility_graph(start: Vector2D, end: Vector2D, obstacles: List[List[Vector2D]], boundary: List[Vector2D]) -> \
List[Tuple[Vector2D, Vector2D]]:
    """
    Creates a visibility graph using a robust check that correctly handles lines of sight that touch obstacle vertices.
    """

    def is_segment_valid(p1: Vector2D, p2: Vector2D, obstacles_to_check: List[List[Vector2D]]) -> bool:
        """Helper to check if a line segment is valid."""
        p1_shrunk = p1.interpolate(p2, 0.001)
        p2_shrunk = p2.interpolate(p1, 0.001)
        for obs_poly in obstacles_to_check:
            midpoint = p1.interpolate(p2, 0.5)
            if point_in_polygon(midpoint, obs_poly):
                return False
            if line_segment_intersects_polygon(p1_shrunk, p2_shrunk, obs_poly):
                return False
        return True

    points = [start, end]
    for obs_poly in obstacles:
        points.extend(obs_poly)

    unique_points = list(set(points))

    graph_edges = []
    for i in range(len(unique_points)):
        for j in range(i + 1, len(unique_points)):
            p1, p2 = unique_points[i], unique_points[j]

            is_an_obstacle_edge = any(are_adjacent(p1, p2, obs) for obs in obstacles)
            if is_an_obstacle_edge:
                continue

            if is_segment_valid(p1, p2, obstacles):
                graph_edges.append((p1, p2))

    return graph_edges


def iterative_path_refinement(
        start_gear: Gear,
        end_gear: Gear,
        boundary: List[Vector2D],
        obstacles: List[List[Vector2D]],
        initial_path: List[Vector2D],
        max_iterations: int = 2000,
        tolerance: float = 0.1,
        learning_rate: float = 0.05
) -> Tuple[List[Vector2D], List[float]]:
    """
    Refines a gear path using a stable, obstacle-aware error-correction method.
    """
    path = [Vector2D(p.x, p.y) for p in list(initial_path)]
    if not path:
        return [], []

    for iteration in range(max_iterations):
        temp_path = [start_gear.center] + path + [end_gear.center]

        radii = [start_gear.radius]
        for i in range(len(temp_path) - 1):
            dist = math.sqrt((temp_path[i + 1].x - temp_path[i].x) ** 2 + (temp_path[i + 1].y - temp_path[i].y) ** 2)
            next_r = dist - radii[-1]
            if next_r <= 0.01:
                radii = None
                break
            radii.append(next_r)

        if not radii:
            continue

        required_last_dist = radii[-2] + end_gear.radius
        actual_last_dist = math.sqrt(
            (temp_path[-1].x - temp_path[-2].x) ** 2 + (temp_path[-1].y - temp_path[-2].y) ** 2)
        if abs(required_last_dist - actual_last_dist) < tolerance:
            break

        for i in range(len(path)):
            current_point = temp_path[i + 1]
            prev_point = temp_path[i]
            next_point = temp_path[i + 2]

            dist1 = math.sqrt((current_point.x - prev_point.x) ** 2 + (current_point.y - prev_point.y) ** 2)
            error1 = (radii[i] + radii[i + 1]) - dist1

            dist2 = math.sqrt((next_point.x - current_point.x) ** 2 + (next_point.y - current_point.y) ** 2)
            error2 = (radii[i + 1] + radii[i + 2]) - dist2

            vec1 = Vector2D(current_point.x - prev_point.x, current_point.y - prev_point.y).normalized()
            vec2 = Vector2D(current_point.x - next_point.x, current_point.y - next_point.y).normalized()

            proposed_pos = Vector2D(
                current_point.x + (vec1.x * error1 + vec2.x * error2) * learning_rate,
                current_point.y + (vec1.y * error1 + vec2.y * error2) * learning_rate
            )

            # --- THIS IS THE CRITICAL COLLISION CHECK ---
            is_move_valid = True
            for obs_poly in obstacles:
                if line_segment_intersects_polygon(prev_point, proposed_pos, obs_poly) or \
                        line_segment_intersects_polygon(proposed_pos, next_point, obs_poly):
                    is_move_valid = False
                    break

            if is_move_valid:
                path[i] = proposed_pos
            # ----------------------------------------------

    if iteration == max_iterations - 1:
        print(f"Warning: Path refinement failed to converge after {max_iterations} iterations.")

    final_radii = radii[1:-1] if 'radii' in locals() and radii and len(radii) > 2 else []

    return path, final_radii


def generate_gear_path(start: Vector2D, end: Vector2D, boundary: List[Vector2D], obstacles: List[List[Vector2D]]) -> \
Tuple[List[Vector2D], List[Tuple[Vector2D, Vector2D]]]:
    """
    Generates a viable gear path, using A* and falling back to a detour if necessary.
    Returns the final path and the visibility graph edges for debugging.
    """
    # Initialize graph_edges to ensure it always has a value
    graph_edges = []

    for obs_poly in obstacles:
        if point_in_polygon(start, obs_poly) or point_in_polygon(end, obs_poly):
            print(f"Error: Start or end point is inside an obstacle.")
            return [], graph_edges  # Return both values

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
        return [], graph_edges  # Return both values

    # Call the refinement function
    path, _ = iterative_path_refinement(start_gear, end_gear, boundary, obstacles, initial_path)

    # --- THIS IS THE CORRECTED LINE ---
    # Return the path AND the graph_edges
    return path, graph_edges