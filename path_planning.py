import numpy as np
import math
from typing import List, Tuple
from components import Vector2D
from physics import line_segment_intersects_polygon, is_gear_inside_boundary
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
def iterative_path_refinement(
        start_gear,
        end_gear,
        boundary: List[Vector2D],
        obstacles: List[List[Vector2D]],
        initial_path: List[Vector2D],  # <-- ADDED: Accepts an initial path
        max_iterations: int = 500,
        tolerance: float = 1e-2,
) -> Tuple[List[Vector2D], List[float]]:
    """
    Refines a given gear path to satisfy meshing and boundary constraints.
    """
    # Start and end points with fixed radii
    start = start_gear.center
    end = end_gear.center
    r_start = start_gear.radius
    r_end = end_gear.radius

    # Get workspace boundaries from config
    half_workspace = config.WORKSPACE_SIZE / 2.0

    # Work with a copy of the initial path
    path = list(initial_path)

    # --- THE OLD PATH GENERATION LOGIC HAS BEEN REMOVED ---

    iteration = 0
    converged = False
    radii = []  # Initialize radii here

    while iteration < max_iterations and not converged:
        # Forward radius propagation
        radii = [r_start]
        failure_index = None
        next_radius = 0  # Initialize next_radius

        for i in range(len(path) - 1):
            dist = math.sqrt(
                (path[i + 1].x - path[i].x) ** 2 + (path[i + 1].y - path[i].y) ** 2
            )
            next_radius = dist - radii[-1]

            if next_radius <= config.MIN_RADIUS / 2:  # Check against a minimum
                failure_index = i + 1
                break
            radii.append(next_radius)

        if failure_index is not None:
            direction = Vector2D(
                path[failure_index].x - path[failure_index - 1].x,
                path[failure_index].y - path[failure_index - 1].y,
            ).normalized()

            # Move vertex to create space
            move_distance = (config.MIN_RADIUS - next_radius) + tolerance
            path[failure_index] = Vector2D(
                path[failure_index].x + direction.x * move_distance,
                path[failure_index].y + direction.y * move_distance,
            )
            # Clamp to workspace boundaries
            path[failure_index].x = max(-half_workspace, min(half_workspace, path[failure_index].x))
            path[failure_index].y = max(-half_workspace, min(half_workspace, path[failure_index].y))

            iteration += 1
            continue

        # Endpoint mismatch check
        # This check is only valid if radius propagation completed successfully
        last_dist = math.sqrt(
            (path[-1].x - path[-2].x) ** 2 + (path[-1].y - path[-2].y) ** 2
        )
        # The last calculated radius is radii[-1]
        required_last_dist = radii[-1] + r_end
        E = required_last_dist - last_dist

        if abs(E) < tolerance:
            converged = True
            break

        # Distribute error across intermediate joints
        num_intermediate_joints = len(path) - 2
        if num_intermediate_joints > 0:
            adjustment_per_joint = E / num_intermediate_joints

            for i in range(1, len(path) - 1):
                segment_vec = Vector2D(
                    path[i].x - path[i - 1].x, path[i].y - path[i - 1].y
                ).normalized()

                path[i] = Vector2D(
                    path[i].x + segment_vec.x * adjustment_per_joint,
                    path[i].y + segment_vec.y * adjustment_per_joint,
                )
                # Clamp to workspace boundaries
                path[i].x = max(-half_workspace, min(half_workspace, path[i].x))
                path[i].y = max(-half_workspace, min(half_workspace, path[i].y))

        iteration += 1

    if not converged:
        print(
            f"Warning: Path refinement failed to converge after {max_iterations} iterations."
        )

    # Final radii calculation on the resulting path
    final_radii = [r_start]
    for i in range(len(path) - 1):
        dist = math.sqrt(
            (path[i + 1].x - path[i].x) ** 2 + (path[i + 1].y - path[i].y) ** 2
        )
        final_radii.append(dist - final_radii[-1])

    return path, final_radii


# =================================================================================
# STEP 2: Corrected generate_gear_path function
# =================================================================================
# This version now CREATES an intelligent initial path before calling the refiner.
def build_visibility_graph(start: Vector2D, end: Vector2D, obstacles: List[List[Vector2D]], boundary: List[Vector2D]) -> \
List[Tuple[Vector2D, Vector2D]]:
    """
    Creates a visibility graph for A* pathfinding.
    Nodes are start, end, and all vertices of the obstacles.
    Edges connect nodes that have a clear line of sight.
    """

    def is_segment_valid(p1: Vector2D, p2: Vector2D, obstacles_to_check: List[List[Vector2D]]) -> bool:
        """Helper to check if a line segment intersects any obstacle."""
        for obs_poly in obstacles_to_check:
            if line_segment_intersects_polygon(p1, p2, obs_poly):
                return False
        return True

    # Collect all points of interest
    points = [start, end]
    for obs_poly in obstacles:
        points.extend(obs_poly)

    graph_edges = []
    # Check for line of sight between every pair of points
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            p1 = points[i]
            p2 = points[j]
            if is_segment_valid(p1, p2, obstacles):
                graph_edges.append((p1, p2))

    return graph_edges


def generate_gear_path(start: Vector2D, end: Vector2D, boundary: List[Vector2D], obstacles: List[List[Vector2D]]) -> \
List[Vector2D]:
    """
    Generates a viable gear path, creating a simple path for direct connections
    and using A* to navigate around obstacles.
    """
    start_gear = Gear(id=-1, center=start, num_teeth=config.MIN_TEETH, module=config.GEAR_MODULE)
    end_gear = Gear(id=-2, center=end, num_teeth=config.MIN_TEETH, module=config.GEAR_MODULE)

    initial_path = []

    if not obstacles:
        # Case 1: No obstacles. Create a simple 3-point path.
        print("No obstacles found. Generating direct path.")
        midpoint = Vector2D((start.x + end.x) / 2, (start.y + end.y) / 2)
        initial_path = [start, midpoint, end]
    else:
        # Case 2: Obstacles are present. Build a graph and use A*.
        print("Obstacles found. Building visibility graph and running A*.")
        try:
            graph_edges = build_visibility_graph(start, end, obstacles, boundary)
            initial_path = a_star_path(start, end, graph_edges)
        except Exception as e:
            print(f"A* Pathfinding failed with an error: {e}")
            return []

    if not initial_path:
        print("Warning: Pathfinding did not return a valid path.")
        return []

    # Call the refinement function with the generated initial_path
    path, _ = iterative_path_refinement(
        start_gear, end_gear, boundary, obstacles, initial_path
    )
    return path

def a_star_path(start: Vector2D, end: Vector2D, graph_edges: List[Tuple[Vector2D, Vector2D]]) -> List[Vector2D]:
    """
    Find the shortest path from start to end using A* algorithm on the visibility graph.
    Returns a list of vertices representing the path.
    """
    # Create graph representation (adjacency list)
    graph = {}
    for edge in graph_edges:
        n1, n2 = edge
        dist = math.sqrt((n2.x - n1.x)**2 + (n2.y - n1.y)**2)
        
        if n1 not in graph:
            graph[n1] = []
        if n2 not in graph:
            graph[n2] = []
            
        graph[n1].append((n2, dist))
        graph[n2].append((n1, dist))
    
    # A* algorithm
    open_set = {start}
    came_from = {}
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0
    
    f_score = {node: float('inf') for node in graph}
    f_score[start] = math.sqrt((end.x - start.x)**2 + (end.y - start.y)**2)
    
    while open_set:
        current = min(open_set, key=lambda node: f_score[node])
        
        if current == end:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return list(reversed(path))
        
        open_set.remove(current)
        
        for neighbor, dist in graph[current]:
            tentative_g = g_score[current] + dist
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + math.sqrt((end.x - neighbor.x)**2 + (end.y - neighbor.y)**2)
                if neighbor not in open_set:
                    open_set.add(neighbor)
    
    return []  # No path found
