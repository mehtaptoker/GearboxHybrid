import numpy as np
import math
from typing import List, Tuple
from components import Vector2D
from physics import line_segment_intersects_polygon, is_gear_inside_boundary
import config

def iterative_path_refinement(start_gear, end_gear, boundary: List[Vector2D], 
                             obstacles: List[List[Vector2D]], 
                             max_iterations: int = 500, tolerance: float = 1e-2) -> Tuple[List[Vector2D], List[float]]:
    """
    Generate a viable gear path between start and end gears with iterative refinement.
    Returns a tuple of (path, radii) where path is list of vertices and radii is list of gear radii.
    
    Implements the gap-filling algorithm:
    1. Initialize path with A* or straight line
    2. Iterative refinement for radius propagation and path adjustment
    """
    # Start and end points with fixed radii
    start = start_gear.center
    end = end_gear.center
    r_start = start_gear.radius
    r_end = end_gear.radius
    
    # Initialize path - always include at least one intermediate point
    path = [start]
    
    # Calculate the required distance for direct connection
    direct_dist = math.sqrt((end.x - start.x)**2 + (end.y - start.y)**2)
    min_required_dist = r_start + r_end
    
    # Get workspace boundaries from config
    half_workspace = config.WORKSPACE_SIZE / 2.0
    
    # Generate initial path with minimum required gear spacing
    path = [start]
    direction = Vector2D(end.x - start.x, end.y - start.y)
    if direction.x == 0 and direction.y == 0:
        direction = Vector2D(1, 0)  # Default direction
    else:
        magnitude = math.sqrt(direction.x**2 + direction.y**2)
        direction.x /= magnitude
        direction.y /= magnitude

    # Calculate minimum required distances for each segment
    min_segment_lengths = [r_start + config.MIN_RADIUS]
    for _ in range(num_joints - 1):
        min_segment_lengths.append(2 * config.MIN_RADIUS)
    min_segment_lengths.append(config.MIN_RADIUS + r_end)
    
    # Place points with minimum required spacing
    current_pos = start
    for i, min_length in enumerate(min_segment_lengths):
        next_pos = Vector2D(
            current_pos.x + direction.x * min_length,
            current_pos.y + direction.y * min_length
        )
        # Clamp to workspace boundaries
        next_pos.x = max(-half_workspace, min(half_workspace, next_pos.x))
        next_pos.y = max(-half_workspace, min(half_workspace, next_pos.y))
        path.append(next_pos)
        current_pos = next_pos
    
    # Adjust last point to be exactly at end position
    path[-1] = end
    
    iteration = 0
    converged = False
    
    while iteration < max_iterations and not converged:
        # Forward radius propagation
        radii = [r_start]
        failure_index = None
        
        # Propagate radii from start to end
        for i in range(len(path) - 1):
            dist = math.sqrt((path[i+1].x - path[i].x)**2 + (path[i+1].y - path[i].y)**2)
            next_radius = dist - radii[-1]
            
            # Mid-chain failure detection
            if next_radius <= 0:
                failure_index = i+1
                break
            radii.append(next_radius)
        
        if failure_index is not None:
            # Adjust segment for mid-chain failure
            direction = Vector2D(path[failure_index].x - path[failure_index-1].x,
                                path[failure_index].y - path[failure_index-1].y)
            if direction.x == 0 and direction.y == 0:
                direction = Vector2D(1, 0)  # Default direction
            
            magnitude = math.sqrt(direction.x**2 + direction.y**2)
            if magnitude > 0:
                direction.x /= magnitude
                direction.y /= magnitude
            
        # Move vertex to create space (minimum radius buffer)
        move_distance = config.MIN_RADIUS - next_radius + 1.0
        path[failure_index] = Vector2D(
            path[failure_index].x + direction.x * move_distance,
            path[failure_index].y + direction.y * move_distance
        )
        # Clamp to workspace boundaries
        path[failure_index].x = max(-half_workspace, min(half_workspace, path[failure_index].x))
        path[failure_index].y = max(-half_workspace, min(half_workspace, path[failure_index].y))
        iteration += 1
        continue
        
        # Endpoint mismatch check
        last_dist = math.sqrt((path[-1].x - path[-2].x)**2 + (path[-1].y - path[-2].y)**2)
        E = (radii[-1] + r_end) - last_dist
        
        if abs(E) < tolerance:
            converged = True
            break
        
        # Distribute error across intermediate joints
        num_joints = len(path) - 2  # Intermediate joints only
        if num_joints > 0:
            adjustment_per_joint = E / num_joints
            for i in range(1, len(path)-1):
                # Move joint along the path direction
                segment_vec = Vector2D(path[i+1].x - path[i-1].x, 
                                      path[i+1].y - path[i-1].y)
                if segment_vec.x == 0 and segment_vec.y == 0:
                    segment_vec = Vector2D(1, 0)  # Default direction
                
                magnitude = math.sqrt(segment_vec.x**2 + segment_vec.y**2)
                if magnitude > 0:
                    segment_vec.x /= magnitude
                    segment_vec.y /= magnitude
                
                path[i] = Vector2D(
                    path[i].x + segment_vec.x * adjustment_per_joint,
                    path[i].y + segment_vec.y * adjustment_per_joint
                )
                # Clamp to workspace boundaries
                path[i].x = max(-half_workspace, min(half_workspace, path[i].x))
                path[i].y = max(-half_workspace, min(half_workspace, path[i].y))
        
        iteration += 1
    
    # Final validation
    if not converged:
        # Instead of failing, return the best path we have with a warning
        print(f"Warning: Path refinement failed to converge after {max_iterations} iterations, using best path found")
    
    # Ensure we have at least one intermediate gear
    if len(path) <= 2:
        # Fallback: add midpoint if algorithm converged to direct path
        midpoint = Vector2D((start.x + end.x)/2, (start.y + end.y)/2)
        path.insert(1, midpoint)
        # Estimate radius for midpoint
        dist1 = math.sqrt((midpoint.x - start.x)**2 + (midpoint.y - start.y)**2)
        dist2 = math.sqrt((end.x - midpoint.x)**2 + (end.y - midpoint.y)**2)
        radii = [r_start, dist1 - r_start, dist2 - (dist1 - r_start)]
    
    return path, radii

from components import Gear

def generate_gear_path(start: Vector2D, end: Vector2D, boundary: List[Vector2D], obstacles: List[List[Vector2D]]) -> List[Vector2D]:
    """
    Generate a viable gear path between start and end points, avoiding obstacles.
    Returns a polyline (list of vertices) for gear placement.
    """
    # Create temporary Gear objects for the start and end points
    start_gear = Gear(id=-1, center=start, num_teeth=config.MIN_TEETH, module=config.GEAR_MODULE)
    end_gear = Gear(id=-2, center=end, num_teeth=config.MIN_TEETH, module=config.GEAR_MODULE)
    
    # Use iterative refinement with default input radius
    path, _ = iterative_path_refinement(start_gear, end_gear, boundary, obstacles, config.GEAR_MODULE * config.MIN_TEETH / 2)
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
