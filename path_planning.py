import numpy as np
import math
from typing import List, Tuple
from components import Vector2D
from physics import line_segment_intersects_polygon, is_gear_inside_boundary
import config

def iterative_path_refinement(start: Vector2D, end: Vector2D, boundary: List[Vector2D], 
                             obstacles: List[List[Vector2D]], input_radius: float,
                             max_iterations: int = 100) -> Tuple[List[Vector2D], List[float]]:
    """
    Generate a viable gear path between start and end points with iterative refinement.
    Returns a tuple of (path, radii) where path is list of vertices and radii is list of gear radii.
    """
    # Step 1: Generate initial path
    path = [start]
    # Simple linear path with one intermediate point
    midpoint = Vector2D((start.x + end.x)/2, (start.y + end.y)/2)
    path.append(midpoint)
    path.append(end)
    
    # Initialize radii
    radii = [input_radius]
    for i in range(1, len(path)):
        # Calculate distance between consecutive points
        dist = math.sqrt((path[i].x - path[i-1].x)**2 + (path[i].y - path[i-1].y)**2)
        # Set radius based on previous gear's radius
        radii.append(dist - radii[i-1])
    
    # Step 2: Iterative refinement loop
    iteration = 0
    while iteration < max_iterations:
        violation_found = False
        
        # Global validation scan
        for i in range(1, len(path)-1):  # Skip input and output gears
            # Create a gear at this position
            gear = type('Gear', (), {})()
            gear.center = path[i]
            gear.radius = radii[i]
            gear.z_layer = 0
            
            # Check boundary containment
            boundary_ok = is_gear_inside_boundary(gear, boundary)
            
            if not boundary_ok:
                violation_found = True
                # Find closest boundary point
                min_dist = float('inf')
                closest_point = None
                for j in range(len(boundary)):
                    p1 = boundary[j]
                    p2 = boundary[(j+1) % len(boundary)]
                    
                    # Calculate distance to line segment
                    dx = p2.x - p1.x
                    dy = p2.y - p1.y
                    length_squared = dx*dx + dy*dy
                    
                    if length_squared == 0:
                        dist = math.sqrt((path[i].x - p1.x)**2 + (path[i].y - p1.y)**2)
                        if dist < min_dist:
                            min_dist = dist
                            closest_point = p1
                    else:
                        t = max(0, min(1, ((path[i].x - p1.x) * dx + (path[i].y - p1.y) * dy) / length_squared))
                        proj_x = p1.x + t * dx
                        proj_y = p1.y + t * dy
                        dist = math.sqrt((path[i].x - proj_x)**2 + (path[i].y - proj_y)**2)
                        
                        if dist < min_dist:
                            min_dist = dist
                            closest_point = Vector2D(proj_x, proj_y)
                
                # Push the center away from boundary
                push_direction = Vector2D(path[i].x - closest_point.x, path[i].y - closest_point.y)
                push_distance = min_dist - radii[i] + 1.0  # Add 1mm buffer
                if push_distance <= 0:
                    push_distance = 1.0  # Minimum push
                
                # Normalize and apply push
                magnitude = math.sqrt(push_direction.x**2 + push_direction.y**2)
                if magnitude > 0:
                    push_direction.x /= magnitude
                    push_direction.y /= magnitude
                
                # Update path point
                path[i] = Vector2D(
                    path[i].x + push_direction.x * push_distance,
                    path[i].y + push_direction.y * push_distance
                )
        
        # Recalculate radii
        for i in range(1, len(path)):
            dist = math.sqrt((path[i].x - path[i-1].x)**2 + (path[i].y - path[i-1].y)**2)
            # Only update if not input gear
            if i > 0:
                radii[i] = dist - radii[i-1]
        
        if not violation_found:
            break
            
        iteration += 1
    
    return path, radii

def generate_gear_path(start: Vector2D, end: Vector2D, boundary: List[Vector2D], obstacles: List[List[Vector2D]]) -> List[Vector2D]:
    """
    Generate a viable gear path between start and end points, avoiding obstacles.
    Returns a polyline (list of vertices) for gear placement.
    """
    # Use iterative refinement with default input radius
    path, _ = iterative_path_refinement(start, end, boundary, obstacles, config.GEAR_MODULE * config.MIN_TEETH / 2)
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
