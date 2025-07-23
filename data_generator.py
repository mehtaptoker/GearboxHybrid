import random
import math
from typing import List, Dict

import config
from components import Vector2D
from physics import is_inside_boundary

def _generate_random_simple_polygon(num_vertices: int, avg_radius: float, irregularity: float, spikeyness: float) -> List[Vector2D]:
    """Generate a random simple polygon (Jordan polygon).
    
    Ensures the polygon boundary does not intersect itself.
    
    Args:
        num_vertices: Number of vertices
        avg_radius: Average distance from center to vertices
        irregularity: Amount of irregularity (0-1)
        spikeyness: Amount of spikeyness (0-1)
        
    Returns:
        List of polygon vertices (simple polygon)
    """
    max_attempts = 100
    for _ in range(max_attempts):
        # Generate candidate polygon
        angles = []
        for _ in range(num_vertices):
            angles.append(random.uniform(0, 2*math.pi))
        angles.sort()
        
        # Generate radii with irregularity and spikeyness
        radii = []
        for i in range(num_vertices):
            radius = avg_radius * (1 + random.uniform(-irregularity, irregularity))
            radius = radius * (1 + random.uniform(-spikeyness, spikeyness))
            radii.append(radius)
        
        # Create vertices
        vertices = []
        for i in range(num_vertices):
            x = radii[i] * math.cos(angles[i])
            y = radii[i] * math.sin(angles[i])
            vertices.append(Vector2D(x, y))
            
        # Check if simple polygon (no self-intersections)
        if _is_simple_polygon(vertices):
            return vertices
            
    # Fallback to convex polygon if unable to generate simple polygon
    return _generate_convex_polygon(num_vertices, avg_radius)

def _is_simple_polygon(vertices: List[Vector2D]) -> bool:
    """Check if a polygon is simple (non-self-intersecting).
    
    Args:
        vertices: List of polygon vertices in order
        
    Returns:
        True if simple polygon, False otherwise
    """
    n = len(vertices)
    if n < 3:
        return False
        
    for i in range(n):
        edge1 = (vertices[i], vertices[(i+1) % n])
        for j in range(i+1, n):
            edge2 = (vertices[j], vertices[(j+1) % n])
            if _edges_intersect(edge1, edge2):
                return False
    return True

def _edges_intersect(edge1: tuple, edge2: tuple) -> bool:
    """Check if two line segments intersect.
    
    Args:
        edge1: Tuple of two Vector2D (start, end)
        edge2: Tuple of two Vector2D (start, end)
        
    Returns:
        True if segments intersect, False otherwise
    """
    a, b = edge1
    c, d = edge2
    
    # Special case: shared vertices don't count as intersection
    if a in (c, d) or b in (c, d):
        return False
        
    def ccw(A, B, C):
        return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)
        
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

def _generate_convex_polygon(num_vertices: int, avg_radius: float) -> List[Vector2D]:
    """Generate a guaranteed convex polygon."""
    # Generate points on a circle
    angles = sorted([random.uniform(0, 2*math.pi) for _ in range(num_vertices)])
    return [Vector2D(avg_radius * math.cos(a), avg_radius * math.sin(a)) for a in angles]

def _generate_point_in_poly(boundary_poly: List[Vector2D], margin: float = 10.0) -> Vector2D:
    """Generate a random point inside a polygon with margin from boundary.
    
    Args:
        boundary_poly: Polygon vertices
        margin: Minimum distance from boundary (mm)
        
    Returns:
        Random point inside polygon with margin
    """
    # Find bounding box with margin
    min_x = min(v.x for v in boundary_poly) + margin
    max_x = max(v.x for v in boundary_poly) - margin
    min_y = min(v.y for v in boundary_poly) + margin
    max_y = max(v.y for v in boundary_poly) - margin
    
    # Generate points until one is inside with margin
    max_attempts = 1000
    for _ in range(max_attempts):
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        point = Vector2D(x, y)
        
        if is_inside_boundary(point, boundary_poly):
            # Check distance to all boundary points
            min_dist = min(math.sqrt((point.x - v.x)**2 + (point.y - v.y)**2) for v in boundary_poly)
            if min_dist >= margin:
                return point

    # Fallback to centroid if no point is found
    centroid_x = sum(v.x for v in boundary_poly) / len(boundary_poly)
    centroid_y = sum(v.y for v in boundary_poly) / len(boundary_poly)
    return Vector2D(centroid_x, centroid_y)

def generate_scenario(data_dir=None) -> Dict:
    """Generate a random scenario with a random boundary and shafts."""
    # 1. Generate a random boundary polygon
    num_vertices = random.randint(5, 8)
    avg_radius = config.WORKSPACE_SIZE / 2 * 0.9  # Use 90% of workspace
    irregularity = 0.6
    spikeyness = 0.3
    boundary_poly = _generate_random_simple_polygon(
        num_vertices, avg_radius, irregularity, spikeyness
    )

    # 2. Generate random shaft positions inside the polygon
    min_shaft_dist = config.WORKSPACE_SIZE / 3
    max_attempts = 100
    for _ in range(max_attempts):
        input_shaft = _generate_point_in_poly(boundary_poly, margin=20.0)
        output_shaft = _generate_point_in_poly(boundary_poly, margin=20.0)
        dist_sq = (input_shaft.x - output_shaft.x)**2 + (input_shaft.y - output_shaft.y)**2
        if dist_sq > min_shaft_dist**2:
            break
    
    # 3. Generate a random target ratio
    possible_ratios = [1.5, 2.0, 2.5, 0.5, 0.75]
    target_ratio = random.choice(possible_ratios)
    
    return {
        "boundary_poly": boundary_poly,
        "input_shaft": input_shaft,
        "output_shaft": output_shaft,
        "target_ratio": target_ratio,
    }
