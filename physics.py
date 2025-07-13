import math
from collections import deque
from typing import List, Optional, Dict

from components import Gear, Vector2D
import config

def check_meshing(gear1: Gear, gear2: Gear) -> bool:
    """Check if two gears can mesh.
    
    Args:
        gear1: First gear
        gear2: Second gear
        
    Returns:
        True if gears can mesh, False otherwise
    """
    # Must be on same z-layer and have same module
    if gear1.z_layer != gear2.z_layer or not math.isclose(gear1.module, gear2.module):
        return False
    
    # Calculate center distance
    dx = gear1.center.x - gear2.center.x
    dy = gear1.center.y - gear2.center.y
    distance = math.sqrt(dx*dx + dy*dy)
    
    # Calculate sum of radii
    sum_radii = gear1.radius + gear2.radius
    
    # Check if distance matches sum of radii within tolerance
    return math.isclose(distance, sum_radii, rel_tol=config.MESHING_TOLERANCE)

def check_collision(new_gear: Gear, existing_gears: List[Gear]) -> bool:
    """Check if new gear collides with existing gears.
    
    Args:
        new_gear: Gear to check
        existing_gears: List of existing gears
        
    Returns:
        True if collision detected, False otherwise
    """
    for gear in existing_gears:
        # Only check gears on same z-layer
        if gear.z_layer != new_gear.z_layer:
            continue
            
        # Calculate center distance
        dx = new_gear.center.x - gear.center.x
        dy = new_gear.center.y - gear.center.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Calculate sum of radii
        sum_radii = new_gear.radius + gear.radius
        
        # Collision if distance < sum of radii (not equal, as that would be meshing)
        if distance < sum_radii and not math.isclose(distance, sum_radii):
            return True
            
    return False

def is_inside_boundary(point: Vector2D, boundary_poly: List[Vector2D]) -> bool:
    """Check if point is inside boundary polygon using ray casting algorithm.
    
    Args:
        point: Point to check
        boundary_poly: List of polygon vertices
        
    Returns:
        True if point is inside polygon, False otherwise
    """
    n = len(boundary_poly)
    inside = False
    
    # Start from last vertex
    p1 = boundary_poly[0]
    
    for i in range(n + 1):
        p2 = boundary_poly[i % n]
        if point.y > min(p1.y, p2.y):
            if point.y <= max(p1.y, p2.y):
                if point.x <= max(p1.x, p2.x):
                    if p1.y != p2.y:
                        x_intersect = (point.y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y) + p1.x
                    if p1.x == p2.x or point.x <= x_intersect:
                        inside = not inside
        p1 = p2
        
    return inside

def calculate_gear_train(gears: List[Gear], input_gear_id: int, output_gear_id: int) -> Optional[float]:
    """Calculate gear train ratio between input and output gears.
    
    Args:
        gears: List of all gears
        input_gear_id: ID of input gear
        output_gear_id: ID of output gear
        
    Returns:
        Gear ratio if path exists, None otherwise
    """
    # Build adjacency list
    graph: Dict[int, List[int]] = {gear.id: [] for gear in gears}
    gear_map = {gear.id: gear for gear in gears}
    
    # Add edges for meshing gears
    for i, gear1 in enumerate(gears):
        for gear2 in gears[i+1:]:
            if check_meshing(gear1, gear2):
                graph[gear1.id].append(gear2.id)
                graph[gear2.id].append(gear1.id)
    
    # BFS setup
    queue = deque([(input_gear_id, 1.0)])
    visited = set([input_gear_id])
    
    while queue:
        current_id, current_ratio = queue.popleft()
        
        # Found output gear
        if current_id == output_gear_id:
            return current_ratio
        
        # Visit neighbors
        for neighbor_id in graph[current_id]:
            if neighbor_id not in visited:
                visited.add(neighbor_id)
                # Calculate new ratio: ratio * (current_teeth / neighbor_teeth)
                new_ratio = current_ratio * (gear_map[current_id].num_teeth / gear_map[neighbor_id].num_teeth)
                queue.append((neighbor_id, new_ratio))
    
    # No path found
    return None

def validate_gear(gear: Gear) -> bool:
    """Validate gear parameters for physical sensibility.
    
    Args:
        gear: Gear to validate
        
    Returns:
        True if gear parameters are physically sensible, False otherwise
    """
    # Check diameter relationships
    if not (gear.tip_diameter > gear.reference_diameter > 
            gear.base_diameter > gear.root_diameter):
        return False
    
    # Check pressure angle range (typical values 14.5°-25°)
    if not (14.5 <= gear.pressure_angle <= 25.0):
        return False
    
    # Check module and teeth count
    if gear.module <= 0 or gear.num_teeth < 12:
        return False
    
    return True
