import math
from collections import deque
from typing import List, Optional, Dict

from components import Gear, Vector2D
import config

def point_in_polygon(point: Vector2D, polygon: List[Vector2D]) -> bool:
    """Check if a point is inside a polygon using ray casting algorithm.
    
    Args:
        point: Point to check
        polygon: List of polygon vertices
        
    Returns:
        True if point is inside polygon, False otherwise
    """
    x, y = point.x, point.y
    n = len(polygon)
    inside = False
    
    p1 = polygon[0]
    for i in range(n+1):
        p2 = polygon[i % n]
        if y > min(p1.y, p2.y):
            if y <= max(p1.y, p2.y):
                if x <= max(p1.x, p2.x):
                    if p1.y != p2.y:
                        xinters = (y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y) + p1.x
                    if p1.x == p2.x or x <= xinters:
                        inside = not inside
        p1 = p2
        
    return inside

def distance_to_line_segment(point: Vector2D, line_start: Vector2D, line_end: Vector2D) -> float:
    """Calculate the shortest distance from a point to a line segment.
    
    Args:
        point: The point to calculate distance for
        line_start: Start point of the line segment
        line_end: End point of the line segment
        
    Returns:
        The shortest distance from the point to the line segment
    """
    # Vector from line_start to line_end
    line_vec = Vector2D(line_end.x - line_start.x, line_end.y - line_start.y)
    # Vector from line_start to point
    point_vec = Vector2D(point.x - line_start.x, point.y - line_start.y)
    
    # Length of the line segment
    line_len = math.sqrt(line_vec.x**2 + line_vec.y**2)
    
    # If line length is zero, return distance to start point
    if line_len == 0:
        return math.sqrt(point_vec.x**2 + point_vec.y**2)
    
    # Normalize line vector
    line_vec_norm = Vector2D(line_vec.x / line_len, line_vec.y / line_len)
    
    # Calculate projection of point_vec onto line_vec_norm
    proj_length = point_vec.x * line_vec_norm.x + point_vec.y * line_vec_norm.y
    
    # Clamp projection to line segment
    proj_length = max(0, min(line_len, proj_length))
    
    # Calculate closest point on line segment
    closest_point = Vector2D(
        line_start.x + line_vec_norm.x * proj_length,
        line_start.y + line_vec_norm.y * proj_length
    )
    
    # Calculate distance between point and closest point
    dx = point.x - closest_point.x
    dy = point.y - closest_point.y
    return math.sqrt(dx**2 + dy**2)

def check_meshing(gear1: Gear, gear2: Gear, abs_tol: float = 0.5) -> bool:
    """Check if two gears can mesh.
    
    Args:
        gear1: First gear
        gear2: Second gear
        abs_tol: Absolute tolerance for distance comparison
        
    Returns:
        True if gears can mesh, False otherwise
    """
    # Must be on same z-layer and have same module
    if gear1.z_layer != gear2.z_layer or not math.isclose(gear1.module, gear2.module, abs_tol=0.01):
        return False
    
    # Calculate center distance
    dx = gear1.center.x - gear2.center.x
    dy = gear1.center.y - gear2.center.y
    distance = math.sqrt(dx*dx + dy*dy)
    
    # Calculate sum of radii
    sum_radii = gear1.radius + gear2.radius
    
    # Check if distance matches sum of radii within tolerance
    return math.isclose(distance, sum_radii, abs_tol=abs_tol)

def check_collision(new_gear: Gear, obstacles: List[object], return_reason=False) -> bool:
    """Check if new gear collides with existing obstacles.
    
    Args:
        new_gear: Gear to check
        obstacles: List of obstacles (gears or Vector2D points) OR a single obstacle
        return_reason: If True, returns (result, reason) tuple
        
    Returns:
        True if collision detected, False otherwise
        If return_reason=True, returns (bool, str) tuple
    """
    # Ensure obstacles is always a list
    if not isinstance(obstacles, list):
        obstacles = [obstacles]
        
    for obstacle in obstacles:
        # Handle Vector2D points
        if isinstance(obstacle, Vector2D):
            # Calculate distance to point
            dx = new_gear.center.x - obstacle.x
            dy = new_gear.center.y - obstacle.y
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Collision if distance < gear radius minus tolerance
            if distance < new_gear.radius - config.MESHING_TOLERANCE:
                reason = f"Collision with point obstacle at ({obstacle.x}, {obstacle.y})"
                return (True, reason) if return_reason else True
                
        # Handle Gear objects
        elif isinstance(obstacle, Gear):
            # Only check gears on same z-layer
            if obstacle.z_layer != new_gear.z_layer:
                continue
                
            # Calculate center distance
            dx = new_gear.center.x - obstacle.center.x
            dy = new_gear.center.y - obstacle.center.y
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Calculate sum of radii
            sum_radii = new_gear.radius + obstacle.radius
            
            # Collision if distance < sum of radii minus tolerance
            if distance < sum_radii - config.MESHING_TOLERANCE:
                reason = f"Collision with gear {obstacle.id} (distance={distance:.2f} < sum_radii={sum_radii:.2f})"
                return (True, reason) if return_reason else True
                
    return (False, "") if return_reason else False

from shapely.geometry import Point, Polygon
from shapely.geometry.polygon import orient

def is_inside_boundary(point: Vector2D, boundary_poly: List[Vector2D]) -> bool:
    """Check if point is inside boundary polygon using shapely.
    
    Args:
        point: Point to check
        boundary_poly: List of polygon vertices
        
    Returns:
        True if point is inside polygon, False otherwise
    """
    # Convert boundary to shapely polygon and ensure correct orientation
    poly_coords = [(v.x, v.y) for v in boundary_poly]
    polygon = orient(Polygon(poly_coords))
    p = Point(point.x, point.y)
    return polygon.contains(p)

def is_gear_inside_boundary(gear: Gear, boundary_poly: List[Vector2D], return_reason=False) -> bool:
    """Check if entire gear is within boundary polygon using shapely.
    
    Args:
        gear: Gear to check
        boundary_poly: List of polygon vertices
        return_reason: If True, returns (result, reason) tuple
        
    Returns:
        True if gear is fully inside boundary, False otherwise
        If return_reason=True, returns (bool, str) tuple
    """
    # Convert boundary to shapely polygon and ensure correct orientation
    poly_coords = [(v.x, v.y) for v in boundary_poly]
    polygon = orient(Polygon(poly_coords))
    
    # Create a circle representing the gear with higher resolution
    gear_circle = Point(gear.center.x, gear.center.y).buffer(gear.radius, resolution=32)
    
    # Check if the gear circle is fully contained within the boundary polygon
    if polygon.covers(gear_circle):
        return (True, "") if return_reason else True
        
    # If not contained, provide detailed reason
    # First check if center is inside
    center_point = Point(gear.center.x, gear.center.y)
    if not polygon.contains(center_point):
        reason = f"Gear center ({gear.center.x:.2f}, {gear.center.y:.2f}) is outside boundary"
        return (False, reason) if return_reason else False
        
    # Calculate minimum distance from center to boundary
    min_distance = polygon.boundary.distance(center_point)
    
    # If min_distance < gear radius, gear extends beyond boundary
    if min_distance < gear.radius * 0.9:  # Allow 10% overlap
        reason = f"Gear extends beyond boundary (min_distance={min_distance:.2f} < radius={gear.radius:.2f})"
        return (False, reason) if return_reason else False
        
    # Check multiple points around the circumference for more accuracy
    angles = [0, 45, 90, 135, 180, 225, 270, 315]  # Check 8 points
    for angle in angles:
        rad = math.radians(angle)
        x = gear.center.x + gear.radius * math.cos(rad)
        y = gear.center.y + gear.radius * math.sin(rad)
        point = Point(x, y)
        
        if not polygon.contains(point):
                # Calculate distance to boundary for more detailed error
                distance_to_boundary = polygon.boundary.distance(point)
                # Only reject if significantly beyond boundary
                if distance_to_boundary < -gear.radius * 0.1:  # Allow 10% overlap
                    reason = (f"Gear edge point at ({x:.2f}, {y:.2f}) is outside boundary "
                              f"(distance={distance_to_boundary:.5f})")
                    return (False, reason) if return_reason else False
            
    return (True, "") if return_reason else True

def calculate_gear_train(gears: List[Gear], start_index: int, end_index: int) -> float:
    """Calculate gear train ratio between two gears in a chain.
    
    Args:
        gears: List of Gear objects in the train
        start_index: Index of the starting gear
        end_index: Index of the ending gear
        
    Returns:
        Overall gear ratio (output teeth / input teeth)
    """
    # Validate indices
    if start_index < 0 or end_index < 0 or start_index >= len(gears) or end_index >= len(gears):
        return 0.0
    
    ratio = 1.0
    current_index = start_index
    
    # Traverse the chain from start to end
    while current_index != end_index:
        next_index = current_index + 1 if current_index < end_index else current_index - 1
        if next_index < 0 or next_index >= len(gears):
            return 0.0
        
        # Calculate ratio for this pair
        ratio *= gears[current_index].num_teeth / gears[next_index].num_teeth
        current_index = next_index
    
    return ratio

def calculate_gear_train_ratio(gears: List[Gear], driver_id: int) -> float:
    """Calculate gear train ratio for a chain of gears.
    
    Args:
        gears: List of Gear objects in the train
        driver_id: ID of the driver gear
        
    Returns:
        Overall gear ratio (output teeth / input teeth)
    """
    # Find the driver gear
    driver_gear = next((g for g in gears if g.id == driver_id), None)
    if not driver_gear:
        return 0.0
    
    # Find the output gear (last gear in the chain)
    output_gear = gears[-1]
    
    return output_gear.num_teeth / driver_gear.num_teeth

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

def calculate_gear_train(gears: List[Gear], start_index: int, end_index: int) -> float:
    """Calculate gear train ratio between two gears in a chain.
    
    Args:
        gears: List of Gear objects in the train
        start_index: Index of the starting gear
        end_index: Index of the ending gear
        
    Returns:
        Overall gear ratio (output teeth / input teeth)
    """
    # Validate indices
    if start_index < 0 or end_index < 0 or start_index >= len(gears) or end_index >= len(gears):
        return 0.0
    
    ratio = 1.0
    current_index = start_index
    
    # Traverse the chain from start to end
    while current_index != end_index:
        next_index = current_index + 1 if current_index < end_index else current_index - 1
        if next_index < 0 or next_index >= len(gears):
            return 0.0
        
        # Calculate ratio for this pair
        ratio *= gears[current_index].num_teeth / gears[next_index].num_teeth
        current_index = next_index
    
    return ratio

def line_segments_intersect(a: Vector2D, b: Vector2D, c: Vector2D, d: Vector2D) -> bool:
    """Check if line segment AB intersects line segment CD."""
    def ccw(A, B, C):
        return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)
    
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

def line_segment_intersects_polygon(p1: Vector2D, p2: Vector2D, poly: List[Vector2D]) -> bool:
    """Check if a line segment intersects a polygon."""
    n = len(poly)
    for i in range(n):
        q1 = poly[i]
        q2 = poly[(i + 1) % n]
        if line_segments_intersect(p1, p2, q1, q2):
            return True
    return False
