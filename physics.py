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

def check_collision(new_gear: Gear, existing_gears: List[Gear], return_reason=False) -> bool:
    """Check if new gear collides with existing gears.
    
    Args:
        new_gear: Gear to check
        existing_gears: List of existing gears
        return_reason: If True, returns (result, reason) tuple
        
    Returns:
        True if collision detected, False otherwise
        If return_reason=True, returns (bool, str) tuple
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
            reason = f"Collision with gear {gear.id} (distance={distance:.2f} < sum_radii={sum_radii:.2f})"
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
    if min_distance < gear.radius:
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
            reason = (f"Gear edge point at ({x:.2f}, {y:.2f}) is outside boundary "
                      f"(distance={distance_to_boundary:.5f})")
            return (False, reason) if return_reason else False
            
    return (True, "") if return_reason else True

def calculate_gear_train_ratio(input_gear: Gear, output_gear: Gear) -> float:
    """Calculate gear train ratio between input and output gears.
    
    This implementation uses the approximation that the torque ratio is equal to the 
    ratio of the number of teeth on the output gear to the number of teeth on the input gear.
    
    Args:
        input_gear: Input gear
        output_gear: Output gear
        
    Returns:
        Approximate gear ratio (output_teeth / input_teeth)
    """
    return output_gear.num_teeth / input_gear.num_teeth

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
