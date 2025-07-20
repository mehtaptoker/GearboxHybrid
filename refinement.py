import math
from typing import List, Tuple, Optional
from components import Gear, Vector2D
from physics import check_meshing, check_collision, is_gear_inside_boundary
import config

def propagate_radii(
    start_gear: Gear,
    end_gear: Gear,
    path: List[Vector2D],
    module: float
) -> Tuple[List[float], float]:
    """
    Propagate gear radii along the path from start to end gear,
    accounting for integer tooth constraints and proper meshing.
    
    Args:
        start_gear: Fixed start gear
        end_gear: Fixed end gear
        path: List of intermediate joint positions
        module: Gear module (tooth size)
        
    Returns:
        Tuple containing:
        - List of radii for intermediate gears
        - Final endpoint mismatch error
    """
    radii = []
    
    # Calculate tooth counts instead of radii first
    teeth_counts = []
    
    # Calculate tooth counts based on actual distances
    teeth_counts = []
    
    # Calculate tooth counts based on actual distances
    teeth_counts = []
    
    # First gear: distance = (start_gear.num_teeth + teeth0) * module / 2
    dx0 = path[0].x - start_gear.center.x
    dy0 = path[0].y - start_gear.center.y
    distance0 = math.sqrt(dx0*dx0 + dy0*dy0)
    teeth0 = round(2 * distance0 / module - start_gear.num_teeth)
    if teeth0 < 1:
        teeth0 = 1
    teeth_counts.append(teeth0)
    radius0 = teeth0 * module / 2
    radii.append(radius0)
    
    # Intermediate gears: distance = (teeth_counts[i-1] + teeth) * module / 2
    for i in range(1, len(path)):
        dx = path[i].x - path[i-1].x
        dy = path[i].y - path[i-1].y
        distance = math.sqrt(dx*dx + dy*dy)
        teeth = round(2 * distance / module - teeth_counts[i-1])
        if teeth < 1:
            teeth = 1
        teeth_counts.append(teeth)
        radius = teeth * module / 2
        radii.append(radius)
    
    # Last gear to end gear: distance = (teeth_counts[-1] + end_gear.num_teeth) * module / 2
    dx_end = end_gear.center.x - path[-1].x
    dy_end = end_gear.center.y - path[-1].y
    end_distance = math.sqrt(dx_end*dx_end + dy_end*dy_end)
    endpoint_error = abs((teeth_counts[-1] + end_gear.num_teeth) * module / 2 - end_distance)
    
    
    return radii, endpoint_error

def validate_gear_train(
    start_gear: Gear,
    end_gear: Gear,
    path: List[Vector2D],
    radii: List[float],
    existing_gears: List[Gear],
    module: float,
    tolerance: float = config.MESHING_TOLERANCE  # Use tolerance from config
) -> Tuple[bool, Optional[str]]:
    """
    Validate the generated gear train.
    
    Args:
        start_gear: Fixed start gear
        end_gear: Fixed end gear
        path: List of intermediate joint positions
        radii: List of radii for intermediate gears
        existing_gears: List of existing gears in the system
        module: Gear module
        tolerance: Acceptable error tolerance
        
    Returns:
        Tuple (valid, reason) where:
        - valid: True if gear train is valid, False otherwise
        - reason: None if valid, otherwise error reason
    """
    # Check for non-positive radii
    if any(r <= 0 for r in radii):
        return False, "Non-positive radius detected in intermediate gears"
    
    # Create intermediate gears with proper tooth counts
    intermediate_gears = []
    for i, (pos, radius) in enumerate(zip(path, radii)):
        # Calculate tooth count based on radius
        num_teeth = round(2 * radius / module)
        # Ensure minimum tooth count
        if num_teeth < 1:
            num_teeth = 1
        gear = Gear(
            id=1000 + i,  # Temporary ID
            center=pos,
            num_teeth=num_teeth,
            module=module
        )
        intermediate_gears.append(gear)
    
    # Check meshing between start and first gear with tolerance
    if not check_meshing(start_gear, intermediate_gears[0], abs_tol=tolerance):
        # Calculate required distance
        required_distance = start_gear.radius + intermediate_gears[0].radius
        actual_distance = math.sqrt(
            (start_gear.center.x - intermediate_gears[0].center.x)**2 +
            (start_gear.center.y - intermediate_gears[0].center.y)**2
        )
        return False, f"Meshing failure between start gear and first intermediate: required {required_distance:.3f}, actual {actual_distance:.3f}"
    
    # Check meshing between intermediate gears with tolerance
    for i in range(len(intermediate_gears) - 1):
        gear1 = intermediate_gears[i]
        gear2 = intermediate_gears[i+1]
        if not check_meshing(gear1, gear2, abs_tol=tolerance):
            # Calculate required distance
            required_distance = gear1.radius + gear2.radius
            actual_distance = math.sqrt(
                (gear1.center.x - gear2.center.x)**2 +
                (gear1.center.y - gear2.center.y)**2
            )
            return False, f"Meshing failure between intermediate gears {i} and {i+1}: required {required_distance:.3f}, actual {actual_distance:.3f}"
    
    # Check meshing between last intermediate and end gear with tolerance
    if not check_meshing(intermediate_gears[-1], end_gear, abs_tol=tolerance):
        # Calculate required distance
        required_distance = intermediate_gears[-1].radius + end_gear.radius
        actual_distance = math.sqrt(
            (intermediate_gears[-1].center.x - end_gear.center.x)**2 +
            (intermediate_gears[-1].center.y - end_gear.center.y)**2
        )
        return False, f"Meshing failure between last intermediate and end gear: required {required_distance:.3f}, actual {actual_distance:.3f}"
    
    # Check collisions with existing gears
    for gear in intermediate_gears:
        if check_collision(gear, existing_gears):
            return False, f"Collision detected for intermediate gear at ({gear.center.x}, {gear.center.y})"
    
    # Check collisions between intermediate gears
    # Check collisions between intermediate gears with tolerance
    for i in range(len(intermediate_gears)):
        for j in range(i + 1, len(intermediate_gears)):
            dx = intermediate_gears[i].center.x - intermediate_gears[j].center.x
            dy = intermediate_gears[i].center.y - intermediate_gears[j].center.y
            distance = math.sqrt(dx*dx + dy*dy)
            min_distance = intermediate_gears[i].radius + intermediate_gears[j].radius
            if distance < min_distance - tolerance:
                return False, f"Collision between intermediate gears {i} and {j} (distance: {distance:.3f} < min: {min_distance:.3f})"
    
    return True, None

def adjust_joints(
    path: List[Vector2D],
    start_gear: Gear,
    end_gear: Gear,
    radii: List[float],
    endpoint_error: float,
    adjustment_factor: float = 0.05  # Reduced adjustment factor for finer control
) -> List[Vector2D]:
    """
    Adjust joint positions to ensure proper meshing distances and reduce endpoint mismatch.
    
    Args:
        path: Current intermediate joint positions
        start_gear: Fixed start gear
        end_gear: Fixed end gear
        radii: Current intermediate gear radii
        endpoint_error: Current endpoint mismatch
        adjustment_factor: Scaling factor for adjustments
        
    Returns:
        Adjusted path
    """
    if not path:
        return path
    
    # Adjust first joint for proper meshing with start gear
    start_dir = Vector2D(
        path[0].x - start_gear.center.x,
        path[0].y - start_gear.center.y
    ).normalized()
    required_start_distance = start_gear.radius + radii[0]
    current_start_distance = math.sqrt(
        (path[0].x - start_gear.center.x)**2 +
        (path[0].y - start_gear.center.y)**2
    )
    start_adjustment = (required_start_distance - current_start_distance) * adjustment_factor
    path[0] = Vector2D(
        start_gear.center.x + start_dir.x * (current_start_distance + start_adjustment),
        start_gear.center.y + start_dir.y * (current_start_distance + start_adjustment)
    )
    
    # Adjust intermediate joints for proper meshing with neighbors
    for i in range(1, len(path)):
        # Calculate vector between current and previous joint
        vec = Vector2D(
            path[i].x - path[i-1].x,
            path[i].y - path[i-1].y
        )
        current_distance = math.sqrt(vec.x**2 + vec.y**2)
        required_distance = radii[i-1] + radii[i]
        adjustment = (required_distance - current_distance) * adjustment_factor
        
        if current_distance > 0:
            unit_vec = Vector2D(vec.x / current_distance, vec.y / current_distance)
            # Move current joint away from previous to increase distance
            path[i] = Vector2D(
                path[i].x + unit_vec.x * adjustment,
                path[i].y + unit_vec.y * adjustment
            )
    
    # Adjust last joint for proper meshing with end gear
    end_dir = Vector2D(
        end_gear.center.x - path[-1].x,
        end_gear.center.y - path[-1].y
    ).normalized()
    required_end_distance = radii[-1] + end_gear.radius
    current_end_distance = math.sqrt(
        (path[-1].x - end_gear.center.x)**2 +
        (path[-1].y - end_gear.center.y)**2
    )
    end_adjustment = (required_end_distance - current_end_distance) * adjustment_factor
    path[-1] = Vector2D(
        path[-1].x + end_dir.x * end_adjustment,
        path[-1].y + end_dir.y * end_adjustment
    )
    
    return path

def generate_gear_train(
    start_gear: Gear,
    end_gear: Gear,
    path: List[Vector2D],
    existing_gears: List[Gear],
    boundary_poly: List[Vector2D],
    obstacles: List[Gear],
    module: float,
    max_iterations: int = 1000,  # Increased to allow more convergence time
    tolerance: float = 0.5  # Tolerance in mm
) -> Optional[List[Gear]]:
    """
    Generate a tangent gear train between start and end gears using iterative refinement.
    
    Args:
        start_gear: Fixed start gear
        end_gear: Fixed end gear
        path: Initial intermediate joint positions
        existing_gears: List of existing gears in the system
        boundary_poly: Boundary polygon
        obstacles: List of obstacle gears
        module: Gear module
        max_iterations: Maximum refinement iterations
        tolerance: Acceptable error tolerance
        
    Returns:
        List of intermediate gears if successful, None otherwise
    """
    # If no intermediate joints, check direct meshing
    if not path:
        if check_meshing(start_gear, end_gear, abs_tol=tolerance):
            return []
        else:
            return None

    # We'll work with a copy of the path to avoid modifying the original
    path = [Vector2D(p.x, p.y) for p in path]
    num_joints = len(path)
    
    # Step 1: Pre-calculate the required distances between consecutive joints and endpoints
    # We don't know the radii yet, so we'll recalculate in each iteration
    
    for iteration in range(max_iterations):
        # Propagate radii along the path
        radii, endpoint_error = propagate_radii(start_gear, end_gear, path, module)
        
        # Debugging info
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: endpoint_error={endpoint_error:.3f}")
        
        # Check for non-positive radii
        if any(r <= 0 for r in radii):
            print(f"Iteration {iteration}: Non-positive radius detected")
            # Adjust the path to reduce curvature: move each joint towards the midpoint of its neighbors
            for i in range(1, num_joints-1):
                prev_point = path[i-1]
                next_point = path[i+1]
                midpoint = Vector2D((prev_point.x + next_point.x)/2, (prev_point.y + next_point.y)/2)
                # Move the joint 10% towards the midpoint
                path[i] = path[i].interpolate(midpoint, 0.1)
            continue

        # Check endpoint connection: distance from last joint to end gear center should be approximately last_radius + end_gear.radius
        last_radius = radii[-1]
        required_end_distance = last_radius + end_gear.radius
        dx_end = end_gear.center.x - path[-1].x
        dy_end = end_gear.center.y - path[-1].y
        actual_end_distance = math.sqrt(dx_end*dx_end + dy_end*dy_end)
        end_error = required_end_distance - actual_end_distance
        
        # If endpoint error is too large, adjust the last joint
        if abs(end_error) > tolerance:
            print(f"Iteration {iteration}: Endpoint error too large: {end_error:.3f}")
            # Move the last joint along the line to the end gear
            direction = Vector2D(dx_end, dy_end).normalized()
            # Use a dynamic adjustment factor that decreases over iterations
            adj_factor = max(0.1, 0.5 * (1 - iteration/max_iterations))
            adjustment = end_error * adj_factor
            path[-1] = Vector2D(path[-1].x + direction.x * adjustment, path[-1].y + direction.y * adjustment)
            continue

        # Check the distances between consecutive joints
        adjustment_needed = False
        for i in range(num_joints-1):
            r1 = radii[i]
            r2 = radii[i+1]
            required_distance = r1 + r2
            dx = path[i+1].x - path[i].x
            dy = path[i+1].y - path[i].y
            actual_distance = math.sqrt(dx*dx + dy*dy)
            distance_error = abs(required_distance - actual_distance)
            
            if distance_error > tolerance:
                print(f"Iteration {iteration}: Distance error at joint {i}: {distance_error:.3f}")
                adjustment_needed = True
                # Adjust the joint i+1 to correct the distance
                direction = Vector2D(dx, dy).normalized()
                # Use a dynamic adjustment factor that decreases over iterations
                adj_factor = max(0.1, 0.5 * (1 - iteration/max_iterations))
                adjustment = (required_distance - actual_distance) * adj_factor
                # Move the joint i+1 away from joint i if too close, or closer if too far
                path[i+1] = Vector2D(path[i+1].x + direction.x * adjustment, path[i+1].y + direction.y * adjustment)
        
        # If we adjusted any joint, continue to next iteration without further checks
        if adjustment_needed:
            for i in range(1, num_joints-1):
                prev_point = path[i-1]
                next_point = path[i+1]
                midpoint = Vector2D((prev_point.x + next_point.x)/2, (prev_point.y + next_point.y)/2)
                path[i] = path[i].interpolate(midpoint, 0.1)
            continue

        # Check boundary constraints
        boundary_violation = False
        for i, (pos, radius) in enumerate(zip(path, radii)):
            gear = Gear(
                id=1000 + i,
                center=pos,
                num_teeth=round(2 * radius / module),
                module=module
            )
            if not is_gear_inside_boundary(gear, boundary_poly):
                boundary_violation = True
                # Move the gear inside the boundary by pushing it towards the center of the boundary?
                # For simplicity, we break and hope the next iteration adjusts
                break
        
        if boundary_violation:
            # Adjust by moving all points towards the center of the boundary? 
            # Instead, we break and try again with a slight adjustment
            # Move each joint towards the centroid of the boundary
            centroid = Vector2D(0, 0)
            for point in boundary_poly:
                centroid.x += point.x
                centroid.y += point.y
            centroid.x /= len(boundary_poly)
            centroid.y /= len(boundary_poly)
            for i in range(num_joints):
                direction = Vector2D(centroid.x - path[i].x, centroid.y - path[i].y).normalize()
                path[i] = Vector2D(path[i].x + direction.x * 1.0, path[i].y + direction.y * 1.0)
            continue

        # If we passed all checks, create the intermediate gears
        intermediate_gears = []
        for i, (pos, radius) in enumerate(zip(path, radii)):
            num_teeth = round(2 * radius / module)
            gear = Gear(
                id=1000 + i,
                center=pos,
                num_teeth=num_teeth,
                module=module
            )
            intermediate_gears.append(gear)
        return intermediate_gears

    print(f"Failed to generate gear train after {max_iterations} iterations")
    return None
