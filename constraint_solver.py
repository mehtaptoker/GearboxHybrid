import math
from typing import List, Optional, Tuple
from components import Gear, Vector2D
from physics import check_meshing, check_collision, is_gear_inside_boundary
import config

def solve_gear_chain(
    start_gear: Gear,
    end_gear: Gear,
    initial_path: List[Vector2D],
    existing_gears: List[Gear],
    boundary_poly: List[Vector2D],
    obstacles: List[Gear],
    module: float,
    tolerance: float = config.MESHING_TOLERANCE,
    max_iterations: int = 100
) -> Optional[List[Gear]]:
    """
    Solve gear chain constraints using a backtracking algorithm with geometric constraints
    
    Args:
        start_gear: Fixed start gear
        end_gear: Fixed end gear
        initial_path: Initial intermediate joint positions
        existing_gears: List of existing gears in the system
        boundary_poly: Boundary polygon
        obstacles: List of obstacle gears
        module: Gear module
        tolerance: Acceptable error tolerance
        max_iterations: Maximum iterations for backtracking
        
    Returns:
        List of intermediate gears if successful, None otherwise
    """
    num_joints = len(initial_path)
    if num_joints == 0:
        if check_meshing(start_gear, end_gear, abs_tol=tolerance):
            return []
        return None
    
    # Create mutable path and calculate target positions
    path = [Vector2D(p.x, p.y) for p in initial_path]
    target_positions = calculate_target_positions(start_gear, end_gear, path, module)
    
    # Try to satisfy constraints with backtracking
    return backtrack_solution(
        start_gear,
        end_gear,
        path,
        target_positions,
        existing_gears + obstacles,
        boundary_poly,
        module,
        tolerance,
        max_iterations
    )

def calculate_target_positions(
    start_gear: Gear,
    end_gear: Gear,
    path: List[Vector2D],
    module: float
) -> List[Vector2D]:
    """
    Calculate ideal target positions for intermediate joints based on gear constraints
    
    Args:
        start_gear: Fixed start gear
        end_gear: Fixed end gear
        path: Intermediate joint positions
        module: Gear module
        
    Returns:
        List of target positions for each joint
    """
    # Propagate ideal tooth counts without rounding
    teeth_counts = []
    targets = []
    
    # Start to first joint
    dx0 = path[0].x - start_gear.center.x
    dy0 = path[0].y - start_gear.center.y
    dist0 = math.sqrt(dx0*dx0 + dy0*dy0)
    teeth0 = 2 * dist0 / module - start_gear.num_teeth
    teeth_counts.append(teeth0)
    
    # Intermediate joints
    for i in range(1, len(path)):
        dx = path[i].x - path[i-1].x
        dy = path[i].y - path[i-1].y
        dist = math.sqrt(dx*dx + dy*dy)
        teeth = 2 * dist / module - teeth_counts[i-1]
        teeth_counts.append(teeth)
    
    # Last joint to end
    dx_end = end_gear.center.x - path[-1].x
    dy_end = end_gear.center.y - path[-1].y
    dist_end = math.sqrt(dx_end*dx_end + dy_end*dy_end)
    teeth_end = 2 * dist_end / module - teeth_counts[-1]
    
    # Calculate target positions based on ideal tooth counts
    # Start from beginning and propagate positions
    current_pos = start_gear.center
    for i in range(len(path)):
        # Calculate direction to next joint
        if i < len(path) - 1:
            dir_vec = path[i+1] - path[i]
        else:
            dir_vec = end_gear.center - path[i]
        
        if dir_vec.magnitude > 0:
            dir_vec = dir_vec.normalized()
        
        # Calculate target distance based on ideal tooth count
        if i == 0:
            target_dist = (start_gear.num_teeth + teeth_counts[i]) * module / 2
        else:
            target_dist = (teeth_counts[i-1] + teeth_counts[i]) * module / 2
        
        # Scale the direction vector by the target distance
        scaled_dir = Vector2D(dir_vec.x * target_dist, dir_vec.y * target_dist)
        target_pos = Vector2D(current_pos.x + scaled_dir.x, current_pos.y + scaled_dir.y)
        targets.append(target_pos)
        current_pos = target_pos
    
    return targets

def backtrack_solution(
    start_gear: Gear,
    end_gear: Gear,
    path: List[Vector2D],
    targets: List[Vector2D],
    obstacles: List[Gear],
    boundary_poly: List[Vector2D],
    module: float,
    tolerance: float,
    max_iterations: int,
    current_iter: int = 0,
    step_size: float = 1.0
) -> Optional[List[Gear]]:
    """
    Recursive backtracking algorithm to find valid gear positions
    
    Args:
        start_gear: Fixed start gear
        end_gear: Fixed end gear
        path: Current intermediate joint positions
        targets: Ideal target positions for joints
        obstacles: List of obstacle gears
        boundary_poly: Boundary polygon
        module: Gear module
        tolerance: Acceptable error tolerance
        max_iterations: Maximum iterations allowed
        current_iter: Current iteration count
        step_size: Adjustment step size
        
    Returns:
        List of intermediate gears if successful, None otherwise
    """
    if current_iter >= max_iterations:
        return None
    
    # Try current configuration
    radii = []
    for i, pos in enumerate(path):
        # Calculate radius based on tooth count
        if i == 0:
            dist = math.sqrt((pos.x - start_gear.center.x)**2 + 
                             (pos.y - start_gear.center.y)**2)
            teeth = max(1, round(2 * dist / module - start_gear.num_teeth))
        elif i == len(path) - 1:
            dist = math.sqrt((pos.x - path[i-1].x)**2 + 
                             (pos.y - path[i-1].y)**2)
            teeth = max(1, round(2 * dist / module - radii[i-1] * 2 / module))
        else:
            dist = math.sqrt((pos.x - path[i-1].x)**2 + 
                             (pos.y - path[i-1].y)**2)
            teeth = max(1, round(2 * dist / module - radii[i-1] * 2 / module))
        
        radius = teeth * module / 2
        radii.append(radius)
    
    # Create intermediate gears
    intermediate_gears = []
    for i, (pos, radius) in enumerate(zip(path, radii)):
        teeth = round(2 * radius / module)
        gear = Gear(
            id=1000 + i,
            center=pos,
            num_teeth=teeth,
            module=module
        )
        intermediate_gears.append(gear)
    
    # Validate the gear train
    valid, _ = validate_gear_train(
        start_gear,
        end_gear,
        path,
        radii,
        obstacles,
        module,
        tolerance
    )
    
    if valid:
        return intermediate_gears
    
    # Adjust positions toward targets
    new_path = []
    for i, (current, target) in enumerate(zip(path, targets)):
        # Move toward target with decreasing step size
        step = step_size * (0.9 ** current_iter)
        dx = target.x - current.x
        dy = target.y - current.y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist > 0:
            dx = dx / dist * min(step, dist)
            dy = dy / dist * min(step, dist)
        new_pos = Vector2D(current.x + dx, current.y + dy)
        
        # Ensure within boundary
        temp_gear = Gear(
            id=0,
            center=new_pos,
            num_teeth=intermediate_gears[i].num_teeth,
            module=module
        )
        if not is_gear_inside_boundary(temp_gear, boundary_poly):
            # Move toward centroid if out of bounds
            centroid = Vector2D(0, 0)
            for point in boundary_poly:
                centroid.x += point.x
                centroid.y += point.y
            centroid.x /= len(boundary_poly)
            centroid.y /= len(boundary_poly)
            
            dir_vec = centroid - new_pos
            if dir_vec.magnitude > 0:
                dir_vec = dir_vec.normalized()
                # Scale the direction vector by the step size
                scaled_dir = Vector2D(dir_vec.x * step, dir_vec.y * step)
                new_pos = Vector2D(new_pos.x + scaled_dir.x, new_pos.y + scaled_dir.y)
        
        new_path.append(new_pos)
    
    # Recursive call with updated path
    return backtrack_solution(
        start_gear,
        end_gear,
        new_path,
        targets,
        obstacles,
        boundary_poly,
        module,
        tolerance,
        max_iterations,
        current_iter + 1,
        step_size
    )

def validate_gear_train(
    start_gear: Gear,
    end_gear: Gear,
    path: List[Vector2D],
    radii: List[float],
    obstacles: List[Gear],
    module: float,
    tolerance: float = config.MESHING_TOLERANCE
) -> Tuple[bool, Optional[str]]:
    """
    Validate the generated gear train.
    
    Args:
        start_gear: Fixed start gear
        end_gear: Fixed end gear
        path: List of intermediate joint positions
        radii: List of radii for intermediate gears
        obstacles: List of obstacle gears (including existing gears)
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
    
    # Check collisions with obstacles
    for gear in intermediate_gears:
        if check_collision(gear, obstacles):
            return False, f"Collision detected for intermediate gear at ({gear.center.x}, {gear.center.y})"
    
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
