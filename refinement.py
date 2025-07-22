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
    using floating-point calculations to reduce quantization errors.
    
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
    
    # First gear: distance = start_gear.radius + radius0
    dx0 = path[0].x - start_gear.center.x
    dy0 = path[0].y - start_gear.center.y
    distance0 = math.sqrt(dx0*dx0 + dy0*dy0)
    radius0 = distance0 - start_gear.radius
    
    # Calculate tooth count based on radius
    num_teeth0 = max(1, round(radius0 * 2 / module))
    radius0 = (num_teeth0 * module) / 2
    radii.append(radius0)
    
    # Intermediate gears: distance = current_radius + next_radius
    for i in range(1, len(path)):
        dx = path[i].x - path[i-1].x
        dy = path[i].y - path[i-1].y
        distance = math.sqrt(dx*dx + dy*dy)
        radius = distance - radii[i-1]
        
        # Calculate tooth count based on radius
        num_teeth = max(1, round(radius * 2 / module))
        radius = (num_teeth * module) / 2
        radii.append(radius)
    
    # Last gear to end gear: distance = last_radius + end_gear.radius
    dx_end = end_gear.center.x - path[-1].x
    dy_end = end_gear.center.y - path[-1].y
    end_distance = math.sqrt(dx_end*dx_end + dy_end*dy_end)
    required_end_distance = radii[-1] + end_gear.radius
    endpoint_error = abs(required_end_distance - end_distance)
    
    return radii, endpoint_error

def validate_gear_train(
    start_gear: Gear,
    end_gear: Gear,
    path: List[Vector2D],
    radii: List[float],
    existing_gears: List[Gear],
    module: float,
    tolerance: float = config.MESHING_TOLERANCE,  # Use tolerance from config
    intermediate_optimization: bool = False  # New parameter
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
        intermediate_optimization: If True, skip collision checks during optimization
        
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
    
    # Skip collision checks during intermediate optimization steps
    if not intermediate_optimization:
        # Check collisions with existing gears
        for gear in intermediate_gears:
            if check_collision(gear, existing_gears):
                return False, f"Collision detected for intermediate gear at ({gear.center.x}, {gear.center.y})"
        
        # Check collisions between intermediate gears
        for i in range(len(intermediate_gears)):
            for j in range(i + 1, len(intermediate_gears)):
                dx = intermediate_gears[i].center.x - intermediate_gears[j].center.x
                dy = intermediate_gears[i].center.y - intermediate_gears[j].center.y
                distance = math.sqrt(dx*dx + dy*dy)
                min_distance = intermediate_gears[i].radius + intermediate_gears[j].radius
                if distance < min_distance - tolerance:
                    return False, f"Collision between intermediate gears {i} and {j} (distance: {distance:.3f} < min: {min_distance:.3f})"
    
    return True, None


def generate_gear_train(
        start_gear: Gear,
        end_gear: Gear,
        path: List[Vector2D],
        existing_gears: List[Gear],
        boundary_poly: List[Vector2D],
        obstacles: List[Gear],
        module: float,
        max_iterations: int = 2000,  # Increased iterations for complex problems
        tolerance: float = 0.1,  # A reasonable tolerance for meshing
        learning_rate: float = 0.1  # The step size for gradient descent
) -> Optional[List[Gear]]:
    """
    Generate a tangent gear train using gradient descent optimization.
    """
    if not path:
        if check_meshing(start_gear, end_gear, abs_tol=tolerance):
            return []
        else:
            return None

    path = [Vector2D(p.x, p.y) for p in path]
    num_joints = len(path)

    # Main optimization loop
    for iteration in range(max_iterations):
        # 1. Propagate radii to determine the size of each gear
        radii, endpoint_error = propagate_radii(start_gear, end_gear, path, module)

        # Check for invalid radii which can happen if gears are too far apart
        if any(r <= 0 for r in radii):
            # If we have an invalid radius, it means the path is stretched too far.
            # A simple recovery is to reset the path by spacing the joints
            # evenly between the start and end gears.
            print(f"Iteration {iteration}: Invalid radius detected. Resetting path.")
            start_pos = start_gear.center
            end_pos = end_gear.center
            for i in range(num_joints):
                t = (i + 1) / (num_joints + 1)  # Interpolation factor
                path[i] = start_pos.interpolate(end_pos, t)
            continue  # Retry with the new path

        # --- LOSS CALCULATION ---
        # The loss is the sum of squared errors of all meshing distances.
        total_error = 0
        all_gears = [start_gear] + [Gear(id=0, center=p, num_teeth=int(r * 2 / module), module=module) for p, r in
                                    zip(path, radii)] + [end_gear]

        for i in range(len(all_gears) - 1):
            gear1 = all_gears[i]
            gear2 = all_gears[i + 1]
            actual_dist = gear1.center.magnitude(gear2.center)
            expected_dist = gear1.radius + gear2.radius
            error = actual_dist - expected_dist
            total_error += error ** 2

        # 2. Check for convergence
        if total_error < tolerance:
            print(f"Converged after {iteration} iterations with error {total_error:.4f}")
            break

        # --- GRADIENT CALCULATION ---
        # Calculate the gradient of the loss function for each joint.
        gradients = [Vector2D(0, 0) for _ in range(num_joints)]

        # We iterate through each joint and calculate its influence on the total error.
        for i in range(num_joints):
            # The position of joint 'i' affects two connections:
            #   - The connection to the previous gear (or start_gear)
            #   - The connection to the next gear (or end_gear)

            # Previous gear connection
            prev_gear = all_gears[i]
            current_gear = all_gears[i + 1]

            actual_dist = prev_gear.center.magnitude(current_gear.center)
            expected_dist = prev_gear.radius + current_gear.radius
            error = actual_dist - expected_dist

            # Gradient contribution from this connection
            grad_x = 2 * error * (current_gear.center.x - prev_gear.center.x) / actual_dist if actual_dist > 0 else 0
            grad_y = 2 * error * (current_gear.center.y - prev_gear.center.y) / actual_dist if actual_dist > 0 else 0
            gradients[i].x += grad_x
            gradients[i].y += grad_y

            # Next gear connection
            next_gear = all_gears[i + 2]

            actual_dist = current_gear.center.magnitude(next_gear.center)
            expected_dist = current_gear.radius + next_gear.radius
            error = actual_dist - expected_dist

            # Gradient contribution from this connection
            grad_x = 2 * error * (current_gear.center.x - next_gear.center.x) / actual_dist if actual_dist > 0 else 0
            grad_y = 2 * error * (current_gear.center.y - next_gear.center.y) / actual_dist if actual_dist > 0 else 0
            gradients[i].x += grad_x
            gradients[i].y += grad_y

        # 3. Update joint positions (take a step opposite the gradient)
        for i in range(num_joints):
            path[i].x -= learning_rate * gradients[i].x
            path[i].y -= learning_rate * gradients[i].y

    # --- Final Validation ---
    final_radii, final_error = propagate_radii(start_gear, end_gear, path, module)
    valid, reason = validate_gear_train(start_gear, end_gear, path, final_radii, existing_gears + obstacles, module,
                                        tolerance)

    if valid and final_error < tolerance:
        print("Successfully generated and validated gear train.")
        intermediate_gears = []
        for i, (pos, radius) in enumerate(zip(path, final_radii)):
            num_teeth = round(2 * radius / module)
            gear = Gear(id=1000 + i, center=pos, num_teeth=num_teeth, module=module)
            intermediate_gears.append(gear)
        return intermediate_gears
    else:
        print(f"Failed to generate valid gear train. Reason: {reason}, Final Error: {final_error:.4f}")
        # Optionally, you could still try the constraint_solver as a fallback here.
        return None

def generate_gear_train(
    start_gear: Gear,
    end_gear: Gear,
    path: List[Vector2D],
    existing_gears: List[Gear],
    boundary_poly: List[Vector2D],
    obstacles: List[Gear],
    module: float,
    max_iterations: int = 1000,  # Increased to allow more convergence time
    tolerance: float = 5.0  # Increased tolerance to 5.0 mm to match physics tolerance
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
    
    # Pre-calculate the target total length of the path
    total_path_length = 0
    for i in range(num_joints - 1):
        dx = path[i+1].x - path[i].x
        dy = path[i+1].y - path[i].y
        total_path_length += math.sqrt(dx*dx + dy*dy)
    
    for iteration in range(max_iterations):
        # Propagate radii along the path
        radii, endpoint_error = propagate_radii(start_gear, end_gear, path, module)
        
        # Check for convergence
        if endpoint_error < tolerance:
            break
        
        # Debugging info
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: endpoint_error={endpoint_error:.3f}")
        
        # Check for non-positive radii
        if any(r <= 0 for r in radii):
            print(f"Iteration {iteration}: Non-positive radius detected")
            # Instead of moving towards midpoint, adjust the path proportionally
            # Calculate the direction from start to end
            total_dx = end_gear.center.x - start_gear.center.x
            total_dy = end_gear.center.y - start_gear.center.y
            total_distance = math.sqrt(total_dx**2 + total_dy**2)
            
            # Normalize direction
            if total_distance > 0:
                total_dir = Vector2D(total_dx/total_distance, total_dy/total_distance)
            else:
                total_dir = Vector2D(1, 0)  # Default direction if start and end are at same point
            
            # Redistribute points along the line from start to end
            for i in range(num_joints):
                t = (i + 1) / (num_joints + 1)
                path[i] = Vector2D(
                    start_gear.center.x + total_dir.x * total_distance * t,
                    start_gear.center.y + total_dir.y * total_distance * t
                )
            continue

        # Calculate total error for the entire chain
        total_error = 0
        # Start point error
        dx_start = path[0].x - start_gear.center.x
        dy_start = path[0].y - start_gear.center.y
        start_distance = math.sqrt(dx_start**2 + dy_start**2)
        start_error = abs(start_gear.radius + radii[0] - start_distance)
        
        # End point error
        dx_end = end_gear.center.x - path[-1].x
        dy_end = end_gear.center.y - path[-1].y
        end_distance = math.sqrt(dx_end**2 + dy_end**2)
        end_error = abs(radii[-1] + end_gear.radius - end_distance)
        
        # Intermediate errors
        intermediate_errors = []
        for i in range(num_joints-1):
            dx = path[i+1].x - path[i].x
            dy = path[i+1].y - path[i].y
            distance = math.sqrt(dx**2 + dy**2)
            required = radii[i] + radii[i+1]
            error = abs(required - distance)
            intermediate_errors.append(error)
            total_error += error
            
        total_error += start_error + end_error
        
        # If total error is below tolerance, we're done
        if total_error < tolerance * (num_joints + 1):
            break
            
        # Gradient descent optimization
        learning_rate = 0.05
        max_steps = 10
        current_error = total_error
        
        for _ in range(max_steps):
            # Calculate gradients for each point
            gradients = [Vector2D(0, 0) for _ in range(num_joints)]
            
            # Gradient for start point
            dx_start = path[0].x - start_gear.center.x
            dy_start = path[0].y - start_gear.center.y
            start_distance = math.sqrt(dx_start**2 + dy_start**2)
            start_error = start_gear.radius + radii[0] - start_distance
            if start_distance > 0:
                grad_x = 2 * start_error * (dx_start / start_distance)
                grad_y = 2 * start_error * (dy_start / start_distance)
                gradients[0].x += grad_x
                gradients[0].y += grad_y
            
            # Gradient for end point
            dx_end = end_gear.center.x - path[-1].x
            dy_end = end_gear.center.y - path[-1].y
            end_distance = math.sqrt(dx_end**2 + dy_end**2)
            end_error = radii[-1] + end_gear.radius - end_distance
            if end_distance > 0:
                grad_x = 2 * end_error * (-dx_end / end_distance)
                grad_y = 2 * end_error * (-dy_end / end_distance)
                gradients[-1].x += grad_x
                gradients[-1].y += grad_y
            
            # Gradients for intermediate points
            for i in range(num_joints-1):
                dx = path[i+1].x - path[i].x
                dy = path[i+1].y - path[i].y
                distance = math.sqrt(dx**2 + dy**2)
                segment_error = (radii[i] + radii[i+1]) - distance
                
                if distance > 0:
                    # Gradient for point i
                    grad_x_i = 2 * segment_error * (-dx / distance)
                    grad_y_i = 2 * segment_error * (-dy / distance)
                    gradients[i].x += grad_x_i
                    gradients[i].y += grad_y_i
                    
                    # Gradient for point i+1
                    grad_x_i1 = 2 * segment_error * (dx / distance)
                    grad_y_i1 = 2 * segment_error * (dy / distance)
                    gradients[i+1].x += grad_x_i1
                    gradients[i+1].y += grad_y_i1
            
            # Apply gradients with learning rate
            new_path = []
            for i in range(num_joints):
                new_x = path[i].x - learning_rate * gradients[i].x
                new_y = path[i].y - learning_rate * gradients[i].y
                new_path.append(Vector2D(new_x, new_y))
            
            # Calculate new error
            new_radii, new_endpoint_error = propagate_radii(start_gear, end_gear, new_path, module)
            new_total_error = new_endpoint_error
            
            # Check for non-positive radii
            if any(r <= 0 for r in new_radii):
                break
                
            # Calculate intermediate errors
            for i in range(num_joints-1):
                dx = new_path[i+1].x - new_path[i].x
                dy = new_path[i+1].y - new_path[i].y
                distance = math.sqrt(dx**2 + dy**2)
                segment_error = abs((new_radii[i] + new_radii[i+1]) - distance)
                new_total_error += segment_error
            
            # If error decreased, accept the step
            if new_total_error < current_error:
                path = new_path
                radii = new_radii
                current_error = new_total_error
                endpoint_error = new_endpoint_error
            else:
                # Reduce learning rate if error increased
                learning_rate *= 0.7
        
        # Update path with optimized positions
        path = new_path

        # During optimization, skip collision checks
        valid, reason = validate_gear_train(
            start_gear, 
            end_gear, 
            path, 
            radii, 
            existing_gears + obstacles, 
            module, 
            tolerance,
            intermediate_optimization=True  # Skip collision checks during optimization
        )
        if not valid:
            if iteration % 100 == 0: # Avoid spamming the console
                print(f"Validation failed: {reason}")
            # The main adjustment logic should handle this failure in the next iteration.
            # The previous ad-hoc adjustment here was likely causing instability.
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
                print(f"Iteration {iteration}: Boundary violation at joint {i}")
                # Move the gear inside the boundary by pushing it towards the center of the boundary
                centroid = Vector2D(0, 0)
                for point in boundary_poly:
                    centroid.x += point.x
                    centroid.y += point.y
                centroid.x /= len(boundary_poly)
                centroid.y /= len(boundary_poly)
                
                # FIXED: Use normalized() instead of normalize()
                direction = Vector2D(centroid.x - pos.x, centroid.y - pos.y).normalized()
                # Move further inside if violation is severe
                path[i] = Vector2D(pos.x + direction.x * 2.0, pos.y + direction.y * 2.0)
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
                # FIXED: Use normalized() instead of normalize()
                direction = Vector2D(centroid.x - path[i].x, centroid.y - path[i].y).normalized()
                path[i] = Vector2D(path[i].x + direction.x * 1.0, path[i].y + direction.y * 1.0)
            continue

        # Final validation with full checks
        valid, reason = validate_gear_train(
            start_gear, 
            end_gear, 
            path, 
            radii, 
            existing_gears + obstacles, 
            module, 
            tolerance,
            intermediate_optimization=False  # Full collision checks
        )
        
        if valid:
            # Create the intermediate gears
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
        else:
            print(f"Validation failed after optimization: {reason}")
            # Continue optimization if validation fails

    print(f"Failed to generate gear train after {max_iterations} iterations")
    # Try constraint solver as fallback
    from constraint_solver import solve_gear_chain
    print("Attempting constraint solver as fallback...")
    intermediate_gears = solve_gear_chain(
        start_gear,
        end_gear,
        path,
        existing_gears,
        boundary_poly,
        obstacles,
        module,
        tolerance=tolerance,
        max_iterations=500
    )
    
    if intermediate_gears is None:
        print("Constraint solver also failed to generate gear train")
        return None
    
    # Validate the solution from constraint solver
    valid, reason = validate_gear_train(
        start_gear,
        end_gear,
        [g.center for g in intermediate_gears],
        [g.radius for g in intermediate_gears],
        existing_gears + obstacles,
        module,
        tolerance
    )
    
    if not valid:
        print(f"Constraint solver solution failed validation: {reason}")
        return None
    
    print("Constraint solver successfully generated gear train")
    return intermediate_gears
