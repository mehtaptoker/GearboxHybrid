# Physics and Constraint System

## Gear Representation
Each gear is represented as:
```python
class Gear:
    id: int
    center: Vector2D  # (x, y) position
    num_teeth: int
    module: float      # Determines tooth size
    z_layer: int       # For handling shaft crossings
    is_driver: bool    # True for input gears
```

## Physical Constraints
### 1. Meshing Condition
For two gears to mesh:
```
distance(center_A, center_B) = radius_A + radius_B
radius = (module * num_teeth) / 2
```

### 2. Rotation Ratio
```
speed_B / speed_A = num_teeth_A / num_teeth_B
```

### 3. Torque Ratio Approximation
```
torque_B / torque_A ≈ diameter_B / diameter_A = num_teeth_B / num_teeth_A
```
We approximate the torque ratio using the reference diameter ratio, which is equivalent to the teeth ratio since all gears share the same module.

### 4. Non-Intersection
- No gear can intersect another on the same z-layer
- Implemented via Shapely polygon collision detection

### 5. Boundary Adherence
All gears must lie entirely within the boundary polygon
- Boundary margin enforced during placement

## Constraint Encoding
Constraints are defined in JSON:
```json
{
  "torque_ratio": "1:2",
  "boundary_margin": 10.0,
  "min_gear_teeth": 15,
  "max_gear_teeth": 50,
  "max_gears": 10
}
```

## Two-Step Gear Generation
Our system uses a two-step approach for generating gears:

1. **Geometric Placement**: 
   - Determines reference diameter and positions
   - Ensures gears fit within boundaries and avoid collisions
   - Optimizes placement using reinforcement learning

2. **Teeth Assignment**:
   - Assigns teeth count based on desired gear ratios
   - Ensures meshing compatibility between gears
   - Optimizes for efficiency and manufacturability

This decoupled approach allows for more efficient exploration of valid gear configurations.

## Physics Validation
Key validation functions:
```python
def check_meshing(gear_A, gear_B):
    """Verify distance matches sum of radii"""
    distance = np.linalg.norm(gear_A.center - gear_B.center)
    return abs(distance - (gear_A.radius + gear_B.radius)) < 1e-5

def calculate_gear_train_ratio(start_gear, end_gear):
    """Traverse gear graph to calculate final ratio"""
    ratio = 1.0
    current_gear = start_gear
    while current_gear != end_gear:
        next_gear = current_gear.meshed_gears[0]  # Simplified
        ratio *= current_gear.num_teeth / next_gear.num_teeth
        current_gear = next_gear
    return ratio

def check_collisions(gears):
    """Detect gear intersections using Shapely"""
    for i, gear1 in enumerate(gears):
        for gear2 in gears[i+1:]:
            if gear1.polygon.intersects(gear2.polygon):
                return False
    return True

def check_boundaries(gears, boundary_poly):
    """Verify all gears within boundary"""
    for gear in gears:
        if not boundary_poly.contains(gear.polygon):
            return False
    return True
```

## Workflow Integration
The physics validation functions are integrated into our workflow:

1. **Training** (`run_train.sh`): 
   - Uses physics engine to simulate gear interactions
   - Validates each step using the above constraints
   - Terminates episodes when constraints are violated

2. **Testing** (`test.sh`):
   - Runs demo cases with full physics validation
   - Generates reports showing constraint violations

3. **Evaluation** (`run_eval.sh`):
   - Evaluates models by running physics-validated episodes
   - Generates detailed reports including physics metrics

To run a full physics validation:
```bash
./test.sh --physics-only
```
