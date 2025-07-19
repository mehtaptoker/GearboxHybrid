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
    # Verify distance matches sum of radii
    pass

def calculate_gear_train_ratio(start_gear, end_gear):
    # Traverse gear graph to calculate final ratio
    pass

def check_collisions(gears):
    # Detect gear intersections
    pass

def check_boundaries(gears, boundary_poly):
    # Verify all gears within boundary
    pass
