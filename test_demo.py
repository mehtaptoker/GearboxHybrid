import random
from components import Gear, Vector2D, SystemState
from data_generator import generate_scenario
from physics import calculate_gear_train, check_meshing, is_inside_boundary
from visualization import render

# Generate a mock scenario
scenario = generate_scenario()
print("Generated scenario:")
print(f"Input shaft: ({scenario['input_shaft'].x}, {scenario['input_shaft'].y})")
print(f"Output shaft: ({scenario['output_shaft'].x}, {scenario['output_shaft'].y})")
print(f"Target ratio: {scenario['target_ratio']}")

# Calculate gear radii
driver_radius = 20 * 1.0 / 2  # num_teeth * module / 2
idler_radius = 30 * 1.0 / 2
output_radius = 40 * 1.0 / 2

# Calculate required distance between gears
driver_to_idler_dist = driver_radius + idler_radius
idler_to_output_dist = idler_radius + output_radius

# Create a simple gear train
driver_gear = Gear(
    id=1,
    center=Vector2D(scenario['input_shaft'].x, scenario['input_shaft'].y),
    num_teeth=20,
    module=1.0,
    z_layer=0,
    is_driver=True
)

# Calculate idler position
input_to_output = Vector2D(
    scenario['output_shaft'].x - scenario['input_shaft'].x,
    scenario['output_shaft'].y - scenario['input_shaft'].y
)
total_distance = input_to_output.magnitude

# Calculate position along the line from input to output
idler_x = scenario['input_shaft'].x + (driver_to_idler_dist / total_distance) * input_to_output.x
idler_y = scenario['input_shaft'].y + (driver_to_idler_dist / total_distance) * input_to_output.y

idler_gear = Gear(
    id=2,
    center=Vector2D(idler_x, idler_y),
    num_teeth=30,
    module=1.0,
    z_layer=0
)

# Adjust output position to match idler-output distance
output_dir = Vector2D(
    scenario['output_shaft'].x - idler_x,
    scenario['output_shaft'].y - idler_y
).normalized()
output_x = idler_x + output_dir.x * idler_to_output_dist
output_y = idler_y + output_dir.y * idler_to_output_dist

output_gear = Gear(
    id=3,
    center=Vector2D(output_x, output_y),
    num_teeth=40,
    module=1.0,
    z_layer=0
)

# Create system state
system_state = SystemState(
    boundary_poly=scenario['boundary_poly'],
    gears=[driver_gear, idler_gear, output_gear],
    input_shaft=scenario['input_shaft'],
    output_shaft=scenario['output_shaft'],
    target_ratio=scenario['target_ratio']
)

# Test physics calculations
print("\nPhysics tests:")
print(f"Driver-idler meshing: {check_meshing(driver_gear, idler_gear)}")
print(f"Idler-output meshing: {check_meshing(idler_gear, output_gear)}")
all_inside = all(is_inside_boundary(g.center, system_state.boundary_poly) for g in system_state.gears)
print(f"Gears inside boundary: {all_inside}")

ratio = calculate_gear_train(system_state.gears, driver_gear.id, output_gear.id)
print(f"Calculated gear ratio: {ratio}")

# Visualize the result
print("\nRendering visualization...")
render(system_state, save_path="test_visualization.png")
print("Visualization saved to test_visualization.png")
