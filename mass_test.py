from components import Gear, Vector2D
import config
from visualization import render_system

# Create a sample gear
gear = Gear(
    id=1,
    center=Vector2D(0, 0),
    num_teeth=20,
    module=config.GEAR_MODULE,
    is_driver=True
)

# Create a simple system state for visualization
class MockSystemState:
    boundary_poly = [Vector2D(-50, -50), Vector2D(50, -50), Vector2D(50, 50), Vector2D(-50, 50)]
    gears = [gear]
    input_shaft = Vector2D(-40, 0)
    output_shaft = Vector2D(40, 0)
    target_ratio = 2.0

# Calculate and display mass
mass = gear.mass(config.GEAR_THICKNESS, config.GEAR_DENSITY)
print(f"Gear with {gear.num_teeth} teeth has mass: {mass:.2f}g")

# Visualize the gear
system_state = MockSystemState()
render_system(system_state, "mass_test.png")
print("Visualization saved to mass_test.png")
