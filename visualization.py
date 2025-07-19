import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.patches import Circle, Polygon
from components import SystemState, Gear
import config
def generate_report(state: SystemState, report_dir: str = "reports"):
    """Generate a comprehensive report for a system state."""
    # Create timestamped report directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(report_dir, f"report_{timestamp}")
    os.makedirs(report_path, exist_ok=True)
    
    # Save visualizations
    system_img = os.path.join(report_path, "system.png")
    render_system(state, system_img)
    
    # Save gear details for each gear
    for gear in state.gears:
        gear_img = os.path.join(report_path, f"gear_{gear.id}.png")
        render_gear_detail(gear, gear_img)
    
    # Calculate metrics
    total_mass = sum(gear.mass(config.GEAR_THICKNESS, config.GEAR_DENSITY) for gear in state.gears)
    num_gears = len(state.gears)
    driver_gears = [g for g in state.gears if g.is_driver]
    driven_gears = [g for g in state.gears if not g.is_driver]
    
    # Create report data
    report = {
        "timestamp": timestamp,
        "input_shaft": {"x": state.input_shaft.x, "y": state.input_shaft.y},
        "output_shaft": {"x": state.output_shaft.x, "y": state.output_shaft.y},
        "target_ratio": state.target_ratio,
        "boundary_points": [{"x": p.x, "y": p.y} for p in state.boundary_poly],
        "total_mass": total_mass,
        "num_gears": num_gears,
        "driver_gears": len(driver_gears),
        "driven_gears": len(driven_gears),
        "gears": [
            {
                "id": gear.id,
                "x": gear.center.x,
                "y": gear.center.y,
                "num_teeth": gear.num_teeth,
                "module": gear.module,
                "z_layer": gear.z_layer,
                "is_driver": gear.is_driver,
                "mass": gear.mass(config.GEAR_THICKNESS, config.GEAR_DENSITY)
            } for gear in state.gears
        ]
    }
    
    # Save JSON report
    report_json = os.path.join(report_path, "report.json")
    with open(report_json, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report_path

def render_system(state: SystemState, save_path: str = None):
    """
    Render system overview showing boundary, gear centers with IDs/types, 
    and reference circles only.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw boundary
    if state.boundary_poly:
        boundary_points = [(p.x, p.y) for p in state.boundary_poly]
        if boundary_points:  # Ensure we have points to draw
            boundary_poly = Polygon(boundary_points, closed=True, fill=None, edgecolor='black', linewidth=2)
            ax.add_patch(boundary_poly)
    
    # Draw large, distinct input and output shafts for detection
    if state.input_shaft:
        # Large green circle with thick border
        circle = Circle((state.input_shaft.x, state.input_shaft.y), 15, 
                        facecolor='green', edgecolor='black', linewidth=2, 
                        label='Input Shaft', alpha=0.9)
        ax.add_patch(circle)
        # Label inside the circle
        ax.text(state.input_shaft.x, state.input_shaft.y, 'IN', 
                ha='center', va='center', fontsize=12, color='white', 
                fontweight='bold')
    if state.output_shaft:
        # Large red circle with thick border
        circle = Circle((state.output_shaft.x, state.output_shaft.y), 15, 
                        facecolor='red', edgecolor='black', linewidth=2, 
                        label='Output Shaft', alpha=0.9)
        ax.add_patch(circle)
        # Label inside the circle
        ax.text(state.output_shaft.x, state.output_shaft.y, 'OUT', 
                ha='center', va='center', fontsize=12, color='white', 
                fontweight='bold')
    
    # Draw gears (system view)
    for gear in state.gears:
        color = 'blue' if gear.is_driver else 'cyan'
        
        # Draw reference circle only
        circle = Circle((gear.center.x, gear.center.y), gear.reference_radius, 
                        fill=False, edgecolor=color, linewidth=1.5)
        ax.add_patch(circle)
        
        # Draw center marker and ID
        ax.plot(gear.center.x, gear.center.y, 'ko', markersize=3)
        gear_type = "D" if gear.is_driver else "d"
        ax.text(gear.center.x, gear.center.y, f"{gear.id}{gear_type}", 
                ha='center', va='center', fontsize=8, 
                bbox=dict(facecolor='white', alpha=0.7))
    
    # Set plot limits and labels
    ax.set_xlim(-config.WORKSPACE_SIZE/2, config.WORKSPACE_SIZE/2)
    ax.set_ylim(-config.WORKSPACE_SIZE/2, config.WORKSPACE_SIZE/2)
    ax.set_aspect('equal')
    ax.set_title('Gear System Overview')
    ax.legend()
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def render_gear_detail(gear: Gear, save_path: str = None):
    """
    Render detailed gear visualization showing all circles and diameters.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw all gear circles
    circles = [
        (gear.reference_radius, 'solid', 'blue', 'Reference Circle'),
        (gear.tip_radius, 'dashed', 'red', 'Tip Circle'),
        (gear.base_radius, 'dotted', 'green', 'Base Circle'),
        (gear.root_radius, 'dashdot', 'purple', 'Root Circle')
    ]
    
    for radius, linestyle, color, label in circles:
        circle = Circle((0, 0), radius, 
                        fill=False, edgecolor=color, linestyle=linestyle, 
                        linewidth=1.5, label=label)
        ax.add_patch(circle)
    
    # Draw center marker
    ax.plot(0, 0, 'ko', markersize=5)
    
    # Add diameter lines with labels
    diameters = [
        (gear.reference_diameter, 'blue', 'Reference Diameter'),
        (gear.tip_diameter, 'red', 'Tip Diameter'),
        (gear.base_diameter, 'green', 'Base Diameter'),
        (gear.root_diameter, 'purple', 'Root Diameter')
    ]
    
    for diameter, color, label in diameters:
        linestyle = 'solid' if label == 'Reference Diameter' else 'dashed'
        ax.plot([-diameter/2, diameter/2], [0, 0], color=color, linestyle=linestyle, 
                linewidth=2, label=label)
    
    # Add gear info
    info = (
        f"Gear ID: {gear.id}\n"
        f"Teeth: {gear.num_teeth}\n"
        f"Module: {gear.module}mm\n"
        f"Pressure Angle: {gear.pressure_angle}°\n"
        f"Driver: {'Yes' if gear.is_driver else 'No'}"
    )
    ax.text(0, gear.tip_radius + 10, info, ha='center', va='bottom', 
            fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    # Set plot limits and labels
    max_radius = max(gear.tip_radius, gear.root_radius, gear.base_radius, gear.reference_radius)
    ax.set_xlim(-max_radius*1.2, max_radius*1.2)
    ax.set_ylim(-max_radius*1.2, max_radius*1.2)
    ax.set_aspect('equal')
    ax.set_title(f'Gear {gear.id} Detail')
    ax.legend(loc='upper right')
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
