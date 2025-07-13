import os
import json
import xml.etree.ElementTree as ET
from typing import List, Dict
from components import Vector2D
import config

def process_svg_directory(input_dir: str, output_dir: str):
    """Process all SVG files in a directory and save as JSON"""
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".svg"):
            svg_path = os.path.join(input_dir, filename)
            data = parse_svg_file(svg_path)
            
            base_name = os.path.splitext(filename)[0]
            
            # Save as JSON
            json_path = os.path.join(output_dir, f"{base_name}.json")
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Create system visualization image
            from visualization import render_system
            from components import SystemState
            
            # Create minimal system state for visualization
            boundary_vectors = [Vector2D(p['x'], p['y']) for p in data['boundary_poly']]
            input_shaft_vec = Vector2D(data['input_shaft']['x'], data['input_shaft']['y']) if data['input_shaft'] else None
            output_shaft_vec = Vector2D(data['output_shaft']['x'], data['output_shaft']['y']) if data['output_shaft'] else None
            
            system_state = SystemState(
                boundary_poly=boundary_vectors,
                input_shaft=input_shaft_vec,
                output_shaft=output_shaft_vec,
                target_ratio=1.0,
                gears=[]  # Add empty gears list
            )
            
            # Save system visualization
            image_path = os.path.join(output_dir, f"{base_name}.png")
            render_system(system_state, image_path)

def normalize_to_100x100(points: List[Dict]) -> List[Dict]:
    """Normalize points to fit within a 100x100 area while preserving aspect ratio"""
    if not points:
        return points
        
    # Find bounding box
    min_x = min(p['x'] for p in points)
    max_x = max(p['x'] for p in points)
    min_y = min(p['y'] for p in points)
    max_y = max(p['y'] for p in points)
    
    width = max_x - min_x
    height = max_y - min_y
    max_dim = max(width, height)
    
    if max_dim == 0:
        return points
        
    # Calculate scaling factor to fit within 100x100
    scale = 100.0 / max_dim
    
    # Normalize all points
    normalized = []
    for p in points:
        normalized.append({
            'x': (p['x'] - min_x) * scale,
            'y': (p['y'] - min_y) * scale
        })
        
    return normalized

def parse_svg_file(svg_path: str) -> Dict:
    """Parse an SVG file to extract boundary, input shaft, and output shaft"""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
    # Extract raw boundary points
    raw_boundary = []
    polygons = root.findall('.//{http://www.w3.org/2000/svg}polygon')
    if polygons:
        points_str = polygons[0].get('points').strip()
        points = points_str.split()
        for point in points:
            if ',' in point:
                x, y = map(float, point.split(','))
                raw_boundary.append({"x": x, "y": y})
    
    # Normalize boundary points
    boundary = normalize_to_100x100(raw_boundary)
    
    # Extract shafts
    input_shaft = None
    output_shaft = None
    
    # Calculate normalization parameters from raw_boundary
    if raw_boundary:
        min_x = min(p['x'] for p in raw_boundary)
        min_y = min(p['y'] for p in raw_boundary)
        max_x = max(p['x'] for p in raw_boundary)
        max_y = max(p['y'] for p in raw_boundary)
        max_dim = max(max_x - min_x, max_y - min_y)
        scale = 100.0 / max_dim if max_dim > 0 else 1.0
    else:
        min_x = min_y = 0
        scale = 1.0
    
    for circle in root.findall('.//{http://www.w3.org/2000/svg}circle'):
        fill = circle.get('fill', '').lower()
        raw_cx = float(circle.get('cx'))
        raw_cy = float(circle.get('cy'))
        
        # Normalize shaft positions using same scaling as boundary
        cx = (raw_cx - min_x) * scale
        cy = (raw_cy - min_y) * scale
        
        if 'green' in fill or '#00ff00' in fill:
            input_shaft = {"x": cx, "y": cy}
        elif 'red' in fill or '#ff0000' in fill:
            output_shaft = {"x": cx, "y": cy}
    
    # Load constraints from JSON file if available
    constraints_path = svg_path.replace('.svg', '_constraints.json')
    constraints = {
        "torque_ratio": "1:1",
        "mass_space_ratio": 0.5
    }
    
    if os.path.exists(constraints_path):
        with open(constraints_path, 'r') as f:
            constraints = json.load(f)
    
    # Convert Vector2D objects to serializable dictionaries
    def vector_to_dict(v):
        return {"x": v.x, "y": v.y} if v else None

    return {
        "boundary_poly": [vector_to_dict(Vector2D(p['x'], p['y'])) for p in boundary],
        "input_shaft": vector_to_dict(Vector2D(input_shaft['x'], input_shaft['y'])) if input_shaft else None,
        "output_shaft": vector_to_dict(Vector2D(output_shaft['x'], output_shaft['y'])) if output_shaft else None,
        "constraints": constraints
    }

# Example usage
if __name__ == "__main__":
    input_dir = "data"
    output_dir = "data/intermediate"
    process_svg_directory(input_dir, output_dir)
