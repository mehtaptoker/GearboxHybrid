import cv2
import numpy as np
import pytesseract
from components import Vector2D
import json
import os
import re
from typing import Dict, List

def process_png_directory(input_dir: str, output_dir: str):
    """Process all PNG files in a directory and save as JSON"""
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            png_path = os.path.join(input_dir, filename)
            data = parse_png_file(png_path)
            
            base_name = os.path.splitext(filename)[0]
            
            # Save as JSON
            json_path = os.path.join(output_dir, f"{base_name}.json")
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Generate system-level image
            img_path = os.path.join(output_dir, f"{base_name}_system.png")
            generate_system_image(data, img_path)

def parse_png_file(png_path: str) -> Dict:
    """Parse a PNG file to extract boundary, input shaft, and output shaft"""
    # Load image
    img = cv2.imread(png_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for green and red
    green_lower = np.array([35, 100, 100])
    green_upper = np.array([85, 255, 255])
    red_lower1 = np.array([0, 100, 100])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 100, 100])
    red_upper2 = np.array([179, 255, 255])
    
    # Create masks for green and red
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # Find green contours (input shaft)
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    input_shaft = None
    if green_contours:
        largest_green = max(green_contours, key=cv2.contourArea)
        M = cv2.moments(largest_green)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            input_shaft = {"x": float(cx), "y": float(cy)}
    
    # Find red contours (output shaft)
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_shaft = None
    if red_contours:
        largest_red = max(red_contours, key=cv2.contourArea)
        M = cv2.moments(largest_red)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            output_shaft = {"x": float(cx), "y": float(cy)}
    
    # Extract boundary using contour detection on grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    boundary = extract_boundary(gray)
    
    # Load constraints from JSON file if available
    constraints_path = png_path.replace('.png', '_constraints.json')
    constraints = {
        "torque_ratio": "1:1",
        "mass_space_ratio": 0.5
    }
    
    if os.path.exists(constraints_path):
        with open(constraints_path, 'r') as f:
            constraints = json.load(f)
    
    return {
        "boundary_poly": boundary,
        "input_shaft": input_shaft,
        "output_shaft": output_shaft,
        "constraints": constraints
    }

def generate_system_image(data: Dict, save_path: str):
    """Generate a system-level image based on parsed JSON data"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw boundary
    if data['boundary_poly']:
        boundary_points = [(p['x'], p['y']) for p in data['boundary_poly']]
        boundary_poly = Polygon(boundary_points, closed=True, fill=None, edgecolor='black', linewidth=2)
        ax.add_patch(boundary_poly)
    
    # Draw input and output shafts
    if data['input_shaft']:
        ax.plot(data['input_shaft']['x'], data['input_shaft']['y'], 'go', markersize=15, label='Input Shaft')
    if data['output_shaft']:
        ax.plot(data['output_shaft']['x'], data['output_shaft']['y'], 'ro', markersize=15, label='Output Shaft')
    
    # Set plot limits and labels
    ax.autoscale()
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Flip the image to match original orientation
    ax.set_title('System Constraints')
    ax.legend()
    
    # Save image
    plt.savefig(save_path)
    plt.close()

def extract_shaft_position(text: str, shaft_name: str) -> Dict:
    """Extract shaft position from OCR text"""
    pattern = rf"{shaft_name}.*?\((-?\d+\.\d+),\s*(-?\d+\.\d+)\)"
    match = re.search(pattern, text)
    if match:
        return {"x": float(match.group(1)), "y": float(match.group(2))}
    return None

def extract_boundary(gray_img: np.ndarray) -> List[Dict]:
    """Detect boundary using contour detection"""
    # Apply thresholding
    _, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find largest contour (assumed to be the boundary)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Simplify contour
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Convert to list of points
        boundary = []
        for point in approx:
            x, y = point[0]
            boundary.append({"x": float(x), "y": float(y)})
            
        return boundary
    return []

if __name__ == "__main__":
    # Process PNG files directly from the data directory
    input_dir = "data"
    output_dir = "data/intermediate"
    process_png_directory(input_dir, output_dir)
