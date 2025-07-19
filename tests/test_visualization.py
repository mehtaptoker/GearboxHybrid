import cv2
import numpy as np
import json
import os
from PIL import Image, ImageDraw

def main():
    # Load the original image
    original_image_path = 'data/Example1.png'
    original_image = Image.open(original_image_path).convert("RGBA")

    # Load the scenario data
    scenario_path = 'data/intermediate/Example1.json'
    with open(scenario_path, 'r') as f:
        scenario = json.load(f)

    # Create a transparent overlay
    overlay = Image.new("RGBA", original_image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Draw the boundary polygon
    boundary_poly = [(p['x'], p['y']) for p in scenario['boundary_poly']]
    draw.polygon(boundary_poly, outline=(255, 0, 0, 255), width=3)

    # Draw input and output shafts
    input_shaft = (scenario['input_shaft']['x'], scenario['input_shaft']['y'])
    output_shaft = (scenario['output_shaft']['x'], scenario['output_shaft']['y'])
    
    draw.ellipse([input_shaft[0]-5, input_shaft[1]-5, input_shaft[0]+5, input_shaft[1]+5], fill=(0, 255, 0, 255))
    draw.ellipse([output_shaft[0]-5, output_shaft[1]-5, output_shaft[0]+5, output_shaft[1]+5], fill=(0, 0, 255, 255))

    # Composite the images
    combined = Image.alpha_composite(original_image, overlay)

    # Save the result
    output_path = 'test_visualization.png'
    combined.save(output_path)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    main()
