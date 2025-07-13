import os
import json
from components import SystemState
from typing import List

def load_training_data(data_dir: str) -> List[SystemState]:
    """Load training data from intermediate JSON files"""
    scenarios = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            # Convert to SystemState
            boundary = [Vector2D(p['x'], p['y']) for p in data['boundary_poly']]
            input_shaft = Vector2D(data['input_shaft']['x'], data['input_shaft']['y'])
            output_shaft = Vector2D(data['output_shaft']['x'], data['output_shaft']['y'])
            
            # Parse constraints
            constraints = data.get('constraints', {})
            
            # Convert torque ratio string to float
            if 'torque_ratio' in constraints:
                n, d = constraints['torque_ratio'].split(':')
                target_ratio = float(n) / float(d)
            else:
                target_ratio = 1.0  # Default ratio
                
            # Create SystemState with constraints
            system_state = SystemState(
                boundary_poly=boundary,
                input_shaft=input_shaft,
                output_shaft=output_shaft,
                target_ratio=target_ratio
            )
            system_state.constraints = constraints
            
            scenarios.append(system_state)
    return scenarios
