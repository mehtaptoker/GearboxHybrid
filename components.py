from dataclasses import dataclass, field
from typing import List, Any  # Add Any import
import math
import config

@dataclass
class Vector2D:
    x: float
    y: float
    
    @property
    def magnitude(self) -> float:
        """Calculate the magnitude of the vector"""
        return (self.x ** 2 + self.y ** 2) ** 0.5
        
    def normalized(self) -> 'Vector2D':
        """Return a normalized version of the vector"""
        mag = self.magnitude
        if mag == 0:
            return Vector2D(0, 0)
        return Vector2D(self.x / mag, self.y / mag)
        
    def __add__(self, other: 'Vector2D') -> 'Vector2D':
        return Vector2D(self.x + other.x, self.y + other.y)
        
    def __sub__(self, other: 'Vector2D') -> 'Vector2D':
        return Vector2D(self.x - other.x, self.y - other.y)
        
    def interpolate(self, other: 'Vector2D', t: float) -> 'Vector2D':
        """Linearly interpolate between two vectors"""
        return Vector2D(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t
        )
        
    def __hash__(self):
        return hash((self.x, self.y))
        
    def __eq__(self, other):
        if not isinstance(other, Vector2D):
            return False
        return math.isclose(self.x, other.x) and math.isclose(self.y, other.y)
        
    def to_dict(self) -> dict:
        """Convert Vector2D object to dictionary for serialization"""
        return {'x': self.x, 'y': self.y}

@dataclass
class Gear:
    id: int
    center: Vector2D
    num_teeth: int
    module: float
    pressure_angle: float = 20.0  # Default pressure angle in degrees
    z_layer: int = 0
    is_driver: bool = False
    connected_gears: List[int] = field(default_factory=list)

    @property
    def reference_diameter(self) -> float:
        """Diameter of the reference (pitch) circle"""
        return self.module * self.num_teeth

    @property
    def tip_diameter(self) -> float:
        """Diameter of the tip circle"""
        return self.reference_diameter + 2 * self.module

    @property
    def root_diameter(self) -> float:
        """Diameter of the root circle"""
        return self.reference_diameter - 2.5 * self.module

    @property
    def base_diameter(self) -> float:
        """Diameter of the base circle"""
        import math
        return self.reference_diameter * math.cos(math.radians(self.pressure_angle))

    @property
    def reference_radius(self) -> float:
        return self.reference_diameter / 2.0

    @property
    def tip_radius(self) -> float:
        return self.tip_diameter / 2.0

    @property
    def root_radius(self) -> float:
        return self.root_diameter / 2.0

    @property
    def base_radius(self) -> float:
        return self.base_diameter / 2.0

    @property
    def radius(self) -> float:
        """Default radius (reference radius) for physics calculations"""
        return self.reference_radius
        
    @property
    def area(self) -> float:
        """Calculate the area of the gear (assuming solid disk)"""
        import math
        return math.pi * (self.radius ** 2)
        
    def mass(self, thickness: float, density: float) -> float:
        """Calculate mass of the gear.
        Note: Dimensions are in mm, density in g/cm³.
        Convert mm³ to cm³ by dividing by 1000.
        """
        volume_mm3 = self.area * thickness
        volume_cm3 = volume_mm3 / 1000.0
        return volume_cm3 * density
        
    def to_dict(self) -> dict:
        """Convert Gear object to dictionary for serialization"""
        return {
            'id': self.id,
            'center': {'x': self.center.x, 'y': self.center.y},
            'num_teeth': self.num_teeth,
            'module': self.module,
            'pressure_angle': self.pressure_angle,
            'z_layer': self.z_layer,
            'is_driver': self.is_driver,
            'connected_gears': self.connected_gears
        }

@dataclass
class SystemState:
    boundary_poly: List[Vector2D]
    gears: List[Gear]
    input_shaft: Vector2D
    output_shaft: Vector2D
    target_ratio: float
    obstacles: List[Any] = field(default_factory=list)  # Add obstacles attribute
    connections: List[Any] = field(default_factory=list)  # Add connections attribute

    def calculate_ratio(self) -> float:
        """Calculate the gear ratio of the system."""
        input_gear = next((g for g in self.gears if g.is_driver), None)
        output_gear = next((g for g in self.gears if not g.is_driver), None)
        if input_gear and output_gear:
            return input_gear.num_teeth / output_gear.num_teeth
        return 0.0
        
    def is_connected(self, gear1: Gear, gear2: Gear) -> bool:
        """Check if two gears are connected (meshed)"""
        dist = math.sqrt((gear1.center.x - gear2.center.x)**2 + 
                         (gear1.center.y - gear2.center.y)**2)
        expected_dist = gear1.radius + gear2.radius
        return abs(dist - expected_dist) < config.GEAR_MODULE * 0.5  # 50% tolerance for practical purposes
        
    def are_gears_connected(self, gear1: Gear, gear2: Gear) -> bool:
        """Alias for is_connected method for backward compatibility"""
        return self.is_connected(gear1, gear2)
