import math
from collections import deque
from typing import List, Optional, Dict

from components import Gear, Vector2D
import config



# The shapely library is used in your original file
from shapely.geometry import Point, Polygon
from shapely.geometry.polygon import orient


def point_in_polygon(point: Vector2D, polygon: List[Vector2D]) -> bool:
    """Check if a point is inside a polygon using ray casting algorithm."""
    x, y = point.x, point.y
    n = len(polygon)
    inside = False

    p1 = polygon[0]
    for i in range(n + 1):
        p2 = polygon[i % n]
        if y > min(p1.y, p2.y):
            if y <= max(p1.y, p2.y):
                if x <= max(p1.x, p2.x):
                    if p1.y != p2.y:
                        xinters = (y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y) + p1.x
                    if p1.x == p2.x or x <= xinters:
                        inside = not inside
        p1 = p2

    return inside


def distance_to_line_segment(point: Vector2D, line_start: Vector2D, line_end: Vector2D) -> float:
    """Calculate the shortest distance from a point to a line segment."""
    line_vec = Vector2D(line_end.x - line_start.x, line_end.y - line_start.y)
    point_vec = Vector2D(point.x - line_start.x, point.y - line_start.y)

    line_len_sq = line_vec.x ** 2 + line_vec.y ** 2
    if line_len_sq == 0:
        return math.sqrt(point_vec.x ** 2 + point_vec.y ** 2)

    proj_length = (point_vec.x * line_vec.x + point_vec.y * line_vec.y) / line_len_sq
    proj_length = max(0, min(1, proj_length))

    closest_point = Vector2D(
        line_start.x + line_vec.x * proj_length,
        line_start.y + line_vec.y * proj_length
    )

    dx = point.x - closest_point.x
    dy = point.y - closest_point.y
    return math.sqrt(dx ** 2 + dy ** 2)


def check_meshing(gear1: Gear, gear2: Gear, abs_tol: float = 0.5) -> bool:
    """Check if two gears can mesh."""
    if gear1.z_layer != gear2.z_layer or not math.isclose(gear1.module, gear2.module, abs_tol=0.01):
        return False

    dx = gear1.center.x - gear2.center.x
    dy = gear1.center.y - gear2.center.y
    distance = math.sqrt(dx * dx + dy * dy)
    sum_radii = gear1.radius + gear2.radius

    return math.isclose(distance, sum_radii, abs_tol=abs_tol)


def check_collision(new_gear: Gear, obstacles: List[object], return_reason=False) -> bool:
    """Check if new gear collides with existing obstacles."""
    if not isinstance(obstacles, list):
        obstacles = [obstacles]

    for obstacle in obstacles:
        if isinstance(obstacle, Vector2D):
            dx = new_gear.center.x - obstacle.x
            dy = new_gear.center.y - obstacle.y
            distance = math.sqrt(dx * dx + dy * dy)

            if distance < new_gear.radius - config.MESHING_TOLERANCE:
                reason = f"Collision with point obstacle at ({obstacle.x}, {obstacle.y})"
                return (True, reason) if return_reason else True

        elif isinstance(obstacle, Gear):
            if obstacle.z_layer != new_gear.z_layer:
                continue
            dx = new_gear.center.x - obstacle.center.x
            dy = new_gear.center.y - obstacle.center.y
            distance = math.sqrt(dx * dx + dy * dy)
            sum_radii = new_gear.radius + obstacle.radius

            if distance < sum_radii - config.MESHING_TOLERANCE:
                reason = f"Collision with gear {obstacle.id} (distance={distance:.2f} < sum_radii={sum_radii:.2f})"
                return (True, reason) if return_reason else True

    return (False, "") if return_reason else False


def is_inside_boundary(point: Vector2D, boundary_poly: List[Vector2D]) -> bool:
    """Check if point is inside boundary polygon using shapely."""
    poly_coords = [(v.x, v.y) for v in boundary_poly]
    polygon = orient(Polygon(poly_coords))
    return polygon.contains(Point(point.x, point.y))


def is_gear_inside_boundary(gear: Gear, boundary_poly: List[Vector2D], return_reason=False) -> bool:
    """Check if entire gear is within boundary polygon using shapely."""
    poly_coords = [(v.x, v.y) for v in boundary_poly]
    polygon = orient(Polygon(poly_coords))
    gear_circle = Point(gear.center.x, gear.center.y).buffer(gear.radius)

    if polygon.covers(gear_circle):
        return (True, "") if return_reason else True

    center_point = Point(gear.center.x, gear.center.y)
    if not polygon.contains(center_point):
        reason = f"Gear center ({gear.center.x:.2f}, {gear.center.y:.2f}) is outside boundary"
        return (False, reason) if return_reason else False

    min_distance = polygon.boundary.distance(center_point)
    if min_distance < gear.radius:
        reason = f"Gear extends beyond boundary (min_distance={min_distance:.2f} < radius={gear.radius:.2f})"
        return (False, reason) if return_reason else False

    return (True, "") if return_reason else True


# Note: The file has two definitions for calculate_gear_train, which is unusual.
# I've kept both as they were.
def calculate_gear_train(gears: List[Gear], start_index: int, end_index: int) -> float:
    """Calculate gear train ratio between two gears in a chain."""
    if not (0 <= start_index < len(gears) and 0 <= end_index < len(gears)):
        return 0.0

    ratio = 1.0
    current_index = start_index

    step = 1 if current_index < end_index else -1
    while current_index != end_index:
        next_index = current_index + step
        if not (0 <= next_index < len(gears)):
            return 0.0

        ratio *= gears[current_index].num_teeth / gears[next_index].num_teeth
        current_index = next_index

    return ratio


def calculate_gear_train_ratio(gears: List[Gear], driver_id: int) -> float:
    """Calculate gear train ratio for a chain of gears."""
    driver_gear = next((g for g in gears if g.id == driver_id), None)
    if not driver_gear or not gears:
        return 0.0

    output_gear = gears[-1]
    return output_gear.num_teeth / driver_gear.num_teeth


def validate_gear(gear: Gear) -> bool:
    """Validate gear parameters for physical sensibility."""
    if not (gear.tip_diameter > gear.reference_diameter >
            gear.base_diameter > gear.root_diameter):
        return False
    if not (14.5 <= gear.pressure_angle <= 25.0):
        return False
    if gear.module <= 0 or gear.num_teeth < 12:
        return False

    return True


# ==============================================================================
# CORRECTED ROBUST INTERSECTION LOGIC
# ==============================================================================

def on_segment(p: Vector2D, q: Vector2D, r: Vector2D) -> bool:
    """Given three collinear points p, q, r, check if point q lies on segment 'pr'."""
    return (q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and
            q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y))


def orientation(p: Vector2D, q: Vector2D, r: Vector2D) -> int:
    """Find orientation of ordered triplet (p, q, r).
    Returns: 0 (Collinear), 1 (Clockwise), 2 (Counterclockwise)
    """
    val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
    if val == 0: return 0
    return 1 if val > 0 else 2


def line_segments_intersect(p1: Vector2D, q1: Vector2D, p2: Vector2D, q2: Vector2D) -> bool:
    """Check if line segment 'p1q1' and 'p2q2' intersect."""
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case where orientations on both sides are different
    if o1 != o2 and o3 != o4:
        return True

    # Special Cases for when points are collinear
    if o1 == 0 and on_segment(p1, p2, q1): return True
    if o2 == 0 and on_segment(p1, q2, q1): return True
    if o3 == 0 and on_segment(p2, p1, q2): return True
    if o4 == 0 and on_segment(p2, q1, q2): return True

    return False


def line_segment_intersects_polygon(p1: Vector2D, p2: Vector2D, poly: List[Vector2D]) -> bool:
    """Check if a line segment p1-p2 intersects with any edge of a polygon."""
    n = len(poly)
    for i in range(n):
        q1 = poly[i]
        q2 = poly[(i + 1) % n]  # Next vertex, wraps around
        if line_segments_intersect(p1, p2, q1, q2):
            return True
    return False