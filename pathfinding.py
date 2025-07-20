import math
from typing import List, Dict, Set, Tuple, Optional
from components import Vector2D
from physics import line_segment_intersects_polygon

class VisibilityGraphPathfinder:
    def __init__(self, obstacles: List[List[Vector2D]]):
        """
        Initialize the pathfinder with obstacle polygons.
        
        Args:
            obstacles: List of obstacle polygons, each represented as a list of Vector2D vertices
        """
        self.obstacles = obstacles
        self.graph: Dict[Vector2D, Set[Vector2D]] = {}
        
    def _line_of_sight(self, p1: Vector2D, p2: Vector2D) -> bool:
        """
        Check if there's a clear line of sight between two points.
        
        Args:
            p1: Start point
            p2: End point
            
        Returns:
            True if line segment between p1 and p2 doesn't intersect any obstacles, False otherwise
        """
        for obstacle in self.obstacles:
            if line_segment_intersects_polygon(p1, p2, obstacle):
                return False
        return True
        
    def build_visibility_graph(self, start: Vector2D, end: Vector2D):
        """
        Build visibility graph for all obstacle vertices plus start and end points.
        
        Args:
            start: Path start point
            end: Path end point
        """
        # Collect all vertices from obstacles plus start and end points
        all_vertices = [start, end]
        for obstacle in self.obstacles:
            all_vertices.extend(obstacle)
            
        # Initialize graph
        self.graph = {vertex: set() for vertex in all_vertices}
        
        # Check visibility between all pairs of vertices
        for i, v1 in enumerate(all_vertices):
            for j, v2 in enumerate(all_vertices):
                if i != j and self._line_of_sight(v1, v2):
                    self.graph[v1].add(v2)
                    self.graph[v2].add(v1)
                    
    def find_shortest_path(self, start: Vector2D, end: Vector2D) -> Optional[List[Vector2D]]:
        """
        Find shortest path using Dijkstra's algorithm.
        
        Args:
            start: Path start point
            end: Path end point
            
        Returns:
            List of vertices representing the shortest path, or None if no path exists
        """
        # Build visibility graph if not already built
        if not self.graph:
            self.build_visibility_graph(start, end)
            
        # Dijkstra's algorithm implementation
        distances: Dict[Vector2D, float] = {vertex: float('inf') for vertex in self.graph}
        prev: Dict[Vector2D, Optional[Vector2D]] = {vertex: None for vertex in self.graph}
        distances[start] = 0
        
        unvisited = set(self.graph.keys())
        
        while unvisited:
            # Find vertex with minimum distance
            current = min(unvisited, key=lambda v: distances[v])
            unvisited.remove(current)
            
            # Stop if we've reached the end
            if current == end:
                break
                
            # Update distances to neighbors
            for neighbor in self.graph[current]:
                if neighbor in unvisited:
                    dx = current.x - neighbor.x
                    dy = current.y - neighbor.y
                    alt = distances[current] + math.sqrt(dx*dx + dy*dy)
                    if alt < distances[neighbor]:
                        distances[neighbor] = alt
                        prev[neighbor] = current
                        
        # Reconstruct path if we found one
        if distances[end] == float('inf'):
            return None
            
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = prev[current]
        return path[::-1]  # Reverse to get start->end order

def generate_collision_free_path(
    start: Vector2D, 
    end: Vector2D, 
    obstacles: List[List[Vector2D]]
) -> Optional[List[Vector2D]]:
    """
    Generate a collision-free path between two points avoiding obstacles.
    
    Args:
        start: Path start point
        end: Path end point
        obstacles: List of obstacle polygons
        
    Returns:
        List of waypoints representing the path, or None if no path exists
    """
    pathfinder = VisibilityGraphPathfinder(obstacles)
    return pathfinder.find_shortest_path(start, end)
