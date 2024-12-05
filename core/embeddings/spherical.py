"""Spherical embedding implementation."""
import numpy as np
from typing import Dict, List, Tuple
import networkx as nx
from .base import BaseEmbedding

class SphericalEmbedding(BaseEmbedding):
    """Spherical space embedding using graph structure."""
    
    def __init__(self, dim: int = 3):
        """Initialize embedding with dimension (minimum 3 for sphere)."""
        super().__init__(max(3, dim))  # Ensure at least 3D for sphere
        
    def _initialize_embeddings(self, graph: nx.Graph) -> Dict[int, np.ndarray]:
        """Initialize embeddings uniformly on the unit sphere."""
        pos = {}
        
        # Generate points using Fibonacci sphere method for uniform distribution
        n = len(graph.nodes())
        phi = np.pi * (3 - np.sqrt(5))  # golden angle in radians
        
        for i, node in enumerate(graph.nodes()):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i  # golden angle increment
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            # Create the full dimensional vector
            coords = np.zeros(self.dim)
            coords[0] = x
            coords[1] = y
            coords[2] = z
            
            pos[node] = coords
            
        return pos
    
    def _update_embeddings(self, graph: nx.Graph, iterations: int = 50):
        """Update embeddings using Riemannian optimization on sphere."""
        learning_rate = 0.1
        
        def _project_to_sphere(v):
            """Project vector onto unit sphere."""
            norm = np.linalg.norm(v)
            if norm > 0:
                return v / norm
            return v
        
        def _exp_map(x, v):
            """Exponential map on sphere."""
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-7:
                return x
            return np.cos(v_norm) * x + np.sin(v_norm) * v / v_norm
        
        for _ in range(iterations):
            # Store old positions
            old_pos = {node: self.embeddings[node].copy() for node in graph.nodes()}
            
            # Update each node's position
            for u in graph.nodes():
                grad = np.zeros(self.dim)
                
                # Attractive forces from neighbors
                for v in graph.neighbors(u):
                    # Vector in tangent space
                    diff = old_pos[v] - old_pos[u] * np.dot(old_pos[u], old_pos[v])
                    grad += diff
                
                # Repulsive forces from non-neighbors
                for v in graph.nodes():
                    if v != u and v not in graph.neighbors(u):
                        diff = old_pos[v] - old_pos[u] * np.dot(old_pos[u], old_pos[v])
                        grad -= 0.1 * diff
                
                # Update position using exponential map
                tangent_vector = learning_rate * grad
                self.embeddings[u] = _exp_map(old_pos[u], tangent_vector)
                
                # Ensure we stay on the sphere
                self.embeddings[u] = _project_to_sphere(self.embeddings[u])
    
    def compute_distance(self, node1: int, node2: int) -> float:
        """Compute geodesic distance between two nodes on the sphere."""
        # Get normalized vectors
        x = self.embeddings[node1]
        y = self.embeddings[node2]
        
        # Compute cosine of angle between vectors
        cos_angle = np.clip(np.dot(x, y), -1.0, 1.0)
        
        # Return great circle distance
        return float(np.arccos(cos_angle))
    
    def get_visualization_coords(self, node: int) -> Tuple[float, float]:
        """Get 2D coordinates for visualization using stereographic projection."""
        x = self.embeddings[node]
        
        # Use stereographic projection from north pole
        scale = 1 / (1 + x[2])  # z coordinate is index 2
        return float(x[0] * scale), float(x[1] * scale)
