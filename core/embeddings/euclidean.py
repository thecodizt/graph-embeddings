"""Euclidean embedding implementation."""
import numpy as np
from typing import Dict, List, Tuple
import networkx as nx
from .base import BaseEmbedding

class EuclideanEmbedding(BaseEmbedding):
    """Euclidean space embedding using graph structure."""
    
    def __init__(self, dim: int = 2):
        """Initialize embedding with dimension."""
        super().__init__(dim)
        
    def _initialize_embeddings(self, graph: nx.Graph) -> Dict[int, np.ndarray]:
        """Initialize node embeddings using spectral layout for better starting positions."""
        try:
            # Try spectral layout first
            pos = nx.spectral_layout(graph, dim=min(self.dim, len(graph.nodes()) - 1))
            if pos[list(graph.nodes())[0]].shape[0] < self.dim:
                # Pad with zeros if needed
                return {node: np.pad(pos[node], (0, self.dim - pos[node].shape[0])) for node in graph.nodes()}
            return {node: pos[node] for node in graph.nodes()}
        except:
            # Fall back to random initialization if spectral layout fails
            return {node: np.random.normal(0, 0.1, self.dim) for node in graph.nodes()}
    
    def _update_embeddings(self, graph: nx.Graph, iterations: int = 50):
        """Update embeddings using force-directed algorithm with graph structure."""
        learning_rate = 0.1
        repulsion = 0.1
        
        for _ in range(iterations):
            # Store old positions
            old_pos = {node: self.embeddings[node].copy() for node in graph.nodes()}
            
            # Calculate attractive forces between connected nodes
            for u, v in graph.edges():
                delta = self.embeddings[u] - self.embeddings[v]
                dist = np.linalg.norm(delta)
                if dist > 0:
                    force = delta * learning_rate
                    self.embeddings[u] -= force
                    self.embeddings[v] += force
            
            # Calculate repulsive forces between all nodes
            for u in graph.nodes():
                force = np.zeros(self.dim)
                for v in graph.nodes():
                    if u != v:
                        delta = self.embeddings[u] - self.embeddings[v]
                        dist = np.linalg.norm(delta)
                        if dist > 0:
                            force += (delta / dist) * repulsion
                self.embeddings[u] += force
            
            # Normalize to prevent exploding gradients
            for node in graph.nodes():
                norm = np.linalg.norm(self.embeddings[node])
                if norm > 0:
                    self.embeddings[node] /= norm
    
    def compute_distance(self, node1: int, node2: int) -> float:
        """Compute Euclidean distance between two nodes."""
        return float(np.linalg.norm(self.embeddings[node1] - self.embeddings[node2]))
    
    def get_visualization_coords(self, node: int) -> Tuple[float, float]:
        """Get 2D coordinates for visualization."""
        coords = self.embeddings[node]
        return float(coords[0]), float(coords[1])
