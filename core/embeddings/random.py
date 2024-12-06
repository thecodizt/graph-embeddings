"""Random embedder for testing."""
import numpy as np
from typing import Dict, List, Tuple
import networkx as nx
from .base import BaseEmbedding

class RandomEmbedder(BaseEmbedding):
    """Random embedder that assigns random vectors to nodes."""
    
    def __init__(self, graph: nx.Graph, dim: int = 64, seed: int = None):
        """
        Initialize random embedder.
        
        Args:
            graph: NetworkX graph
            dim: Embedding dimension
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.graph = graph
        self.dim = dim
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize embeddings
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize random embeddings for each node."""
        self.embeddings = {
            node: np.random.normal(0, 1, self.dim)
            for node in self.graph.nodes()
        }
        
        # Normalize embeddings to unit length
        for node in self.embeddings:
            self.embeddings[node] = self.embeddings[node] / np.linalg.norm(self.embeddings[node])
    
    def _update_embeddings(self, num_iterations: int = 1):
        """No updates needed for random embeddings."""
        pass
    
    def get_embedding(self, node: int) -> np.ndarray:
        """Get embedding for a node."""
        return self.embeddings[node]
    
    def get_node_embedding(self, node: int) -> np.ndarray:
        """Alias for get_embedding to maintain API compatibility."""
        return self.get_embedding(node)
    
    def compute_distance(self, node1: int, node2: int) -> float:
        """Compute Euclidean distance between node embeddings."""
        return np.linalg.norm(self.embeddings[node1] - self.embeddings[node2])
    
    def compute_distance_from_embeddings(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute Euclidean distance between embeddings."""
        return np.linalg.norm(emb1 - emb2)
    
    def get_visualization_coords(self) -> Dict[int, Tuple[float, float]]:
        """Get 2D coordinates for visualization."""
        if self.dim < 2:
            raise ValueError("Embedding dimension must be at least 2 for visualization")
        
        return {
            node: (self.embeddings[node][0], self.embeddings[node][1])
            for node in self.graph.nodes()
        }
    
    def train(self, graph: nx.Graph = None):
        """No training needed for random embeddings."""
        if graph is not None:
            self.graph = graph
            self._initialize_embeddings()
