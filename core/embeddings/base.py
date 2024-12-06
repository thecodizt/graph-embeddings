"""Base class for graph embeddings."""
from abc import ABC, abstractmethod
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any

class BaseEmbedding(ABC):
    """Base class for all embedding types."""
    
    def __init__(self, dim: int = 2):
        """Initialize embedding parameters."""
        self.dim = dim
        self.embeddings = {}
        
    def train(self, graph: nx.Graph, iterations: int = 50) -> None:
        """Train the embeddings on a graph."""
        # Initialize embeddings for all nodes
        init_embeddings = self._initialize_embeddings(graph)
        
        # Ensure all nodes have embeddings
        for node in graph.nodes():
            if node not in init_embeddings:
                # Initialize missing nodes with random values
                init_embeddings[node] = np.random.normal(0, 0.1, self.dim)
        
        self.embeddings = init_embeddings
        
        # Update embeddings
        self._update_embeddings(graph, iterations)
        
        # Final check to ensure all nodes have embeddings
        for node in graph.nodes():
            if node not in self.embeddings:
                raise ValueError(f"Node {node} missing from embeddings after training")
    
    @abstractmethod
    def _initialize_embeddings(self, graph: nx.Graph) -> Dict[int, np.ndarray]:
        """Initialize embeddings for all nodes in the graph."""
        pass
    
    @abstractmethod
    def _update_embeddings(self, graph: nx.Graph, iterations: int) -> None:
        """Update embeddings using graph structure."""
        pass
    
    def get_embedding(self, node: int) -> np.ndarray:
        """Get the embedding vector for a node."""
        return self.embeddings[node]
    
    @abstractmethod
    def compute_distance(self, node1: int, node2: int) -> float:
        """Compute distance between two nodes in the embedding space."""
        pass
    
    @abstractmethod
    def get_visualization_coords(self, node: int) -> Tuple[float, float]:
        """Get 2D coordinates for visualization."""
        pass
