"""Community detection algorithms with and without embedding awareness."""
from typing import Dict, List, Set
import networkx as nx
from collections import defaultdict
from .base import GraphAlgorithm, EmbeddingAwareAlgorithm

class LabelPropagationTraditional(GraphAlgorithm):
    """Traditional Label Propagation algorithm for community detection."""
    
    @GraphAlgorithm.measure_execution_time
    def run(self, max_iterations: int = 10) -> Dict[int, int]:
        """
        Run label propagation algorithm.
        
        Args:
            max_iterations: Maximum number of iterations
            
        Returns:
            Dictionary mapping node IDs to community IDs
        """
        # Initialize each node with unique label
        labels = {node: i for i, node in enumerate(self.graph.nodes())}
        
        for _ in range(max_iterations):
            # Store old labels for convergence check
            old_labels = labels.copy()
            
            # Update each node's label
            for node in self.graph.nodes():
                neighbor_labels = defaultdict(int)
                for neighbor in self.graph.neighbors(node):
                    neighbor_labels[labels[neighbor]] += 1
                
                # Choose most common label among neighbors
                if neighbor_labels:
                    labels[node] = max(neighbor_labels.items(), key=lambda x: x[1])[0]
            
            # Check for convergence
            if old_labels == labels:
                break
                
        return labels

class LabelPropagationEmbedding(EmbeddingAwareAlgorithm):
    """Embedding-aware Label Propagation algorithm for community detection."""
    
    def _get_neighbor_weight(self, node: int, neighbor: int) -> float:
        """Calculate weight between node and neighbor based on embedding distance."""
        edge_weight = self.graph[node][neighbor].get('weight', 1)
        embedding_distance = self.embedder.compute_distance(node, neighbor)
        # Convert distance to similarity (closer = higher weight)
        embedding_similarity = 1 / (1 + embedding_distance)
        return edge_weight * embedding_similarity
    
    @GraphAlgorithm.measure_execution_time
    def run(self, max_iterations: int = 10) -> Dict[int, int]:
        """
        Run embedding-aware label propagation algorithm.
        
        Args:
            max_iterations: Maximum number of iterations
            
        Returns:
            Dictionary mapping node IDs to community IDs
        """
        # Initialize each node with unique label
        labels = {node: i for i, node in enumerate(self.graph.nodes())}
        
        for _ in range(max_iterations):
            # Store old labels for convergence check
            old_labels = labels.copy()
            
            # Update each node's label
            for node in self.graph.nodes():
                neighbor_labels = defaultdict(float)
                for neighbor in self.graph.neighbors(node):
                    weight = self._get_neighbor_weight(node, neighbor)
                    neighbor_labels[labels[neighbor]] += weight
                
                # Choose label with highest weighted sum
                if neighbor_labels:
                    labels[node] = max(neighbor_labels.items(), key=lambda x: x[1])[0]
            
            # Check for convergence
            if old_labels == labels:
                break
                
        return labels
