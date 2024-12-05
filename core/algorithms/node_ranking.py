"""Node ranking algorithms with and without embedding awareness."""
from typing import Dict
import networkx as nx
from .base import GraphAlgorithm, EmbeddingAwareAlgorithm

class PageRankTraditional(GraphAlgorithm):
    """Traditional PageRank implementation."""
    
    @GraphAlgorithm.measure_execution_time
    def run(self, alpha: float = 0.85, max_iterations: int = 100, tol: float = 1e-6) -> Dict[int, float]:
        """
        Compute PageRank for all nodes.
        
        Args:
            alpha: Damping parameter
            max_iterations: Maximum number of iterations
            tol: Convergence tolerance
            
        Returns:
            Dictionary mapping node IDs to PageRank scores
        """
        n = self.graph.number_of_nodes()
        if n == 0:
            return {}
            
        # Initialize scores
        scores = {node: 1/n for node in self.graph.nodes()}
        
        for _ in range(max_iterations):
            prev_scores = scores.copy()
            
            # Update each node's score
            for node in self.graph.nodes():
                score_sum = sum(prev_scores[in_neighbor] / self.graph.out_degree(in_neighbor)
                              for in_neighbor in self.graph.predecessors(node))
                scores[node] = (1 - alpha) / n + alpha * score_sum
            
            # Check convergence
            error = sum(abs(scores[node] - prev_scores[node]) for node in scores)
            if error < tol:
                break
                
        return scores

class PageRankEmbedding(EmbeddingAwareAlgorithm):
    """Embedding-aware PageRank implementation."""
    
    def _get_edge_weight(self, source: int, target: int) -> float:
        """Calculate edge weight based on embedding distance."""
        embedding_distance = self.embedder.compute_distance(source, target)
        # Convert distance to similarity (closer = higher weight)
        return 1 / (1 + embedding_distance)
    
    @GraphAlgorithm.measure_execution_time
    def run(self, alpha: float = 0.85, max_iterations: int = 100, tol: float = 1e-6) -> Dict[int, float]:
        """
        Compute embedding-aware PageRank for all nodes.
        
        Args:
            alpha: Damping parameter
            max_iterations: Maximum number of iterations
            tol: Convergence tolerance
            
        Returns:
            Dictionary mapping node IDs to PageRank scores
        """
        n = self.graph.number_of_nodes()
        if n == 0:
            return {}
            
        # Initialize scores
        scores = {node: 1/n for node in self.graph.nodes()}
        
        # Precompute edge weights based on embeddings
        edge_weights = {}
        for edge in self.graph.edges():
            edge_weights[edge] = self._get_edge_weight(*edge)
        
        for _ in range(max_iterations):
            prev_scores = scores.copy()
            
            # Update each node's score
            for node in self.graph.nodes():
                weighted_sum = 0
                for in_neighbor in self.graph.predecessors(node):
                    # Calculate weighted contribution
                    weight = edge_weights[(in_neighbor, node)]
                    out_weight_sum = sum(edge_weights[(in_neighbor, out_neighbor)]
                                       for out_neighbor in self.graph.successors(in_neighbor))
                    weighted_sum += prev_scores[in_neighbor] * weight / out_weight_sum
                
                scores[node] = (1 - alpha) / n + alpha * weighted_sum
            
            # Check convergence
            error = sum(abs(scores[node] - prev_scores[node]) for node in scores)
            if error < tol:
                break
                
        return scores
