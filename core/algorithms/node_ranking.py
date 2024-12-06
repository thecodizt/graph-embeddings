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
        
        # Handle both directed and undirected graphs
        is_directed = self.graph.is_directed()
        
        for _ in range(max_iterations):
            prev_scores = scores.copy()
            
            # Update each node's score
            for node in self.graph.nodes():
                if is_directed:
                    in_neighbors = self.graph.predecessors(node)
                    out_degree = lambda x: self.graph.out_degree(x)
                else:
                    in_neighbors = self.graph.neighbors(node)
                    out_degree = lambda x: self.graph.degree(x)
                
                score_sum = sum(prev_scores[in_neighbor] / out_degree(in_neighbor)
                              for in_neighbor in in_neighbors)
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
        
        # Handle both directed and undirected graphs
        is_directed = self.graph.is_directed()
        
        # Precompute edge weights based on embeddings
        edge_weights = {}
        for edge in self.graph.edges():
            weight = self._get_edge_weight(*edge)
            edge_weights[edge] = weight
            if not is_directed:
                edge_weights[edge[::-1]] = weight  # Add reverse edge for undirected graphs
        
        for _ in range(max_iterations):
            prev_scores = scores.copy()
            
            # Update each node's score
            for node in self.graph.nodes():
                weighted_sum = 0
                
                # Get incoming neighbors based on graph type
                if is_directed:
                    in_neighbors = self.graph.predecessors(node)
                    get_out_neighbors = lambda x: self.graph.successors(x)
                else:
                    in_neighbors = self.graph.neighbors(node)
                    get_out_neighbors = lambda x: self.graph.neighbors(x)
                
                for in_neighbor in in_neighbors:
                    # Calculate weighted contribution
                    weight = edge_weights[(in_neighbor, node)]
                    out_weight_sum = sum(edge_weights[(in_neighbor, out_neighbor)]
                                       for out_neighbor in get_out_neighbors(in_neighbor))
                    weighted_sum += prev_scores[in_neighbor] * weight / out_weight_sum
                
                scores[node] = (1 - alpha) / n + alpha * weighted_sum
            
            # Check convergence
            error = sum(abs(scores[node] - prev_scores[node]) for node in scores)
            if error < tol:
                break
                
        return scores
