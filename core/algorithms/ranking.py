"""Node ranking algorithms with and without embedding awareness."""
from typing import Dict, List, Tuple
import networkx as nx
import numpy as np
from .base import GraphAlgorithm, EmbeddingAwareAlgorithm

class PageRankTraditional(GraphAlgorithm):
    """Traditional PageRank algorithm implementation."""
    
    def run(self, damping_factor: float = 0.85, max_iterations: int = 100, tolerance: float = 1e-6) -> Dict[int, float]:
        """Run PageRank algorithm.
        
        Args:
            damping_factor: Probability of following an outgoing edge (default: 0.85)
            max_iterations: Maximum number of iterations (default: 100)
            tolerance: Convergence tolerance (default: 1e-6)
            
        Returns:
            Dict mapping node IDs to PageRank scores
        """
        n = len(self.graph)
        scores = {node: 1.0 / n for node in self.graph.nodes()}
        
        for _ in range(max_iterations):
            prev_scores = scores.copy()
            
            # Compute new scores
            for node in self.graph.nodes():
                in_neighbors = list(self.graph.predecessors(node))
                score = (1 - damping_factor) / n
                
                # Sum contributions from incoming edges
                for neighbor in in_neighbors:
                    out_degree = self.graph.out_degree(neighbor)
                    if out_degree > 0:  # Avoid division by zero
                        score += damping_factor * prev_scores[neighbor] / out_degree
                
                scores[node] = score
            
            # Check convergence
            total_diff = sum(abs(scores[node] - prev_scores[node]) for node in self.graph.nodes())
            if total_diff < tolerance:
                break
        
        return scores


class PageRankEmbedding(EmbeddingAwareAlgorithm):
    """Embedding-aware PageRank algorithm implementation."""
    
    def run(self, damping_factor: float = 0.85, max_iterations: int = 100, tolerance: float = 1e-6) -> Dict[int, float]:
        """Run PageRank algorithm using embeddings for optimization.
        
        Args:
            damping_factor: Probability of following an outgoing edge (default: 0.85)
            max_iterations: Maximum number of iterations (default: 100)
            tolerance: Convergence tolerance (default: 1e-6)
            
        Returns:
            Dict mapping node IDs to PageRank scores
        """
        n = len(self.graph)
        scores = {node: 1.0 / n for node in self.graph.nodes()}
        
        # Get embeddings for all nodes
        embeddings = {node: self.embedder.get_node_embedding(node) for node in self.graph.nodes()}
        
        # Sort nodes by embedding similarity to optimize iteration order
        nodes = list(self.graph.nodes())
        nodes.sort(key=lambda n: np.linalg.norm(embeddings[n]))
        
        for _ in range(max_iterations):
            prev_scores = scores.copy()
            
            # Compute new scores
            for node in nodes:
                in_neighbors = list(self.graph.predecessors(node))
                score = (1 - damping_factor) / n
                
                # Filter neighbors based on embedding similarity
                similar_neighbors = [
                    neighbor for neighbor in in_neighbors
                    if np.linalg.norm(embeddings[node] - embeddings[neighbor]) < 2.0
                ]
                
                # Sum contributions from similar incoming edges
                for neighbor in similar_neighbors:
                    out_degree = self.graph.out_degree(neighbor)
                    if out_degree > 0:  # Avoid division by zero
                        score += damping_factor * prev_scores[neighbor] / out_degree
                
                scores[node] = score
            
            # Check convergence
            total_diff = sum(abs(scores[node] - prev_scores[node]) for node in self.graph.nodes())
            if total_diff < tolerance:
                break
        
        return scores


class HITSTraditional(GraphAlgorithm):
    """Traditional HITS algorithm implementation."""
    
    def run(self, max_iterations: int = 100, tolerance: float = 1e-6) -> Tuple[Dict[int, float], Dict[int, float]]:
        """Run HITS algorithm.
        
        Args:
            max_iterations: Maximum number of iterations (default: 100)
            tolerance: Convergence tolerance (default: 1e-6)
            
        Returns:
            Tuple of (hub_scores, authority_scores) dicts mapping node IDs to scores
        """
        n = len(self.graph)
        hub_scores = {node: 1.0 / n for node in self.graph.nodes()}
        auth_scores = {node: 1.0 / n for node in self.graph.nodes()}
        
        for _ in range(max_iterations):
            prev_hub = hub_scores.copy()
            prev_auth = auth_scores.copy()
            
            # Update authority scores
            for node in self.graph.nodes():
                in_neighbors = list(self.graph.predecessors(node))
                auth_scores[node] = sum(prev_hub[neighbor] for neighbor in in_neighbors)
            
            # Normalize authority scores
            auth_sum = sum(score * score for score in auth_scores.values()) ** 0.5
            if auth_sum > 0:
                for node in auth_scores:
                    auth_scores[node] /= auth_sum
            
            # Update hub scores
            for node in self.graph.nodes():
                out_neighbors = list(self.graph.successors(node))
                hub_scores[node] = sum(auth_scores[neighbor] for neighbor in out_neighbors)
            
            # Normalize hub scores
            hub_sum = sum(score * score for score in hub_scores.values()) ** 0.5
            if hub_sum > 0:
                for node in hub_scores:
                    hub_scores[node] /= hub_sum
            
            # Check convergence
            hub_diff = sum(abs(hub_scores[node] - prev_hub[node]) for node in self.graph.nodes())
            auth_diff = sum(abs(auth_scores[node] - prev_auth[node]) for node in self.graph.nodes())
            if hub_diff < tolerance and auth_diff < tolerance:
                break
        
        return hub_scores, auth_scores


class HITSEmbedding(EmbeddingAwareAlgorithm):
    """Embedding-aware HITS algorithm implementation."""
    
    def run(self, max_iterations: int = 100, tolerance: float = 1e-6) -> Tuple[Dict[int, float], Dict[int, float]]:
        """Run HITS algorithm using embeddings for optimization.
        
        Args:
            max_iterations: Maximum number of iterations (default: 100)
            tolerance: Convergence tolerance (default: 1e-6)
            
        Returns:
            Tuple of (hub_scores, authority_scores) dicts mapping node IDs to scores
        """
        n = len(self.graph)
        hub_scores = {node: 1.0 / n for node in self.graph.nodes()}
        auth_scores = {node: 1.0 / n for node in self.graph.nodes()}
        
        # Get embeddings for all nodes
        embeddings = {node: self.embedder.get_node_embedding(node) for node in self.graph.nodes()}
        
        # Sort nodes by embedding similarity to optimize iteration order
        nodes = list(self.graph.nodes())
        nodes.sort(key=lambda n: np.linalg.norm(embeddings[n]))
        
        for _ in range(max_iterations):
            prev_hub = hub_scores.copy()
            prev_auth = auth_scores.copy()
            
            # Update authority scores
            for node in nodes:
                in_neighbors = list(self.graph.predecessors(node))
                # Filter neighbors based on embedding similarity
                similar_neighbors = [
                    neighbor for neighbor in in_neighbors
                    if np.linalg.norm(embeddings[node] - embeddings[neighbor]) < 2.0
                ]
                auth_scores[node] = sum(prev_hub[neighbor] for neighbor in similar_neighbors)
            
            # Normalize authority scores
            auth_sum = sum(score * score for score in auth_scores.values()) ** 0.5
            if auth_sum > 0:
                for node in auth_scores:
                    auth_scores[node] /= auth_sum
            
            # Update hub scores
            for node in nodes:
                out_neighbors = list(self.graph.successors(node))
                # Filter neighbors based on embedding similarity
                similar_neighbors = [
                    neighbor for neighbor in out_neighbors
                    if np.linalg.norm(embeddings[node] - embeddings[neighbor]) < 2.0
                ]
                hub_scores[node] = sum(auth_scores[neighbor] for neighbor in similar_neighbors)
            
            # Normalize hub scores
            hub_sum = sum(score * score for score in hub_scores.values()) ** 0.5
            if hub_sum > 0:
                for node in hub_scores:
                    hub_scores[node] /= hub_sum
            
            # Check convergence
            hub_diff = sum(abs(hub_scores[node] - prev_hub[node]) for node in self.graph.nodes())
            auth_diff = sum(abs(auth_scores[node] - prev_auth[node]) for node in self.graph.nodes())
            if hub_diff < tolerance and auth_diff < tolerance:
                break
        
        return hub_scores, auth_scores
