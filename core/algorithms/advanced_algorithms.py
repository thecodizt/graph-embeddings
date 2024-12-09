"""Advanced graph algorithms leveraging embeddings for performance gains."""
import numpy as np
import networkx as nx
from typing import Dict, List, Set, Tuple
from sklearn.neighbors import NearestNeighbors
from .base import GraphAlgorithm, EmbeddingAwareAlgorithm

class AnomalyDetection(EmbeddingAwareAlgorithm):
    """Detect anomalous nodes using embedding space properties."""
    
    @GraphAlgorithm.measure_execution_time
    def run(self, contamination: float = 0.1) -> Dict[int, float]:
        """
        Detect anomalous nodes using Local Outlier Factor in embedding space.
        
        Args:
            contamination: Expected proportion of outliers in the data
            
        Returns:
            Dictionary mapping node IDs to anomaly scores (higher = more anomalous)
        """
        # Get embeddings for all nodes
        nodes = list(self.graph.nodes())
        embeddings = np.array([self.embedder.get_embedding(node) for node in nodes])
        
        # Compute LOF scores
        k = min(20, len(nodes) - 1)  # Number of neighbors
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)
        
        # Compute local reachability density
        lrd = np.zeros(len(nodes))
        for i in range(len(nodes)):
            k_distances = distances[indices[i]]
            reach_dist = np.maximum(k_distances, distances[i])
            lrd[i] = k / np.sum(reach_dist)
        
        # Compute LOF scores
        lof_scores = np.zeros(len(nodes))
        for i in range(len(nodes)):
            lof_scores[i] = np.mean([lrd[j] for j in indices[i]]) / lrd[i]
        
        return dict(zip(nodes, lof_scores))

class SubgraphMatching(EmbeddingAwareAlgorithm):
    """Fast approximate subgraph matching using embeddings."""
    
    @GraphAlgorithm.measure_execution_time
    def run(self, pattern: nx.Graph, max_candidates: int = 100) -> List[Set[int]]:
        """
        Find approximate matches of a pattern graph in the main graph.
        
        Args:
            pattern: The pattern graph to search for
            max_candidates: Maximum number of candidate matches to consider
            
        Returns:
            List of sets of nodes that approximately match the pattern
        """
        # Get embeddings for pattern and main graph
        pattern_nodes = list(pattern.nodes())
        pattern_embeddings = np.array([self.embedder.get_embedding(node) for node in pattern_nodes])
        
        main_nodes = list(self.graph.nodes())
        main_embeddings = np.array([self.embedder.get_embedding(node) for node in main_nodes])
        
        # Adjust max_candidates to not exceed graph size
        max_candidates = min(max_candidates, len(main_nodes))
        
        # Find candidate matches for each pattern node
        nbrs = NearestNeighbors(n_neighbors=max_candidates, algorithm='ball_tree').fit(main_embeddings)
        candidates = []
        
        for pattern_emb in pattern_embeddings:
            distances, indices = nbrs.kneighbors([pattern_emb])
            candidates.append({main_nodes[idx] for idx in indices[0]})
        
        # Find consistent matches
        matches = []
        def find_matches(pattern_idx: int, current_match: Set[int]):
            if pattern_idx == len(pattern_nodes):
                matches.append(current_match.copy())
                return
                
            for candidate in candidates[pattern_idx]:
                if candidate not in current_match:
                    # Verify edge consistency
                    valid = True
                    for i, matched in enumerate(current_match):
                        if pattern.has_edge(pattern_nodes[i], pattern_nodes[pattern_idx]):
                            if not self.graph.has_edge(matched, candidate):
                                valid = False
                                break
                    
                    if valid:
                        current_match.add(candidate)
                        find_matches(pattern_idx + 1, current_match)
                        current_match.remove(candidate)
        
        find_matches(0, set())
        return matches

class GraphEditDistance(EmbeddingAwareAlgorithm):
    """Fast approximate graph edit distance using embeddings."""
    
    @GraphAlgorithm.measure_execution_time
    def run(self, other_graph: nx.Graph) -> float:
        """
        Compute approximate graph edit distance using embedding space.
        
        Args:
            other_graph: Graph to compare with
            
        Returns:
            Approximate graph edit distance
        """
        # Get embeddings for both graphs
        nodes1 = list(self.graph.nodes())
        nodes2 = list(other_graph.nodes())
        
        emb1 = np.array([self.embedder.get_embedding(node) for node in nodes1])
        emb2 = np.array([self.embedder.get_embedding(node) for node in nodes2])
        
        # Compute cost matrix
        cost_matrix = np.zeros((len(nodes1), len(nodes2)))
        for i, e1 in enumerate(emb1):
            for j, e2 in enumerate(emb2):
                cost_matrix[i,j] = np.linalg.norm(e1 - e2)
        
        # Use Hungarian algorithm for optimal assignment
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Compute final distance including structural differences
        distance = cost_matrix[row_ind, col_ind].sum()
        
        # Add penalties for edge differences
        matched_edges = 0
        total_edges = self.graph.number_of_edges() + other_graph.number_of_edges()
        
        for i, j in zip(row_ind, col_ind):
            n1, n2 = nodes1[i], nodes2[j]
            for neighbor1 in self.graph.neighbors(n1):
                idx1 = nodes1.index(neighbor1)
                if idx1 in row_ind:
                    matched_n1 = nodes2[col_ind[list(row_ind).index(idx1)]]
                    if other_graph.has_edge(n2, matched_n1):
                        matched_edges += 2  # Count each matched edge twice since we added both graphs' edges
        
        edge_distance = (total_edges - matched_edges) / 2  # Divide by 2 since we counted each edge twice
        
        return distance + edge_distance * np.mean(cost_matrix)  # Weight edge differences by average node distance
