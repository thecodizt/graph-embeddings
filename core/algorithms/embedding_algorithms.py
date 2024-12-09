"""Algorithms that leverage graph embeddings for efficient computation."""
import numpy as np
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from .base import EmbeddingAwareAlgorithm


class ApproximatePersonalizedPageRank(EmbeddingAwareAlgorithm):
    """Fast approximate Personalized PageRank using embedding space."""
    
    @EmbeddingAwareAlgorithm.measure_execution_time
    def run(self, source_node: int, num_neighbors: int = 10) -> Dict[int, float]:
        """
        Compute approximate Personalized PageRank scores using embedding space similarity.
        This is much faster than traditional PPR computation for large graphs.
        
        Args:
            source_node: Starting node for PPR computation
            num_neighbors: Number of nearest neighbors to consider in embedding space
            
        Returns:
            Dictionary mapping node IDs to their PPR scores
        """
        # Get embeddings for all nodes
        nodes = list(self.graph.nodes())
        embeddings = np.array([self.embedder.get_embedding(node) for node in nodes])
        source_embedding = self.embedder.get_embedding(source_node)
        
        # Find nearest neighbors in embedding space
        nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree').fit(embeddings)
        distances, indices = nbrs.kneighbors([source_embedding])
        
        # Compute approximate PPR scores based on embedding similarity
        scores = {}
        max_distance = np.max(distances[0]) + 1e-6  # Avoid division by zero
        for node, distance in zip(np.array(nodes)[indices[0]], distances[0]):
            scores[node] = 1.0 - (distance / max_distance)
            
        return scores


class FastNodeClassification(EmbeddingAwareAlgorithm):
    """Efficient node classification using embedding space."""
    
    @EmbeddingAwareAlgorithm.measure_execution_time
    def run(self, labeled_nodes: Dict[int, str], k: int = 5) -> Dict[int, str]:
        """
        Classify nodes based on their k-nearest labeled neighbors in embedding space.
        Much faster than traditional label propagation algorithms.
        
        Args:
            labeled_nodes: Dictionary mapping node IDs to their labels
            k: Number of nearest neighbors to consider
            
        Returns:
            Dictionary mapping all node IDs to predicted labels
        """
        # Get embeddings
        all_nodes = list(self.graph.nodes())
        all_embeddings = np.array([self.embedder.get_embedding(node) for node in all_nodes])
        
        # Create labeled embeddings dataset
        labeled_indices = [i for i, node in enumerate(all_nodes) if node in labeled_nodes]
        labeled_embeddings = all_embeddings[labeled_indices]
        labels = [labeled_nodes[all_nodes[i]] for i in labeled_indices]
        
        # Find k-nearest neighbors for all unlabeled nodes
        nbrs = NearestNeighbors(n_neighbors=min(k, len(labeled_indices)), 
                               algorithm='ball_tree').fit(labeled_embeddings)
        
        predictions = {}
        for node, embedding in zip(all_nodes, all_embeddings):
            if node in labeled_nodes:
                predictions[node] = labeled_nodes[node]
            else:
                distances, indices = nbrs.kneighbors([embedding])
                # Majority voting weighted by inverse distance
                weights = 1.0 / (distances[0] + 1e-6)
                neighbor_labels = [labels[i] for i in indices[0]]
                unique_labels = set(neighbor_labels)
                label_scores = {label: sum(weights[i] for i, l in enumerate(neighbor_labels) if l == label)
                              for label in unique_labels}
                predictions[node] = max(label_scores.items(), key=lambda x: x[1])[0]
                
        return predictions


class EfficientCommunityDetection(EmbeddingAwareAlgorithm):
    """Fast community detection using embedding space clustering."""
    
    @EmbeddingAwareAlgorithm.measure_execution_time
    def run(self, min_community_size: int = 5, eps: float = 0.5) -> List[Set[int]]:
        """
        Detect communities by clustering nodes in embedding space using DBSCAN.
        Much faster than traditional community detection algorithms for large graphs.
        
        Args:
            min_community_size: Minimum number of nodes in a community
            eps: Maximum distance between nodes in the same neighborhood
            
        Returns:
            List of sets, where each set contains node IDs in the same community
        """
        # Get embeddings for all nodes
        nodes = list(self.graph.nodes())
        embeddings = np.array([self.embedder.get_embedding(node) for node in nodes])
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_community_size).fit(embeddings)
        labels = clustering.labels_
        
        # Group nodes by cluster
        communities = {}
        for node, label in zip(nodes, labels):
            if label != -1:  # Ignore noise points
                if label not in communities:
                    communities[label] = set()
                communities[label].add(node)
                
        return list(communities.values())


class LinkPrediction(EmbeddingAwareAlgorithm):
    """Fast link prediction using embedding similarity."""
    
    @EmbeddingAwareAlgorithm.measure_execution_time
    def run(self, source_node: int, num_candidates: int = 10) -> List[Tuple[int, float]]:
        """
        Predict most likely new edges for a given node using embedding similarity.
        Much faster than traditional link prediction methods.
        
        Args:
            source_node: Node to predict new links for
            num_candidates: Number of candidates to return
            
        Returns:
            List of tuples (node_id, score) for most likely new edges
        """
        # Get embeddings
        nodes = list(self.graph.nodes())
        embeddings = np.array([self.embedder.get_embedding(node) for node in nodes])
        source_embedding = self.embedder.get_embedding(source_node)
        
        # Find nearest neighbors in embedding space
        nbrs = NearestNeighbors(n_neighbors=num_candidates + 1, algorithm='ball_tree').fit(embeddings)
        distances, indices = nbrs.kneighbors([source_embedding])
        
        # Filter out existing edges and self-loops
        existing_edges = set(self.graph.neighbors(source_node))
        candidates = []
        for node_idx, distance in zip(indices[0], distances[0]):
            node = nodes[node_idx]
            if node != source_node and node not in existing_edges:
                similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity score
                candidates.append((node, similarity))
                
        # Sort by similarity score
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:num_candidates]
