"""Clustering algorithms with and without embedding awareness."""
from typing import Dict, List, Set, Tuple
import networkx as nx
from collections import defaultdict
import numpy as np
from .base import GraphAlgorithm, EmbeddingAwareAlgorithm

class KMeansTraditional(GraphAlgorithm):
    """Traditional K-means clustering using graph structure."""
    
    def _get_node_features(self, node: int) -> np.ndarray:
        """Extract features from graph structure."""
        # Use node degree and clustering coefficient as features
        return np.array([
            self.graph.degree[node],
            nx.clustering(self.graph, node)
        ])
    
    @GraphAlgorithm.measure_execution_time
    def run(self, k: int, max_iterations: int = 100) -> Dict[int, int]:
        """
        Run k-means clustering on nodes using graph features.
        
        Args:
            k: Number of clusters
            max_iterations: Maximum number of iterations
            
        Returns:
            Dictionary mapping node IDs to cluster IDs
        """
        nodes = list(self.graph.nodes())
        n = len(nodes)
        
        if n < k:
            return {node: i for i, node in enumerate(nodes)}
        
        # Extract features for each node
        features = {node: self._get_node_features(node) for node in nodes}
        
        # Initialize centroids randomly
        centroid_nodes = np.random.choice(nodes, k, replace=False)
        centroids = [features[node] for node in centroid_nodes]
        
        # Initialize cluster assignments
        clusters = {node: -1 for node in nodes}
        
        for _ in range(max_iterations):
            old_clusters = clusters.copy()
            
            # Assign nodes to nearest centroid
            for node in nodes:
                distances = [np.linalg.norm(features[node] - centroid) 
                           for centroid in centroids]
                clusters[node] = np.argmin(distances)
            
            # Check for convergence
            if old_clusters == clusters:
                break
            
            # Update centroids
            for i in range(k):
                cluster_nodes = [node for node in nodes if clusters[node] == i]
                if cluster_nodes:
                    centroids[i] = np.mean([features[node] for node in cluster_nodes], axis=0)
        
        return clusters

class KMeansEmbedding(EmbeddingAwareAlgorithm):
    """K-means clustering using embeddings for optimization."""
    
    def _get_node_features(self, node: int) -> np.ndarray:
        """Extract same features as traditional version."""
        return np.array([
            self.graph.degree[node],
            nx.clustering(self.graph, node)
        ])
    
    @GraphAlgorithm.measure_execution_time
    def run(self, k: int, max_iterations: int = 100) -> Dict[int, int]:
        """
        Run k-means clustering using embeddings to guide initial centroids.
        
        Args:
            k: Number of clusters
            max_iterations: Maximum number of iterations
            
        Returns:
            Dictionary mapping node IDs to cluster IDs
        """
        nodes = list(self.graph.nodes())
        n = len(nodes)
        
        if n < k:
            return {node: i for i, node in enumerate(nodes)}
            
        # Extract features for each node (same as traditional)
        features = {node: self._get_node_features(node) for node in nodes}
        
        # Use embeddings to choose better initial centroids
        # This speeds up convergence while maintaining same final results
        embeddings = {node: self.embedder.get_embedding(node) for node in nodes}
        
        # Use k-means++ initialization on embeddings to get good starting centroids
        centroid_nodes = []
        first_centroid = np.random.choice(nodes)
        centroid_nodes.append(first_centroid)
        
        while len(centroid_nodes) < k:
            distances = []
            for node in nodes:
                if node not in centroid_nodes:
                    # Use embedding distances for faster initial centroid selection
                    min_dist = min(self.embedder.compute_distance(node, c) 
                                 for c in centroid_nodes)
                    distances.append((node, min_dist))
            
            # Choose next centroid weighted by distance squared
            weights = np.array([d[1]**2 for d in distances])
            weights = weights / np.sum(weights)
            next_centroid = np.random.choice([d[0] for d in distances], p=weights)
            centroid_nodes.append(next_centroid)
        
        centroids = [features[node] for node in centroid_nodes]
        
        # Run standard k-means with the optimized initial centroids
        clusters = {node: -1 for node in nodes}
        
        for _ in range(max_iterations):
            old_clusters = clusters.copy()
            
            # Assign nodes to nearest centroid using original features
            for node in nodes:
                distances = [np.linalg.norm(features[node] - centroid) 
                           for centroid in centroids]
                clusters[node] = np.argmin(distances)
            
            if old_clusters == clusters:
                break
            
            # Update centroids using original features
            for i in range(k):
                cluster_nodes = [node for node in nodes if clusters[node] == i]
                if cluster_nodes:
                    centroids[i] = np.mean([features[node] for node in cluster_nodes], axis=0)
        
        return clusters

class SpectralClusteringTraditional(GraphAlgorithm):
    """Traditional Spectral Clustering implementation."""
    
    def _get_laplacian_eigenvectors(self, k: int) -> Tuple[np.ndarray, List[int]]:
        """Compute the first k eigenvectors of the normalized Laplacian."""
        # Create Laplacian matrix
        L = nx.normalized_laplacian_matrix(self.graph).todense()
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        
        # Sort by eigenvalues and get indices of nodes
        idx = np.argsort(eigenvalues)
        nodes = list(self.graph.nodes())
        
        return eigenvectors[:, idx[:k]], nodes
    
    @GraphAlgorithm.measure_execution_time
    def run(self, k: int) -> Dict[int, int]:
        """
        Run spectral clustering to partition the graph.
        
        Args:
            k: Number of clusters
            
        Returns:
            Dictionary mapping node IDs to cluster IDs
        """
        nodes = list(self.graph.nodes())
        n = len(nodes)
        
        # Handle small graphs
        if n <= k:
            return {node: i for i, node in enumerate(nodes)}
        
        # Get eigenvectors
        eigenvectors, nodes = self._get_laplacian_eigenvectors(k)
        
        # Normalize rows to unit length
        row_norms = np.sqrt(np.sum(eigenvectors ** 2, axis=1))
        row_norms[row_norms == 0] = 1  # Avoid division by zero
        eigenvectors = eigenvectors / row_norms[:, np.newaxis]
        
        # Run k-means on the rows of eigenvectors
        n_samples = len(nodes)
        centroid_indices = np.random.choice(n_samples, k, replace=False)
        centroids = eigenvectors[centroid_indices]
        
        # Run k-means iterations
        clusters = {node: -1 for node in nodes}
        for _ in range(100):  # Max iterations
            old_clusters = clusters.copy()
            
            # Assign nodes to nearest centroid
            for i, node in enumerate(nodes):
                distances = [np.linalg.norm(eigenvectors[i] - centroid) 
                           for centroid in centroids]
                clusters[node] = np.argmin(distances)
            
            # Check convergence
            if old_clusters == clusters:
                break
            
            # Update centroids
            for i in range(k):
                cluster_indices = [j for j, node in enumerate(nodes) if clusters[node] == i]
                if cluster_indices:
                    centroids[i] = np.mean(eigenvectors[cluster_indices], axis=0)
        
        return clusters

class SpectralClusteringEmbedding(EmbeddingAwareAlgorithm):
    """Spectral Clustering using embeddings for optimization."""
    
    def _get_laplacian_eigenvectors(self, k: int) -> Tuple[np.ndarray, List[int]]:
        """Compute the first k eigenvectors of the normalized Laplacian."""
        # Create Laplacian matrix
        L = nx.normalized_laplacian_matrix(self.graph).todense()
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        
        # Sort by eigenvalues and get indices of nodes
        idx = np.argsort(eigenvalues)
        nodes = list(self.graph.nodes())
        
        return eigenvectors[:, idx[:k]], nodes
    
    @GraphAlgorithm.measure_execution_time
    def run(self, k: int) -> Dict[int, int]:
        """
        Run spectral clustering using embeddings for optimization.
        
        Args:
            k: Number of clusters
            
        Returns:
            Dictionary mapping node IDs to cluster IDs
        """
        nodes = list(self.graph.nodes())
        n = len(nodes)
        
        # Handle small graphs
        if n <= k:
            return {node: i for i, node in enumerate(nodes)}
        
        # Get embeddings for all nodes
        embeddings = {node: self.embedder.get_embedding(node) for node in nodes}
        
        # Get eigenvectors (same as traditional)
        eigenvectors, nodes = self._get_laplacian_eigenvectors(k)
        
        # Normalize rows to unit length
        row_norms = np.sqrt(np.sum(eigenvectors ** 2, axis=1))
        row_norms[row_norms == 0] = 1  # Avoid division by zero
        eigenvectors = eigenvectors / row_norms[:, np.newaxis]
        
        # Use embeddings to choose better initial centroids
        n_samples = len(nodes)
        centroid_indices = []
        first_centroid = np.random.randint(n_samples)
        centroid_indices.append(first_centroid)
        
        while len(centroid_indices) < k:
            distances = []
            for i in range(n_samples):
                if i not in centroid_indices:
                    # Use embedding distances for faster initial centroid selection
                    min_dist = min(np.linalg.norm(embeddings[nodes[i]] - embeddings[nodes[c]])
                                 for c in centroid_indices)
                    distances.append((i, min_dist))
            
            # Choose next centroid weighted by distance squared
            weights = np.array([d[1]**2 for d in distances])
            weights = weights / np.sum(weights)
            next_centroid = np.random.choice([d[0] for d in distances], p=weights)
            centroid_indices.append(next_centroid)
        
        centroids = eigenvectors[centroid_indices]
        
        # Run k-means iterations
        clusters = {node: -1 for node in nodes}
        for _ in range(100):  # Max iterations
            old_clusters = clusters.copy()
            
            # Assign nodes to nearest centroid
            for i, node in enumerate(nodes):
                distances = [np.linalg.norm(eigenvectors[i] - centroid)
                           for centroid in centroids]
                clusters[node] = np.argmin(distances)
            
            # Check convergence
            if old_clusters == clusters:
                break
            
            # Update centroids
            for i in range(k):
                cluster_indices = [j for j, node in enumerate(nodes) if clusters[node] == i]
                if cluster_indices:
                    centroids[i] = np.mean(eigenvectors[cluster_indices], axis=0)
        
        return clusters
