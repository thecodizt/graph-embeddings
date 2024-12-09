"""Traditional implementations of graph algorithms for benchmarking comparison."""
import numpy as np
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, Counter
from .base import GraphAlgorithm


class TraditionalPersonalizedPageRank(GraphAlgorithm):
    """Traditional implementation of Personalized PageRank."""
    
    @GraphAlgorithm.measure_execution_time
    def run(self, source_node: int, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> Dict[int, float]:
        """
        Compute Personalized PageRank scores using power iteration method.
        
        Args:
            source_node: Starting node for PPR computation
            alpha: Damping parameter
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            
        Returns:
            Dictionary mapping node IDs to their PPR scores
        """
        nodes = list(self.graph.nodes())
        n = len(nodes)
        node_idx = {node: idx for idx, node in enumerate(nodes)}
        
        # Initialize probability vector
        x = np.zeros(n)
        x[node_idx[source_node]] = 1.0
        
        # Power iteration
        for _ in range(max_iter):
            prev_x = x.copy()
            
            # Random walk
            new_x = np.zeros(n)
            for i, node in enumerate(nodes):
                neighbors = list(self.graph.neighbors(node))
                if neighbors:
                    weight = x[i] / len(neighbors)
                    for neighbor in neighbors:
                        new_x[node_idx[neighbor]] += weight
            
            # Teleport
            x = alpha * new_x + (1 - alpha) * (x == x[node_idx[source_node]])
            
            # Check convergence
            if np.sum(np.abs(x - prev_x)) < tol:
                break
                
        return {node: score for node, score in zip(nodes, x)}


class TraditionalNodeClassification(GraphAlgorithm):
    """Traditional implementation of Label Propagation for node classification."""
    
    @GraphAlgorithm.measure_execution_time
    def run(self, labeled_nodes: Dict[int, str], max_iter: int = 30) -> Dict[int, str]:
        """
        Classify nodes using Label Propagation Algorithm.
        
        Args:
            labeled_nodes: Dictionary mapping node IDs to their labels
            max_iter: Maximum number of iterations
            
        Returns:
            Dictionary mapping all node IDs to predicted labels
        """
        current_labels = labeled_nodes.copy()
        nodes = list(self.graph.nodes())
        
        # Initialize unlabeled nodes randomly
        all_labels = list(set(labeled_nodes.values()))
        for node in nodes:
            if node not in current_labels:
                current_labels[node] = np.random.choice(all_labels)
        
        # Iterate until convergence or max iterations
        for _ in range(max_iter):
            changes = 0
            # Shuffle nodes to prevent cyclic behavior
            np.random.shuffle(nodes)
            
            for node in nodes:
                if node in labeled_nodes:
                    continue
                    
                # Get neighbor labels
                neighbor_labels = [current_labels[neighbor] 
                                 for neighbor in self.graph.neighbors(node)]
                if not neighbor_labels:
                    continue
                
                # Update to most common neighbor label
                label_counts = Counter(neighbor_labels)
                max_count = max(label_counts.values())
                most_common = [label for label, count in label_counts.items() 
                             if count == max_count]
                new_label = np.random.choice(most_common)
                
                if new_label != current_labels[node]:
                    changes += 1
                    current_labels[node] = new_label
            
            if changes == 0:
                break
                
        return current_labels


class TraditionalCommunityDetection(GraphAlgorithm):
    """Traditional implementation of Louvain method for community detection."""
    
    def _modularity(self, communities: Dict[int, int]) -> float:
        """Calculate modularity of the partition."""
        m = self.graph.number_of_edges()
        if m == 0:
            return 0.0
            
        q = 0.0
        for node in self.graph.nodes():
            ki = self.graph.degree(node)
            community = communities[node]
            for neighbor in self.graph.neighbors(node):
                if communities[neighbor] == community:
                    kj = self.graph.degree(neighbor)
                    q += 1 - (ki * kj) / (2 * m)
        return q / (2 * m)
    
    @GraphAlgorithm.measure_execution_time
    def run(self, min_community_size: int = 5) -> List[Set[int]]:
        """
        Detect communities using Louvain method.
        
        Args:
            min_community_size: Minimum number of nodes in a community
            
        Returns:
            List of sets, where each set contains node IDs in the same community
        """
        # Initialize each node in its own community
        communities = {node: i for i, node in enumerate(self.graph.nodes())}
        
        while True:
            improvement = False
            # Consider moving each node
            for node in self.graph.nodes():
                current_community = communities[node]
                best_modularity = self._modularity(communities)
                best_community = current_community
                
                # Try moving to each neighbor's community
                neighbor_communities = set(communities[neighbor] 
                                        for neighbor in self.graph.neighbors(node))
                
                for new_community in neighbor_communities:
                    if new_community != current_community:
                        communities[node] = new_community
                        modularity = self._modularity(communities)
                        if modularity > best_modularity:
                            best_modularity = modularity
                            best_community = new_community
                        communities[node] = current_community
                
                if best_community != current_community:
                    communities[node] = best_community
                    improvement = True
            
            if not improvement:
                break
        
        # Group nodes by community
        community_sets = defaultdict(set)
        for node, community in communities.items():
            community_sets[community].add(node)
        
        # Filter small communities
        return [nodes for nodes in community_sets.values() 
                if len(nodes) >= min_community_size]


class TraditionalLinkPrediction(GraphAlgorithm):
    """Traditional implementation of link prediction using common neighbors and Adamic-Adar."""
    
    def _common_neighbors_score(self, u: int, v: int) -> float:
        """Compute common neighbors score."""
        u_neighbors = set(self.graph.neighbors(u))
        v_neighbors = set(self.graph.neighbors(v))
        return len(u_neighbors & v_neighbors)
    
    def _adamic_adar_score(self, u: int, v: int) -> float:
        """Compute Adamic-Adar score."""
        u_neighbors = set(self.graph.neighbors(u))
        v_neighbors = set(self.graph.neighbors(v))
        score = 0.0
        for w in u_neighbors & v_neighbors:
            score += 1.0 / np.log(self.graph.degree(w))
        return score
    
    @GraphAlgorithm.measure_execution_time
    def run(self, source_node: int, num_candidates: int = 10) -> List[Tuple[int, float]]:
        """
        Predict most likely new edges for a given node.
        
        Args:
            source_node: Node to predict new links for
            num_candidates: Number of candidates to return
            
        Returns:
            List of tuples (node_id, score) for most likely new edges
        """
        candidates = []
        existing_edges = set(self.graph.neighbors(source_node))
        
        # Consider all nodes as potential candidates
        for target in self.graph.nodes():
            if target != source_node and target not in existing_edges:
                # Combine common neighbors and Adamic-Adar scores
                cn_score = self._common_neighbors_score(source_node, target)
                aa_score = self._adamic_adar_score(source_node, target)
                score = cn_score + aa_score
                candidates.append((target, score))
        
        # Sort by score and return top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:num_candidates]
