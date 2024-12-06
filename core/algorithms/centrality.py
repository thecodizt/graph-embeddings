"""Centrality algorithms with and without embedding awareness."""
from typing import Dict, Set
import math
import networkx as nx
from .base import GraphAlgorithm, EmbeddingAwareAlgorithm

class BetweennessCentralityTraditional(GraphAlgorithm):
    """Traditional Betweenness Centrality implementation."""
    
    @GraphAlgorithm.measure_execution_time
    def run(self, normalized: bool = True) -> Dict[int, float]:
        """
        Compute betweenness centrality for all nodes.
        
        Args:
            normalized: Whether to normalize the centrality values
            
        Returns:
            Dictionary mapping node IDs to centrality scores
        """
        betweenness = {node: 0.0 for node in self.graph.nodes()}
        nodes = list(self.graph.nodes())
        
        for s in nodes:
            # Single-source shortest paths
            S = []  # Stack of nodes in order of non-increasing distance from s
            P = {w: [] for w in nodes}  # Predecessors on shortest paths from s
            sigma = {w: 0 for w in nodes}  # Number of shortest paths from s to w
            sigma[s] = 1
            D = {w: -1 for w in nodes}  # Distance from s to w
            D[s] = 0
            Q = [s]  # Use as queue
            
            while Q:  # Use BFS to find shortest paths
                v = Q.pop(0)
                S.append(v)
                for w in self.graph.neighbors(v):
                    # Path discovery
                    if D[w] < 0:  # First time to find w
                        Q.append(w)
                        D[w] = D[v] + 1
                    # Path counting
                    if D[w] == D[v] + 1:  # Edge is on a shortest path
                        sigma[w] += sigma[v]
                        P[w].append(v)
            
            # Accumulation phase
            delta = {w: 0 for w in nodes}
            while S:  # Back propagation
                w = S.pop()
                for v in P[w]:
                    delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
                if w != s:
                    betweenness[w] += delta[w]
        
        if normalized:
            n = len(nodes)
            if n <= 2:
                return betweenness
            scale = 1.0 / ((n - 1) * (n - 2))
            for v in nodes:
                betweenness[v] *= scale
        
        return betweenness

class BetweennessCentralityEmbedding(EmbeddingAwareAlgorithm):
    """Embedding-aware Betweenness Centrality implementation."""
    
    def _get_edge_weight(self, u: int, v: int) -> float:
        """Get edge weight based on embedding distance."""
        return self.embedder.compute_distance(u, v)
    
    @GraphAlgorithm.measure_execution_time
    def run(self, normalized: bool = True) -> Dict[int, float]:
        """
        Compute betweenness centrality using embedding distances.
        
        Args:
            normalized: Whether to normalize the centrality values
            
        Returns:
            Dictionary mapping node IDs to centrality scores
        """
        betweenness = {node: 0.0 for node in self.graph.nodes()}
        nodes = list(self.graph.nodes())
        
        for s in nodes:
            # Single-source shortest paths using embedding distances
            S = []
            P = {w: [] for w in nodes}
            sigma = {w: 0 for w in nodes}
            sigma[s] = 1
            D = {w: float('infinity') for w in nodes}
            D[s] = 0
            Q = [(0, s)]  # Priority queue with distances
            
            while Q:  # Use Dijkstra for shortest paths
                d, v = min(Q)
                Q.remove((d, v))
                S.append(v)
                
                for w in self.graph.neighbors(v):
                    # Path discovery using embedding distances
                    dw = D[v] + self._get_edge_weight(v, w)
                    
                    if D[w] == float('infinity'):  # First time to find w
                        Q.append((dw, w))
                        D[w] = dw
                        sigma[w] = sigma[v]
                        P[w] = [v]
                    elif dw == D[w]:  # Another shortest path
                        sigma[w] += sigma[v]
                        P[w].append(v)
            
            # Accumulation phase
            delta = {w: 0 for w in nodes}
            while S:  # Back propagation
                w = S.pop()
                for v in P[w]:
                    delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
                if w != s:
                    betweenness[w] += delta[w]
        
        if normalized:
            n = len(nodes)
            if n <= 2:
                return betweenness
            scale = 1.0 / ((n - 1) * (n - 2))
            for v in nodes:
                betweenness[v] *= scale
        
        return betweenness

class ClosenessCentralityTraditional(GraphAlgorithm):
    """Traditional Closeness Centrality implementation."""
    
    @GraphAlgorithm.measure_execution_time
    def run(self, normalized: bool = True) -> Dict[int, float]:
        """
        Compute closeness centrality for all nodes.
        
        Args:
            normalized: Whether to normalize the centrality values
            
        Returns:
            Dictionary mapping node IDs to centrality scores
        """
        closeness = {}
        nodes = list(self.graph.nodes())
        n = len(nodes)
        
        for node in nodes:
            # Compute shortest path lengths from node to all others
            path_lengths = nx.single_source_dijkstra_path_length(self.graph, node)
            
            if len(path_lengths) > 1:
                # Sum of shortest paths
                total = sum(path_lengths.values())
                if total > 0:
                    closeness[node] = (len(path_lengths) - 1) / total
                    if normalized:
                        closeness[node] *= (len(path_lengths) - 1) / (n - 1)
                else:
                    closeness[node] = 0.0
            else:
                closeness[node] = 0.0
        
        return closeness

class ClosenessCentralityEmbedding(EmbeddingAwareAlgorithm):
    """Embedding-aware Closeness Centrality implementation."""
    
    def _get_shortest_paths(self, source: int) -> Dict[int, float]:
        """Compute shortest paths using embedding distances."""
        distances = {node: float('infinity') for node in self.graph.nodes()}
        distances[source] = 0
        visited = set()
        pq = [(0, source)]
        
        while pq:
            d, node = min(pq)
            pq.remove((d, node))
            
            if node in visited:
                continue
            visited.add(node)
            
            for neighbor in self.graph.neighbors(node):
                if neighbor in visited:
                    continue
                weight = self.embedder.compute_distance(node, neighbor)
                distance = distances[node] + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    pq.append((distance, neighbor))
        
        return distances
    
    @GraphAlgorithm.measure_execution_time
    def run(self, normalized: bool = True) -> Dict[int, float]:
        """
        Compute closeness centrality using embedding distances.
        
        Args:
            normalized: Whether to normalize the centrality values
            
        Returns:
            Dictionary mapping node IDs to centrality scores
        """
        closeness = {}
        nodes = list(self.graph.nodes())
        n = len(nodes)
        
        for node in nodes:
            # Compute shortest path lengths using embedding distances
            path_lengths = self._get_shortest_paths(node)
            
            if len(path_lengths) > 1:
                # Remove unreachable nodes (infinity distances)
                reachable_lengths = {k: v for k, v in path_lengths.items() 
                                  if v != float('infinity') and k != node}
                
                if reachable_lengths:
                    # Sum of shortest paths in embedding space
                    total = sum(reachable_lengths.values())
                    if total > 0:
                        closeness[node] = len(reachable_lengths) / total
                        if normalized:
                            closeness[node] *= len(reachable_lengths) / (n - 1)
                    else:
                        closeness[node] = 0.0
                else:
                    closeness[node] = 0.0
            else:
                closeness[node] = 0.0
        
        return closeness
