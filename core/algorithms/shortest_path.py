"""Shortest path algorithms with and without embedding awareness."""
from typing import Dict, List, Optional, Set, Tuple, Any
import heapq
import networkx as nx
import numpy as np
from .base import GraphAlgorithm, EmbeddingAwareAlgorithm

class DijkstraTraditional(GraphAlgorithm):
    """Traditional Dijkstra's algorithm implementation."""
    
    @GraphAlgorithm.measure_execution_time
    def run(self, source: int, target: int) -> Dict[str, Any]:
        """
        Find shortest path between source and target nodes.
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            Dictionary containing:
                - path: List of nodes in the shortest path
                - distance: Length of the shortest path
        """
        distances = {node: float('infinity') for node in self.graph.nodes()}
        predecessors = {node: None for node in self.graph.nodes()}
        distances[source] = 0
        pq = [(0, source)]
        visited = set()
        
        while pq:
            current_distance, current = heapq.heappop(pq)
            
            if current == target:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = predecessors[current]
                path.reverse()
                
                return {
                    'path': path,
                    'distance': distances[target]
                }
            
            if current in visited:
                continue
            
            visited.add(current)
            
            for neighbor in self.graph.neighbors(current):
                distance = current_distance + 1  # Using unit weights
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current
                    heapq.heappush(pq, (distance, neighbor))
        
        return {
            'path': [],
            'distance': float('infinity')
        }

class DijkstraEmbedding(EmbeddingAwareAlgorithm):
    """Embedding-aware Dijkstra's algorithm implementation.
    
    This implementation uses embedding distances for path selection,
    which means it will find different paths than traditional Dijkstra.
    The paths found will be optimal in the embedding space rather than
    the original graph space.
    """
    
    def _get_edge_weight(self, u: int, v: int) -> float:
        """
        Get the weight between two nodes based on their embedding distance.
        
        Args:
            u: First node ID
            v: Second node ID
            
        Returns:
            Edge weight based on embedding distance
        """
        return self.embedder.compute_distance(u, v)
    
    @GraphAlgorithm.measure_execution_time
    def run(self, source: int, target: int) -> Dict[str, Any]:
        """
        Find shortest path between source and target nodes using embedding distances
        for path selection. This will find paths that are optimal in the embedding
        space rather than the original graph space.
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            Dictionary containing:
                - path: List of nodes in the shortest path
                - distance: Length of the shortest path
        """
        distances = {node: float('infinity') for node in self.graph.nodes()}
        predecessors = {node: None for node in self.graph.nodes()}
        distances[source] = 0
        pq = [(0, source)]
        visited = set()
        
        while pq:
            current_distance, current = heapq.heappop(pq)
            
            if current == target:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = predecessors[current]
                path.reverse()
                
                return {
                    'path': path,
                    'distance': distances[target]
                }
            
            if current in visited:
                continue
            
            visited.add(current)
            
            for neighbor in self.graph.neighbors(current):
                weight = self._get_edge_weight(current, neighbor)
                distance = current_distance + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current
                    heapq.heappush(pq, (distance, neighbor))
        
        return {
            'path': [],
            'distance': float('infinity')
        }

class BellmanFordTraditional(GraphAlgorithm):
    """Traditional Bellman-Ford algorithm implementation."""
    
    @GraphAlgorithm.measure_execution_time
    def run(self, source: int, target: int) -> Dict[str, Any]:
        """
        Find shortest path between source and target nodes, works with negative weights.
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            Dictionary containing:
                - path: List of nodes in the shortest path
                - distance: Length of the shortest path
        """
        distances = {node: float('infinity') for node in self.graph.nodes()}
        predecessors = {node: None for node in self.graph.nodes()}
        distances[source] = 0
        
        # Relax edges |V| - 1 times
        for _ in range(len(self.graph.nodes()) - 1):
            for u, v in self.graph.edges():
                weight = self.graph[u][v].get('weight', 1)
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    predecessors[v] = u
        
        # Check for negative cycles
        for u, v in self.graph.edges():
            weight = self.graph[u][v].get('weight', 1)
            if distances[u] + weight < distances[v]:
                raise ValueError("Graph contains negative cycles")
        
        # Reconstruct path
        path = []
        current = target
        while current is not None:
            path.append(current)
            current = predecessors[current]
        path.reverse()
        
        return {
            'path': path,
            'distance': distances[target]
        }

class BellmanFordEmbedding(EmbeddingAwareAlgorithm):
    """Embedding-aware Bellman-Ford algorithm implementation."""
    
    def _get_edge_weight(self, u: int, v: int) -> float:
        """Get edge weight based on embedding distance."""
        return self.embedder.compute_distance(u, v)
    
    @GraphAlgorithm.measure_execution_time
    def run(self, source: int, target: int) -> Dict[str, Any]:
        """
        Find shortest path using embedding distances, works with negative weights.
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            Dictionary containing:
                - path: List of nodes in the shortest path
                - distance: Length of the shortest path
        """
        distances = {node: float('infinity') for node in self.graph.nodes()}
        predecessors = {node: None for node in self.graph.nodes()}
        distances[source] = 0
        
        # Relax edges |V| - 1 times using embedding distances
        for _ in range(len(self.graph.nodes()) - 1):
            for u, v in self.graph.edges():
                weight = self._get_edge_weight(u, v)
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    predecessors[v] = u
        
        # Check for negative cycles using embedding distances
        for u, v in self.graph.edges():
            weight = self._get_edge_weight(u, v)
            if distances[u] + weight < distances[v]:
                raise ValueError("Embedding space contains negative cycles")
        
        # Reconstruct path
        path = []
        current = target
        while current is not None:
            path.append(current)
            current = predecessors[current]
        path.reverse()
        
        return {
            'path': path,
            'distance': distances[target]
        }

class FloydWarshallTraditional(GraphAlgorithm):
    """Traditional Floyd-Warshall algorithm implementation."""
    
    def run(self) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """Run Floyd-Warshall algorithm to find all-pairs shortest paths.
        
        Returns:
            dict: A dictionary mapping (source, target) tuples to shortest paths.
        """
        # Initialize distances and paths
        dist = {(u, v): float('inf') for u in self.graph.nodes() for v in self.graph.nodes()}
        next_node = {(u, v): None for u in self.graph.nodes() for v in self.graph.nodes()}
        
        # Set initial distances
        for u in self.graph.nodes():
            dist[(u, u)] = 0
            next_node[(u, u)] = u
        
        for u, v in self.graph.edges():
            dist[(u, v)] = self.graph[u][v]['weight']
            next_node[(u, v)] = v
        
        # Floyd-Warshall algorithm
        for k in self.graph.nodes():
            for i in self.graph.nodes():
                for j in self.graph.nodes():
                    if dist[(i, k)] + dist[(k, j)] < dist[(i, j)]:
                        dist[(i, j)] = dist[(i, k)] + dist[(k, j)]
                        next_node[(i, j)] = next_node[(i, k)]
        
        # Reconstruct paths
        paths = {}
        for u in self.graph.nodes():
            for v in self.graph.nodes():
                if dist[(u, v)] != float('inf'):
                    path = []
                    curr = u
                    while curr != v:
                        path.append(curr)
                        curr = next_node[(curr, v)]
                    path.append(v)
                    paths[(u, v)] = {
                        'path': path,
                        'distance': dist[(u, v)]
                    }
        
        return paths


class FloydWarshallEmbedding(EmbeddingAwareAlgorithm):
    """Embedding-aware Floyd-Warshall algorithm implementation."""
    
    def run(self) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """Run Floyd-Warshall algorithm using embeddings for optimization.
        
        Returns:
            dict: A dictionary mapping (source, target) tuples to shortest paths.
        """
        # Get embeddings for all nodes
        embeddings = {node: self.embedder.get_node_embedding(node) for node in self.graph.nodes()}
        
        # Initialize distances and paths
        dist = {(u, v): float('inf') for u in self.graph.nodes() for v in self.graph.nodes()}
        next_node = {(u, v): None for u in self.graph.nodes() for v in self.graph.nodes()}
        
        # Set initial distances
        for u in self.graph.nodes():
            dist[(u, u)] = 0
            next_node[(u, u)] = u
        
        for u, v in self.graph.edges():
            dist[(u, v)] = self.graph[u][v]['weight']
            next_node[(u, v)] = v
        
        # Sort nodes by embedding similarity to optimize iteration order
        nodes = list(self.graph.nodes())
        nodes.sort(key=lambda n: np.linalg.norm(embeddings[n]))
        
        # Floyd-Warshall algorithm with optimized iteration order
        for k in nodes:
            for i in nodes:
                # Skip if i and k are far in embedding space
                if np.linalg.norm(embeddings[i] - embeddings[k]) > 2.0:
                    continue
                for j in nodes:
                    # Skip if k and j are far in embedding space
                    if np.linalg.norm(embeddings[k] - embeddings[j]) > 2.0:
                        continue
                    if dist[(i, k)] + dist[(k, j)] < dist[(i, j)]:
                        dist[(i, j)] = dist[(i, k)] + dist[(k, j)]
                        next_node[(i, j)] = next_node[(i, k)]
        
        # Reconstruct paths
        paths = {}
        for u in self.graph.nodes():
            for v in self.graph.nodes():
                if dist[(u, v)] != float('inf'):
                    path = []
                    curr = u
                    while curr != v:
                        path.append(curr)
                        curr = next_node[(curr, v)]
                    path.append(v)
                    paths[(u, v)] = {
                        'path': path,
                        'distance': dist[(u, v)]
                    }
        
        return paths

class AStarTraditional(GraphAlgorithm):
    """Traditional A* algorithm implementation."""
    
    def _heuristic(self, node: int, target: int) -> float:
        """
        Compute admissible heuristic for A*.
        Uses graph structure to estimate remaining distance.
        """
        # Use graph-theoretic properties for heuristic
        # Here we use degree difference as a simple admissible heuristic
        return abs(self.graph.degree[node] - self.graph.degree[target]) * 0.1
    
    @GraphAlgorithm.measure_execution_time
    def run(self, source: int, target: int) -> Dict[str, Any]:
        """
        Find shortest path between source and target nodes using A*.
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            Dictionary containing:
                - path: List of nodes in the shortest path
                - distance: Length of the shortest path
        """
        distances = {node: float('infinity') for node in self.graph.nodes()}
        predecessors = {node: None for node in self.graph.nodes()}
        distances[source] = 0
        
        # Priority queue entries are (f_score, node)
        # f_score = g_score + heuristic
        pq = [(self._heuristic(source, target), source)]
        visited = set()
        
        while pq:
            _, current = heapq.heappop(pq)
            
            if current in visited:
                continue
                
            visited.add(current)
            
            if current == target:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = predecessors[current]
                path.reverse()
                
                return {
                    'path': path,
                    'distance': distances[target]
                }
            
            for neighbor in self.graph.neighbors(current):
                weight = self.graph[current][neighbor].get('weight', 1)
                distance = distances[current] + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current
                    f_score = distance + self._heuristic(neighbor, target)
                    heapq.heappush(pq, (f_score, neighbor))
        
        return {
            'path': [],
            'distance': float('infinity')
        }

class AStarEmbedding(EmbeddingAwareAlgorithm):
    """Embedding-aware A* algorithm implementation."""
    
    def _get_edge_weight(self, u: int, v: int) -> float:
        """Get edge weight based on embedding distance."""
        return self.embedder.compute_distance(u, v)
    
    def _heuristic(self, node: int, target: int) -> float:
        """
        Compute admissible heuristic for A*.
        Uses embedding distance as the heuristic.
        """
        return self.embedder.compute_distance(node, target)
    
    @GraphAlgorithm.measure_execution_time
    def run(self, source: int, target: int) -> Dict[str, Any]:
        """
        Find shortest path between source and target nodes using A* with embeddings.
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            Dictionary containing:
                - path: List of nodes in the shortest path
                - distance: Length of the shortest path
        """
        distances = {node: float('infinity') for node in self.graph.nodes()}
        predecessors = {node: None for node in self.graph.nodes()}
        distances[source] = 0
        
        # Priority queue entries are (f_score, node)
        # f_score = g_score + heuristic
        pq = [(self._heuristic(source, target), source)]
        visited = set()
        
        while pq:
            _, current = heapq.heappop(pq)
            
            if current in visited:
                continue
                
            visited.add(current)
            
            if current == target:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = predecessors[current]
                path.reverse()
                
                return {
                    'path': path,
                    'distance': distances[target]
                }
            
            for neighbor in self.graph.neighbors(current):
                weight = self._get_edge_weight(current, neighbor)
                distance = distances[current] + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current
                    f_score = distance + self._heuristic(neighbor, target)
                    heapq.heappush(pq, (f_score, neighbor))
        
        return {
            'path': [],
            'distance': float('infinity')
        }
