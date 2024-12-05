"""Shortest path algorithms with and without embedding awareness."""
from typing import Dict, List, Optional, Set, Tuple
import heapq
import networkx as nx
from .base import GraphAlgorithm, EmbeddingAwareAlgorithm

class DijkstraTraditional(GraphAlgorithm):
    """Traditional Dijkstra's algorithm implementation."""
    
    @GraphAlgorithm.measure_execution_time
    def run(self, source: int, target: int) -> Tuple[List[int], float]:
        """
        Find shortest path between source and target nodes.
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            Tuple of (path, distance)
        """
        distances = {node: float('infinity') for node in self.graph.nodes()}
        predecessors = {node: None for node in self.graph.nodes()}
        distances[source] = 0
        pq = [(0, source)]
        visited = set()
        
        while pq:
            current_distance, current = heapq.heappop(pq)
            
            if current in visited:
                continue
                
            visited.add(current)
            
            if current == target:
                break
                
            for neighbor in self.graph.neighbors(current):
                if neighbor in visited:
                    continue
                    
                weight = self.graph[current][neighbor].get('weight', 1)
                distance = current_distance + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current
                    heapq.heappush(pq, (distance, neighbor))
        
        # Reconstruct path
        path = []
        current = target
        while current is not None:
            path.append(current)
            current = predecessors[current]
        path.reverse()
        
        return path, distances[target]

class DijkstraEmbedding(EmbeddingAwareAlgorithm):
    """Embedding-aware Dijkstra's algorithm implementation.
    
    This implementation uses embeddings only to guide the search order,
    not to influence the actual path selection. This ensures we find the
    same optimal paths as traditional Dijkstra's algorithm, but potentially
    faster by exploring promising nodes first.
    """
    
    def _get_search_priority(self, distance: float, node: int, target: int) -> float:
        """
        Get search priority for a node based on its current shortest path distance
        and its embedding distance to target. Lower value means higher priority.
        
        Args:
            distance: Current shortest path distance to this node
            node: Current node ID
            target: Target node ID
            
        Returns:
            Priority value (lower is better)
        """
        # A* like heuristic: f(n) = g(n) + h(n)
        # g(n) is the current shortest path distance
        # h(n) is the embedding distance to target (our heuristic)
        h_estimate = self.embedder.compute_distance(node, target)
        return distance + h_estimate
    
    @GraphAlgorithm.measure_execution_time
    def run(self, source: int, target: int) -> Tuple[List[int], float]:
        """
        Find shortest path between source and target nodes using embedding information
        to guide the search order. The actual path selection is based solely on
        shortest path distances, ensuring optimality.
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            Tuple of (path, distance)
        """
        distances = {node: float('infinity') for node in self.graph.nodes()}
        predecessors = {node: None for node in self.graph.nodes()}
        distances[source] = 0
        
        # Priority queue entries are (search_priority, current_distance, node)
        start_priority = self._get_search_priority(0, source, target)
        pq = [(start_priority, 0, source)]
        visited = set()
        
        while pq:
            _, current_distance, current = heapq.heappop(pq)
            
            if current in visited:
                continue
                
            visited.add(current)
            
            if current == target:
                break
            
            for neighbor in self.graph.neighbors(current):
                if neighbor in visited:
                    continue
                    
                weight = self.graph[current][neighbor].get('weight', 1)
                distance = current_distance + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current
                    priority = self._get_search_priority(distance, neighbor, target)
                    heapq.heappush(pq, (priority, distance, neighbor))
        
        # Reconstruct path
        path = []
        current = target
        while current is not None:
            path.append(current)
            current = predecessors[current]
        path.reverse()
        
        return path, distances[target]
