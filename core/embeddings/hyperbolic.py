"""Hyperbolic embedding implementation."""
import numpy as np
from typing import Dict, List, Tuple
import networkx as nx
from .base import BaseEmbedding

class HyperbolicEmbedding(BaseEmbedding):
    """Hyperbolic space (Poincaré disk) embedding."""
    
    def __init__(self, dim: int = 2):
        """Initialize embedding with dimension."""
        super().__init__(max(2, dim))  # Ensure at least 2D for Poincaré disk
        
    def _initialize_embeddings(self, graph: nx.Graph) -> Dict[int, np.ndarray]:
        """Initialize embeddings in the Poincaré disk using tree-like structure."""
        pos = {}
        nodes = list(graph.nodes())
        
        if not nodes:
            return pos
            
        # Convert to undirected for component analysis
        G = graph.to_undirected() if graph.is_directed() else graph
            
        # Handle disconnected components separately
        components = list(nx.connected_components(G))
        
        for component_idx, component in enumerate(components):
            # Create a subgraph for this component
            subgraph = graph.subgraph(component)
            
            # Choose a root node for this component
            root = list(component)[0]
            
            # For directed graphs, use weakly connected components
            if graph.is_directed():
                bfs_tree = nx.bfs_tree(subgraph.to_undirected(), root)
            else:
                bfs_tree = nx.bfs_tree(subgraph, root)
            
            # Calculate offset for this component to separate it from others
            offset = np.array([0.3 * component_idx, 0.3 * component_idx] + [0] * (self.dim - 2))
            
            # Place root with offset
            pos[root] = offset + np.zeros(self.dim)
            
            # Place other nodes level by level
            level = 1
            nodes_at_level = {0: [root]}
            current_level_nodes = [root]
            
            while current_level_nodes:
                next_level_nodes = []
                nodes_at_level[level] = []
                
                for parent in current_level_nodes:
                    # For directed graphs, consider both in and out neighbors
                    if graph.is_directed():
                        children = list(set(bfs_tree.neighbors(parent)) - set(pos.keys()))
                    else:
                        children = list(set(bfs_tree.neighbors(parent)) - set(pos.keys()))
                        
                    if not children:
                        continue
                        
                    # Place children around their parent
                    angle_step = 2 * np.pi / len(children)
                    for i, child in enumerate(children):
                        angle = angle_step * i
                        
                        # Calculate position in polar coordinates
                        r = np.tanh(0.4 * level)  # Radius increases with level but stays in disk
                        x = r * np.cos(angle)
                        y = r * np.sin(angle)
                        
                        # Create position vector
                        position = np.zeros(self.dim)
                        position[0] = x
                        position[1] = y
                        
                        # Add offset and store
                        pos[child] = offset + position
                        nodes_at_level[level].append(child)
                        next_level_nodes.append(child)
                
                current_level_nodes = next_level_nodes
                level += 1
        
        # Initialize any remaining nodes randomly near the origin
        for node in graph.nodes():
            if node not in pos:
                rand_vec = np.random.normal(0, 0.1, self.dim)
                pos[node] = rand_vec / (np.linalg.norm(rand_vec) + 1e-6) * 0.5
        
        return pos
    
    def _update_embeddings(self, graph: nx.Graph, iterations: int = 50):
        """Update embeddings using Riemannian optimization."""
        learning_rate = 0.01
        
        def _mobius_addition(u, v):
            """Möbius addition in the Poincaré disk."""
            uv = np.sum(u*v)
            u_norm_sq = np.sum(u*u)
            v_norm_sq = np.sum(v*v)
            denominator = 1 + 2*uv + u_norm_sq*v_norm_sq
            return ((1 + 2*uv + v_norm_sq)*u + (1 - u_norm_sq)*v) / denominator
        
        def _exp_map(x, v):
            """Exponential map in the Poincaré disk."""
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-7:
                return x
            x_norm = np.linalg.norm(x)
            c = 1 - x_norm**2
            return _mobius_addition(x, np.tanh(v_norm/c/2) * v/v_norm)
        
        def _project_to_disk(x):
            """Project point onto Poincaré disk."""
            norm = np.linalg.norm(x)
            if norm >= 1:
                return x / norm * 0.99
            return x
        
        # Ensure all embeddings are in the disk
        for node in graph.nodes():
            self.embeddings[node] = _project_to_disk(self.embeddings[node])
        
        for _ in range(iterations):
            # Store old positions
            old_pos = {node: self.embeddings[node].copy() for node in graph.nodes()}
            
            # Update each node's position
            for u in graph.nodes():
                grad = np.zeros(self.dim)
                
                # For directed graphs, consider both in and out neighbors
                neighbors = (list(graph.predecessors(u)) + list(graph.successors(u))) if graph.is_directed() else list(graph.neighbors(u))
                neighbors = list(set(neighbors))  # Remove duplicates
                
                # Attractive forces from neighbors
                for v in neighbors:
                    diff = old_pos[v] - old_pos[u]
                    dist = self.compute_distance(u, v)
                    if dist > 0:
                        grad += diff / (dist + 1e-6)
                
                # Repulsive forces from non-neighbors
                for v in graph.nodes():
                    if v != u and v not in neighbors:
                        diff = old_pos[v] - old_pos[u]
                        dist = self.compute_distance(u, v)
                        if dist > 0:
                            grad -= 0.1 * diff / (dist + 1e-6)
                
                # Update position using Riemannian gradient descent
                self.embeddings[u] = _exp_map(old_pos[u], learning_rate * grad)
                
                # Project back to Poincaré disk
                self.embeddings[u] = _project_to_disk(self.embeddings[u])
    
    def compute_distance(self, node1: int, node2: int) -> float:
        """Compute hyperbolic distance between two nodes in the Poincaré disk."""
        x = self.embeddings[node1]
        y = self.embeddings[node2]
        
        # Compute the hyperbolic distance in the Poincaré disk
        xy = np.sum((x - y) * (x - y))
        x_norm = np.sum(x * x)
        y_norm = np.sum(y * y)
        
        num = 2 * xy
        den = (1 - x_norm) * (1 - y_norm)
        
        return float(np.arccosh(1 + num/den))
    
    def get_visualization_coords(self, node: int) -> Tuple[float, float]:
        """Get 2D coordinates for visualization."""
        coords = self.embeddings[node]
        return float(coords[0]), float(coords[1])
