import streamlit as st
import os
from pathlib import Path

def read_html_file(file_path):
    """Read HTML file and return its content."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def main():
    st.set_page_config(page_title="Graph Embeddings Documentation", layout="wide")
    
    # Documentation sections
    st.title("Graph Embeddings Documentation")
    
    st.markdown("""
    ## Overview
    
    Graph Embeddings is a powerful library for analyzing and visualizing graphs using various embedding techniques.
    
    ### Features
    
    - Multiple embedding types (Euclidean, Spherical, Hyperbolic)
    - Graph algorithms optimized for embeddings
    - Interactive visualization
    - Streamlit-based web interface
    
    ### Core Components
    
    #### Embeddings
    
    The library provides several embedding types:
    
    1. **Base Embedding**
       - Abstract base class defining the embedding interface
       - Common utility methods for all embedding types
    
    2. **Euclidean Embedding**
       - Embeds graphs in Euclidean space
       - Uses standard Euclidean distance metrics
       - Suitable for general-purpose graph embedding
    
    3. **Spherical Embedding**
       - Embeds graphs on the surface of a sphere
       - Uses geodesic distances
       - Useful for graphs with hierarchical structure
    
    4. **Hyperbolic Embedding**
       - Embeds graphs in hyperbolic space (Poincaré disk)
       - Particularly effective for scale-free and hierarchical networks
       - Preserves both local and global structure
    
    #### Algorithms
    
    1. **Shortest Path Algorithms**
       - Traditional Dijkstra's algorithm
       - Embedding-aware Dijkstra's algorithm
       - Path visualization and comparison tools
    
    2. **Graph Properties**
       - Connectivity analysis
       - Clustering coefficients
       - Density calculations
       - Support for both directed and undirected graphs
    """)

    st.markdown("""
    ### Usage Examples
    ```python
    from core.embeddings import EuclideanEmbedding
    from core.algorithms import DijkstraEmbedding
    
    # Create an embedding
    embedding = EuclideanEmbedding(dim=2)
    embedding.train(graph)
    
    # Use embedding-aware algorithms
    dijkstra = DijkstraEmbedding(embedding)
    path = dijkstra.shortest_path(source, target)
    ```
    """)

    st.markdown("""
    ### API Reference
    
    #### Embeddings
    
    **BaseEmbedding**
    ```python
    class BaseEmbedding:
        def __init__(self, dim: int):
            '''Initialize embedding with dimension.'''
            
        def train(self, graph: nx.Graph):
            '''Train the embedding on a graph.'''
            
        def compute_distance(self, node1: int, node2: int) -> float:
            '''Compute distance between two nodes in the embedding space.'''
    ```
    
    **EuclideanEmbedding**
    ```python
    class EuclideanEmbedding(BaseEmbedding):
        def compute_distance(self, node1: int, node2: int) -> float:
            '''Compute Euclidean distance between two nodes.'''
    ```
    
    **SphericalEmbedding**
    ```python
    class SphericalEmbedding(BaseEmbedding):
        def compute_distance(self, node1: int, node2: int) -> float:
            '''Compute geodesic distance between two nodes on the sphere.'''
    ```
    
    **HyperbolicEmbedding**
    ```python
    class HyperbolicEmbedding(BaseEmbedding):
        def compute_distance(self, node1: int, node2: int) -> float:
            '''Compute hyperbolic distance between two nodes in the Poincaré disk.'''
    ```
    
    #### Algorithms
    
    **DijkstraEmbedding**
    ```python
    class DijkstraEmbedding:
        def __init__(self, embedding: BaseEmbedding):
            '''Initialize with an embedding.'''
            
        def shortest_path(self, source: int, target: int) -> List[int]:
            '''Find shortest path using embedding-aware algorithm.'''
    ```
    """)

if __name__ == "__main__":
    main()
