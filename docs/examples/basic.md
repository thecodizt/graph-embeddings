# Basic Examples

This page contains basic examples of using the Graph Embeddings library.

## Creating and Training Embeddings

```python
import networkx as nx
from core.embeddings import EuclideanEmbedding

# Create a sample graph
G = nx.karate_club_graph()

# Initialize embedding
embedding = EuclideanEmbedding(dim=2)

# Train the embedding
embedding.train(G)

# Access node embeddings
node_0_embedding = embedding.get_embedding(0)
print(f"Node 0 embedding: {node_0_embedding}")
```

## Finding Shortest Paths

```python
from core.algorithms import DijkstraEmbedding, DijkstraTraditional

# Initialize algorithms
dijkstra_embedding = DijkstraEmbedding(embedding)
dijkstra_traditional = DijkstraTraditional()

# Find shortest path using embedding-aware algorithm
path_embedding = dijkstra_embedding.shortest_path(0, 33)
print(f"Embedding-aware path: {path_embedding}")

# Compare with traditional algorithm
path_traditional = dijkstra_traditional.shortest_path(G, 0, 33)
print(f"Traditional path: {path_traditional}")
```

## Different Embedding Types

### Spherical Embedding

```python
from core.embeddings import SphericalEmbedding

# Initialize and train spherical embedding
spherical = SphericalEmbedding(dim=3)  # Must be 3 dimensions for sphere
spherical.train(G)

# Access embeddings
node_embeddings = {node: spherical.get_embedding(node) for node in G.nodes()}
```

### Hyperbolic Embedding

```python
from core.embeddings import HyperbolicEmbedding

# Initialize and train hyperbolic embedding
hyperbolic = HyperbolicEmbedding(dim=2)  # 2D Poincaré disk
hyperbolic.train(G)

# Compute distance between nodes
distance = hyperbolic.compute_distance(0, 1)
print(f"Hyperbolic distance between nodes 0 and 1: {distance}")
```

## Visualization

```python
import streamlit as st
from core.visualization import visualize_graph

# Create tabs for different visualizations
tab1, tab2, tab3 = st.tabs(["Euclidean", "Spherical", "Hyperbolic"])

with tab1:
    st.header("Euclidean Embedding")
    visualize_graph(G, euclidean_embedding)

with tab2:
    st.header("Spherical Embedding")
    visualize_graph(G, spherical_embedding)

with tab3:
    st.header("Hyperbolic Embedding")
    visualize_graph(G, hyperbolic_embedding)
```
