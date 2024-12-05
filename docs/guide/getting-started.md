# Getting Started

This guide will help you get started with the Graph Embeddings library.

## Prerequisites

- Python 3.8 or higher
- NetworkX
- NumPy
- Streamlit (for visualization)

## Installation

Install the library using pip:

```bash
pip install -r requirements.txt
```

## Basic Usage

### 1. Creating an Embedding

```python
from core.embeddings import EuclideanEmbedding
import networkx as nx

# Create a sample graph
G = nx.karate_club_graph()

# Initialize the embedding
embedding = EuclideanEmbedding(dim=2)

# Train the embedding
embedding.train(G)
```

### 2. Using Embedding-Aware Algorithms

```python
from core.algorithms import DijkstraEmbedding

# Initialize the algorithm with the embedding
dijkstra = DijkstraEmbedding(embedding)

# Find shortest path
path = dijkstra.shortest_path(source=0, target=33)
print(f"Shortest path: {path}")
```

### 3. Visualizing Results

```python
import streamlit as st
from core.visualization import visualize_graph

# Visualize the graph with embeddings
st.title("Graph Visualization")
visualize_graph(G, embedding)
```

## Choosing an Embedding Type

The library provides three types of embeddings:

1. **Euclidean Embedding**: Best for general-purpose graph embedding
2. **Spherical Embedding**: Suitable for graphs with hierarchical structure
3. **Hyperbolic Embedding**: Optimal for scale-free networks

Choose the embedding type based on your graph's properties and your specific needs.

## Next Steps

- Learn about [advanced usage](../examples/advanced.md)
- Explore the [API reference](../api/reference.md)
- Check out [example notebooks](../examples/basic.md)
