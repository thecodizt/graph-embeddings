# Graph Embeddings Library

Welcome to the Graph Embeddings library documentation! This library provides a comprehensive suite of tools for embedding graphs in various geometric spaces and performing graph algorithms that leverage these embeddings.

## Features

- Multiple embedding types:
    - Euclidean space embeddings
    - Spherical surface embeddings
    - Hyperbolic (Poincaré disk) embeddings
- Embedding-aware graph algorithms
- Interactive visualization
- Support for both directed and undirected graphs

## Quick Start

```python
from core.embeddings import EuclideanEmbedding
from core.algorithms import DijkstraEmbedding
import networkx as nx

# Create a sample graph
G = nx.karate_club_graph()

# Create and train the embedding
embedding = EuclideanEmbedding(dim=2)
embedding.train(G)

# Use embedding-aware algorithms
dijkstra = DijkstraEmbedding(embedding)
path = dijkstra.shortest_path(0, 33)
```

## Why Graph Embeddings?

Graph embeddings provide a way to represent graph nodes in continuous vector spaces while preserving graph properties such as node similarity and graph structure. This enables:

1. **Efficient Algorithms**: Use geometric properties to speed up graph algorithms
2. **Visualization**: Natural way to visualize high-dimensional graph data
3. **Machine Learning**: Bridge between graph structures and traditional ML methods

## Project Structure

```
graph-embeddings/
├── core/
│   ├── embeddings/       # Embedding implementations
│   ├── algorithms/       # Graph algorithms
│   └── visualization/    # Visualization tools
├── docs/                 # Documentation
└── examples/            # Usage examples
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](guide/contributing.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/yourusername/graph-embeddings/blob/main/LICENSE) file for details.
