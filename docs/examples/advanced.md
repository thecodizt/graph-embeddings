# Advanced Usage

This page demonstrates advanced usage patterns and techniques for the Graph Embeddings library.

## Custom Distance Functions

You can create custom embeddings by subclassing `BaseEmbedding` and implementing your own distance function:

```python
import numpy as np
from core.embeddings import BaseEmbedding

class CustomEmbedding(BaseEmbedding):
    def compute_distance(self, node1: int, node2: int) -> float:
        """Custom distance function using Manhattan distance."""
        v1 = self.embeddings[node1]
        v2 = self.embeddings[node2]
        return np.sum(np.abs(v1 - v2))

    def _initialize_embeddings(self, graph):
        """Custom initialization strategy."""
        pos = {}
        for node in graph.nodes():
            pos[node] = np.random.uniform(-1, 1, self.dim)
        return pos
```

## Working with Large Graphs

For large graphs, you can use batching and optimization techniques:

```python
import networkx as nx
from core.embeddings import EuclideanEmbedding

# Create a large graph
G = nx.barabasi_albert_graph(10000, 3)

# Initialize embedding with optimization parameters
embedding = EuclideanEmbedding(
    dim=2,
    batch_size=128,
    learning_rate=0.01,
    num_epochs=50
)

# Train with progress tracking
embedding.train(G, verbose=True)
```

## Embedding Evaluation

Evaluate embedding quality using various metrics:

```python
from core.evaluation import (
    compute_stress,
    compute_distortion,
    evaluate_link_prediction
)

# Compute stress (how well distances are preserved)
stress = compute_stress(G, embedding)
print(f"Embedding stress: {stress}")

# Compute distortion
distortion = compute_distortion(G, embedding)
print(f"Average distortion: {distortion}")

# Evaluate link prediction
auc_score = evaluate_link_prediction(G, embedding)
print(f"Link prediction AUC: {auc_score}")
```

## Combining Multiple Embeddings

Create ensemble embeddings by combining different types:

```python
from core.embeddings import EuclideanEmbedding, HyperbolicEmbedding
import numpy as np

class EnsembleEmbedding:
    def __init__(self, embeddings, weights=None):
        self.embeddings = embeddings
        self.weights = weights or [1/len(embeddings)] * len(embeddings)
    
    def compute_distance(self, node1, node2):
        distances = [
            emb.compute_distance(node1, node2)
            for emb in self.embeddings
        ]
        return np.average(distances, weights=self.weights)

# Create ensemble
euclidean = EuclideanEmbedding(dim=2)
hyperbolic = HyperbolicEmbedding(dim=2)

euclidean.train(G)
hyperbolic.train(G)

ensemble = EnsembleEmbedding(
    embeddings=[euclidean, hyperbolic],
    weights=[0.7, 0.3]
)
```

## Custom Visualization

Create custom visualizations using the embedding coordinates:

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_embedding(G, embedding):
    """Create a 3D visualization of the embedding."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get node positions
    pos = {
        node: embedding.get_embedding(node)
        for node in G.nodes()
    }
    
    # Plot nodes
    xs = [pos[node][0] for node in G.nodes()]
    ys = [pos[node][1] for node in G.nodes()]
    zs = [pos[node][2] for node in G.nodes()]
    ax.scatter(xs, ys, zs)
    
    # Plot edges
    for edge in G.edges():
        x = [pos[edge[0]][0], pos[edge[1]][0]]
        y = [pos[edge[0]][1], pos[edge[1]][1]]
        z = [pos[edge[0]][2], pos[edge[1]][2]]
        ax.plot(x, y, z, 'gray', alpha=0.5)
    
    plt.show()

# Use the custom visualization
spherical = SphericalEmbedding(dim=3)
spherical.train(G)
plot_3d_embedding(G, spherical)
```
