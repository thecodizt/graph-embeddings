# Mathematical Background of Graph Embeddings

## Introduction to Graph Embeddings

Graph embeddings map nodes from a graph into a continuous vector space while preserving structural information. Formally, given a graph G = (V, E), we want to find a mapping:

$$f: V \to \mathbb{R}^d$$

where d is the dimension of the embedding space.

The key challenge is preserving graph distances in the embedded space. For nodes u, v ∈ V, we want:

$$d_G(u,v) \approx d_E(f(u), f(v))$$

where $d_G$ is the graph distance and $d_E$ is the distance in the embedding space.

## Euclidean Embeddings

### Theory

Euclidean embeddings map nodes into standard Euclidean space $\mathbb{R}^d$ with the Euclidean distance metric:

$$d_E(x, y) = \sqrt{\sum_{i=1}^d (x_i - y_i)^2}$$

Properties:
- **Symmetry**: $d(x,y) = d(y,x)$
- **Triangle Inequality**: $d(x,z) \leq d(x,y) + d(y,z)$
- **Translation Invariance**: Distance preserved under translations

### Limitations

Euclidean space has constant curvature = 0, which can limit its ability to represent hierarchical structures. The number of points that can be embedded at a fixed distance from a central point grows polynomially with the radius.

## Spherical Embeddings

### Theory

Spherical embeddings map nodes onto the surface of a $d$-dimensional sphere $\mathbb{S}^{d-1}$. Points are represented as unit vectors, with distance measured by great circle distance or angular distance:

$$d_S(x, y) = \arccos(\langle x, y \rangle)$$

where $\langle x, y \rangle$ is the dot product of unit vectors.

Properties:
- **Positive Curvature**: The space has constant positive curvature
- **Bounded**: All distances are bounded by $\pi$
- **Symmetry**: Distances are symmetric and rotationally invariant

### Applications

Particularly useful for:
- Data with inherent spherical structure
- Problems requiring bounded distances
- Circular or periodic relationships

## Hyperbolic Embeddings

### Theory

Hyperbolic embeddings map nodes into hyperbolic space $\mathbb{H}^d$, which has constant negative curvature. We typically use the Poincaré ball model:

$$d_H(x, y) = \text{arcosh}\left(1 + 2\frac{\|x-y\|^2}{(1-\|x\|^2)(1-\|y\|^2)}\right)$$

Properties:
- **Negative Curvature**: Space expands exponentially
- **Tree-Like**: Natural representation for hierarchical structures
- **Exponential Growth**: The space available at distance $r$ grows exponentially with $r$

### Advantages

Hyperbolic geometry is particularly well-suited for:
- Hierarchical structures
- Scale-free networks
- Trees and tree-like graphs

The volume of a ball in hyperbolic space grows exponentially with its radius, allowing efficient embedding of hierarchical structures.

## Comparison of Embedding Spaces

### Geometric Properties

| Property | Euclidean | Spherical | Hyperbolic |
|----------|-----------|-----------|------------|
| Curvature | 0 | +1 | -1 |
| Volume Growth | Polynomial | Bounded | Exponential |
| Distance Type | Unbounded | Bounded | Unbounded |
| Best For | Flat structures | Circular patterns | Hierarchies |

### Distortion Bounds

For a graph G with n nodes:

1. **Euclidean Space**:
   - Some graphs require $\Omega(\log n)$ distortion
   - Complete binary trees require $\Omega(\log n)$ dimensions

2. **Spherical Space**:
   - All distances bounded by $\pi$
   - Best for graphs with bounded diameter

3. **Hyperbolic Space**:
   - Trees can be embedded with zero distortion
   - Many real-world networks can be embedded in low dimensions

## Implementation Considerations

### Optimization Objective

The general form of the optimization objective is:

$$\min_f \sum_{(u,v) \in E} \|d_G(u,v) - d_E(f(u), f(v))\|^2$$

Additional terms might include:

1. **Stress term**:
   $$\sum_{u,v \in V} (d_G(u,v) - d_E(f(u), f(v)))^2$$

2. **Regularization**:
   $$\lambda \sum_{u \in V} \|f(u)\|^2$$

3. **Negative sampling**:
   $$\sum_{(u,v) \not\in E} \max(0, \gamma - d_E(f(u), f(v)))$$

### Initialization Strategies

1. **Random Initialization**:
   - Euclidean: Sample from $\mathcal{N}(0, \sigma^2)$
   - Spherical: Normalize random vectors
   - Hyperbolic: Sample from truncated normal in Poincaré ball

2. **Spectral Initialization**:
   - Use top-$d$ eigenvectors of graph Laplacian
   - Project onto appropriate manifold if needed

3. **Landmark-based**:
   - Choose landmark nodes
   - Initialize other nodes based on distances to landmarks
