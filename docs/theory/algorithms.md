# Graph Algorithms with Embeddings

## Shortest Path Algorithms

### Traditional Dijkstra's Algorithm

The classical Dijkstra's algorithm finds shortest paths in a weighted graph $G=(V,E)$ with non-negative edge weights $w: E \rightarrow \mathbb{R}^+$.

#### Algorithm Description

For source vertex $s$:
1. Initialize distances: $d[v] = \infty$ for all $v \neq s$, $d[s] = 0$
2. Initialize priority queue $Q$ with $(s,0)$
3. While $Q$ not empty:
   - Extract vertex $u$ with minimum distance
   - For each neighbor $v$ of $u$:
     - If $d[v] > d[u] + w(u,v)$:
       - $d[v] = d[u] + w(u,v)$
       - Update $v$ in $Q$

Time complexity: $O((|V| + |E|)\log|V|)$ with binary heap

### Embedding-Aware Dijkstra's Algorithm

This variant leverages geometric information from embeddings to improve path finding.

#### Mathematical Foundation

Given an embedding $f: V \rightarrow X$ where $X$ is the embedding space:

1. **Edge Weights**: Combine graph and geometric distances:
   $$w'(u,v) = \alpha w(u,v) + (1-\alpha)d_X(f(u), f(v))$$
   where $d_X$ is the distance in embedding space and $\alpha \in [0,1]$

2. **Heuristic Function**: Use embedding distance as heuristic:
   $$h(v,t) = \beta d_X(f(v), f(t))$$
   where $t$ is the target vertex and $\beta$ is a scaling factor

#### Algorithm Description

1. Initialize as in traditional Dijkstra
2. Modify priority function to include heuristic:
   ```
   Priority = d[v] + h(v,t)
   ```
3. Use modified edge weights $w'(u,v)$

#### Theoretical Properties

1. **Admissibility**:
   - If $\beta d_X(f(u), f(v)) \leq d_G(u,v)$ for all $u,v$
   - Then algorithm finds optimal paths

2. **Consistency**:
   - If $h(u,t) \leq w'(u,v) + h(v,t)$ for all edges $(u,v)$
   - Then algorithm explores minimum number of nodes

## Distance Computation

### Embedding Space Distances

For vertices $u,v \in V$, compute $d_X(f(u), f(v))$ based on embedding type:

1. **Euclidean**:

   $d_E(x,y) = \sqrt{\sum_{i=1}^d (x_i - y_i)^2}$

2. **Spherical**:

   $d_S(x,y) = \arccos\left(\frac{\langle x,y \rangle}{\|x\|\|y\|}\right)$

3. **Hyperbolic** (Poincaré model):

   $d_H(x,y) = \text{arcosh}\left(1 + 2\frac{\|x-y\|^2}{(1-\|x\|^2)(1-\|y\|^2)}\right)$

### Graph Distances

Multiple approaches for computing graph distances:

1. **All-Pairs Shortest Paths**:
   - Floyd-Warshall: $O(|V|^3)$
   - Johnson's: $O(|V||E| + |V|^2\log|V|)$

2. **Approximate Distances**:
   - Landmark-based: Select $k$ landmarks
   - Estimate $d(u,v) \approx \min_{\ell \in L} (d(u,\ell) + d(\ell,v))$
   - Complexity: $O(k|V|)$ after preprocessing

## Performance Analysis

### Time Complexity

| Algorithm | Traditional | Embedding-Aware |
|-----------|-------------|----------------|
| Preprocessing | O(1) | O(V d) |
| Query | O((V+E)\log V) | O((V+E)\log V) |
| Distance Computation | O(1) | O(d) |

where $d$ is embedding dimension

### Space Complexity

| Component | Memory Usage |
|-----------|-------------|
| Graph | O(V + E) |
| Embeddings | O(V d) |
| Priority Queue | O(V) |

### Practical Considerations

1. **Embedding Dimension**:
   - Higher $d$ → Better accuracy
   - Higher $d$ → More computation
   - Typical range: 8-128

2. **Parameter Tuning**:
   - $\alpha$: Balance between graph and geometric information
   - $\beta$: Control heuristic strength
   - Cross-validation on path lengths

3. **Optimization**:
   - Cache frequently computed distances
   - Use approximate nearest neighbors
   - Parallel computation of distances
