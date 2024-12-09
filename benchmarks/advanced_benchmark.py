"""Benchmark advanced embedding-based algorithms against traditional approaches."""
import os
import sys
import time
from typing import Dict, List

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt
    from sklearn.neighbors import LocalOutlierFactor
    from networkx.algorithms.similarity import graph_edit_distance
except ImportError as e:
    print(f"Please install required packages: {e}")
    print("Run: pip install numpy networkx matplotlib scikit-learn scipy")
    sys.exit(1)

from core.embeddings.euclidean import EuclideanEmbedding
from core.algorithms.advanced_algorithms import AnomalyDetection, SubgraphMatching, GraphEditDistance

def generate_test_graph(n: int, p: float = 0.1) -> nx.Graph:
    """Generate random graph with planted anomalies and patterns."""
    G = nx.erdos_renyi_graph(n, p)
    
    # Add some anomalous structures
    anomaly_nodes = np.random.choice(n, size=int(0.05 * n), replace=False)
    for node in anomaly_nodes:
        # Create unusual degree patterns
        neighbors = list(G.neighbors(node))
        G.remove_edges_from([(node, neigh) for neigh in neighbors])
        new_neighbors = np.random.choice(list(range(n)), size=int(n * 0.5), replace=False)
        G.add_edges_from([(node, neigh) for neigh in new_neighbors if neigh != node])
    
    return G

def traditional_anomaly_detection(G: nx.Graph, contamination: float = 0.1) -> Dict[int, float]:
    """Traditional anomaly detection using graph statistics."""
    features = []
    nodes = list(G.nodes())
    
    for node in nodes:
        # Compute various node statistics
        degree = G.degree(node)
        clustering = nx.clustering(G, node)
        centrality = nx.betweenness_centrality(G, k=min(10, len(nodes)))[node]
        features.append([degree, clustering, centrality])
    
    # Use LOF on graph statistics
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    scores = -lof.fit_predict(features)
    return dict(zip(nodes, scores))

def traditional_subgraph_matching(G: nx.Graph, pattern: nx.Graph) -> List[List[int]]:
    """Traditional subgraph isomorphism."""
    import networkx.algorithms.isomorphism as iso
    matcher = iso.GraphMatcher(G, pattern)
    return [list(match.keys()) for match in matcher.subgraph_isomorphisms_iter()]

def benchmark_algorithms(sizes: List[int], trials: int = 3):
    """Benchmark algorithms across different graph sizes."""
    results = {
        'anomaly': {'traditional': [], 'embedding': []},
        'subgraph': {'traditional': [], 'embedding': []},
        'edit': {'traditional': [], 'embedding': []}
    }
    
    for n in sizes:
        print(f"\nBenchmarking graphs of size {n}")
        trad_times = {'anomaly': [], 'subgraph': [], 'edit': []}
        emb_times = {'anomaly': [], 'subgraph': [], 'edit': []}
        
        for trial in range(trials):
            # Generate test graphs
            G1 = generate_test_graph(n)
            G2 = generate_test_graph(n)
            pattern = generate_test_graph(min(5, n//10))
            
            # Create embeddings
            embedder = EuclideanEmbedding(dim=8)
            embedder.train(G1)
            
            # Anomaly Detection
            start = time.time()
            _ = traditional_anomaly_detection(G1)
            trad_times['anomaly'].append(time.time() - start)
            
            detector = AnomalyDetection(G1, embedder)
            start = time.time()
            _ = detector.run()
            emb_times['anomaly'].append(time.time() - start)
            
            # Subgraph Matching
            if n <= 100:  # Traditional method is too slow for larger graphs
                start = time.time()
                _ = traditional_subgraph_matching(G1, pattern)
                trad_times['subgraph'].append(time.time() - start)
            else:
                trad_times['subgraph'].append(float('inf'))
            
            matcher = SubgraphMatching(G1, embedder)
            start = time.time()
            _ = matcher.run(pattern)
            emb_times['subgraph'].append(time.time() - start)
            
            # Graph Edit Distance
            if n <= 50:  # Traditional method is too slow for larger graphs
                start = time.time()
                _ = graph_edit_distance(G1, G2)
                trad_times['edit'].append(time.time() - start)
            else:
                trad_times['edit'].append(float('inf'))
            
            edit_dist = GraphEditDistance(G1, embedder)
            start = time.time()
            _ = edit_dist.run(G2)
            emb_times['edit'].append(time.time() - start)
        
        # Store average times
        for algo in ['anomaly', 'subgraph', 'edit']:
            results[algo]['traditional'].append(np.mean(trad_times[algo]))
            results[algo]['embedding'].append(np.mean(emb_times[algo]))
    
    return results

def plot_results(sizes: List[int], results: Dict):
    """Plot benchmark results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    algorithms = ['anomaly', 'subgraph', 'edit']
    titles = ['Anomaly Detection', 'Subgraph Matching', 'Graph Edit Distance']
    
    for ax, algo, title in zip(axes, algorithms, titles):
        trad_times = results[algo]['traditional']
        emb_times = results[algo]['embedding']
        
        ax.plot(sizes, trad_times, 'o-', label='Traditional')
        ax.plot(sizes, emb_times, 'o-', label='Embedding')
        ax.set_title(title)
        ax.set_xlabel('Graph Size (nodes)')
        ax.set_ylabel('Time (seconds)')
        ax.set_yscale('log')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    plt.close()

if __name__ == '__main__':
    # Use smaller graph sizes and fewer trials for faster testing
    sizes = [10, 20, 50, 100]
    print("\nRunning benchmark with graph sizes:", sizes)
    results = benchmark_algorithms(sizes, trials=2)
    plot_results(sizes, results)
    print("\nBenchmark complete! Results saved to benchmark_results.png")
