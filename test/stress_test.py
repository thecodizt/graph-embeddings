import networkx as nx
import numpy as np
import time
import random
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Import core algorithms
from core.algorithms.shortest_path import (
    DijkstraTraditional, DijkstraEmbedding,
    BellmanFordTraditional, BellmanFordEmbedding,
    FloydWarshallTraditional, FloydWarshallEmbedding,
    AStarTraditional, AStarEmbedding
)
from core.algorithms.clustering import (
    KMeansTraditional, KMeansEmbedding,
    SpectralClusteringTraditional, SpectralClusteringEmbedding
)
from core.algorithms.ranking import (
    PageRankTraditional, PageRankEmbedding,
    HITSTraditional, HITSEmbedding
)
from core.embeddings.random import RandomEmbedder

def generate_random_graph(n_nodes, edge_probability=0.1):
    """Generate a random graph with weighted edges."""
    G = nx.erdos_renyi_graph(n=n_nodes, p=edge_probability)
    
    # Add random weights to edges
    for (u, v) in G.edges():
        G[u][v]['weight'] = np.random.uniform(0.1, 10.0)
    
    return G

def run_algorithm_with_timing(algo_instance, graph_type, size, **kwargs):
    """Run algorithm and measure execution time with proper parameters based on algorithm type."""
    start_time = time.time()
    try:
        if isinstance(algo_instance, (DijkstraTraditional, DijkstraEmbedding,
                                    BellmanFordTraditional, BellmanFordEmbedding,
                                    AStarTraditional, AStarEmbedding)):
            # Shortest path algorithms
            source = kwargs.get('source', 0)
            target = kwargs.get('target', size-1)
            result = algo_instance.run(source, target)
            success = True
            metric = result.get('distance', float('inf'))
        
        elif isinstance(algo_instance, (FloydWarshallTraditional, FloydWarshallEmbedding)):
            # Floyd-Warshall computes all pairs
            result = algo_instance.run()
            success = True
            # Calculate average path length as metric
            distances = [result[s, t]['distance'] for s, t in result.keys()]
            metric = sum(distances) / len(distances) if distances else float('inf')
        
        elif isinstance(algo_instance, (KMeansTraditional, KMeansEmbedding,
                                      SpectralClusteringTraditional, SpectralClusteringEmbedding)):
            # Clustering algorithms
            n_clusters = min(int(size/10), 10)  # Reasonable number of clusters
            result = algo_instance.run(k=n_clusters)
            success = True
            metric = len(set(result.values())) if result else 0
        
        elif isinstance(algo_instance, (PageRankTraditional, PageRankEmbedding)):
            # PageRank algorithms
            result = algo_instance.run()
            success = True
            metric = np.mean(list(result.values())) if result else 0
        
        elif isinstance(algo_instance, (HITSTraditional, HITSEmbedding)):
            # HITS algorithms
            hub_scores, auth_scores = algo_instance.run()
            success = True
            # Use average of hub and authority scores as metric
            hub_mean = np.mean(list(hub_scores.values())) if hub_scores else 0
            auth_mean = np.mean(list(auth_scores.values())) if auth_scores else 0
            metric = (hub_mean + auth_mean) / 2
        
        else:
            raise ValueError(f"Unknown algorithm type: {type(algo_instance)}")
            
    except Exception as e:
        print(f"Error running {algo_instance.__class__.__name__}: {str(e)}")
        success = False
        metric = float('inf') if isinstance(algo_instance, (DijkstraTraditional, DijkstraEmbedding,
                                                          BellmanFordTraditional, BellmanFordEmbedding,
                                                          FloydWarshallTraditional, FloydWarshallEmbedding,
                                                          AStarTraditional, AStarEmbedding)) else 0
    
    execution_time = time.time() - start_time
    return {
        'execution_time': execution_time,
        'success': success,
        'metric': metric,
        'algorithm': algo_instance.__class__.__name__,
        'graph_type': graph_type,
        'size': size
    }

def run_stress_test(sizes=[100, 500, 1000], num_trials=3, edge_probability=0.1):
    """Run stress test for different graph sizes."""
    # Create results directory if it doesn't exist
    results_dir = Path("test/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create CSV file with headers
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_file = results_dir / f"stress_test_results_{timestamp}.csv"
    headers = ['execution_time', 'success', 'metric', 'algorithm', 'graph_type', 'size']
    with open(results_file, 'w') as f:
        pd.DataFrame(columns=headers).to_csv(f, index=False)
    
    algorithm_groups = {
        'shortest_path': {
            'directed': False,
            'algorithms': {
                'dijkstra': (DijkstraTraditional, DijkstraEmbedding),
                'bellman_ford': (BellmanFordTraditional, BellmanFordEmbedding),
                'floyd_warshall': (FloydWarshallTraditional, FloydWarshallEmbedding),
                'astar': (AStarTraditional, AStarEmbedding)
            }
        },
        'clustering': {
            'directed': False,
            'algorithms': {
                'kmeans': (KMeansTraditional, KMeansEmbedding),
                'spectral': (SpectralClusteringTraditional, SpectralClusteringEmbedding)
            }
        },
        'ranking': {
            'directed': True,
            'algorithms': {
                'pagerank': (PageRankTraditional, PageRankEmbedding),
                'hits': (HITSTraditional, HITSEmbedding)
            }
        }
    }

    for size in tqdm(sizes, desc="Testing graph sizes"):
        for group_name, group_info in algorithm_groups.items():
            # Generate appropriate graph type
            if group_info['directed']:
                graph = nx.erdos_renyi_graph(size, edge_probability, directed=True)
                graph_type = 'directed'
                
                # Ensure strong connectivity for directed graph
                if not nx.is_strongly_connected(graph):
                    largest = max(nx.strongly_connected_components(graph), key=len)
                    graph = nx.DiGraph(graph.subgraph(largest))
            else:
                graph = nx.erdos_renyi_graph(size, edge_probability, directed=False)
                graph_type = 'undirected'
                
                # Ensure connectivity for undirected graph
                if not nx.is_connected(graph):
                    largest = max(nx.connected_components(graph), key=len)
                    graph = nx.Graph(graph.subgraph(largest))
            
            # Add random weights
            for u, v in graph.edges():
                graph[u][v]['weight'] = random.uniform(0.1, 1.0)
            
            # Create embedder
            embedder = RandomEmbedder(graph, dim=64)
            
            # Run algorithms in this group
            for algo_name, (trad_class, emb_class) in group_info['algorithms'].items():
                for trial in range(num_trials):
                    # Initialize algorithms
                    trad_algo = trad_class(graph)
                    emb_algo = emb_class(graph, embedder)
                    
                    # Random source and target for path algorithms
                    kwargs = {}
                    if group_name == 'shortest_path' and algo_name != 'floyd_warshall':
                        nodes = list(graph.nodes())
                        kwargs = {
                            'source': random.choice(nodes),
                            'target': random.choice(nodes)
                        }
                    
                    # Run and time both traditional and embedding versions
                    trad_result = run_algorithm_with_timing(trad_algo, graph_type, size, **kwargs)
                    emb_result = run_algorithm_with_timing(emb_algo, graph_type, size, **kwargs)
                    
                    # Save results incrementally
                    pd.DataFrame([trad_result, emb_result]).to_csv(results_file, mode='a', header=False, index=False)
    
    # Read and return the complete results
    return pd.read_csv(results_file)

def main():
    """Main function to run stress tests."""
    print("Starting stress tests...")
    sizes = [i for i in range(100, 1001, 50)]  # Graph sizes to test
    num_trials = 3  # Number of trials per configuration
    
    results = run_stress_test(sizes=sizes, num_trials=num_trials)
    
    print("\nTest completed. Summary of results:")
    print("\nMean execution times by algorithm and graph size:")
    summary = results.groupby(['algorithm', 'size'])['execution_time'].agg(['mean', 'std']).round(4)
    print(summary)
    
    print("\nSuccess rates by algorithm:")
    success_rates = results.groupby('algorithm')['success'].mean() * 100
    print(success_rates.round(2))

if __name__ == "__main__":
    main()
