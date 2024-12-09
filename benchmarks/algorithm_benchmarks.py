"""Benchmark suite comparing traditional vs embedding-based algorithms."""
import time
import psutil
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Callable
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import sys

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from core.algorithms.embedding_algorithms import (
    ApproximatePersonalizedPageRank,
    FastNodeClassification,
    EfficientCommunityDetection,
    LinkPrediction
)
from core.algorithms.traditional_algorithms import (
    TraditionalPersonalizedPageRank,
    TraditionalNodeClassification,
    TraditionalCommunityDetection,
    TraditionalLinkPrediction
)


class GraphGenerator:
    """Utility class for generating different types of graphs for benchmarking."""
    
    @staticmethod
    def generate_erdos_renyi(n: int, p: float) -> nx.Graph:
        """Generate Erdős-Rényi random graph."""
        return nx.erdos_renyi_graph(n, p)
    
    @staticmethod
    def generate_barabasi_albert(n: int, m: int) -> nx.Graph:
        """Generate Barabási-Albert preferential attachment graph."""
        return nx.barabasi_albert_graph(n, m)
    
    @staticmethod
    def generate_watts_strogatz(n: int, k: int, p: float) -> nx.Graph:
        """Generate Watts-Strogatz small-world graph."""
        return nx.watts_strogatz_graph(n, k, p)
    
    @staticmethod
    def generate_powerlaw_cluster(n: int, m: int, p: float) -> nx.Graph:
        """Generate Holme-Kim powerlaw clustered graph."""
        return nx.powerlaw_cluster_graph(n, m, p)


class MockEmbedder:
    """Mock embedder for testing embedding-based algorithms."""
    
    def __init__(self, graph: nx.Graph, dim: int = 128):
        """Initialize with random embeddings."""
        self.embeddings = {
            node: np.random.normal(0, 1, dim) 
            for node in graph.nodes()
        }
    
    def get_embedding(self, node: int) -> np.ndarray:
        """Get embedding for a node."""
        return self.embeddings[node]


class BenchmarkSuite:
    """Suite for benchmarking traditional vs embedding-based algorithms."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """Initialize benchmark suite."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results_file = self.output_dir / "benchmark_results.csv"
        
        # Create empty results file with headers if it doesn't exist
        if not self.results_file.exists():
            pd.DataFrame(columns=[
                "graph_type", "graph_size", "algorithm",
                "traditional_time", "embedding_time",
                "traditional_memory", "embedding_memory",
                "time_speedup", "memory_reduction"
            ]).to_csv(self.results_file, index=False)
    
    def _measure_memory(self, func: Callable) -> float:
        """Measure peak memory usage of a function in MB."""
        process = psutil.Process()
        baseline = process.memory_info().rss / 1024 / 1024  # Convert to MB
        func()
        peak = process.memory_info().rss / 1024 / 1024  # Convert to MB
        return peak - baseline
    
    def benchmark_algorithm(self, 
                          graph: nx.Graph,
                          traditional_algo: Any,
                          embedding_algo: Any,
                          graph_type: str,
                          graph_size: int,
                          **kwargs) -> Dict:
        """Benchmark a pair of algorithms."""
        # Traditional algorithm
        start_time = time.time()
        trad_memory = self._measure_memory(
            lambda: traditional_algo.run(**kwargs)
        )
        trad_time = time.time() - start_time
        
        # Embedding-based algorithm
        start_time = time.time()
        emb_memory = self._measure_memory(
            lambda: embedding_algo.run(**kwargs)
        )
        emb_time = time.time() - start_time
        
        return {
            "graph_type": graph_type,
            "graph_size": graph_size,
            "traditional_time": trad_time,
            "embedding_time": emb_time,
            "traditional_memory": trad_memory,
            "embedding_memory": emb_memory,
            "time_speedup": trad_time / emb_time if emb_time > 0 else float('inf'),
            "memory_reduction": trad_memory / emb_memory if emb_memory > 0 else float('inf')
        }
    
    def append_result(self, result: Dict, algorithm: str):
        """Append a single result to the CSV file."""
        result_df = pd.DataFrame([{
            'graph_type': result['graph_type'],
            'graph_size': result['graph_size'],
            'algorithm': algorithm,
            'traditional_time': result['traditional_time'],
            'embedding_time': result['embedding_time'],
            'traditional_memory': result['traditional_memory'],
            'embedding_memory': result['embedding_memory'],
            'time_speedup': result['traditional_time'] / result['embedding_time'] if result['embedding_time'] > 0 else float('inf'),
            'memory_reduction': result['traditional_memory'] / result['embedding_memory'] if result['embedding_memory'] > 0 else float('inf')
        }])
        
        result_df.to_csv(self.results_file, mode='a', header=False, index=False)
    
    def run_benchmarks(self, sizes: List[int]):
        """Run comprehensive benchmarks on different graph types and sizes."""
        graph_generators = {
            "erdos_renyi": lambda n: GraphGenerator.generate_erdos_renyi(n, 0.1),
            "barabasi_albert": lambda n: GraphGenerator.generate_barabasi_albert(n, 3),
            "watts_strogatz": lambda n: GraphGenerator.generate_watts_strogatz(n, 6, 0.1),
            "powerlaw_cluster": lambda n: GraphGenerator.generate_powerlaw_cluster(n, 3, 0.1)
        }
        
        # Calculate total iterations for progress bar
        total_iterations = len(sizes) * len(graph_generators) * 4  # 4 algorithms
        
        with tqdm(total=total_iterations, desc="Running benchmarks") as pbar:
            for size in sizes:
                for graph_type, generator in graph_generators.items():
                    graph = generator(size)
                    embedder = MockEmbedder(graph)
                    
                    # 1. PageRank
                    source_node = list(graph.nodes())[0]
                    result = self.benchmark_algorithm(
                        graph=graph,
                        traditional_algo=TraditionalPersonalizedPageRank(graph),
                        embedding_algo=ApproximatePersonalizedPageRank(graph, embedder),
                        graph_type=graph_type,
                        graph_size=size,
                        source_node=source_node
                    )
                    self.append_result(result, "pagerank")
                    pbar.update(1)
                    
                    # 2. Node Classification
                    labeled_nodes = {
                        node: f"class_{node % 3}"
                        for node in list(graph.nodes())[:size//10]
                    }
                    result = self.benchmark_algorithm(
                        graph=graph,
                        traditional_algo=TraditionalNodeClassification(graph),
                        embedding_algo=FastNodeClassification(graph, embedder),
                        graph_type=graph_type,
                        graph_size=size,
                        labeled_nodes=labeled_nodes
                    )
                    self.append_result(result, "node_classification")
                    pbar.update(1)
                    
                    # 3. Community Detection
                    result = self.benchmark_algorithm(
                        graph=graph,
                        traditional_algo=TraditionalCommunityDetection(graph),
                        embedding_algo=EfficientCommunityDetection(graph, embedder),
                        graph_type=graph_type,
                        graph_size=size
                    )
                    self.append_result(result, "community_detection")
                    pbar.update(1)
                    
                    # 4. Link Prediction
                    result = self.benchmark_algorithm(
                        graph=graph,
                        traditional_algo=TraditionalLinkPrediction(graph),
                        embedding_algo=LinkPrediction(graph, embedder),
                        graph_type=graph_type,
                        graph_size=size,
                        source_node=source_node
                    )
                    self.append_result(result, "link_prediction")
                    pbar.update(1)
        
        # Generate summary statistics
        df = pd.read_csv(self.results_file)
        summary = df.groupby("algorithm").agg({
            "time_speedup": ["mean", "std", "min", "max"],
            "memory_reduction": ["mean", "std", "min", "max"]
        })
        summary.to_csv(self.output_dir / "summary_statistics.csv")
        
        print("\nBenchmarking complete! Run 'python benchmarks/visualize_results.py' to generate interactive visualizations.")


if __name__ == "__main__":
    # Run benchmarks on different graph sizes
    sizes = [100, 500, 1000, 5000, 10000]
    suite = BenchmarkSuite()
    suite.run_benchmarks(sizes)
