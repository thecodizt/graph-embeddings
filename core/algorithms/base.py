"""Base classes for graph algorithms."""
from abc import ABC, abstractmethod
import time
from typing import Any, Dict, List, Optional, Set, Tuple

class GraphAlgorithm(ABC):
    """Base class for all graph algorithms."""
    
    def __init__(self, graph):
        """Initialize algorithm with a graph."""
        self.graph = graph
        self.execution_time = 0.0
        
    def measure_execution_time(func):
        """Decorator to measure execution time of algorithms."""
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = func(self, *args, **kwargs)
            self.execution_time = time.time() - start_time
            return result
        return wrapper
    
    @abstractmethod
    def run(self, *args, **kwargs):
        """Run the algorithm."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the algorithm execution."""
        return {
            "execution_time": self.execution_time,
        }

class EmbeddingAwareAlgorithm(GraphAlgorithm):
    """Base class for embedding-aware algorithms."""
    
    def __init__(self, graph, embedder):
        """Initialize algorithm with a graph and embedder."""
        super().__init__(graph)
        self.embedder = embedder
