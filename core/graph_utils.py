import networkx as nx
import random
import json
from typing import Dict, Any, Tuple

def generate_graph(config: Dict[str, Any]) -> Tuple[nx.Graph, Dict[str, Any]]:
    """
    Generate a graph based on the provided configuration.
    
    Args:
        config: Dictionary containing graph configuration parameters
        
    Returns:
        tuple: (Generated graph, Statistics about the graph)
    """
    n_nodes = config['n_nodes']
    avg_degree = config['avg_degree']
    is_cyclic = config['is_cyclic']
    is_directed = config['is_directed']
    graph_type = config['graph_type']
    allow_unconnected = config.get('allow_unconnected', True)
    
    if is_directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
        
    # Add nodes
    G.add_nodes_from(range(n_nodes))
    
    if graph_type == 'random':
        # Calculate probability based on average degree
        p = avg_degree / (n_nodes - 1)
        G = nx.erdos_renyi_graph(n_nodes, p, directed=is_directed)
    
    elif graph_type == 'scale_free':
        # Generate scale-free graph
        G = nx.barabasi_albert_graph(n_nodes, int(avg_degree/2))
        if is_directed:
            G = G.to_directed()
    
    elif graph_type == 'small_world':
        # Generate small-world graph
        k = int(avg_degree)
        p = config.get('rewiring_prob', 0.3)  # Get rewiring probability from config or use default
        G = nx.watts_strogatz_graph(n_nodes, k, p)
        if is_directed:
            G = G.to_directed()
            
    # Ensure the graph is connected if required
    if not allow_unconnected:
        ensure_connected(G)
            
    # Ensure cyclic property if requested
    if is_cyclic and not has_cycle(G):
        add_cycle(G)
    elif not is_cyclic and has_cycle(G):
        remove_cycles(G)
    
    # Calculate statistics
    stats = calculate_graph_stats(G)
    
    return G, stats

def has_cycle(G: nx.Graph) -> bool:
    """Check if graph has a cycle."""
    try:
        nx.find_cycle(G)
        return True
    except nx.NetworkXNoCycle:
        return False

def add_cycle(G: nx.Graph) -> None:
    """Add a cycle to the graph if it doesn't have one."""
    if not has_cycle(G):
        nodes = list(G.nodes())
        for i in range(len(nodes)-1):
            G.add_edge(nodes[i], nodes[i+1])
        G.add_edge(nodes[-1], nodes[0])

def remove_cycles(G: nx.Graph) -> None:
    """Remove cycles from the graph by removing edges."""
    while has_cycle(G):
        cycle = nx.find_cycle(G)
        # Remove random edge from cycle
        edge = random.choice(cycle)
        G.remove_edge(*edge)

def ensure_connected(G: nx.Graph) -> None:
    """Ensure the graph is connected by adding edges between components."""
    if nx.is_directed(G):
        components = list(nx.weakly_connected_components(G))
    else:
        components = list(nx.connected_components(G))
    
    if len(components) > 1:
        # Connect each component to the next one
        for i in range(len(components) - 1):
            # Get a random node from each component
            node1 = random.choice(list(components[i]))
            node2 = random.choice(list(components[i + 1]))
            G.add_edge(node1, node2)

def calculate_graph_stats(G: nx.Graph) -> Dict[str, Any]:
    """Calculate various statistics about the graph."""
    stats = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'average_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
        'density': nx.density(G),
        'is_connected': nx.is_connected(G) if not G.is_directed() else nx.is_weakly_connected(G),
        'has_cycles': has_cycle(G),
        'clustering_coefficient': nx.average_clustering(G),
    }
    
    try:
        stats['average_shortest_path'] = nx.average_shortest_path_length(G)
    except:
        stats['average_shortest_path'] = float('inf')  # For disconnected graphs
        
    return stats

def save_graph_json(G: nx.Graph, filename: str) -> None:
    """Save graph in NetworkX JSON format."""
    data = nx.node_link_data(G)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
