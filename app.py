import streamlit as st
import networkx as nx
import json
import tempfile
from pathlib import Path
import plotly.graph_objects as go
import numpy as np
from core.graph_utils import generate_graph, save_graph_json
from core.embeddings.euclidean import EuclideanEmbedding
from core.embeddings.hyperbolic import HyperbolicEmbedding
from core.embeddings.spherical import SphericalEmbedding
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
from core.algorithms.community_detection import LabelPropagationTraditional, LabelPropagationEmbedding
from core.algorithms.node_ranking import PageRankTraditional, PageRankEmbedding


def create_plotly_graph(G, layout_type, node_size_scale=3, edge_width=0.5):
    """Create a Plotly figure for the graph visualization."""
    
    # Get the layout positions
    if layout_type == "spring":
        pos = nx.spring_layout(G)
    elif layout_type == "circular":
        pos = nx.circular_layout(G)
    elif layout_type == "shell":
        pos = nx.shell_layout(G)
    elif layout_type == "kamada_kawai":
        try:
            pos = nx.kamada_kawai_layout(G)
        except:
            st.warning("Kamada-Kawai layout failed. Falling back to spring layout.")
            pos = nx.spring_layout(G)
    else:  # random
        pos = nx.random_layout(G)
    
    # Create edge trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=edge_width, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Create node trace
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    # Calculate node sizes based on degree
    node_degrees = [G.degree(node) for node in G.nodes()]
    node_sizes = [min(20 + node_size_scale * deg, 50) for deg in node_degrees]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=node_sizes,
            color=node_degrees,
            colorbar=dict(
                thickness=15,
                title='Node Degree',
                xanchor='left',
                titleside='right'
            )
        ),
        text=[f'Node {node}<br>Degree: {G.degree(node)}' for node in G.nodes()],
    )

    # Create the figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title='Graph Visualization',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    
    return fig


def suggest_embedding_type(graph):
    """Suggest the most suitable embedding type based on graph properties."""
    # Convert to undirected for analysis if needed
    G = graph.to_undirected() if graph.is_directed() else graph
    
    # Get graph properties
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = nx.density(G)
    avg_clustering = nx.average_clustering(G)
    
    # Check connectivity
    components = list(nx.connected_components(G))
    n_components = len(components)
    largest_component_size = len(max(components, key=len))
    component_ratio = largest_component_size / n_nodes
    
    # Analyze degree distribution
    degrees = [d for _, d in G.degree()]
    max_degree = max(degrees) if degrees else 0
    avg_degree = sum(degrees) / len(degrees) if degrees else 0
    degree_variance = sum((d - avg_degree) ** 2 for d in degrees) / len(degrees) if degrees else 0
    
    # Decision logic
    if n_components > 1 and component_ratio < 0.8:
        return "hyperbolic", "Graph has multiple significant components. Hyperbolic space can better preserve hierarchical structure and handle disconnected components."
    
    if avg_clustering > 0.3 and density < 0.3:
        return "hyperbolic", "High clustering with low density suggests hierarchical community structure, well-suited for hyperbolic embedding."
    
    if max_degree > 5 * avg_degree:
        return "hyperbolic", "Presence of hub nodes (high degree variance) suggests scale-free properties, better represented in hyperbolic space."
    
    if density > 0.5 or (avg_clustering < 0.1 and density > 0.3):
        return "euclidean", "High density or low clustering suggests a more uniform structure, well-suited for Euclidean space."
    
    if avg_clustering > 0.5 and degree_variance < avg_degree:
        return "spherical", "High clustering with uniform degree distribution suggests spherical topology."
    
    # Default to Euclidean as it's most general
    return "euclidean", "Default choice for general graph structure. Use this as a starting point and experiment with other types if needed."


def get_graph_properties(G):
    """Get graph properties handling both directed and undirected graphs."""
    properties = {
        "Nodes": G.number_of_nodes(),
        "Edges": G.number_of_edges(),
        "Density": round(nx.density(G), 3),
        "Is directed": G.is_directed()
    }
    
    try:
        if G.is_directed():
            properties["Is weakly connected"] = nx.is_weakly_connected(G)
            properties["Number of weakly connected components"] = nx.number_weakly_connected_components(G)
            properties["Average clustering"] = round(nx.average_clustering(G.to_undirected()), 3)
        else:
            properties["Is connected"] = nx.is_connected(G)
            properties["Number of connected components"] = nx.number_connected_components(G)
            properties["Average clustering"] = round(nx.average_clustering(G), 3)
    except Exception as e:
        st.warning(f"Some graph metrics could not be computed: {str(e)}")
    
    return properties


# Set page title and layout
st.set_page_config(
    page_title="Graph Embedding & Analysis Platform",
    page_icon="🔍",
    layout="wide"
)

st.title("Graph Embedding & Analysis Platform")
st.markdown("""
Analyze and visualize graphs using advanced embedding techniques and graph algorithms:
- **Geometric Embeddings**: Euclidean, Hyperbolic (Poincaré disk), and Spherical spaces
- **Graph Analysis**: Shortest paths, community detection, and node ranking
- **Performance Comparison**: Traditional vs. embedding-aware algorithms
""")

# Initialize session state
if 'graph' not in st.session_state:
    st.session_state['graph'] = None
if 'embedder' not in st.session_state:
    st.session_state['embedder'] = None
if 'pos' not in st.session_state:
    st.session_state['pos'] = None

# Sidebar configuration
with st.sidebar:
    st.title("Graph Embedding & Analysis Platform")
    st.markdown("""
A platform for analyzing graphs using various embedding techniques.
""")
    
    # Add documentation link
    # st.markdown("---")
    # st.markdown("[📚 View Documentation](http://localhost:8502)")
    
    # File uploader at the top of sidebar
    uploaded_file = st.file_uploader("Upload NetworkX JSON", type=['json'])
    
    if uploaded_file is not None:
        try:
            # Read and parse JSON
            graph_data = json.load(uploaded_file)
            G = nx.node_link_graph(graph_data)
            st.session_state['graph'] = G
            st.success("Graph loaded successfully!")
            
            # Display basic graph info using the new function
            properties = get_graph_properties(G)
            for key, value in properties.items():
                st.write(f"{key}: {value}")
                
        except Exception as e:
            st.error(f"Error loading graph: {str(e)}")
            st.session_state['graph'] = None
    
    st.header("Or Generate Graph")
    
    # Basic configurations
    n_nodes = st.slider("Number of Nodes", min_value=5, max_value=2000, value=50)
    avg_degree = st.slider("Average Degree", min_value=2, max_value=min(50, n_nodes-1), value=4)
    
    # Graph type selection
    graph_type = st.selectbox(
        "Graph Type",
        ["random", "scale_free", "small_world"],
        help="Random: Erdős-Rényi random graph\nScale-free: Barabási-Albert model\nSmall-world: Watts-Strogatz model"
    )
    
    # Advanced configurations
    with st.expander("Advanced Configuration"):
        is_directed = st.checkbox("Directed Graph", value=False)
        is_cyclic = st.checkbox("Ensure Cyclic", value=True)
        allow_unconnected = st.checkbox("Allow Unconnected Nodes", value=True,
                                    help="If unchecked, the generator will ensure all nodes are connected")
        
        if graph_type == "small_world":
            rewiring_prob = st.slider("Rewiring Probability", min_value=0.0, max_value=1.0, value=0.3, step=0.1,
                                    help="Probability of rewiring edges in the Watts-Strogatz model")
        else:
            rewiring_prob = 0.3
        
        # Layout options
        layout_type = st.selectbox(
            "Layout Algorithm",
            ["spring", "kamada_kawai", "circular", "random", "shell"],
            help="Choose the layout algorithm for graph visualization"
        )
        
        # Node size configuration
        node_size_scale = st.slider("Node Size Scale", min_value=1, max_value=10, value=3,
                                help="Scale factor for node sizes based on degree")
        
        # Edge width configuration
        edge_width = st.slider("Edge Width", min_value=0.1, max_value=3.0, value=0.5, step=0.1,
                            help="Width of the edges in the visualization")
    
    # Create configuration dictionary
    config = {
        'n_nodes': n_nodes,
        'avg_degree': avg_degree,
        'is_cyclic': is_cyclic,
        'is_directed': is_directed,
        'graph_type': graph_type,
        'rewiring_prob': rewiring_prob,
        'allow_unconnected': allow_unconnected
    }
    
    # Generate graph button
    if st.button("Generate Graph"):
        # Generate the graph
        G, stats = generate_graph(config)
        
        # Store the graph in session state
        st.session_state['graph'] = G
        st.session_state['stats'] = stats
        
        # Clear any existing embeddings when new graph is generated
        if 'embedder' in st.session_state:
            del st.session_state['embedder']
        
        st.success("Graph generated successfully!")

# Main area with tabs
if st.session_state['graph'] is not None:
    tab1, tab2, tab3 = st.tabs(["Graph Visualization", "Embeddings", "Algorithms"])
    
    with tab1:
        if st.session_state['graph'] is not None:
            fig = create_plotly_graph(st.session_state['graph'], layout_type)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please upload or generate a graph first.")
    
    with tab2:
        if st.session_state['graph'] is not None:
            # Embedding configuration
            st.header("Embedding Configuration")
            
            # Get embedding suggestion
            suggested_embedding = suggest_embedding_type(st.session_state['graph'])
            
            embedding_type = suggested_embedding[0] if suggested_embedding else None
            embedding_description = suggested_embedding[1] if suggested_embedding else ""
            
            selected_embedding = st.selectbox(
                "Embedding Type",
                ["euclidean", "hyperbolic", "spherical"],
                index=0 if embedding_type is None else ["euclidean", "hyperbolic", "spherical"].index(embedding_type),
                help=f"Suggested embedding type: {embedding_type}\n{embedding_description}"
            )
            
            embedding_dim = st.slider("Embedding Dimension", min_value=2, max_value=128, value=64)
            
            if st.button("Generate Embeddings"):
                with st.spinner("Generating embeddings..."):
                    if selected_embedding == "euclidean":
                        embedder = EuclideanEmbedding(dim=embedding_dim)
                    elif selected_embedding == "hyperbolic":
                        embedder = HyperbolicEmbedding(dim=embedding_dim)
                    else:  # spherical
                        embedder = SphericalEmbedding(dim=embedding_dim)
                    
                    # Train the embedder on the graph
                    embedder.train(st.session_state['graph'])
                    
                    st.session_state['embedder'] = embedder
                    st.success("Embeddings generated successfully!")
        else:
            st.info("Please upload or generate a graph first.")
    
    with tab3:
        if st.session_state['graph'] is not None:
            st.header("Graph Algorithms")
            
            # Algorithm selection
            algo_type = st.selectbox(
                "Algorithm Type",
                ["Shortest Path", "Clustering", "Ranking"]
            )
            
            if algo_type == "Shortest Path":
                algo = st.selectbox(
                    "Algorithm",
                    ["Dijkstra", "Bellman-Ford", "Floyd-Warshall", "A*"]
                )
                
                # Source and target node selection
                nodes = list(st.session_state['graph'].nodes())
                source_node = st.selectbox("Source Node", nodes, index=0)
                target_node = st.selectbox("Target Node", nodes, index=min(1, len(nodes)-1))
                
                if st.button("Run Algorithm"):
                    # Traditional version
                    if algo == "Dijkstra":
                        algorithm_class = DijkstraTraditional
                    elif algo == "Bellman-Ford":
                        algorithm_class = BellmanFordTraditional
                    elif algo == "Floyd-Warshall":
                        algorithm_class = FloydWarshallTraditional
                    else:  # A*
                        algorithm_class = AStarTraditional
                    
                    # Run traditional algorithm
                    trad_algo = algorithm_class(st.session_state['graph'])
                    trad_result = trad_algo.run(source_node, target_node)
                    
                    st.subheader("Traditional Algorithm Result")
                    st.write(f"Execution time: {trad_algo.execution_time:.4f} seconds")
                    st.write(f"Path: {trad_result['path']}")
                    st.write(f"Path Length: {trad_result['distance']}")
                    
                    # Run embedding version if embedder exists
                    if st.session_state['embedder']:
                        if algo == "Dijkstra":
                            algorithm_class = DijkstraEmbedding
                        elif algo == "Bellman-Ford":
                            algorithm_class = BellmanFordEmbedding
                        elif algo == "Floyd-Warshall":
                            algorithm_class = FloydWarshallEmbedding
                        else:  # A*
                            algorithm_class = AStarEmbedding
                        
                        emb_algo = algorithm_class(st.session_state['graph'], st.session_state['embedder'])
                        emb_result = emb_algo.run(source_node, target_node)
                        
                        st.subheader("Embedding-aware Algorithm Result")
                        st.write(f"Execution time: {emb_algo.execution_time:.4f} seconds")
                        st.write(f"Path: {emb_result['path']}")
                        st.write(f"Path Length: {emb_result['distance']}")
            
            elif algo_type == "Clustering":
                algo = st.selectbox(
                    "Algorithm",
                    ["K-Means", "Spectral Clustering"]
                )
                
                k = st.slider("Number of Clusters", min_value=2, max_value=min(10, len(st.session_state['graph'])), value=3)
                
                if st.button("Run Algorithm"):
                    if algo == "K-Means":
                        trad_algo = KMeansTraditional(st.session_state['graph'])
                        emb_algo = KMeansEmbedding(st.session_state['graph'], st.session_state['embedder']) if st.session_state['embedder'] else None
                    else:  # Spectral
                        trad_algo = SpectralClusteringTraditional(st.session_state['graph'])
                        emb_algo = SpectralClusteringEmbedding(st.session_state['graph'], st.session_state['embedder']) if st.session_state['embedder'] else None
                    
                    # Run traditional version
                    trad_clusters = trad_algo.run(k)
                    st.write("Traditional Algorithm Result:")
                    st.write("Clusters:", trad_clusters)
                    
                    # Run embedding version if embedder exists
                    if st.session_state['embedder'] and emb_algo:
                        emb_clusters = emb_algo.run(k)
                        st.write("\nEmbedding-aware Algorithm Result:")
                        st.write("Clusters:", emb_clusters)
            
            else:  # Ranking
                algo = st.selectbox(
                    "Algorithm",
                    ["PageRank", "HITS"]
                )
                
                if st.button("Run Algorithm"):
                    if algo == "PageRank":
                        trad_algo = PageRankTraditional(st.session_state['graph'])
                        emb_algo = PageRankEmbedding(st.session_state['graph'], st.session_state['embedder']) if st.session_state['embedder'] else None
                        
                        # Run traditional version
                        import time
                        start_time = time.time()
                        trad_scores = trad_algo.run()
                        end_time = time.time()
                        trad_algo.execution_time = end_time - start_time
                        
                        st.subheader("Traditional Algorithm Results")
                        st.write(f"Execution time: {trad_algo.execution_time:.4f} seconds")
                        
                        # Create columns for traditional results
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Top 5 nodes by score:")
                            top_nodes = sorted(trad_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                            for node, score in top_nodes:
                                st.write(f"Node {node}: {score:.4f}")
                        
                        with col2:
                            st.write("Score distribution:")
                            fig = go.Figure(data=[go.Histogram(x=list(trad_scores.values()))])
                            fig.update_layout(title="Score Distribution", xaxis_title="Score", yaxis_title="Count")
                            st.plotly_chart(fig)
                        
                        # Run embedding version if embedder exists
                        if st.session_state['embedder'] and emb_algo:
                            start_time = time.time()
                            emb_scores = emb_algo.run()
                            end_time = time.time()
                            emb_algo.execution_time = end_time - start_time
                            
                            st.subheader("Embedding-Aware Algorithm Results")
                            st.write(f"Execution time: {emb_algo.execution_time:.4f} seconds")
                            
                            # Create columns for embedding-aware results
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("Top 5 nodes by score:")
                                top_nodes = sorted(emb_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                                for node, score in top_nodes:
                                    st.write(f"Node {node}: {score:.4f}")
                            
                            with col2:
                                st.write("Score distribution:")
                                fig = go.Figure(data=[go.Histogram(x=list(emb_scores.values()))])
                                fig.update_layout(title="Score Distribution", xaxis_title="Score", yaxis_title="Count")
                                st.plotly_chart(fig)
                    
                    else:  # HITS
                        trad_algo = HITSTraditional(st.session_state['graph'])
                        emb_algo = HITSEmbedding(st.session_state['graph'], st.session_state['embedder']) if st.session_state['embedder'] else None
                        
                        # Run traditional version
                        trad_scores = trad_algo.run()
                        
                        st.subheader("Traditional Algorithm Results")
                        st.write(f"Execution time: {trad_algo.execution_time:.4f} seconds")
                        
                        # Show hub and authority scores
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Top 5 nodes by hub score:")
                            top_hubs = sorted(trad_scores['hub_scores'].items(), key=lambda x: x[1], reverse=True)[:5]
                            for node, score in top_hubs:
                                st.write(f"Node {node}: {score:.4f}")
                        
                        with col2:
                            st.write("Top 5 nodes by authority score:")
                            top_auths = sorted(trad_scores['authority_scores'].items(), key=lambda x: x[1], reverse=True)[:5]
                            for node, score in top_auths:
                                st.write(f"Node {node}: {score:.4f}")
                        
                        # Run embedding version if embedder exists
                        if st.session_state['embedder'] and emb_algo:
                            emb_scores = emb_algo.run()
                            
                            st.subheader("Embedding-Aware Algorithm Results")
                            st.write(f"Execution time: {emb_algo.execution_time:.4f} seconds")
                            
                            # Show hub and authority scores
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("Top 5 nodes by hub score:")
                                top_hubs = sorted(emb_scores['hub_scores'].items(), key=lambda x: x[1], reverse=True)[:5]
                                for node, score in top_hubs:
                                    st.write(f"Node {node}: {score:.4f}")
                            
                            with col2:
                                st.write("Top 5 nodes by authority score:")
                                top_auths = sorted(emb_scores['authority_scores'].items(), key=lambda x: x[1], reverse=True)[:5]
                                for node, score in top_auths:
                                    st.write(f"Node {node}: {score:.4f}")
        
        else:
            st.info("Please upload or generate a graph first.")
else:
    st.info("Please upload or generate a graph to begin")
