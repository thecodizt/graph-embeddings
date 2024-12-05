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
from core.algorithms.shortest_path import DijkstraTraditional, DijkstraEmbedding
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
    tab1, tab2, tab3 = st.tabs(["Graph Analysis", "Embeddings", "Algorithm Comparison"])
    
    with tab1:
        # Graph visualization
        st.header("Graph Visualization")
        fig = create_plotly_graph(st.session_state['graph'], layout_type, 
                                node_size_scale=node_size_scale, 
                                edge_width=edge_width)
        st.plotly_chart(fig, use_container_width=True)
        
        # Graph properties using the new function
        st.header("Graph Properties")
        properties = get_graph_properties(st.session_state['graph'])
        
        # Display properties in two columns
        col1, col2 = st.columns(2)
        props_per_column = len(properties) // 2 + len(properties) % 2
        
        with col1:
            for key, value in list(properties.items())[:props_per_column]:
                st.write(f"{key}: {value}")
        
        with col2:
            for key, value in list(properties.items())[props_per_column:]:
                st.write(f"{key}: {value}")
        
        # Download section
        st.header("Download Graph")
        if st.button("Download Graph (JSON)"):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp:
                save_graph_json(st.session_state['graph'], tmp.name)
                with open(tmp.name, 'r') as f:
                    graph_json = f.read()
                
                st.download_button(
                    label="Click to Download",
                    data=graph_json,
                    file_name="graph.json",
                    mime="application/json"
                )
    
    with tab2:
        st.header("Embedding Configuration")
        
        # Get suggested embedding type
        suggested_type, suggestion_reason = suggest_embedding_type(st.session_state['graph'])
        
        # Convert suggested_type to lowercase for consistency
        suggested_type = suggested_type.lower()
        
        # Display suggestion with explanation
        st.info(f"Suggested embedding type: **{suggested_type.capitalize()}**\n\nReason: {suggestion_reason}")
        
        # Let user choose embedding type
        embedding_type = st.selectbox(
            "Select Embedding Type",
            ["Euclidean", "Hyperbolic", "Spherical"],
            index=["Euclidean", "Hyperbolic", "Spherical"].index(suggested_type.capitalize())
        ).lower()
        
        # Embedding dimension
        embedding_dim = st.slider("Embedding Dimension", min_value=2, max_value=10, value=2)
        
        # Create embeddings button
        if st.button("Create Embeddings"):
            # Create embedder based on type
            if embedding_type == "euclidean":
                embedder = EuclideanEmbedding(dim=embedding_dim)
            elif embedding_type == "hyperbolic":
                embedder = HyperbolicEmbedding(dim=embedding_dim)
            else:  # spherical
                embedder = SphericalEmbedding(dim=embedding_dim)
            
            # Train embeddings
            embedder.train(st.session_state['graph'])
            
            # Store embedder in session state
            st.session_state['embedder'] = embedder
            
            st.success("Embeddings created successfully!")
        
        if 'embedder' in st.session_state:
            try:
                # Embedding visualization
                st.header("Embedding Visualization")
                
                if embedding_dim == 2:
                    # 2D visualization
                    fig = create_plotly_graph(st.session_state['graph'], layout_type, 
                                            node_size_scale=node_size_scale, 
                                            edge_width=edge_width)
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif embedding_dim == 3:
                    # 3D visualization
                    # fig = create_embedding_plot_3d(st.session_state['graph'],
                    #                             st.session_state['embedder'],
                    #                             node_size_scale=node_size_scale,
                    #                             edge_width=edge_width)
                    # st.plotly_chart(fig, use_container_width=True)
                    st.warning("3D visualization not implemented")
                    
                else:
                    st.warning("Visualization only available for 2D and 3D embeddings")
                    
                # Embedding statistics
                st.header("Embedding Statistics")
                
                # Calculate and display distortion
                # distortion = calculate_distortion(st.session_state['graph'], 
                #                                 st.session_state['embedder'])
                # st.write(f"Average Distortion: {distortion:.4f}")
                
                # Calculate and display stress
                # stress = calculate_stress(st.session_state['graph'],
                #                         st.session_state['embedder'])
                # st.write(f"Embedding Stress: {stress:.4f}")
                
            except Exception as e:
                st.error(f"Error visualizing embeddings: {str(e)}")
                if 'embedder' in st.session_state:
                    del st.session_state['embedder']
    
    with tab3:
        st.header("Algorithm Comparison")
        
        if 'graph' not in st.session_state:
            st.warning("Please generate or upload a graph first")
        else:
            algorithm_type = st.selectbox(
                "Select Algorithm",
                ["Shortest Path", "Community Detection", "Node Ranking"]
            )
            
            if algorithm_type == "Shortest Path":
                # Source and target selection
                source = st.selectbox("Select Source Node", 
                                    options=list(st.session_state['graph'].nodes()),
                                    key="sp_source")
                target = st.selectbox("Select Target Node",
                                    options=list(st.session_state['graph'].nodes()),
                                    key="sp_target")
                
                if st.button("Run Shortest Path Comparison"):
                    # Traditional algorithm
                    trad_algo = DijkstraTraditional(st.session_state['graph'])
                    trad_path, trad_dist = trad_algo.run(source, target)
                    trad_time = trad_algo.execution_time
                    
                    # Embedding-aware algorithm (if embeddings exist)
                    if 'embedder' in st.session_state:
                        emb_algo = DijkstraEmbedding(st.session_state['graph'], st.session_state['embedder'])
                        emb_path, emb_dist = emb_algo.run(source, target)
                        emb_time = emb_algo.execution_time
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Traditional Algorithm")
                            st.write(f"Path: {' → '.join(map(str, trad_path))}")
                            st.write(f"Distance: {trad_dist:.4f}")
                            st.write(f"Time: {trad_time:.6f} seconds")
                        
                        with col2:
                            st.subheader("Embedding-Aware Algorithm")
                            st.write(f"Path: {' → '.join(map(str, emb_path))}")
                            st.write(f"Distance: {emb_dist:.4f}")
                            st.write(f"Time: {emb_time:.6f} seconds")
                            
                        # Performance comparison
                        st.subheader("Performance Comparison")
                        speedup = trad_time / emb_time if emb_time > 0 else float('inf')
                        st.write(f"Speedup: {speedup:.2f}x")
                        
                        # Path quality comparison
                        if trad_dist > 0:
                            quality_ratio = emb_dist / trad_dist
                            st.write(f"Path Quality Ratio: {quality_ratio:.2f}x " +
                                   "(1.0 means same quality, >1.0 means longer path)")
                    else:
                        st.error("Please create embeddings first")
                        
            elif algorithm_type == "Community Detection":
                max_iterations = st.slider("Max Iterations", 5, 50, 10)
                
                if st.button("Run Community Detection Comparison"):
                    # Traditional algorithm
                    trad_algo = LabelPropagationTraditional(st.session_state['graph'])
                    trad_communities = trad_algo.run(max_iterations)
                    trad_time = trad_algo.execution_time
                    
                    # Embedding-aware algorithm (if embeddings exist)
                    if 'embedder' in st.session_state:
                        emb_algo = LabelPropagationEmbedding(st.session_state['graph'], st.session_state['embedder'])
                        emb_communities = emb_algo.run(max_iterations)
                        emb_time = emb_algo.execution_time
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Traditional Algorithm")
                            st.write(f"Number of Communities: {len(set(trad_communities.values()))}")
                            st.write(f"Time: {trad_time:.6f} seconds")
                        
                        with col2:
                            st.subheader("Embedding-Aware Algorithm")
                            st.write(f"Number of Communities: {len(set(emb_communities.values()))}")
                            st.write(f"Time: {emb_time:.6f} seconds")
                        
                        # Performance comparison
                        st.subheader("Performance Comparison")
                        speedup = trad_time / emb_time if emb_time > 0 else float('inf')
                        st.write(f"Speedup: {speedup:.2f}x")
                    else:
                        st.error("Please create embeddings first")
                        
            else:  # Node Ranking
                if st.button("Run Node Ranking Comparison"):
                    # Traditional algorithm
                    trad_algo = PageRankTraditional(st.session_state['graph'])
                    trad_ranks = trad_algo.run()
                    trad_time = trad_algo.execution_time
                    
                    # Embedding-aware algorithm (if embeddings exist)
                    if 'embedder' in st.session_state:
                        emb_algo = PageRankEmbedding(st.session_state['graph'], st.session_state['embedder'])
                        emb_ranks = emb_algo.run()
                        emb_time = emb_algo.execution_time
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Traditional Algorithm")
                            st.write("Top 5 Nodes by PageRank:")
                            top_trad = sorted(trad_ranks.items(), key=lambda x: x[1], reverse=True)[:5]
                            for node, rank in top_trad:
                                st.write(f"Node {node}: {rank:.4f}")
                            st.write(f"Time: {trad_time:.6f} seconds")
                        
                        with col2:
                            st.subheader("Embedding-Aware Algorithm")
                            st.write("Top 5 Nodes by PageRank:")
                            top_emb = sorted(emb_ranks.items(), key=lambda x: x[1], reverse=True)[:5]
                            for node, rank in top_emb:
                                st.write(f"Node {node}: {rank:.4f}")
                            st.write(f"Time: {emb_time:.6f} seconds")
                        
                        # Performance comparison
                        st.subheader("Performance Comparison")
                        speedup = trad_time / emb_time if emb_time > 0 else float('inf')
                        st.write(f"Speedup: {speedup:.2f}x")
                        
                        # Rank correlation
                        trad_order = {node: i for i, (node, _) in enumerate(top_trad)}
                        emb_order = {node: i for i, (node, _) in enumerate(top_emb)}
                        common_nodes = set(trad_order.keys()) & set(emb_order.keys())
                        if common_nodes:
                            differences = sum(abs(trad_order[node] - emb_order[node])
                                           for node in common_nodes)
                            similarity = 1 - (differences / (len(common_nodes) * len(common_nodes)))
                            st.write(f"Rank Similarity: {similarity:.2f} " +
                                   "(1.0 means identical rankings)")
                    else:
                        st.error("Please create embeddings first")
else:
    st.info("Please upload or generate a graph to begin")
