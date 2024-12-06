import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import numpy as np

def load_and_process_results(csv_path):
    """Load and process the stress test results."""
    df = pd.read_csv(csv_path)
    # Only include successful runs
    df = df[df['success'] == True]
    return df

def create_algorithm_comparison_plot(df):
    """Create comparison plot between traditional and embedding-based algorithms."""
    # Group algorithms by type
    algo_groups = {
        'Shortest Path': ['Dijkstra', 'BellmanFord', 'FloydWarshall', 'AStar'],
        'Clustering': ['KMeans', 'SpectralClustering'],
        'Ranking': ['PageRank', 'HITS']
    }
    
    # Create subplots for each algorithm type
    fig = make_subplots(
        rows=len(algo_groups),
        cols=1,
        subplot_titles=list(algo_groups.keys()),
        vertical_spacing=0.1
    )
    
    colors = px.colors.qualitative.Set1
    row = 1
    has_data = False
    
    for group_name, algorithms in algo_groups.items():
        for i, algo_base in enumerate(algorithms):
            # Get data for traditional version
            trad_data = df[df['algorithm'] == f'{algo_base}Traditional'].groupby('size')['execution_time'].mean()
            # Get data for embedding version
            emb_data = df[df['algorithm'] == f'{algo_base}Embedding'].groupby('size')['execution_time'].mean()
            
            # Add traces for both versions
            if not trad_data.empty:
                has_data = True
                fig.add_trace(
                    go.Scatter(
                        x=trad_data.index,
                        y=trad_data.values,
                        name=f'{algo_base} (traditional)',
                        line=dict(color=colors[i*2], dash='solid'),
                        showlegend=(row == 1)  # Only show legend for first subplot
                    ),
                    row=row,
                    col=1
                )
            
            if not emb_data.empty:
                has_data = True
                fig.add_trace(
                    go.Scatter(
                        x=emb_data.index,
                        y=emb_data.values,
                        name=f'{algo_base} (embedding)',
                        line=dict(color=colors[i*2], dash='dash'),
                        showlegend=(row == 1)  # Only show legend for first subplot
                    ),
                    row=row,
                    col=1
                )
        
        # Update y-axis label for each subplot
        fig.update_yaxes(title_text="Execution Time (s)", row=row, col=1)
        if row == len(algo_groups):  # Only add x-axis label for bottom subplot
            fig.update_xaxes(title_text="Graph Size (nodes)", row=row, col=1)
        
        row += 1
    
    if not has_data:
        print("Warning: No valid algorithm comparisons could be made.")
        fig = go.Figure()
        fig.add_annotation(
            text="No valid algorithm comparisons available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig
    
    # Update layout
    fig.update_layout(
        height=300 * len(algo_groups),
        width=1000,
        title_text="Algorithm Performance Comparison",
        showlegend=True
    )
    
    return fig

def create_speedup_plot(df):
    """Create plot showing speedup of embedding-based vs traditional algorithms."""
    # Calculate speedup for each algorithm pair
    speedups = []
    
    for size in df['size'].unique():
        size_df = df[df['size'] == size]
        
        for algo_base in ['Dijkstra', 'BellmanFord', 'FloydWarshall', 'AStar',
                         'KMeans', 'SpectralClustering', 'PageRank', 'HITS']:
            trad_data = size_df[size_df['algorithm'] == f'{algo_base}Traditional']
            emb_data = size_df[size_df['algorithm'] == f'{algo_base}Embedding']
            
            if not trad_data.empty and not emb_data.empty:
                trad_time = trad_data['execution_time'].mean()
                emb_time = emb_data['execution_time'].mean()
                
                if pd.notnull(trad_time) and pd.notnull(emb_time) and trad_time > 0 and emb_time > 0:
                    speedup = trad_time / emb_time
                    speedups.append({
                        'size': size,
                        'algorithm': algo_base,
                        'speedup': speedup
                    })
    
    if not speedups:
        print("Warning: No valid speedup comparisons could be made.")
        # Return an empty figure
        fig = go.Figure()
        fig.add_annotation(
            text="No valid speedup comparisons available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig
    
    speedup_df = pd.DataFrame(speedups)
    
    # Create plot
    fig = px.line(
        speedup_df,
        x='size',
        y='speedup',
        color='algorithm',
        title='Speedup of Embedding-based vs Traditional Algorithms',
        labels={
            'size': 'Graph Size (nodes)',
            'speedup': 'Speedup Factor (traditional time / embedding time)',
            'algorithm': 'Algorithm'
        }
    )
    
    # Add reference line for speedup = 1
    fig.add_hline(y=1, line_dash="dash", line_color="gray")
    
    return fig

def create_result_quality_plot(df):
    """Create plot comparing result quality between traditional and embedding versions."""
    quality_metrics = []
    
    for size in df['size'].unique():
        size_df = df[df['size'] == size]
        
        for algo_base in ['Dijkstra', 'BellmanFord', 'FloydWarshall', 'AStar',
                         'KMeans', 'SpectralClustering', 'PageRank', 'HITS']:
            trad_data = size_df[size_df['algorithm'] == f'{algo_base}Traditional']
            emb_data = size_df[size_df['algorithm'] == f'{algo_base}Embedding']
            
            if not trad_data.empty and not emb_data.empty:
                trad_metric = trad_data['metric'].mean()
                emb_metric = emb_data['metric'].mean()
                
                if pd.notnull(trad_metric) and pd.notnull(emb_metric):
                    if np.isinf(trad_metric) and np.isinf(emb_metric):
                        quality = 1.0  # Both infinite means they agree
                    elif np.isinf(trad_metric) or np.isinf(emb_metric):
                        quality = 0.0  # One infinite but not the other means they disagree
                    else:
                        relative_diff = abs(trad_metric - emb_metric) / (abs(trad_metric) + 1e-10)
                        quality = 1.0 if relative_diff < 0.01 else max(0.0, 1.0 - relative_diff)
                    
                    quality_metrics.append({
                        'size': size,
                        'algorithm': algo_base,
                        'quality': quality
                    })
    
    if not quality_metrics:
        print("Warning: No valid quality comparisons could be made.")
        # Return an empty figure
        fig = go.Figure()
        fig.add_annotation(
            text="No valid quality comparisons available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig
    
    quality_df = pd.DataFrame(quality_metrics)
    
    # Create plot
    fig = px.line(
        quality_df,
        x='size',
        y='quality',
        color='algorithm',
        title='Result Quality Comparison',
        labels={
            'size': 'Graph Size (nodes)',
            'quality': 'Result Quality (1 = identical, 0 = very different)',
            'algorithm': 'Algorithm'
        }
    )
    
    return fig

def visualize_results(results_df, output_dir):
    """Create and save visualization plots."""
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plots
    comparison_fig = create_algorithm_comparison_plot(results_df)
    speedup_fig = create_speedup_plot(results_df)
    quality_fig = create_result_quality_plot(results_df)
    
    # Save plots
    comparison_fig.write_html(output_dir / 'algorithm_comparison.html')
    speedup_fig.write_html(output_dir / 'speedup_comparison.html')
    quality_fig.write_html(output_dir / 'quality_comparison.html')
    
    # Save summary statistics
    summary = results_df.groupby('algorithm').agg({
        'execution_time': ['mean', 'std', 'min', 'max'],
        'metric': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    summary.to_csv(output_dir / 'summary_statistics.csv')

def main():
    """Main function to visualize results."""
    # Find the most recent results file
    results_dir = Path('test/results')
    result_files = list(results_dir.glob('stress_test_results_*.csv'))
    if not result_files:
        print("No result files found in test/results/")
        return
    
    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading results from: {latest_file}")
    
    # Load and process results
    df = load_and_process_results(latest_file)
    
    # Create visualizations
    output_dir = results_dir / 'visualizations'
    visualize_results(df, output_dir)
    
    print(f"Visualizations saved to {output_dir}")
    print("\nTo view the results:")
    print(f"1. Algorithm comparison: {output_dir}/algorithm_comparison.html")
    print(f"2. Speedup comparison: {output_dir}/speedup_comparison.html")
    print(f"3. Quality comparison: {output_dir}/quality_comparison.html")
    print(f"4. Summary statistics: {output_dir}/summary_statistics.csv")

if __name__ == "__main__":
    main()
