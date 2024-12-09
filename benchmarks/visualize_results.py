"""Interactive visualization of benchmark results using Plotly."""
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path


def create_time_comparison_plot(df: pd.DataFrame, output_dir: Path):
    """Create interactive boxplot for time comparison."""
    fig = px.box(df, x="algorithm", y="time_speedup",
                 title="Speed Improvement using Embeddings",
                 labels={
                     "algorithm": "Algorithm",
                     "time_speedup": "Speedup Factor (Traditional Time / Embedding Time)"
                 })
    
    fig.update_layout(
        template="plotly_white",
        boxmode="group",
        showlegend=True,
        hovermode="x"
    )
    
    # Save as HTML for interactive viewing
    fig.write_html(output_dir / "time_comparison.html")
    # Save as static image
    fig.write_image(output_dir / "time_comparison.png")


def create_memory_comparison_plot(df: pd.DataFrame, output_dir: Path):
    """Create interactive boxplot for memory comparison."""
    fig = px.box(df, x="algorithm", y="memory_reduction",
                 title="Memory Usage Improvement using Embeddings",
                 labels={
                     "algorithm": "Algorithm",
                     "memory_reduction": "Memory Reduction Factor (Traditional Memory / Embedding Memory)"
                 })
    
    fig.update_layout(
        template="plotly_white",
        boxmode="group",
        showlegend=True,
        hovermode="x"
    )
    
    fig.write_html(output_dir / "memory_comparison.html")
    fig.write_image(output_dir / "memory_comparison.png")


def create_scaling_plots(df: pd.DataFrame, output_dir: Path):
    """Create interactive line plots for scaling behavior."""
    # Ensure numeric columns are float
    df = df.copy()
    numeric_columns = ['time_speedup', 'memory_reduction', 'graph_size']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    metrics = ["time_speedup", "memory_reduction"]
    titles = {
        "time_speedup": "Speed Improvement vs Graph Size",
        "memory_reduction": "Memory Reduction vs Graph Size"
    }
    ylabels = {
        "time_speedup": "Speedup Factor",
        "memory_reduction": "Memory Reduction Factor"
    }
    
    for metric in metrics:
        fig = go.Figure()
        
        for algo in sorted(df["algorithm"].unique()):
            for graph_type in sorted(df["graph_type"].unique()):
                data = df[(df["algorithm"] == algo) & (df["graph_type"] == graph_type)]
                if not data.empty:
                    hover_text = []
                    for size, value in zip(data["graph_size"], data[metric]):
                        try:
                            hover_text.append(
                                f"Graph Size: {int(size)}<br>"
                                f"{ylabels[metric]}: {float(value):.2f}<br>"
                                f"Algorithm: {algo}<br>"
                                f"Graph Type: {graph_type}"
                            )
                        except (ValueError, TypeError):
                            continue
                    
                    fig.add_trace(go.Scatter(
                        x=data["graph_size"],
                        y=data[metric],
                        name=f"{algo} ({graph_type})",
                        mode="lines+markers",
                        hovertext=hover_text,
                        hoverinfo="text"
                    ))
        
        fig.update_layout(
            title=titles[metric],
            xaxis_title="Number of Nodes",
            yaxis_title=ylabels[metric],
            xaxis_type="log",
            yaxis_type="log",
            template="plotly_white",
            hovermode="closest",
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
        
        fig.write_html(output_dir / f"{metric}_vs_size.html")
        fig.write_image(output_dir / f"{metric}_vs_size.png")


def create_performance_heatmaps(df: pd.DataFrame, output_dir: Path):
    """Create interactive heatmaps showing performance across algorithms and graph types."""
    # Ensure numeric columns are float
    df = df.copy()
    numeric_columns = ['time_speedup', 'memory_reduction']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    metrics = {
        "time_speedup": "Speed Improvement",
        "memory_reduction": "Memory Reduction"
    }
    
    for metric, title_prefix in metrics.items():
        # Calculate mean values for each algorithm-graph type combination
        pivot_data = df.pivot_table(
            values=metric,
            index="algorithm",
            columns="graph_type",
            aggfunc="mean"
        ).round(2)
        
        # Create text matrix for hover information
        hover_text = []
        for algo, row in pivot_data.iterrows():
            hover_row = []
            for graph_type, value in row.items():
                try:
                    hover_row.append(
                        f"Algorithm: {algo}<br>"
                        f"Graph Type: {graph_type}<br>"
                        f"{title_prefix} Factor: {float(value):.2f}"
                    )
                except (ValueError, TypeError):
                    hover_row.append(
                        f"Algorithm: {algo}<br>"
                        f"Graph Type: {graph_type}<br>"
                        f"{title_prefix} Factor: N/A"
                    )
            hover_text.append(hover_row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            text=[[f"{v:.2f}" if pd.notnull(v) else "N/A" for v in row] for row in pivot_data.values],
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertext=hover_text,
            hoverinfo="text",
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(
                title=f"{title_prefix} Factor",
                titleside="right"
            )
        ))
        
        fig.update_layout(
            title=f"Average {title_prefix} by Algorithm and Graph Type",
            xaxis_title="Graph Type",
            yaxis_title="Algorithm",
            template="plotly_white",
            height=400,
            width=800,
            xaxis=dict(side="bottom"),
            yaxis=dict(autorange="reversed")  # To match traditional heatmap orientation
        )
        
        fig.write_html(output_dir / f"{metric}_heatmap.html")
        fig.write_image(output_dir / f"{metric}_heatmap.png")


def main():
    """Generate all visualizations."""
    # Create output directory
    output_dir = Path("benchmark_results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read and preprocess the data
    df = pd.read_csv("benchmark_results/benchmark_results.csv")
    
    # Convert numeric columns
    numeric_cols = ['graph_size', 'traditional_time', 'embedding_time', 
                   'traditional_memory', 'embedding_memory', 'time_speedup', 'memory_reduction']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Replace inf values with a large number for visualization
    df = df.replace([float('inf'), -float('inf')], 1000.0)
    
    # Remove rows with NaN values
    df = df.dropna()
    
    # Create visualizations
    create_time_comparison_plot(df, output_dir)
    create_memory_comparison_plot(df, output_dir)
    create_scaling_plots(df, output_dir)
    create_performance_heatmaps(df, output_dir)


if __name__ == "__main__":
    main()
