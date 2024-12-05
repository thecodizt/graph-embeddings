# Graph Generator

A Streamlit application for generating and configuring various types of graphs with different properties.

## Features

- Generate different types of graphs:
  - Random (Erdős-Rényi model)
  - Scale-free (Barabási-Albert model)
  - Small-world (Watts-Strogatz model)
- Configure graph properties:
  - Number of nodes
  - Average degree
  - Directed/Undirected
  - Cyclic/Acyclic
- Download graphs in NetworkX JSON format
- View graph statistics

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

## Graph Configuration Options

- **Number of Nodes**: Control the size of the graph
- **Average Degree**: Set the average number of connections per node
- **Graph Type**: Choose between different graph generation models
- **Advanced Options**:
  - Directed/Undirected graphs
  - Enforce cyclic/acyclic properties

## Output Format

Graphs can be downloaded in NetworkX JSON format, which includes:
- Node information
- Edge information
- Graph properties
