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

## Docker Setup

The application can be run using Docker and Docker Compose. This setup includes both the Streamlit application and the documentation server.

### Prerequisites
- Docker
- Docker Compose

### Running with Docker Compose

1. Build and start the containers:
```bash
docker-compose up --build
```

2. Access the applications:
- Streamlit app: http://localhost:8501
- Documentation: http://localhost:8000

### Modifying Port Mappings

To change the ports on your host machine/VPS:

1. Open `docker-compose.yml`
2. Modify the port mappings under each service:
   - For Streamlit: Change `"8501:8501"` to `"YOUR_PORT:8501"`
   - For Documentation: Change `"8000:8000"` to `"YOUR_PORT:8000"`

Example:
```yaml
ports:
  - "9000:8501"  # Streamlit will be available on port 9000
