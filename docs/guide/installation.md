# Installation Guide

## System Requirements

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

## Dependencies

The Graph Embeddings library requires the following main dependencies:

- NetworkX: For graph operations
- NumPy: For numerical computations
- Streamlit: For interactive visualization
- SciPy: For scientific computing
- Matplotlib: For plotting

## Installation Steps

### 1. Create a Virtual Environment (Recommended)

```bash
# Create a new virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

### 2. Install from Source

Clone the repository and install the package:

```bash
# Clone the repository
git clone https://github.com/yourusername/graph-embeddings.git
cd graph-embeddings

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify Installation

```python
# Test the installation
python -c "from core.embeddings import EuclideanEmbedding; print('Installation successful!')"
```

## Optional Dependencies

For advanced features, you may want to install additional packages:

```bash
# For GPU support
pip install torch

# For advanced visualization
pip install plotly

# For documentation
pip install mkdocs-material mkdocstrings[python] mkdocs-jupyter
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'core'**
   - Make sure you're in the correct directory
   - Check if the package is installed correctly
   - Verify your Python path

2. **Version conflicts**
   - Try creating a fresh virtual environment
   - Update all dependencies to their latest versions

3. **GPU-related errors**
   - Ensure CUDA is installed correctly
   - Check GPU compatibility
   - Verify PyTorch installation

### Getting Help

If you encounter any issues:

1. Check the [GitHub Issues](https://github.com/yourusername/graph-embeddings/issues)
2. Join our [Discord community](https://discord.gg/yourinvitelink)
3. Create a new issue with:
   - Your system information
   - Error message
   - Steps to reproduce
