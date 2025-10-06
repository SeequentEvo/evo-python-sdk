# Evo Code Samples

Welcome to the Evo Python SDK code samples! This directory contains comprehensive Jupyter notebook examples that demonstrate how to use the various Evo APIs for geoscience data management and analysis.

## 📋 Prerequisites

- **Python**: Version 3.10 or higher
- **Evo Application**: You'll need a [registered application in Bentley](https://developer.bentley.com/register/?product=seequent-evo) to obtain a client ID
- **Jupyter Environment**: JupyterLab, VS Code, or another Jupyter notebook editor

## 🚀 Quick Start

### 1. Set Up Your Environment

The recommended approach is to use `uv` (included in this repository) for dependency management:

```bash
# From the root directory of the repository
./scripts/install-uv.sh  # On macOS/Linux
# or
./scripts/install-uv.ps1  # On Windows

# Move to the samples directory
cd samples

# Install the dependencies
uv sync
```

### 2. Start with Authentication

Before diving into specific samples, **start here**:

📁 **[auth-and-evo-discovery](auth-and-evo-discovery/)** - Essential first steps
- Create an Evo access token (choose your authentication method)
- Discover your organization ID and Evo hub URL
- **Start with these notebooks before using any other samples**

### 3. Run the Notebooks

Launch Jupyter and open the desired notebook:

```bash
jupyter lab
# or if using VS Code, simply open the .ipynb files
```

## 📚 Sample Categories

### 🔐 Authentication & Discovery
**📁 [auth-and-evo-discovery](auth-and-evo-discovery/)**

Essential setup for all other samples. Contains:
- `native-app-token.ipynb` - Authentication for desktop applications
- `service-app-token.ipynb` - Authentication for service applications  
- `evo-discovery.ipynb` - Find your organization ID and hub URL

*Required for: All other samples*

### 📁 File Operations
**📁 [files](files/)**

Basic file management operations:
- Upload files to Evo
- Download files from Evo
- List and organize files
- Delete files

**📁 [file-handling](file-handling/)**

Advanced file processing:
- Working with Parquet data files
- Data manipulation and analysis

### 🧊 Block Models
**📁 [blockmodels](blockmodels/)**

Comprehensive block model workflows organized by operation:
- **Create**: Regular and variable octree block models
- **Download**: Entire models or specific bounding box regions
- **Manage**: List, delete, and restore block models
- **Update**: Add, delete, rename, and update columns

### 🌍 Geoscience Objects
**📁 [geoscience-objects](geoscience-objects/)**

Publish and download various geoscience data types:
- Drilling campaigns and downhole collections
- Point sets and triangular meshes
- 2D regular grids
- Complex geoscience data structures

*Note: Some notebooks have platform-specific requirements (e.g., Windows-only dependencies)*

### 🏢 Workspace Management
**📁 [workspaces](workspaces/)**

Administrative operations:
- Manage Evo workspaces
- Handle user roles and permissions

## 🔧 Running a Sample

1. **Complete authentication setup** (auth-and-evo-discovery folder)
2. **Navigate to your chosen sample folder**
3. **Install requirements**: `pip install -r requirements.txt`
4. **Open the notebook** in your preferred editor
5. **Update the first cell** with your client ID (and redirect URL if needed)
6. **Run the authentication cell** - this will open your browser for Bentley ID sign-in
7. **Select your workspace** using the provided widget
8. **Continue with the remaining cells** in order

## 💡 Tips for Success

- **Always start with authentication**: The auth-and-evo-discovery samples are prerequisite for all others
- **Check platform requirements**: Some geoscience-objects samples are Windows-specific
- **Use virtual environments**: Keep dependencies isolated for each project
- **Follow notebook order**: Run cells sequentially for best results
- **Keep credentials secure**: Never commit tokens or credentials to version control

## 📖 Additional Resources

- [Seequent Developer Portal](https://developer.seequent.com/docs/guides/getting-started/quick-start-guide)
- [Evo SDK Documentation](../README.md)
- [Seequent Community](https://community.seequent.com/group/19-evo)
- [API References](https://developer.seequent.com/)

## 🆘 Getting Help

If you encounter issues:
1. Check that you've completed the authentication setup
2. Verify your Python version (3.10+)
3. Ensure all requirements are installed
4. Visit the [Seequent Community](https://community.seequent.com/group/19-evo) for support
5. Check the [GitHub issues](https://github.com/SeequentEvo/evo-python-sdk/issues) for known problems

Happy coding with Evo! 🎉
