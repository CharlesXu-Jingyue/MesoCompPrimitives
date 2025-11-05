# MesoCompPrimitives: Scientific Methods Guide

## Overview

This guide documents the scientific methods and computational approaches implemented in the MesoCompPrimitives project for analyzing neural connectivity data and extracting mesoscale computational primitives.

## Data Source

**Dataset**: FlyWire ring extend neural connectivity
- **Source**: 855 neurons from Drosophila central complex
- **Cell Types**: 68 distinct neuronal cell types
- **Connectivity**: Signed adjacency matrix (excitatory/inhibitory synapses)
- **Focus Subset**: Delta7 (40 neurons) + EPG (47 neurons) = 87 neurons

## Preprocessing Pipeline

### 1. Data Loading and Matrix Construction

```python
# Load connectivity matrices and metadata
connectivity_matrix_signed_df = pd.read_csv('connectivity_matrix_signed.csv')
ctoi_df = pd.read_csv('ctoi_list.csv')  # Cell type mapping
noi_df = pd.read_csv('noi_list.csv')   # Neuron ID mapping
```

**Key Operations:**
- Convert DataFrame to numpy array for efficient computation
- Handle potential index columns and ensure square matrix format
- Log-scale visualization with sign preservation: `log10(|W| + 1) * sign(W)`

### 2. Cell Type Organization

**Alphabetical Grouping:**
- Sort unique cell types alphabetically for consistent ordering
- Reorder neurons to group by cell type
- Create mapping dictionaries for efficient lookup

**Unconnected Node Detection:**
- Identify neurons with zero in-degree and out-degree
- Remove isolated nodes (though analysis shows all neurons are connected)
- Preserve data structure integrity across filtering operations

### 3. Cell Type Selection

**Rationale for Delta7 + EPG Subset:**
- Delta7: Ring neurons involved in compass calculations
- EPG: Ellipsoid body protocerebral bridge glomeruli
- Together form a computationally meaningful circuit for spatial navigation

## Graph Laplacian Analysis

### Mathematical Foundations

Given adjacency matrix **A**, we compute multiple Laplacian variants to capture different aspects of network dynamics:

### 1. Combinatorial Laplacians

**Out-degree Laplacian:**
```
L_out = D_out - A
where D_out = diag(sum(A, axis=1))
```
- **Property**: Row sums equal zero
- **Interpretation**: Models diffusion with outgoing flow normalization

**In-degree Laplacian:**
```
L_in = D_in - A^T
where D_in = diag(sum(A, axis=0))
```
- **Property**: Column sums equal zero
- **Interpretation**: Models diffusion with incoming flow normalization

### 2. Random Walk Laplacian

```
L_rw = I - D_out^(-1) * A
```
- **Properties**: Row-stochastic transition matrix
- **Eigenvalues**: Real, in [0, 2]
- **Interpretation**: Describes random walks on the directed graph

### 3. Symmetric Laplacians

**Symmetrize-then-Normalize:**
```
A_sym = (A + A^T) / 2
D_sym = diag(sum(A_sym, axis=1))
L_sym = I - D_sym^(-1/2) * A_sym * D_sym^(-1/2)
```

**Direct Symmetrization:**
```
L_out_sym = D_out^(-1/2) * (D_out - A_sym) * D_out^(-1/2)
```

### Degree Distribution Analysis

**Metrics Computed:**
- Mean, standard deviation, min/max for in-degree and out-degree
- Heavy-tailed distributions indicate hub neurons
- Degree heterogeneity affects spectral properties

## Community Detection Methods

### Modularity Optimization

**Modularity Definition:**
```
Q = (1/2m) * sum_ij [A_ij - (k_i * k_j)/(2m)] * δ(c_i, c_j)
```
where:
- m = total number of edges
- k_i = degree of node i
- c_i = community assignment of node i
- δ = Kronecker delta

### 1. Louvain Algorithm

**Implementation:**
```python
communities = community.louvain_communities(G.to_undirected(), seed=42)
communities_sorted = sorted(communities, key=len, reverse=True)
```

**Results for Delta7+EPG Subset:**
- **Communities**: 4
- **Sizes**: [24, 24, 20, 19]
- **Modularity**: 0.2123

**Advantages:**
- Fast, scalable algorithm
- Good performance on sparse networks
- Deterministic with fixed random seed

### 2. Leiden Algorithm

**Implementation:**
```python
partition = leidenalg.find_partition(g_ig, leidenalg.ModularityVertexPartition,
                                    weights='weight', seed=42)
```

**Results for Delta7+EPG Subset:**
- **Communities**: 6
- **Sizes**: [23, 20, 13, 13, 10, 8]
- **Modularity**: 0.1853

**Advantages:**
- Better handling of degenerate cases
- More refined community structure
- Avoids poorly connected communities

### Community Quality Metrics

**Community Purity:**
```python
purity = max(cell_type_counts_in_community) / total_nodes_in_community
```

**Cell Type Splitting Analysis:**
- Measures how biological cell types distribute across algorithmic communities
- Indicates alignment between connectivity-based and morphology-based classifications

## Visualization Methods

### 1. Connectivity Matrix Visualization

**Log-scale with Sign Preservation:**
```python
plt.imshow(np.log10(np.abs(A) + 1) * np.sign(A), cmap='bwr')
```
- Red: Excitatory connections
- Blue: Inhibitory connections
- Intensity: Connection strength (log-scaled)

### 2. Community Structure Visualization

**Matrix Reordering:**
- Nodes reordered by community assignment
- Visual block structure reveals modular organization
- Community boundaries marked with dividing lines

### 3. Cell Type Distribution Heatmaps

**2D Distribution Analysis:**
```python
sns.heatmap(distribution_matrix,
            xticklabels=['Comm 0', 'Comm 1', ...],
            yticklabels=cell_types)
```
- Rows: Biological cell types
- Columns: Algorithmic communities
- Values: Number of neurons

## Key Findings

### Connectivity Structure
- **Dense Connectivity**: 1,729 edges among 87 neurons (high connection density)
- **Degree Heterogeneity**: Out-degree range [5, 471], in-degree range [23, 812]
- **Hub Neurons**: Identified through degree distribution analysis

### Community Structure
- **Louvain vs Leiden**: Different granularity of community detection
- **Biological Alignment**: Community structure partially aligns with cell type boundaries
- **Cross-type Communities**: Some algorithmic communities span multiple cell types

### Spectral Properties
- **Laplacian Eigenvalues**: Capture network connectivity patterns
- **Random Walk Dynamics**: Row-stochastic properties enable diffusion analysis
- **Symmetrization Effects**: Different approaches yield distinct spectral signatures

## Computational Implementation

### Performance Considerations
- **Matrix Operations**: Efficient numpy operations for large adjacency matrices
- **Graph Libraries**: NetworkX for Louvain, igraph for Leiden
- **Memory Management**: Selective variable retention for large datasets

### Reproducibility
- **Fixed Random Seeds**: Ensures deterministic community detection
- **Version Control**: Track algorithm implementations and parameter choices
- **Data Provenance**: Clear documentation of data sources and preprocessing steps

## Future Directions

### System Identification
- **Linear Dynamics**: Fit local linear models within communities
- **Normal Forms**: Identify canonical dynamical motifs
- **Port Identification**: Define input/output subspaces for each primitive

### Role Discovery
- **Hub Classification**: Core vs. peripheral nodes
- **Flow Patterns**: Feedforward vs. recurrent connectivity
- **Functional Roles**: Integration, competition, routing, memory

### Synthesis Validation
- **Reconstruction**: Build coarse models from identified primitives
- **Task Performance**: Test functional equivalence on computational tasks
- **Ablation Studies**: Demonstrate necessity of specific primitives

## References

### Algorithms
- Louvain: Blondel et al. (2008) "Fast unfolding of communities in large networks"
- Leiden: Traag et al. (2019) "From Louvain to Leiden: guaranteeing well-connected communities"

### Graph Theory
- Chung (1997) "Spectral Graph Theory"
- von Luxburg (2007) "A tutorial on spectral clustering"

### Neuroscience Applications
- Sporns (2011) "Networks of the Brain"
- Bassett & Sporns (2017) "Network neuroscience"