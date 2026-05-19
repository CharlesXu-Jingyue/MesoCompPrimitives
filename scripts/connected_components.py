"""
Connected Component Analysis of Metagraph
Identifies mesoscale groups/blocks via spectral clustering on Zeta
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from collections import deque


def symmetrize_metagraph(Zeta, method='union'):
    """
    Symmetrize the metagraph adjacency matrix

    Parameters:
    -----------
    Zeta : ndarray (n, n)
        Binary metagraph (may be asymmetric)
    method : str
        'union' : Zeta_sym[i,j] = 1 if Zeta[i,j] OR Zeta[j,i]
        'max' : Zeta_sym[i,j] = max(Zeta[i,j], Zeta[j,i])

    Returns:
    --------
    Zeta_sym : ndarray (n, n)
        Symmetrized metagraph
    """

    if method == 'union':
        Zeta_sym = np.logical_or(Zeta, Zeta.T).astype(int)
    elif method == 'max':
        Zeta_sym = np.maximum(Zeta, Zeta.T)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Remove self-loops (handled implicitly as within-component)
    np.fill_diagonal(Zeta_sym, 0)

    return Zeta_sym


def find_connected_components_dfs(Zeta_sym):
    """
    Find connected components using depth-first search (DFS)

    Parameters:
    -----------
    Zeta_sym : ndarray (n, n)
        Symmetrized undirected metagraph adjacency matrix

    Returns:
    --------
    labels : ndarray (n,)
        Component assignment for each node: labels[i] in {0, 1, ..., m-1}
    n_components : int
        Number of connected components (m)
    component_sizes : dict
        Mapping from component label to size
    """

    n = Zeta_sym.shape[0]
    labels = -np.ones(n, dtype=int)
    n_components = 0

    for start_node in range(n):
        if labels[start_node] >= 0:
            # Already visited
            continue

        # BFS/DFS from start_node
        stack = [start_node]
        labels[start_node] = n_components

        while stack:
            node = stack.pop()
            neighbors = np.where(Zeta_sym[node, :] > 0)[0]

            for neighbor in neighbors:
                if labels[neighbor] < 0:
                    labels[neighbor] = n_components
                    stack.append(neighbor)

        n_components += 1

    # Compute component sizes
    component_sizes = {}
    for comp_id in range(n_components):
        component_sizes[comp_id] = np.sum(labels == comp_id)

    return labels, n_components, component_sizes


def find_connected_components_networkx(Zeta_sym):
    """
    Find connected components using NetworkX (alternative)

    Parameters:
    -----------
    Zeta_sym : ndarray (n, n)
        Symmetrized undirected metagraph adjacency matrix

    Returns:
    --------
    labels : ndarray (n,)
        Component assignment for each node
    n_components : int
        Number of connected components
    component_sizes : dict
        Mapping from component label to size
    """

    G = nx.Graph(Zeta_sym)
    n_components = nx.number_connected_components(G)
    components = list(nx.connected_components(G))

    # Create label mapping
    labels = np.zeros(Zeta_sym.shape[0], dtype=int)
    component_sizes = {}

    for comp_id, component in enumerate(components):
        component_sizes[comp_id] = len(component)
        for node in component:
            labels[node] = comp_id

    return labels, n_components, component_sizes


def visualize_components(Zeta_sym, labels, tau_star, figsize=(14, 6)):
    """
    Visualize metagraph with component coloring

    Parameters:
    -----------
    Zeta_sym : ndarray (n, n)
        Symmetrized metagraph
    labels : ndarray (n,)
        Component assignment
    tau_star : float
        Critical scale parameter
    figsize : tuple
        Figure size

    Returns:
    --------
    fig : matplotlib.figure.Figure
    axes : ndarray of matplotlib.axes.Axes
    """

    n = Zeta_sym.shape[0]
    n_components = len(np.unique(labels))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Metagraph with node coloring by component
    ax = axes[0]

    # Reorder nodes by component for better visualization
    order = np.argsort(labels)
    Zeta_reordered = Zeta_sym[np.ix_(order, order)]
    labels_reordered = labels[order]

    im = ax.imshow(Zeta_reordered, cmap='binary', aspect='auto', interpolation='nearest')
    ax.set_title(f'Meta-Graph with Connected Components (τ* = {tau_star:.4f})',
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Neuron j (reordered by component)', fontsize=11)
    ax.set_ylabel('Neuron i (reordered by component)', fontsize=11)

    # Draw component boundaries
    boundaries = np.where(np.diff(labels_reordered) != 0)[0] + 1
    for boundary in boundaries:
        ax.axhline(y=boundary - 0.5, color='red', linewidth=1.5, alpha=0.7)
        ax.axvline(x=boundary - 0.5, color='red', linewidth=1.5, alpha=0.7)

    # Plot 2: Component size distribution
    ax = axes[1]

    sizes = np.array([len(np.where(labels == c)[0]) for c in range(n_components)])
    colors = plt.cm.tab20(np.arange(n_components) % 20)

    bars = ax.bar(range(n_components), sizes, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Component ID', fontsize=11)
    ax.set_ylabel('Component Size', fontsize=11)
    ax.set_title(f'Connected Component Sizes ({n_components} components)',
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add size labels on bars
    for i, (bar, size) in enumerate(zip(bars, sizes)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               f'{int(size)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()

    return fig, axes


def print_component_summary(labels, n_original_nodes):
    """
    Print summary statistics of connected components

    Parameters:
    -----------
    labels : ndarray (n,)
        Component assignment for each node
    n_original_nodes : int
        Number of nodes in original network
    """

    n_components = len(np.unique(labels))
    sizes = np.array([np.sum(labels == c) for c in range(n_components)])

    print("\n" + "=" * 70)
    print("CONNECTED COMPONENT ANALYSIS")
    print("=" * 70)
    print(f"\nNumber of components: {n_components}")
    print(f"Total nodes: {n_original_nodes}")
    print(f"\nComponent sizes:")

    for comp_id in np.argsort(-sizes):
        size = sizes[comp_id]
        nodes_in_comp = np.where(labels == comp_id)[0]
        print(f"  Component {comp_id:2d}: {size:3d} nodes (indices: {nodes_in_comp[:5]}{'...' if len(nodes_in_comp) > 5 else ''})")

    print(f"\nSize statistics:")
    print(f"  Min: {sizes.min()}, Max: {sizes.max()}, Mean: {sizes.mean():.2f}, Std: {sizes.std():.2f}")
    print(f"  Largest component spans {100*sizes.max()/n_original_nodes:.1f}% of network")
    print("=" * 70)


# ============================================================================
# MAIN: Connected component analysis
# ============================================================================

if __name__ == "__main__":
    # Assuming Zeta, tau_star from metagraph construction

    print("=" * 70)
    print("CONNECTED COMPONENT ANALYSIS OF METAGRAPH")
    print("=" * 70)

    # Step 1: Symmetrize
    print("\nStep 1: Symmetrizing metagraph...")
    Zeta_sym = symmetrize_metagraph(Zeta, method='union')
    n_asymmetric = np.sum(Zeta != Zeta.T)
    print(f"  ✓ Asymmetric entries before symmetrization: {n_asymmetric}")
    print(f"    After symmetrization: density = {np.sum(Zeta_sym) / (Zeta_sym.shape[0]**2):.4f}")

    # Step 2: Find connected components (use DFS for efficiency)
    print("\nStep 2: Finding connected components...")
    labels, n_components, component_sizes = find_connected_components_dfs(Zeta_sym)
    print(f"  ✓ Found {n_components} connected component(s)")

    # Step 3: Print summary
    print_component_summary(labels, Zeta_sym.shape[0])

    # Step 4: Visualize
    print("\nStep 3: Creating visualizations...")
    fig, axes = visualize_components(Zeta_sym, labels, tau_star)
    plt.savefig('metagraph_components.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n✓ Visualization saved to 'metagraph_components.png'")
