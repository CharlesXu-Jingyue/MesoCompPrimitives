"""
Meta-Graph Construction from Diffusion Density Operator
Identifies critical scale structure based on specific heat peak
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.signal import argrelextrema

def find_critical_scale(tau_array, C_array):
    """
    Find the first peak (extremum) in specific heat C(tau)

    Parameters:
    -----------
    tau_array : ndarray
        Array of scale parameters
    C_array : ndarray
        Specific heat values

    Returns:
    --------
    tau_star : float
        Scale parameter at first peak
    idx_star : int
        Index of peak in arrays
    C_peak : float
        Specific heat value at peak
    """

    # Find local extrema of |C(tau)| (peaks in magnitude)
    peaks = argrelextrema(np.abs(C_array), np.greater, order=5)[0]

    if len(peaks) == 0:
        print("Warning: No peaks found. Using maximum of |C|.")
        idx_star = np.argmax(np.abs(C_array))
    else:
        # Take the first (leftmost) peak
        idx_star = peaks[0]

    tau_star = tau_array[idx_star]
    C_peak = C_array[idx_star]

    print(f"Critical scale found:")
    print(f"  τ* = {tau_star:.6f}")
    print(f"  C(τ*) = {C_peak:.6f}")
    print(f"  Index: {idx_star}/{len(tau_array)}")

    return tau_star, idx_star, C_peak


def construct_metagraph(rho_hat, threshold_method='normalize'):
    """
    Construct binary metagraph from density operator

    Parameters:
    -----------
    rho_hat : ndarray (n, n)
        Canonical density operator at critical scale
    threshold_method : str
        Method for thresholding ('normalize' uses diagonal normalization)

    Returns:
    --------
    rho_hat_normalized : ndarray (n, n)
        Normalized diffusion matrix: rho_ij' = rho_ij / min(rho_ii, rho_jj)
    Zeta : ndarray (n, n)
        Binary metagraph: 1 if rho_ij' > 1, else 0
    """

    n = rho_hat.shape[0]
    rho_hat_normalized = np.zeros_like(rho_hat)

    # Normalize by diagonal minimum
    # rho_ij' = rho_ij / min(rho_ii, rho_jj)
    for i in range(n):
        for j in range(n):
            denom = min(rho_hat[i, i], rho_hat[j, j])
            if denom > 1e-12:
                rho_hat_normalized[i, j] = rho_hat[i, j] / denom
            else:
                rho_hat_normalized[i, j] = 0

    # Threshold to binary
    Zeta = (rho_hat_normalized > 1.0).astype(int)

    print(f"\nMeta-graph construction:")
    print(f"  Density operator diagonal range: [{np.diag(rho_hat).min():.6f}, {np.diag(rho_hat).max():.6f}]")
    print(f"  Normalized values range: [{rho_hat_normalized.min():.6f}, {rho_hat_normalized.max():.6f}]")
    print(f"  Edges in metagraph: {np.sum(Zeta)}")
    print(f"  Metagraph density: {np.sum(Zeta) / (n*n):.4f}")

    return rho_hat_normalized, Zeta


def visualize_metagraph(Zeta, rho_hat_normalized=None, node_labels=None, figsize=(12, 5)):
    """
    Visualize the metagraph and diffusion matrix

    Parameters:
    -----------
    Zeta : ndarray (n, n)
        Binary metagraph adjacency matrix
    rho_hat_normalized : ndarray (n, n), optional
        Normalized diffusion matrix for reference
    node_labels : list, optional
        Labels for nodes
    figsize : tuple
        Figure size

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    axes : ndarray of matplotlib.axes.Axes
        Subplot axes
    """

    n = Zeta.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Metagraph adjacency matrix
    ax = axes[0]
    im = ax.imshow(Zeta, cmap='binary', aspect='auto', interpolation='nearest')
    ax.set_title('Binary Meta-Graph Ζ (ρ\'_ij > 1)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Neuron j', fontsize=11)
    ax.set_ylabel('Neuron i', fontsize=11)
    plt.colorbar(im, ax=ax, label='Connection (0/1)')

    # Plot 2: Normalized diffusion matrix
    if rho_hat_normalized is not None:
        ax = axes[1]
        im = ax.imshow(rho_hat_normalized, cmap='RdYlBu_r', aspect='auto',
                       vmin=0, vmax=2)
        ax.set_title('Normalized Diffusion ρ\'_ij = ρ_ij / min(ρ_ii, ρ_jj)',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Neuron j', fontsize=11)
        ax.set_ylabel('Neuron i', fontsize=11)
        cbar = plt.colorbar(im, ax=ax, label='ρ\'_ij')
        cbar.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Threshold')

    plt.tight_layout()

    return fig, axes


def analyze_metagraph_structure(Zeta, name="Meta-graph"):
    """
    Analyze graph properties of the metagraph

    Parameters:
    -----------
    Zeta : ndarray (n, n)
        Binary metagraph adjacency matrix
    name : str
        Name for reporting

    Returns:
    --------
    stats : dict
        Dictionary of statistics
    """

    n = Zeta.shape[0]

    # Convert to NetworkX graph
    G = nx.DiGraph(Zeta)

    # Compute statistics
    stats = {
        'n_nodes': n,
        'n_edges': np.sum(Zeta),
        'density': np.sum(Zeta) / (n * n),
        'n_components': nx.number_weakly_connected_components(G),
        'n_strongly_connected': nx.number_strongly_connected_components(G),
    }

    # In-degree and out-degree
    in_degrees = np.sum(Zeta, axis=0)
    out_degrees = np.sum(Zeta, axis=1)

    stats['in_degree_range'] = (in_degrees.min(), in_degrees.max())
    stats['out_degree_range'] = (out_degrees.min(), out_degrees.max())
    stats['in_degree_mean'] = in_degrees.mean()
    stats['out_degree_mean'] = out_degrees.mean()

    # Reciprocity
    reciprocal_edges = np.sum(Zeta * Zeta.T) / 2
    stats['reciprocity'] = (2 * reciprocal_edges) / stats['n_edges'] if stats['n_edges'] > 0 else 0

    print(f"\n{name} Structure Analysis:")
    print(f"  Nodes: {stats['n_nodes']}")
    print(f"  Edges: {stats['n_edges']}")
    print(f"  Density: {stats['density']:.4f}")
    print(f"  Weakly connected components: {stats['n_components']}")
    print(f"  Strongly connected components: {stats['n_strongly_connected']}")
    print(f"  In-degree range: {stats['in_degree_range']}")
    print(f"  Out-degree range: {stats['out_degree_range']}")
    print(f"  Mean in-degree: {stats['in_degree_mean']:.2f}")
    print(f"  Mean out-degree: {stats['out_degree_mean']:.2f}")
    print(f"  Reciprocity: {stats['reciprocity']:.4f}")

    return stats


# ============================================================================
# MAIN: Construct metagraph from critical scale
# ============================================================================

if __name__ == "__main__":
    # Assuming tau_array, C_array, and rho_hat_list are available from diffusion evolution

    print("=" * 70)
    print("META-GRAPH CONSTRUCTION AT CRITICAL SCALE")
    print("=" * 70)

    # Step 1: Find critical scale (first peak of C)
    tau_star, idx_star, C_peak = find_critical_scale(tau_array, C_array)

    # Step 2: Get density operator at critical scale
    rho_hat_critical = rho_hat_list[idx_star]

    # Step 3: Construct metagraph
    rho_hat_normalized, Zeta = construct_metagraph(rho_hat_critical)

    # Step 4: Analyze structure
    stats = analyze_metagraph_structure(Zeta)

    # Step 5: Visualize
    fig, axes = visualize_metagraph(Zeta, rho_hat_normalized)
    plt.savefig('metagraph_structure.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n" + "=" * 70)
    print("Meta-graph saved to 'metagraph_structure.png'")
    print("=" * 70)
