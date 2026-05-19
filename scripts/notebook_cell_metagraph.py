# ============================================================================
# NOTEBOOK CELL: Meta-Graph Construction at Critical Scale
# Run this after the diffusion dynamics evolution cell
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.signal import argrelextrema

print("=" * 70)
print("META-GRAPH CONSTRUCTION AT CRITICAL SCALE")
print("=" * 70)

# ============================================================================
# Step 1: Find critical scale (first peak of specific heat C)
# ============================================================================

print("\nStep 1: Identifying critical scale...")

# Find local extrema of |C(tau)|
peaks = argrelextrema(np.abs(C_array), np.greater, order=5)[0]

if len(peaks) == 0:
    print("  No peaks found. Using maximum of |C|.")
    idx_star = np.argmax(np.abs(C_array))
else:
    # Take the first (leftmost) peak
    idx_star = peaks[0]

tau_star = tau_array[idx_star]
C_peak = C_array[idx_star]

print(f"  ✓ Critical scale found at:")
print(f"    τ* = {tau_star:.6f}")
print(f"    C(τ*) = {C_peak:.6f}")
print(f"    Index: {idx_star + 1}/{len(tau_array)}")

# ============================================================================
# Step 2: Get density operator at critical scale
# ============================================================================

print("\nStep 2: Extracting density operator at critical scale...")

rho_hat_critical = rho_hat_list[idx_star]
rho_diag_critical = np.diag(rho_hat_critical)

print(f"  ✓ Density operator shape: {rho_hat_critical.shape}")
print(f"    Diagonal range: [{rho_diag_critical.min():.6f}, {rho_diag_critical.max():.6f}]")
print(f"    Trace: {np.trace(rho_hat_critical):.6f}")

# ============================================================================
# Step 3: Normalize diffusion matrix by diagonal minimum
# ============================================================================

print("\nStep 3: Normalizing pairwise diffusion...")

n = rho_hat_critical.shape[0]
rho_hat_normalized = np.zeros_like(rho_hat_critical)

# rho_ij' = rho_ij / min(rho_ii, rho_jj)
for i in range(n):
    for j in range(n):
        denom = min(rho_hat_critical[i, i], rho_hat_critical[j, j])
        if denom > 1e-12:
            rho_hat_normalized[i, j] = rho_hat_critical[i, j] / denom
        else:
            rho_hat_normalized[i, j] = 0

print(f"  ✓ Normalized values range: [{rho_hat_normalized.min():.6f}, {rho_hat_normalized.max():.6f}]")

# ============================================================================
# Step 4: Threshold to binary metagraph
# ============================================================================

print("\nStep 4: Thresholding to binary metagraph...")

# Zeta_ij = 1 if rho_ij' > 1, else 0
Zeta = (rho_hat_normalized > 1.0).astype(int)

n_edges = np.sum(Zeta)
density = n_edges / (n * n)

print(f"  ✓ Metagraph constructed:")
print(f"    Edges in Ζ: {n_edges}")
print(f"    Metagraph density: {density:.4f}")
print(f"    Sparsity: {1 - density:.4f}")

# ============================================================================
# Step 5: Analyze graph structure
# ============================================================================

print("\nStep 5: Analyzing meta-graph structure...")

# Convert to NetworkX
G = nx.DiGraph(Zeta)

# Degree statistics
in_degrees = np.sum(Zeta, axis=0)
out_degrees = np.sum(Zeta, axis=1)

n_components = nx.number_weakly_connected_components(G)
n_scc = nx.number_strongly_connected_components(G)

# Reciprocity
reciprocal_edges = np.sum(Zeta * Zeta.T) / 2
reciprocity = (2 * reciprocal_edges) / n_edges if n_edges > 0 else 0

print(f"  ✓ Graph properties:")
print(f"    Nodes: {n}")
print(f"    Edges: {n_edges}")
print(f"    Weakly connected components: {n_components}")
print(f"    Strongly connected components: {n_scc}")
print(f"    In-degree: {in_degrees.min():.0f} to {in_degrees.max():.0f} (mean: {in_degrees.mean():.2f})")
print(f"    Out-degree: {out_degrees.min():.0f} to {out_degrees.max():.0f} (mean: {out_degrees.mean():.2f})")
print(f"    Reciprocity: {reciprocity:.4f}")

# ============================================================================
# Step 6: Visualize metagraph
# ============================================================================

print("\nStep 6: Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Binary metagraph
ax = axes[0, 0]
im = ax.imshow(Zeta, cmap='binary', aspect='auto', interpolation='nearest')
ax.set_title(f'Binary Meta-Graph Ζ (τ* = {tau_star:.4f})', fontsize=12, fontweight='bold')
ax.set_xlabel('Neuron j', fontsize=11)
ax.set_ylabel('Neuron i', fontsize=11)
plt.colorbar(im, ax=ax, label='Connection (0/1)')

# Plot 2: Normalized diffusion matrix with threshold
ax = axes[0, 1]
im = ax.imshow(rho_hat_normalized, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=2)
ax.set_title('Normalized Diffusion ρ\'_ij = ρ_ij / min(ρ_ii, ρ_jj)', fontsize=12, fontweight='bold')
ax.set_xlabel('Neuron j', fontsize=11)
ax.set_ylabel('Neuron i', fontsize=11)
cbar = plt.colorbar(im, ax=ax, label='ρ\'_ij')
# Add threshold line to colorbar
cbar.ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2)

# Plot 3: In-degree and out-degree distributions
ax = axes[1, 0]
ax.bar(np.arange(n), in_degrees, alpha=0.6, label='In-degree', width=0.8)
ax.bar(np.arange(n), out_degrees, alpha=0.6, label='Out-degree', width=0.8)
ax.set_xlabel('Neuron Index', fontsize=11)
ax.set_ylabel('Degree', fontsize=11)
ax.set_title('Degree Distribution in Meta-Graph', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Scatter of in-degree vs out-degree
ax = axes[1, 1]
scatter = ax.scatter(in_degrees, out_degrees, s=100, c=np.diag(Zeta), cmap='viridis',
                    alpha=0.7, edgecolors='black', linewidth=1)
ax.set_xlabel('In-Degree', fontsize=11)
ax.set_ylabel('Out-Degree', fontsize=11)
ax.set_title('Degree Correlation in Meta-Graph', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Self-loops')

plt.tight_layout()
plt.savefig('metagraph_structure.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# Summary Table
# ============================================================================

print("\n" + "=" * 70)
print("META-GRAPH SUMMARY")
print("=" * 70)
print(f"Critical scale:           τ* = {tau_star:.6f}")
print(f"Specific heat at peak:    C(τ*) = {C_peak:.6f}")
print(f"Network size:             n = {n}")
print(f"Total edges:              {n_edges} / {n*n}")
print(f"Network density:          {density:.4f}")
print(f"Connectivity structure:   {n_components} weakly connected, {n_scc} strongly connected")
print(f"Hub neurons:              {len(np.where(out_degrees > np.median(out_degrees))[0])} (out-degree > median)")
print("=" * 70)
