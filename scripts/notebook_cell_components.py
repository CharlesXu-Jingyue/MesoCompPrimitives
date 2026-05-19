# ============================================================================
# NOTEBOOK CELL: Connected Components of Metagraph
# Run this after the metagraph construction cell
# ============================================================================

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

print("=" * 70)
print("CONNECTED COMPONENT ANALYSIS OF METAGRAPH")
print("=" * 70)

# ============================================================================
# Step 1: Symmetrize Zeta
# ============================================================================

print("\nStep 1: Symmetrizing metagraph Ζ...")

# Union method: Zeta_sym[i,j] = 1 if Zeta[i,j] OR Zeta[j,i]
Zeta_sym = np.logical_or(Zeta, Zeta.T).astype(int)

# Remove self-loops (handled implicitly within components)
np.fill_diagonal(Zeta_sym, 0)

n_asymmetric = np.sum(Zeta != Zeta.T)
density_original = np.sum(Zeta) / (Zeta.shape[0]**2)
density_sym = np.sum(Zeta_sym) / (Zeta_sym.shape[0]**2)

print(f"  ✓ Original metagraph:")
print(f"    - Edges: {np.sum(Zeta)}, Density: {density_original:.4f}")
print(f"  ✓ Asymmetric entries: {n_asymmetric}")
print(f"  ✓ Symmetrized metagraph:")
print(f"    - Edges: {np.sum(Zeta_sym)}, Density: {density_sym:.4f}")

# ============================================================================
# Step 2: Find connected components using DFS
# ============================================================================

print("\nStep 2: Finding connected components via DFS...")

n = Zeta_sym.shape[0]
labels = -np.ones(n, dtype=int)
n_components = 0

# DFS to find connected components
for start_node in range(n):
    if labels[start_node] >= 0:
        continue

    # DFS from start_node
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

print(f"  ✓ Found {n_components} connected component(s)")

# ============================================================================
# Step 3: Compute component statistics
# ============================================================================

print("\nStep 3: Analyzing component structure...")

sizes = np.array([np.sum(labels == c) for c in range(n_components)])
sorted_comps = np.argsort(-sizes)

print(f"  ✓ Component sizes:")
for comp_id in sorted_comps[:min(5, n_components)]:
    size = sizes[comp_id]
    nodes_in_comp = np.where(labels == comp_id)[0]
    pct = 100 * size / n
    print(f"    Component {comp_id}: {size:3d} nodes ({pct:5.1f}%) - indices: {list(nodes_in_comp[:5])}{'...' if len(nodes_in_comp) > 5 else ''}")

if n_components > 5:
    print(f"    ... and {n_components - 5} smaller component(s)")

print(f"\n  ✓ Size statistics:")
print(f"    - Min: {sizes.min()}, Max: {sizes.max()}, Mean: {sizes.mean():.2f}, Std: {sizes.std():.2f}")
print(f"    - Largest component: {100*sizes.max()/n:.1f}% of network")

# ============================================================================
# Step 4: Visualize components
# ============================================================================

print("\nStep 4: Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Reordered metagraph with component boundaries
ax = axes[0, 0]

order = np.argsort(labels)
Zeta_reordered = Zeta_sym[np.ix_(order, order)]
labels_reordered = labels[order]

im = ax.imshow(Zeta_reordered, cmap='binary', aspect='auto', interpolation='nearest')
ax.set_title(f'Metagraph Ζ Reordered by Connected Components', fontsize=12, fontweight='bold')
ax.set_xlabel('Neuron j (reordered by component)', fontsize=11)
ax.set_ylabel('Neuron i (reordered by component)', fontsize=11)

# Draw component boundaries
boundaries = np.where(np.diff(labels_reordered) != 0)[0] + 1
for boundary in boundaries:
    ax.axhline(y=boundary - 0.5, color='red', linewidth=2, alpha=0.8)
    ax.axvline(x=boundary - 0.5, color='red', linewidth=2, alpha=0.8)

plt.colorbar(im, ax=ax, label='Connection (0/1)')

# Plot 2: Component size distribution
ax = axes[0, 1]

colors = plt.cm.tab20(np.arange(n_components) % 20)
bars = ax.bar(range(n_components), sizes, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

ax.set_xlabel('Component ID', fontsize=11)
ax.set_ylabel('Component Size', fontsize=11)
ax.set_title(f'Connected Component Sizes ({n_components} components)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add size labels on bars
for i, (bar, size) in enumerate(zip(bars, sizes)):
    if size > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
               f'{int(size)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 3: Component membership by neuron (stacked bar)
ax = axes[1, 0]

neuron_indices = np.arange(n)
colors_extended = plt.cm.tab20(np.arange(n_components) % 20)

ax.scatter(neuron_indices, labels, c=labels, cmap='tab20', s=100,
          edgecolors='black', linewidth=1, alpha=0.8)

ax.set_xlabel('Neuron Index', fontsize=11)
ax.set_ylabel('Component ID', fontsize=11)
ax.set_title('Component Assignment per Neuron', fontsize=12, fontweight='bold')
ax.set_yticks(range(n_components))
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Cumulative size
ax = axes[1, 1]

sorted_sizes = sizes[sorted_comps]
cumsum = np.cumsum(sorted_sizes)
cumsum_pct = 100 * cumsum / n

ax.bar(range(n_components), sorted_sizes, color=colors[sorted_comps],
      edgecolor='black', linewidth=1.5, alpha=0.7, label='Component size')
ax.plot(range(n_components), cumsum_pct, 'r-o', linewidth=2, markersize=6,
       label='Cumulative %', markeredgecolor='darkred')

ax.set_xlabel('Component (sorted by size)', fontsize=11)
ax.set_ylabel('Size / Cumulative %', fontsize=11)
ax.set_title('Cumulative Size Distribution', fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='right')
ax.grid(True, alpha=0.3, axis='y')

# Add right y-axis for percentage
ax2 = ax.twinx()
ax2.set_ylabel('Cumulative Percentage (%)', fontsize=11)
ax2.set_ylim(ax.get_ylim())

plt.tight_layout()
plt.savefig('metagraph_components.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# Summary and Output
# ============================================================================

print("\n" + "=" * 70)
print("COMPONENT ANALYSIS SUMMARY")
print("=" * 70)
print(f"Symmetrized metagraph vertices (nodes): {n}")
print(f"Connected components found: {n_components}")
print(f"Component labeling: c: {{1,...,{n}}} → {{0,...,{n_components-1}}}")
print(f"\nComponent size distribution:")
for idx, comp_id in enumerate(sorted_comps):
    pct = 100 * sizes[comp_id] / n
    print(f"  c = {comp_id}: {sizes[comp_id]:3d} nodes ({pct:5.1f}%)")
print("=" * 70)

print("\n✓ Component labeling stored in variable: labels")
print("✓ Visualization saved to: metagraph_components.png")
