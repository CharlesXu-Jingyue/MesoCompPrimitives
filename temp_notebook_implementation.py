"""
Complete Notebook Implementation Code for CTRNN and Port Analysis
================================================================

This file contains the complete code that would be added to the poc.ipynb notebook
under the "System identification/Model-based" section.

Usage in notebook:
- Replace W = C_signed_grouped and group_labels = bilrg_labels with your actual variables
- Run each section sequentially for complete analysis
"""

# ============================================================================
# SECTION 1: CTRNN Analysis - Row-Sum Normalized Fixed-Point Pipeline
# ============================================================================

print("=" * 60)
print("CTRNN Analysis: Row-Sum Normalized Fixed-Point Pipeline")
print("=" * 60)

# Import required modules
from src.sysid import CTRNNAnalyzer, PortAnalyzer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# Initialize CTRNN analyzer with parameters
analyzer = CTRNNAnalyzer(
    safety_margin=0.9,      # Contraction safety factor c
    tolerance=1e-6,         # Fixed-point convergence tolerance
    damping=0.5,           # Picard iteration damping
    max_iterations=1000     # Maximum iterations
)

# Use the grouped connectivity matrix and block labels
W = C_signed_grouped  # Your signed connectivity matrix
group_labels = bilrg_labels  # Your block labels

print(f"Network size: {W.shape[0]} neurons")
print(f"Number of blocks: {len(np.unique(group_labels))}")
print(f"Original matrix norm: {np.max(np.sum(np.abs(W), axis=1)):.3f}")

# Run complete CTRNN analysis
print("\nRunning CTRNN analysis...")
ctrnn_results = analyzer.analyze(
    W=W,
    block_labels=group_labels,
    perform_optional_analyses=True
)

print("✓ CTRNN analysis completed")
print(f"Normalization factor α: {ctrnn_results.normalization_factor:.4f}")
print(f"Normalized matrix norm: {np.max(np.sum(np.abs(ctrnn_results.W_normalized), axis=1)):.4f}")

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('CTRNN Analysis: Fixed-Point & Linearization Results', fontsize=16, fontweight='bold')

# 1. Original vs Normalized Weight Matrix
ax1 = axes[0, 0]
im1 = ax1.imshow(W, cmap='RdBu_r', aspect='auto')
ax1.set_title('Original Weight Matrix W')
ax1.set_xlabel('Postsynaptic Neuron')
ax1.set_ylabel('Presynaptic Neuron')
plt.colorbar(im1, ax=ax1, shrink=0.8)

ax2 = axes[0, 1]
im2 = ax2.imshow(ctrnn_results.W_normalized, cmap='RdBu_r', aspect='auto')
ax2.set_title(f'Normalized Matrix W̃ (α={ctrnn_results.normalization_factor:.3f})')
ax2.set_xlabel('Postsynaptic Neuron')
ax2.set_ylabel('Presynaptic Neuron')
plt.colorbar(im2, ax=ax2, shrink=0.8)

# 2. Block Fixed Points
ax3 = axes[0, 2]
block_ids = sorted(ctrnn_results.fixed_points.keys())
fixed_point_values = []
for block_id in block_ids:
    fp = ctrnn_results.fixed_points[block_id]
    fixed_point_values.extend(fp)

ax3.bar(range(len(fixed_point_values)), fixed_point_values, alpha=0.7, color='steelblue')
ax3.set_title('Block Fixed Points x*')
ax3.set_xlabel('Neuron Index (by Block)')
ax3.set_ylabel('Fixed Point Value')
ax3.grid(True, alpha=0.3)

# Add block boundaries
neuron_idx = 0
for i, block_id in enumerate(block_ids):
    block_size = len(ctrnn_results.fixed_points[block_id])
    if i > 0:
        ax3.axvline(neuron_idx - 0.5, color='red', linestyle='--', alpha=0.5)
    neuron_idx += block_size

# 3. Sigmoid Gains
ax4 = axes[1, 0]
sigmoid_gains = []
for block_id in block_ids:
    gains = np.diag(ctrnn_results.sigmoid_gains[block_id])
    sigmoid_gains.extend(gains)

ax4.bar(range(len(sigmoid_gains)), sigmoid_gains, alpha=0.7, color='orange')
ax4.set_title('Sigmoid Gains σ\'(x*)')
ax4.set_xlabel('Neuron Index (by Block)')
ax4.set_ylabel('Gain Value')
ax4.grid(True, alpha=0.3)

# Add block boundaries
neuron_idx = 0
for i, block_id in enumerate(block_ids):
    block_size = len(ctrnn_results.fixed_points[block_id])
    if i > 0:
        ax4.axvline(neuron_idx - 0.5, color='red', linestyle='--', alpha=0.5)
    neuron_idx += block_size

# 4. Linearized Dynamics Matrix
ax5 = axes[1, 1]
im5 = ax5.imshow(ctrnn_results.A_global, cmap='RdBu_r', aspect='auto')
ax5.set_title('Linearized Dynamics Matrix A')
ax5.set_xlabel('State Variable')
ax5.set_ylabel('State Variable')
plt.colorbar(im5, ax=ax5, shrink=0.8)

# Add block boundaries
block_boundaries = []
current_idx = 0
for block_id in sorted(ctrnn_results.A_blocks.keys()):
    block_size = ctrnn_results.A_blocks[block_id].shape[0]
    block_boundaries.append(current_idx)
    current_idx += block_size
block_boundaries.append(current_idx)

for boundary in block_boundaries[1:-1]:
    ax5.axhline(boundary - 0.5, color='white', linewidth=2, alpha=0.7)
    ax5.axvline(boundary - 0.5, color='white', linewidth=2, alpha=0.7)

# 5. Eigenvalue Analysis
if ctrnn_results.eigenvalues is not None:
    ax6 = axes[1, 2]
    eigenvals = ctrnn_results.eigenvalues
    real_parts = np.real(eigenvals)
    imag_parts = np.imag(eigenvals)

    scatter = ax6.scatter(real_parts, imag_parts, alpha=0.7, s=50, c=range(len(eigenvals)), cmap='viridis')
    ax6.axvline(0, color='red', linestyle='--', alpha=0.5, label='Stability Boundary')
    ax6.set_title('Eigenvalues of Linearized System')
    ax6.set_xlabel('Real Part')
    ax6.set_ylabel('Imaginary Part')
    ax6.grid(True, alpha=0.3)
    ax6.legend()

    # Add stability information
    stable_count = np.sum(real_parts < 0)
    ax6.text(0.02, 0.98, f'Stable modes: {stable_count}/{len(eigenvals)}',
             transform=ax6.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
else:
    axes[1, 2].text(0.5, 0.5, 'Eigenvalue analysis\nnot performed',
                    ha='center', va='center', transform=axes[1, 2].transAxes,
                    fontsize=12, style='italic')
    axes[1, 2].set_title('Eigenvalue Analysis')

plt.tight_layout()
plt.show()

# Print convergence diagnostics
print("\n" + "="*40)
print("CONVERGENCE DIAGNOSTICS")
print("="*40)

for block_id, conv_info in ctrnn_results.convergence_info.items():
    print(f"\nBlock {block_id}:")
    print(f"  Converged: {conv_info['converged']}")
    print(f"  Iterations: {conv_info['iterations']}")
    print(f"  Final error: {conv_info['final_error']:.2e}")

# Print stability analysis
if hasattr(ctrnn_results, 'A_global'):
    stability_info = analyzer.check_global_stability(ctrnn_results.A_global)
    print(f"\nGLOBAL STABILITY:")
    print(f"  System is {'stable' if stability_info['is_stable'] else 'unstable'}")
    print(f"  Max real eigenvalue: {stability_info['max_real_eigenvalue']:.4f}")
    print(f"  Unstable modes: {stability_info['num_unstable_modes']}")

# ============================================================================
# SECTION 2: Port Analysis - Inter-Block Control Ports & Controllability
# ============================================================================

print("\n\n" + "=" * 60)
print("Port Analysis: Inter-Block Control Ports & Controllability")
print("=" * 60)

# Initialize port analyzer
port_analyzer = PortAnalyzer()

print("Running port analysis...")
print(f"Number of inter-block connections: {len(ctrnn_results.E_blocks)}")

# Run port analysis using results from CTRNN analysis
port_results = port_analyzer.analyze_ports(
    A_blocks=ctrnn_results.A_blocks,
    E_blocks=ctrnn_results.E_blocks
)

print("✓ Port analysis completed")
print(f"Total ports analyzed: {len(port_results.port_map)}")

# Create comprehensive port visualization
fig = plt.figure(figsize=(20, 24))
gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

# 1. Port Structure Overview
ax1 = fig.add_subplot(gs[0, 0])
# Create port connectivity matrix for visualization
num_blocks = len(ctrnn_results.A_blocks)
port_matrix = np.zeros((num_blocks, num_blocks))
for (dest, src) in port_results.port_map.keys():
    port_matrix[dest, src] = port_results.port_metrics[(dest, src)].trace

im1 = ax1.imshow(port_matrix, cmap='Blues', aspect='auto')
ax1.set_title('Port Structure\n(Trace of Controllability Gramians)', fontweight='bold')
ax1.set_xlabel('Source Block')
ax1.set_ylabel('Destination Block')
ax1.set_xticks(range(num_blocks))
ax1.set_yticks(range(num_blocks))

# Annotate with values
for i in range(num_blocks):
    for j in range(num_blocks):
        if port_matrix[i, j] > 0:
            ax1.text(j, i, f'{port_matrix[i, j]:.2f}', ha='center', va='center',
                    color='white' if port_matrix[i, j] > port_matrix.max()/2 else 'black')

plt.colorbar(im1, ax=ax1, shrink=0.8)

# 2. Port Controllability Gramians Heatmap
ax2 = fig.add_subplot(gs[0, 1])
port_keys = list(port_results.Wc_port.keys())
if port_keys:
    # Select first port for detailed view
    selected_port = port_keys[0]
    selected_gramian = port_results.Wc_port[selected_port]

    im2 = ax2.imshow(selected_gramian, cmap='RdYlBu_r', aspect='auto')
    ax2.set_title(f'Controllability Gramian\nPort {selected_port[1]}→{selected_port[0]}', fontweight='bold')
    ax2.set_xlabel('State Variable')
    ax2.set_ylabel('State Variable')
    plt.colorbar(im2, ax=ax2, shrink=0.8)
else:
    ax2.text(0.5, 0.5, 'No ports\nanalyzed', ha='center', va='center',
             transform=ax2.transAxes, fontsize=12)
    ax2.set_title('Port Controllability Gramian')

# 3. Total vs Port Gramians Comparison
ax3 = fig.add_subplot(gs[0, 2])
if len(ctrnn_results.A_blocks) > 0:
    # Compare total vs sum of port Gramians for first block
    block_id = list(ctrnn_results.A_blocks.keys())[0]
    total_gramian = port_results.Wc_total[block_id]

    # Sum port Gramians
    port_sum = np.zeros_like(total_gramian)
    for (dest, src), W_port in port_results.Wc_port.items():
        if dest == block_id:
            port_sum += W_port

    # Plot difference
    difference = total_gramian - port_sum
    im3 = ax3.imshow(difference, cmap='RdBu_r', aspect='auto')
    ax3.set_title(f'Gramian Additivity Check\nBlock {block_id} (Total - Sum)', fontweight='bold')
    ax3.set_xlabel('State Variable')
    ax3.set_ylabel('State Variable')
    plt.colorbar(im3, ax=ax3, shrink=0.8)

    max_diff = np.max(np.abs(difference))
    ax3.text(0.02, 0.98, f'Max diff: {max_diff:.2e}', transform=ax3.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# 4. Port Metrics Comparison
ax4 = fig.add_subplot(gs[1, :])
if port_results.port_metrics:
    port_labels = [f"{src}→{dest}" for (dest, src) in port_results.port_metrics.keys()]
    metrics_data = {
        'Trace': [m.trace for m in port_results.port_metrics.values()],
        'Max Eigenvalue': [m.lambda_max for m in port_results.port_metrics.values()],
        'Rank': [m.rank for m in port_results.port_metrics.values()],
        'Condition Number': [np.log10(m.condition_number) for m in port_results.port_metrics.values()]
    }

    x = np.arange(len(port_labels))
    width = 0.2

    for i, (metric, values) in enumerate(metrics_data.items()):
        offset = (i - 1.5) * width
        label = metric if metric != 'Condition Number' else 'log₁₀(Cond. Num.)'
        ax4.bar(x + offset, values, width, label=label, alpha=0.8)

    ax4.set_title('Port Controllability Metrics Comparison', fontweight='bold')
    ax4.set_xlabel('Port (Source→Destination)')
    ax4.set_ylabel('Metric Value')
    ax4.set_xticks(x)
    ax4.set_xticklabels(port_labels, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

# 5. Eigenmode Analysis
ax5 = fig.add_subplot(gs[2, 0])
if port_keys:
    selected_gramian = port_results.Wc_port[port_keys[0]]
    eigenvals, eigenvecs = np.linalg.eigh(selected_gramian)
    eigenvals = eigenvals[::-1]  # Sort descending

    ax5.semilogy(eigenvals, 'o-', linewidth=2, markersize=6, color='steelblue')
    ax5.set_title(f'Gramian Eigenvalues\nPort {port_keys[0][1]}→{port_keys[0][0]}', fontweight='bold')
    ax5.set_xlabel('Mode Index')
    ax5.set_ylabel('Eigenvalue (log scale)')
    ax5.grid(True, alpha=0.3)

    # Highlight dominant modes
    dominant_threshold = eigenvals[0] * 0.01  # 1% of max
    dominant_modes = np.sum(eigenvals > dominant_threshold)
    ax5.axhline(dominant_threshold, color='red', linestyle='--', alpha=0.7)
    ax5.text(0.02, 0.98, f'Dominant modes: {dominant_modes}', transform=ax5.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# 6. Port Rankings
ax6 = fig.add_subplot(gs[2, 1])
if port_results.top_ports:
    # Show rankings for first block
    block_id = list(port_results.top_ports.keys())[0]
    rankings = port_results.top_ports[block_id]

    if rankings:
        sources = [f"Block {src}" for src, _ in rankings]
        values = [val for _, val in rankings]

        bars = ax6.bar(sources, values, alpha=0.8, color='lightcoral')
        ax6.set_title(f'Port Rankings for Block {block_id}\n(by Trace)', fontweight='bold')
        ax6.set_xlabel('Source Block')
        ax6.set_ylabel('Controllability Metric')
        ax6.tick_params(axis='x', rotation=45)

        # Annotate bars with values
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(values),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        ax6.grid(True, alpha=0.3, axis='y')

# 7. Block Controllability Summary
ax7 = fig.add_subplot(gs[2, 2])
block_totals = []
block_ids_sorted = sorted(port_results.total_metrics.keys())
for block_id in block_ids_sorted:
    total_metric = port_results.total_metrics[block_id].trace
    block_totals.append(total_metric)

bars = ax7.bar([f"Block {bid}" for bid in block_ids_sorted], block_totals,
               alpha=0.8, color='lightgreen')
ax7.set_title('Total Controllability by Block', fontweight='bold')
ax7.set_xlabel('Block ID')
ax7.set_ylabel('Total Controllability (Trace)')
ax7.tick_params(axis='x', rotation=45)

# Annotate bars
for bar, val in zip(bars, block_totals):
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(block_totals),
            f'{val:.3f}', ha='center', va='bottom', fontsize=9)

ax7.grid(True, alpha=0.3, axis='y')

# 8. Stability and Condition Analysis
ax8 = fig.add_subplot(gs[3, 0])
condition_numbers = [port_results.port_metrics[key].condition_number
                    for key in port_results.port_metrics.keys()]
ranks = [port_results.port_metrics[key].rank
         for key in port_results.port_metrics.keys()]

scatter = ax8.scatter(ranks, np.log10(condition_numbers), alpha=0.7, s=60,
                     c=range(len(ranks)), cmap='viridis')
ax8.set_title('Port Conditioning Analysis', fontweight='bold')
ax8.set_xlabel('Gramian Rank')
ax8.set_ylabel('log₁₀(Condition Number)')
ax8.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax8, label='Port Index', shrink=0.8)

# 9. Port Contribution Analysis
ax9 = fig.add_subplot(gs[3, 1:])
if len(port_results.port_metrics) > 1:
    # Create stacked bar chart of port contributions
    block_contributions = {}
    for (dest, src), metrics in port_results.port_metrics.items():
        if dest not in block_contributions:
            block_contributions[dest] = {}
        block_contributions[dest][src] = metrics.trace

    # Prepare data for stacked bar chart
    dest_blocks = sorted(block_contributions.keys())
    all_sources = set()
    for contrib in block_contributions.values():
        all_sources.update(contrib.keys())
    all_sources = sorted(all_sources)

    # Create stacked bars
    bottom = np.zeros(len(dest_blocks))
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_sources)))

    for i, source in enumerate(all_sources):
        values = []
        for dest in dest_blocks:
            values.append(block_contributions[dest].get(source, 0))

        ax9.bar(dest_blocks, values, bottom=bottom, label=f'From Block {source}',
               color=colors[i], alpha=0.8)
        bottom += values

    ax9.set_title('Port Controllability Contributions by Destination Block', fontweight='bold')
    ax9.set_xlabel('Destination Block')
    ax9.set_ylabel('Controllability Contribution')
    ax9.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax9.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Print detailed port analysis summary
from src.sysid.ports import summarize_port_rankings, validate_port_analysis

print("\n" + "="*50)
print("PORT ANALYSIS SUMMARY")
print("="*50)

# Validation
validation = validate_port_analysis(port_results)
print(f"Analysis valid: {validation['valid']}")
if validation['warnings']:
    print("Warnings:")
    for warning in validation['warnings']:
        print(f"  - {warning}")

print(f"\nTotal ports: {validation['summary']['num_ports']}")
print(f"Total blocks: {validation['summary']['num_blocks']}")

# Detailed rankings
rankings_summary = summarize_port_rankings(port_results, top_k=3)

for block_id, block_summary in rankings_summary.items():
    print(f"\nBlock {block_id}:")
    print(f"  Total incoming ports: {block_summary['total_incoming_ports']}")
    print(f"  Total controllability: {block_summary['total_controllability']:.4f}")
    print("  Top contributing ports:")

    for port_info in block_summary['top_ports']:
        print(f"    Rank {port_info['rank']}: Block {port_info['source_block']} → "
              f"Block {block_id} ({port_info['relative_contribution']:.1f}% contribution, "
              f"metric={port_info['metric_value']:.4f})")

print(f"\n✓ Complete CTRNN and Port Analysis finished!")
print(f"  - Linearized {W.shape[0]}-neuron network into {len(ctrnn_results.A_blocks)} blocks")
print(f"  - Analyzed {len(port_results.port_map)} inter-block control ports")
print(f"  - Computed per-port controllability metrics and rankings")