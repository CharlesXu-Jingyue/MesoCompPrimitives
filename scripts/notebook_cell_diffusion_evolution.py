# ============================================================================
# NOTEBOOK CELL: Diffusion Dynamics Evolution (tau = 0.01 to 100)
# Run this cell to evolve dynamics and plot entropy S and specific heat C
# ============================================================================

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# Log-spaced tau values (avoid tau=0 for numerical stability)
tau_array = np.logspace(-2, 2, 150)  # From 0.01 to 100
N = L_hat.shape[0]

S_array = np.zeros(len(tau_array))
K_hat_list = []
rho_hat_list = []

print(f"Evolving diffusion dynamics for {len(tau_array)} tau values...\n")

# ============================================================================
# Stage 1: Compute Entropy at each tau
# ============================================================================

for idx, tau in enumerate(tau_array):
    if idx % 20 == 0:
        print(f"  tau = {tau:.6f} ({idx+1}/{len(tau_array)})")

    # Diffusion kernel: K(tau) = exp(-tau * L_hat)
    K_hat = la.expm(-tau * L_hat)
    K_hat_list.append(K_hat)

    # Canonical density operator: rho_hat(tau) = diag(K_hat) / Trace(K_hat)
    rho_hat_diag = np.diag(K_hat) / np.trace(K_hat)
    rho_hat = np.diag(rho_hat_diag)
    rho_hat_list.append(rho_hat)

    # Eigenvalues of density operator
    eigenvalues_rho = np.linalg.eigvalsh(rho_hat)
    eigenvalues_rho = eigenvalues_rho[eigenvalues_rho > 1e-15]

    # Entropy: S(tau) = -1/ln(N) * sum(mu_i * ln(mu_i))
    S = -1.0 / np.log(N) * np.sum(eigenvalues_rho * np.log(eigenvalues_rho + 1e-12))
    S_array[idx] = S

print("✓ Entropy computation complete\n")

# ============================================================================
# Stage 2: Compute Specific Heat from Numerical Derivative
# ============================================================================

# C(tau) = -dS/d(ln(tau))
ln_tau = np.log(tau_array)
dS_dln_tau = np.gradient(S_array, ln_tau)
C_array = -dS_dln_tau

print("✓ Specific heat computation complete\n")

# ============================================================================
# Stage 3: Plotting
# ============================================================================

fig, axes = plt.subplots(2, 1, figsize=(11, 8))

# Plot 1: Entropy S(tau)
ax = axes[0]
ax.semilogx(tau_array, S_array, 'b-', linewidth=2.5, label='S(τ)')
ax.scatter(tau_array[::10], S_array[::10], s=30, c='blue', alpha=0.5, edgecolors='darkblue')
ax.fill_between(tau_array, S_array, alpha=0.1, color='blue')
ax.set_ylabel('Normalized Entropy S(τ)', fontsize=12, fontweight='bold')
ax.set_title('Diffusion Dynamics: Thermodynamic Evolution (τ = 0.01 → 100)',
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, which='both', linestyle='--')
ax.legend(fontsize=11, loc='best')
ax.set_ylim([S_array.min() - 0.05, 1.05])

# Add annotations for min/max
idx_max = np.argmax(S_array)
idx_min = np.argmin(S_array)
ax.annotate(f'Max: {S_array[idx_max]:.3f}\nat τ={tau_array[idx_max]:.2f}',
            xy=(tau_array[idx_max], S_array[idx_max]),
            xytext=(tau_array[idx_max]*0.3, S_array[idx_max]-0.08),
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
            fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

# Plot 2: Specific Heat C(tau)
ax = axes[1]
ax.semilogx(tau_array, C_array, 'r-', linewidth=2.5, label='C(τ) = -dS/d(ln τ)')
ax.scatter(tau_array[::10], C_array[::10], s=30, c='red', alpha=0.5, edgecolors='darkred')
ax.fill_between(tau_array, C_array, where=(C_array >= 0), alpha=0.2, color='red', label='C > 0')
ax.fill_between(tau_array, C_array, where=(C_array < 0), alpha=0.2, color='blue', label='C < 0')
ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Inverse Temperature τ', fontsize=12, fontweight='bold')
ax.set_ylabel('Specific Heat C(τ)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, which='both', linestyle='--')
ax.legend(fontsize=11, loc='best')

# Add annotation for critical region
idx_peak = np.argmax(np.abs(C_array))
ax.annotate(f'Peak: {C_array[idx_peak]:.3f}\nat τ={tau_array[idx_peak]:.2f}',
            xy=(tau_array[idx_peak], C_array[idx_peak]),
            xytext=(tau_array[idx_peak]*3, C_array[idx_peak]*0.6),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

plt.tight_layout()
plt.savefig('diffusion_thermodynamics.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# Summary Statistics
# ============================================================================

print("=" * 70)
print("THERMODYNAMIC SUMMARY")
print("=" * 70)
print(f"\nEntropy Statistics:")
print(f"  Range:        S ∈ [{S_array.min():.6f}, {S_array.max():.6f}]")
print(f"  At τ = 0.01:  S = {S_array[0]:.6f}")
print(f"  At τ = 100:   S = {S_array[-1]:.6f}")
print(f"  Change:       ΔS = {S_array[-1] - S_array[0]:+.6f}")

print(f"\nSpecific Heat Statistics:")
print(f"  Range:        C ∈ [{C_array.min():.6f}, {C_array.max():.6f}]")
print(f"  Peak value:   |C|_max = {np.max(np.abs(C_array)):.6f}")
print(f"  Peak at:      τ = {tau_array[np.argmax(np.abs(C_array))]:.4f}")

print(f"\nNetwork Information:")
print(f"  Network size: N = {N}")
print(f"  Laplacian:    L_hat (shape {L_hat.shape})")
print(f"  Tau points:   {len(tau_array)}")
print("=" * 70)
