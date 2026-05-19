"""
Diffusion Dynamics Evolution: Compute Entropy and Specific Heat
Evolves the canonical density operator from tau=0 to tau=100
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# Assuming A, D_out, P_hat, L_hat are available from the notebook
# If not, reconstruct them from the connectivity matrix

def evolve_diffusion_dynamics(L_hat, tau_min=0.01, tau_max=100, n_tau=200):
    """
    Evolve diffusion dynamics over inverse temperature tau

    Parameters:
    -----------
    L_hat : ndarray (n, n)
        Random walk Laplacian = I - P_hat
    tau_min : float
        Starting inverse temperature (avoid exactly 0)
    tau_max : float
        Maximum inverse temperature
    n_tau : int
        Number of tau points to sample (logarithmic spacing)

    Returns:
    --------
    tau_array : ndarray (n_tau,)
        Array of tau values (log-spaced)
    S_array : ndarray (n_tau,)
        Entropy at each tau
    C_array : ndarray (n_tau-1,)
        Specific heat (computed from finite differences)
    K_hat_array : list of ndarrays
        Diffusion kernels at each tau
    rho_hat_array : list of ndarrays
        Density operators at each tau
    """

    # Log-spaced tau values (important for numerical derivatives)
    tau_array = np.logspace(np.log10(tau_min), np.log10(tau_max), n_tau)

    N = L_hat.shape[0]
    S_array = np.zeros(n_tau)
    K_hat_array = []
    rho_hat_array = []
    eigenvalues_rho_array = []

    print(f"Computing diffusion dynamics for {n_tau} tau values...")

    for idx, tau in enumerate(tau_array):
        if idx % 20 == 0:
            print(f"  tau = {tau:.4f} ({idx+1}/{n_tau})")

        # 1. Compute diffusion kernel: K(tau) = exp(-tau * L_hat)
        K_hat = la.expm(-tau * L_hat)
        K_hat_array.append(K_hat)

        # 2. Compute canonical density operator
        # rho_hat(tau) = diag(K_hat) / Trace(K_hat)
        rho_hat_diag = np.diag(K_hat) / np.trace(K_hat)
        rho_hat = np.diag(rho_hat_diag)
        rho_hat_array.append(rho_hat)

        # 3. Compute eigenvalues of density operator
        eigenvalues_rho = np.linalg.eigvalsh(rho_hat)  # Use eigvalsh for symmetric matrices
        eigenvalues_rho = eigenvalues_rho[eigenvalues_rho > 1e-15]  # Filter out numerical zeros
        eigenvalues_rho_array.append(eigenvalues_rho)

        # 4. Compute entropy: S(tau) = -1/ln(N) * sum(mu_i * ln(mu_i))
        # Normalized entropy (0 <= S <= 1)
        S = -1.0 / np.log(N) * np.sum(eigenvalues_rho * np.log(eigenvalues_rho + 1e-12))
        S_array[idx] = S

    # 5. Compute specific heat from numerical derivative
    # C = -dS/d(ln(tau))
    # Using finite differences: C(tau_i) = -(S(tau_{i+1}) - S(tau_{i-1})) / (ln(tau_{i+1}) - ln(tau_{i-1}))

    ln_tau = np.log(tau_array)
    dS_dln_tau = np.gradient(S_array, ln_tau)  # Numerical gradient
    C_array = -dS_dln_tau[:-1]  # Specific heat (one fewer point due to derivative)
    tau_array_C = tau_array[:-1]

    print("Done!")

    return tau_array, S_array, tau_array_C, C_array, K_hat_array, rho_hat_array, eigenvalues_rho_array


def plot_thermodynamic_quantities(tau_array, S_array, tau_array_C, C_array):
    """
    Plot entropy and specific heat

    Parameters:
    -----------
    tau_array : ndarray
        Tau values for entropy
    S_array : ndarray
        Entropy values
    tau_array_C : ndarray
        Tau values for specific heat
    C_array : ndarray
        Specific heat values
    """

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: Entropy S(tau)
    ax = axes[0]
    ax.semilogx(tau_array, S_array, 'b-', linewidth=2, label='S(τ)')
    ax.scatter(tau_array, S_array, s=20, c='blue', alpha=0.6)
    ax.set_xlabel('Inverse Temperature τ = 1/T', fontsize=12)
    ax.set_ylabel('Entropy S(τ)', fontsize=12)
    ax.set_title('Entropy Evolution in Diffusion Dynamics', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.05])

    # Plot 2: Specific Heat C(tau)
    ax = axes[1]
    ax.semilogx(tau_array_C, C_array, 'r-', linewidth=2, label='C(τ) = -dS/d(ln τ)')
    ax.scatter(tau_array_C, C_array, s=20, c='red', alpha=0.6)
    ax.set_xlabel('Inverse Temperature τ = 1/T', fontsize=12)
    ax.set_ylabel('Specific Heat C(τ)', fontsize=12)
    ax.set_title('Specific Heat in Diffusion Dynamics', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()

    return fig, axes


# Main execution (to be run in notebook)
if __name__ == "__main__":
    print("=" * 70)
    print("DIFFUSION DYNAMICS EVOLUTION: τ from 0.01 to 100")
    print("=" * 70)

    # Evolve dynamics (use log-spaced points for better derivative estimation)
    tau_array, S_array, tau_array_C, C_array, K_hat_array, rho_hat_array, ev_array = \
        evolve_diffusion_dynamics(L_hat, tau_min=0.01, tau_max=100, n_tau=150)

    # Create plots
    fig, axes = plot_thermodynamic_quantities(tau_array, S_array, tau_array_C, C_array)
    plt.savefig('diffusion_thermodynamics.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Entropy range:     S ∈ [{S_array.min():.6f}, {S_array.max():.6f}]")
    print(f"Specific heat range: C ∈ [{C_array.min():.6f}, {C_array.max():.6f}]")
    print(f"\nAt τ = 0.01:     S = {S_array[0]:.6f}")
    print(f"At τ = 100:      S = {S_array[-1]:.6f}")
    print(f"\nEntropy change (ΔS): {S_array[-1] - S_array[0]:.6f}")
    print(f"Max specific heat:   {np.max(np.abs(C_array)):.6f} at τ = {tau_array_C[np.argmax(np.abs(C_array))]:.4f}")
