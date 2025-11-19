"""
Continuous-Time Recurrent Neural Network (CTRNN) Analysis.

This module implements row-sum normalized fixed-point and blockwise linearization
pipeline for rate-based RNNs with sigmoid activation functions.
"""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_continuous_lyapunov, schur, solve
from scipy.sparse.linalg import spsolve
from typing import Union, Dict, List, Tuple, Optional, Any
import warnings
from dataclasses import dataclass


@dataclass
class FixedPointAnalysis:
    """Results of fixed-point and linearization analysis."""

    # Global normalization
    W_normalized: np.ndarray
    normalization_factor: float
    original_norm: float
    safety_margin: float

    # Per-block fixed points
    fixed_points: Dict[int, np.ndarray]
    fixed_points_global: Dict[int, np.ndarray]
    sigmoid_gains: Dict[int, np.ndarray]
    convergence_info: Dict[int, Dict[str, Any]]

    # Linearized dynamics
    A_blocks: Dict[int, np.ndarray]  # Diagonal blocks A_rr
    B_blocks: Dict[Tuple[int, int], np.ndarray]  # Off-diagonal blocks B_rs
    C_blocks: Dict[int, np.ndarray]  # Per-block observation matrices C_rr
    E_blocks: Dict[Tuple[int, int], np.ndarray]  # Off-diagonal blocks E_rs
    A_assembled: np.ndarray  # Assembled block matrix A
    A_global: np.ndarray  # Full linearized system matrix
    B_linear: Optional[np.ndarray] = None

    # Optional analyses
    eigenvalues: Optional[Dict[int, np.ndarray]] = None
    schur_decomp: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None
    balanced_truncation: Optional[Dict[int, Dict[str, Any]]] = None


class CTRNNAnalyzer:
    """
    Continuous-Time Recurrent Neural Network Analyzer.

    Implements row-sum normalized fixed-point computation and blockwise
    linearization for rate-based RNNs with sigmoid activation functions.

    Parameters
    ----------
    safety_margin : float, default=0.9
        Safety margin c for global contraction (c < 1)
    tolerance : float, default=1e-6
        Convergence tolerance for fixed-point iteration
    damping : float, default=0.5
        Damping factor η for Picard iteration (0 < η ≤ 1)
    max_iterations : int, default=1000
        Maximum iterations for fixed-point computation
    """

    def __init__(self, safety_margin: float = 0.9, tolerance: float = 1e-6,
                 damping: float = 0.5, max_iterations: int = 1000):

        if not (0 < safety_margin < 1):
            raise ValueError("safety_margin must be in (0, 1)")
        if not (0 < damping <= 1):
            raise ValueError("damping must be in (0, 1]")

        self.safety_margin = safety_margin
        self.tolerance = tolerance
        self.damping = damping
        self.max_iterations = max_iterations

    def analyze(self, W: np.ndarray, block_labels: np.ndarray,
                bias: Optional[np.ndarray] = None,
                time_constants: Optional[Union[np.ndarray, float]] = None,
                input_weights: Optional[np.ndarray] = None,
                perform_optional_analyses: bool = True) -> FixedPointAnalysis:
        """
        Perform complete CTRNN analysis pipeline.

        Parameters
        ----------
        W : ndarray, shape (N, N)
            Signed weight matrix
        block_labels : ndarray, shape (N,)
            Group labels g(i) ∈ {1, ..., k} for each neuron
        bias : ndarray, shape (N,), optional
            Bias vector b. Defaults to zeros.
        time_constants : ndarray, shape (k,) or float, optional
            Time constants τ_r for each block or single value for all blocks.
            Defaults to 1.0.
        input_weights : ndarray, shape (N, M), optional
            External input weights B for linearization
        perform_optional_analyses : bool, default=True
            Whether to perform eigenvalue, Schur, and balanced truncation analyses

        Returns
        -------
        FixedPointAnalysis
            Complete analysis results
        """
        N = W.shape[0]
        if bias is None:
            bias = np.zeros(N)

        # Step 1: Global row-sum normalization
        W_norm, alpha, s_inf = self._global_normalization(W)

        # Get block information
        block_info = self._get_block_info(block_labels)

        # Step 2: Fixed-point computation
        # Per-block
        fixed_points, gains, conv_info = self._compute_fixed_points(
            W_norm, block_info, bias
        )

        # Global
        block_info_global = {0: np.arange(N)}
        fixed_points_global, gains_global, conv_info_global = self._compute_fixed_points(
            W_norm, block_info_global, bias
        )

        # Step 3: Sigmoid gains already computed in step 2

        # Step 4: Construct linearized dynamics
        A_blocks, B_blocks, C_blocks, E_blocks, A_assembled, B_linear = self._construct_linearized_dynamics(
            W_norm, block_info, gains, time_constants, input_weights
        )

        A_global, _, _, _, _, _ = self._construct_linearized_dynamics(
            W_norm, block_info_global, gains_global, time_constants, input_weights
        )
        A_global = A_global[0]

        # Create base results
        results = FixedPointAnalysis(
            W_normalized=W_norm,
            normalization_factor=alpha,
            original_norm=s_inf,
            safety_margin=self.safety_margin,
            fixed_points=fixed_points,
            fixed_points_global=fixed_points_global,
            sigmoid_gains=gains,
            convergence_info=conv_info,
            A_blocks=A_blocks,
            B_blocks=B_blocks,
            C_blocks=C_blocks,
            E_blocks=E_blocks,
            A_assembled=A_assembled,
            A_global=A_global,
            B_linear=B_linear
        )

        # Step 5: Optional analyses
        if perform_optional_analyses:
            results.eigenvalues = self._compute_eigenvalues(A_blocks)
            results.schur_decomp = self._compute_schur_decomposition(A_blocks)

            if input_weights is not None:
                results.balanced_truncation = self._compute_balanced_truncation(
                    A_blocks, input_weights, block_info, time_constants
                )

        return results

    def _global_normalization(self, W: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Step 1: Global row-sum normalization.

        Computes α = 4c/s_∞ where s_∞ = ||W||_∞ and returns W̃ = αW.
        """
        # Compute induced infinity norm (maximum row absolute sum)
        s_inf = np.max(np.sum(np.abs(W), axis=1))

        if s_inf == 0:
            warnings.warn("Weight matrix has zero norm. No normalization applied.")
            return W.copy(), 1.0, 0.0

        # Compute scaling factor: α = 4c/s_∞
        alpha = 4 * self.safety_margin / s_inf

        # Normalized matrix
        W_norm = alpha * W

        return W_norm, alpha, s_inf

    def _get_block_info(self, block_labels: np.ndarray) -> Dict[int, np.ndarray]:
        """Get indices for each block."""
        unique_blocks = np.unique(block_labels)
        return {r: np.where(block_labels == r)[0] for r in unique_blocks}

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function σ(x) = 1/(1 + exp(-x))."""
        # Clip to prevent overflow
        x_clipped = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_clipped))

    def _sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid derivative σ'(x) = σ(x)(1 - σ(x))."""
        sigma_x = self._sigmoid(x)
        return sigma_x * (1.0 - sigma_x)

    def _compute_fixed_points(self, W_norm: np.ndarray,
                              block_info: Dict[int, np.ndarray],
                              bias: np.ndarray) -> Tuple[Dict[int, np.ndarray],
                                                         Dict[int, np.ndarray],
                                                         Dict[int, Dict[str, Any]]]:
        """
        Step 2: Per-block fixed-point computation using Picard iteration.

        For each block r, solve: x_r* = W̃_rr σ(x_r*) + b_r
        """
        fixed_points = {}
        sigmoid_gains = {}
        convergence_info = {}

        for r, indices in block_info.items():
            # Extract block recurrence matrix and bias
            W_rr = W_norm[np.ix_(indices, indices)]
            b_r = bias[indices]

            # Initialize with bias (reasonable starting point)
            x_r = b_r.copy()

            # Picard iteration with damping
            converged = False
            iterations = 0
            errors = []

            for iteration in range(self.max_iterations):
                # Compute next iterate: x^(t+1) = W_rr σ(x^(t)) + b_r
                x_new = W_rr @ self._sigmoid(x_r) + b_r

                # Apply damping: x^(t+1) ← (1-η)x^(t) + η[W_rr σ(x^(t)) + b_r]
                x_new = (1 - self.damping) * x_r + self.damping * x_new

                # Check convergence
                error = np.linalg.norm(x_new - x_r, ord=np.inf)
                errors.append(error)

                if error < self.tolerance:
                    converged = True
                    iterations = iteration + 1
                    break

                x_r = x_new

            if not converged:
                warnings.warn(f"Block {r} did not converge after {self.max_iterations} iterations. "
                             f"Final error: {errors[-1]:.2e}")

            # Store results
            fixed_points[r] = x_r

            # Compute sigmoid gains: Γ_r = diag(σ'(x_r*))
            gains = self._sigmoid_derivative(x_r)
            sigmoid_gains[r] = np.diag(gains)

            # Store convergence info
            convergence_info[r] = {
                'converged': converged,
                'iterations': iterations,
                'final_error': errors[-1] if errors else float('inf'),
                'error_history': errors
            }

        return fixed_points, sigmoid_gains, convergence_info

    def _construct_linearized_dynamics(self, W_norm: np.ndarray,
                                     block_info: Dict[int, np.ndarray],
                                     sigmoid_gains: Dict[int, np.ndarray],
                                     time_constants: Optional[Union[np.ndarray, float]],
                                     input_weights: Optional[np.ndarray]
                                     ) -> Tuple[Dict[int, np.ndarray],
                                               Dict[Tuple[int, int], np.ndarray],
                                               np.ndarray, Optional[np.ndarray]]:
        """
        Step 4: Construct linearized dynamics matrices.

        For each block r: δẋ_r = A_rr δx_r + Σ_s E_rs δx_s + B_r^lin δu_r
        where:
        - A_rr = -I/τ_r + (1/τ_r) W̃_rr Γ_r
        - B_rs = (1/τ_r) W̃_rr
        - C_rr = Γ_r
        - E_rs = (1/τ_r) W̃_rr Γ_s = B_rs C_ss
        - B_r^lin = (1/τ_r) B_r
        """
        N = W_norm.shape[0]
        num_blocks = len(block_info)

        # Handle time constants
        if time_constants is None:
            tau = {r: 1.0 for r in block_info.keys()}
        elif np.isscalar(time_constants):
            tau = {r: float(time_constants) for r in block_info.keys()}
        else:
            tau = {r: time_constants[i] for i, r in enumerate(sorted(block_info.keys()))}

        # Construct block matrices
        A_blocks = {}
        B_blocks = {}
        C_blocks = {}
        E_blocks = {}

        if num_blocks == 1:
            # Single block case
            r = next(iter(block_info))
            indices_r = block_info[r]
            n_r = len(indices_r)
            W_rr = W_norm[np.ix_(indices_r, indices_r)]

            A_rr = (-1.0 / tau[r]) * np.eye(n_r) + (1.0 / tau[r]) * W_rr @ sigmoid_gains[r]
            A_blocks[r] = A_rr

            C_rr = np.zeros((n_r, n_r))
            C_blocks[r] = C_rr

            B_rs = np.zeros((n_r, n_r))
            B_blocks[(r, r)] = B_rs

            E_rs = np.zeros((n_r, n_r))
            E_blocks[(r, r)] = E_rs

        else:
            for r, indices_r in block_info.items():
                n_r = len(indices_r)

                # Diagonal block: A_rr = -I/τ_r + (1/τ_r) W̃_rr Γ_r
                W_rr = W_norm[np.ix_(indices_r, indices_r)]
                A_rr = (-1.0 / tau[r]) * np.eye(n_r) + (1.0 / tau[r]) * W_rr @ sigmoid_gains[r]
                A_blocks[r] = A_rr

                # C_rr = Γ_r
                C_rr = sigmoid_gains[r]
                C_blocks[r] = C_rr

                # Off-diagonal blocks: E_rs = (1/τ_r) W̃_rr Γ_s
                for s, indices_s in block_info.items():
                    if s != r:
                        W_rs = W_norm[np.ix_(indices_r, indices_s)]

                        # B_rs = (1/τ_r) W̃_rr
                        B_rs = (1.0 / tau[r]) * W_rs
                        B_blocks[(r, s)] = B_rs
                        E_rs = B_rs @ sigmoid_gains[s]
                        E_blocks[(r, s)] = E_rs

        # Construct global matrix A
        A_global = np.zeros((N, N))

        # Fill diagonal blocks
        for r, indices_r in block_info.items():
            A_global[np.ix_(indices_r, indices_r)] = A_blocks[r]

        # Fill off-diagonal blocks
        for (r, s), E_rs in E_blocks.items():
            indices_r = block_info[r]
            indices_s = block_info[s]
            A_global[np.ix_(indices_r, indices_s)] = E_rs

        # Handle input weights
        B_linear = None
        if input_weights is not None:
            B_linear = np.zeros_like(input_weights)
            for r, indices_r in block_info.items():
                B_r = input_weights[indices_r]
                B_linear[indices_r] = (1.0 / tau[r]) * B_r

        return A_blocks, B_blocks, C_blocks, E_blocks, A_global, B_linear

    def _compute_eigenvalues(self, A_blocks: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """Step 5a: Compute eigenvalues for each block."""
        eigenvalues = {}
        for r, A_rr in A_blocks.items():
            eigenvalues[r] = np.linalg.eigvals(A_rr)
        return eigenvalues

    def _compute_schur_decomposition(self, A_blocks: Dict[int, np.ndarray]
                                   ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Step 5b: Compute Schur decomposition for each block."""
        schur_decomp = {}
        for r, A_rr in A_blocks.items():
            try:
                T, Q = schur(A_rr, output='complex')
                schur_decomp[r] = (T, Q)
            except Exception as e:
                warnings.warn(f"Schur decomposition failed for block {r}: {e}")
                schur_decomp[r] = (None, None)
        return schur_decomp

    def _compute_balanced_truncation(self, A_blocks: Dict[int, np.ndarray],
                                   input_weights: np.ndarray,
                                   block_info: Dict[int, np.ndarray],
                                   time_constants: Optional[Union[np.ndarray, float]]
                                   ) -> Dict[int, Dict[str, Any]]:
        """Step 5c: Compute balanced truncation analysis for each block."""
        balanced_results = {}

        # Handle time constants
        if time_constants is None:
            tau = {r: 1.0 for r in block_info.keys()}
        elif np.isscalar(time_constants):
            tau = {r: float(time_constants) for r in block_info.keys()}
        else:
            tau = {r: time_constants[i] for i, r in enumerate(sorted(block_info.keys()))}

        for r, A_rr in A_blocks.items():
            try:
                indices_r = block_info[r]
                B_r = (1.0 / tau[r]) * input_weights[indices_r]

                # For simplicity, assume C_r = I (full state observation)
                C_r = np.eye(A_rr.shape[0])

                # Solve Lyapunov equations for controllability and observability Gramians
                # A_rr W_c + W_c A_rr^T + B_r B_r^T = 0
                # A_rr^T W_o + W_o A_rr + C_r^T C_r = 0

                # Check if system is stable (all eigenvalues have negative real parts)
                eigenvals = np.linalg.eigvals(A_rr)
                if np.any(np.real(eigenvals) >= 0):
                    warnings.warn(f"Block {r} is not stable. Skipping balanced truncation.")
                    balanced_results[r] = {'stable': False, 'hankel_svs': None}
                    continue

                W_c = solve_continuous_lyapunov(A_rr, -B_r @ B_r.T)
                W_o = solve_continuous_lyapunov(A_rr.T, -C_r.T @ C_r)

                # Compute Hankel singular values
                hankel_svs = np.sqrt(np.real(np.linalg.eigvals(W_c @ W_o)))
                hankel_svs = np.sort(hankel_svs)[::-1]  # Sort in descending order

                balanced_results[r] = {
                    'stable': True,
                    'controllability_gramian': W_c,
                    'observability_gramian': W_o,
                    'hankel_svs': hankel_svs
                }

            except Exception as e:
                warnings.warn(f"Balanced truncation failed for block {r}: {e}")
                balanced_results[r] = {'stable': False, 'hankel_svs': None, 'error': str(e)}

        return balanced_results

    def check_global_stability(self, A_global: np.ndarray) -> Dict[str, Any]:
        """
        Step 6: Check global stability of the linearized system.

        Returns
        -------
        stability_info : dict
            Contains eigenvalues, stability status, and stability margin
        """
        eigenvals = np.linalg.eigvals(A_global)
        max_real_part = np.max(np.real(eigenvals))
        is_stable = max_real_part < 0
        stability_margin = -max_real_part if is_stable else None

        return {
            'eigenvalues': eigenvals,
            'is_stable': is_stable,
            'max_real_eigenvalue': max_real_part,
            'stability_margin': stability_margin,
            'num_unstable_modes': np.sum(np.real(eigenvals) >= 0)
        }


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Vectorized sigmoid function."""
    x_clipped = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Vectorized sigmoid derivative."""
    sigma_x = sigmoid(x)
    return sigma_x * (1.0 - sigma_x)