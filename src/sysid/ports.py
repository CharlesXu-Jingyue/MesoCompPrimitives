"""
Inter-Block Control Ports & Per-Port Controllability Analysis for CT RNN.

This module extends the existing CTRNN analysis to define state-ports between
blocks and compute per-port controllability Gramians for systematic analysis
of inter-block coupling strength and directionality.
"""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_continuous_lyapunov, eigh, eigvals
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from dataclasses import dataclass, field
from enum import Enum


class GramianMode(Enum):
    """Modes for controllability Gramian computation."""
    INFINITE_HORIZON = "infinite_horizon"
    DISCOUNTED = "discounted"
    FINITE_HORIZON = "finite_horizon"


class CovarianceWeighting(Enum):
    """Covariance weighting options for ports."""
    NONE = None
    STATE = "state"
    RATE = "rate"


@dataclass
class PortConfig:
    """Configuration for port controllability analysis."""
    mode: GramianMode = GramianMode.INFINITE_HORIZON
    discount_lambda: float = 0.1
    horizon_T: float = 2.0
    covariance_weighting: Optional[CovarianceWeighting] = CovarianceWeighting.NONE
    alpha_logdet: float = 1e-2
    top_k_modes: int = 5
    metric: str = "trace"
    stability_check: bool = True
    rank_tolerance: float = 1e-12


@dataclass
class PortMetrics:
    """Metrics for a single controllability port."""
    trace: float
    lambda_max: float
    lambda_min: float
    logdet: float
    rank: int
    condition_number: float
    frobenius_norm: float


@dataclass
class PortModes:
    """Leading eigenmodes of controllability Gramian for a port."""
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    participation_ratios: np.ndarray


@dataclass
class PortAnalysisResults:
    """Results of inter-block port controllability analysis."""

    # Port structure
    port_map: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict)  # B_rs matrices
    block_sizes: Dict[int, int] = field(default_factory=dict)

    # Per-port Gramians
    Wc_port: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict)
    Wo_port: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict)
    Wc_total: Dict[int, np.ndarray] = field(default_factory=dict)

    # Metrics and rankings
    port_metrics: Dict[Tuple[int, int], PortMetrics] = field(default_factory=dict)
    total_metrics: Dict[int, PortMetrics] = field(default_factory=dict)
    top_ports: Dict[int, List[Tuple[int, float]]] = field(default_factory=dict)  # Ranked by metric

    # Leading modes
    port_modes: Dict[Tuple[int, int], PortModes] = field(default_factory=dict)
    total_modes: Dict[int, PortModes] = field(default_factory=dict)

    # Diagnostics
    stability_info: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    config: Optional[PortConfig] = None


class PortAnalyzer:
    """
    Inter-block control port and controllability analyzer.

    Builds on existing CTRNN linearization to analyze state-ports between blocks
    and compute per-port controllability Gramians for understanding inter-block
    coupling strength and directionality.
    """

    def __init__(self, config: Optional[PortConfig] = None):
        self.config = config or PortConfig()

    def analyze_ports(self, A_blocks: Dict[int, np.ndarray],
                      B_blocks: Dict[Tuple[int, int], np.ndarray],
                      C_blocks: Dict[int, np.ndarray],
                      E_blocks: Dict[Tuple[int, int], np.ndarray],
                      time_constants: Optional[Dict[int, float]] = None,
                      covariance_matrices: Optional[Dict[int, np.ndarray]] = None) -> PortAnalysisResults:
        """
        Perform complete inter-block port controllability analysis.

        Parameters
        ----------
        A_blocks : Dict[int, ndarray]
            Diagonal block matrices A_rr for each block r
        B_blocks : Dict[Tuple[int, int], ndarray]
            Off-diagonal coupling matrices B_rs from block s to block r
        C_blocks : Dict[int, ndarray]
            Per-block observation matrices C_rr
        E_blocks : Dict[Tuple[int, int], ndarray]
            Off-diagonal coupling matrices E_rs from block s to block r
        time_constants : Dict[int, float], optional
            Time constants for each block (for diagnostics)
        covariance_matrices : Dict[int, ndarray], optional
            State covariance matrices for each block (for weighting)

        Returns
        -------
        PortAnalysisResults
            Complete port analysis results
        """
        # Initialize results
        results = PortAnalysisResults(config=self.config)

        # Extract block information
        blocks = list(A_blocks.keys())
        results.block_sizes = {r: A_blocks[r].shape[0] for r in blocks}

        # # Build state ports from E_blocks
        results.port_map = self._build_state_ports(E_blocks)

        # Check stability and compute Gramians
        results.stability_info = self._check_stability(A_blocks)
        results.Wc_port = self._compute_controllability_gramians(A_blocks, B_blocks, covariance_matrices)
        results.Wo_port = self._compute_observability_gramians(A_blocks, C_blocks, covariance_matrices)

        # Aggregate total Gramians
        results.Wc_total = self._aggregate_total_gramians(results.Wc_port, blocks)

        # Compute metrics and modes
        results.port_metrics = self._compute_port_metrics(results.Wc_port)
        results.total_metrics = self._compute_port_metrics(results.Wc_total)
        results.port_modes = self._compute_port_modes(results.Wc_port)
        results.total_modes = self._compute_port_modes(results.Wc_total)

        # Rank ports by chosen metric
        results.top_ports = self._rank_ports(results.port_metrics)

        return results

    def _build_state_ports(self, E_blocks: Dict[Tuple[int, int], np.ndarray]
                          ) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Build state-port matrices B_rs = E_rs from coupling blocks.

        Each port (s→r) represents the linearized influence of block s states
        on block r dynamics: δẋ_r = A_rr δx_r + Σ_s B_rs δx_s
        """
        port_map = {}

        for (r, s), E_rs in E_blocks.items():
            # State-port convention: B_rs = E_rs = (1/τ_r) U_rs Γ_s
            port_map[(r, s)] = E_rs.copy()

        return port_map

    def _check_stability(self, A_blocks: Dict[int, np.ndarray]) -> Dict[int, Dict[str, Any]]:
        """Check stability of each block for Gramian computation."""
        stability_info = {}

        for r, A_rr in A_blocks.items():
            eigenvals = eigvals(A_rr)
            max_real = np.max(np.real(eigenvals))
            is_stable = max_real < -1e-12  # Small tolerance for numerical precision

            stability_info[r] = {
                'eigenvalues': eigenvals,
                'max_real_eigenvalue': max_real,
                'is_stable': is_stable,
                'spectral_abscissa': max_real
            }

            if self.config.stability_check and self.config.mode == GramianMode.INFINITE_HORIZON and not is_stable:
                warnings.warn(f"Block {r} is not stable (max Re(λ) = {max_real:.6f}). "
                            f"Consider using discounted mode with λ = {self.config.discount_lambda}")

        return stability_info

    def _compute_controllability_gramians(self, A_blocks: Dict[int, np.ndarray],
                              B_blocks: Dict[Tuple[int, int], np.ndarray],
                              covariance_matrices: Optional[Dict[int, np.ndarray]] = None
                              ) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Compute per-port controllability Gramians.

        For each port (r,s), solve the continuous-time Lyapunov equation:
        A_rr W + W A_rr^T + B̂_rs B̂_rs^T = 0
        """
        Wc_port = {}

        for (r, s), B_rs in B_blocks.items():
            A_rr = A_blocks[r]

            # Apply covariance weighting if specified
            B_hat = self._apply_covariance_weighting(B_rs, s, covariance_matrices)

            # Compute Gramian based on mode
            try:
                if self.config.mode == GramianMode.INFINITE_HORIZON:
                    W = self._solve_infinite_horizon_controllability_gramian(A_rr, B_hat)
                elif self.config.mode == GramianMode.DISCOUNTED:
                    W = self._solve_discounted_controllability_gramian(A_rr, B_hat)
                elif self.config.mode == GramianMode.FINITE_HORIZON:
                    W = self._solve_finite_horizon_controllability_gramian(A_rr, B_hat)
                else:
                    raise ValueError(f"Unknown Gramian mode: {self.config.mode}")

                # Ensure symmetry and positive semidefiniteness
                W = 0.5 * (W + W.T)

                # Check PSD property
                min_eigenval = np.min(np.real(eigvals(W)))
                if min_eigenval < -self.config.rank_tolerance:
                    warnings.warn(f"Port ({r},{s}) Gramian has negative eigenvalue: {min_eigenval:.2e}")

                Wc_port[(r, s)] = W

            except Exception as e:
                warnings.warn(f"Failed to compute Gramian for port ({r},{s}): {e}")
                # Use zero Gramian as fallback
                Wc_port[(r, s)] = np.zeros((A_rr.shape[0], A_rr.shape[0]))

        return Wc_port
    
    def _compute_observability_gramians(self, A_blocks: Dict[int, np.ndarray],
                                        C_blocks: Dict[int, np.ndarray],
                                        covariance_matrices: Optional[Dict[int, np.ndarray]] = None
                                        ) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Compute per-port observability Gramians.

        For each port (r,s), solve the continuous-time Lyapunov equation:
        A_ss^T W + W A_ss + Ĉ_rs^T Ĉ_rs = 0
        """
        Wo_port = {}

        for r in C_blocks.keys():
            for s in C_blocks.keys():
                A_ss = A_blocks[s]
                C_rs = C_blocks[s]

                # Apply covariance weighting if specified
                C_hat = self._apply_covariance_weighting(C_rs, s, covariance_matrices)

                # Compute Gramian based on mode
                try:
                    if self.config.mode == GramianMode.INFINITE_HORIZON:
                        W = self._solve_infinite_horizon_observability_gramian(A_ss, C_hat)
                    elif self.config.mode == GramianMode.DISCOUNTED:
                        W = self._solve_discounted_observability_gramian(A_ss, C_hat)
                    elif self.config.mode == GramianMode.FINITE_HORIZON:
                        W = self._solve_finite_horizon_observability_gramian(A_ss, C_hat)
                    else:
                        raise ValueError(f"Unknown Gramian mode: {self.config.mode}")

                    # Ensure symmetry and positive semidefiniteness
                    W = 0.5 * (W + W.T)

                    # Check PSD property
                    min_eigenval = np.min(np.real(eigvals(W)))
                    if min_eigenval < -self.config.rank_tolerance:
                        warnings.warn(f"Port ({r},{s}) Gramian has negative eigenvalue: {min_eigenval:.2e}")

                    Wo_port[(r, s)] = W

                except Exception as e:
                    warnings.warn(f"Failed to compute Gramian for port ({r},{s}): {e}")
                    # Use zero Gramian as fallback
                    Wo_port[(r, s)] = np.zeros((A_ss.shape[0], A_ss.shape[0]))

        return Wo_port

    def _apply_covariance_weighting(self, B_rs: np.ndarray, s: int,
                                   covariance_matrices: Optional[Dict[int, np.ndarray]]
                                   ) -> np.ndarray:
        """Apply covariance weighting to input matrix."""
        if self.config.covariance_weighting == CovarianceWeighting.NONE or covariance_matrices is None:
            return B_rs

        if s not in covariance_matrices:
            warnings.warn(f"No covariance matrix provided for block {s}. Using unweighted input.")
            return B_rs

        Sigma_s = covariance_matrices[s]

        try:
            # Compute matrix square root of covariance
            eigenvals, eigenvecs = eigh(Sigma_s)
            eigenvals = np.maximum(eigenvals, 0)  # Ensure non-negative
            Sigma_sqrt = eigenvecs @ np.diag(np.sqrt(eigenvals)) @ eigenvecs.T

            # Weight input matrix: B̂_rs = B_rs Σ_s^(1/2)
            B_hat = B_rs @ Sigma_sqrt
            return B_hat

        except Exception as e:
            warnings.warn(f"Failed to apply covariance weighting for block {s}: {e}")
            return B_rs

    def _solve_infinite_horizon_controllability_gramian(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Solve infinite-horizon controllability Gramian: AW + WA^T + BB^T = 0."""
        return solve_continuous_lyapunov(A, -B @ B.T)
    
    def _solve_infinite_horizon_observability_gramian(self, A: np.ndarray, C: np.ndarray) -> np.ndarray:
        """Solve infinite-horizon observability Gramian: A^T W + WA + C^T C = 0."""
        return solve_continuous_lyapunov(A.T, -C.T @ C)

    def _solve_discounted_controllability_gramian(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Solve discounted Gramian: (A + λI)W + W(A + λI)^T + BB^T = 0."""
        A_discounted = A + self.config.discount_lambda * np.eye(A.shape[0])
        return solve_continuous_lyapunov(A_discounted, -B @ B.T)
    
    def _solve_discounted_observability_gramian(self, A: np.ndarray, C: np.ndarray) -> np.ndarray:
        """Solve discounted Gramian: (A + λI)^T W + W(A + λI) + C^T C = 0."""
        A_discounted = A + self.config.discount_lambda * np.eye(A.shape[0])
        return solve_continuous_lyapunov(A_discounted.T, -C.T @ C)

    def _solve_finite_horizon_controllability_gramian(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Solve finite-horizon Gramian: W(T) = ∫₀ᵀ e^{At} BB^T e^{A^T t} dt.

        Uses the Van Loan method for numerical integration.
        """
        n = A.shape[0]
        T = self.config.horizon_T

        # Build augmented matrix for Van Loan method
        # M = [[A,    BB^T],
        #      [0,   -A^T]]
        M = np.zeros((2*n, 2*n))
        M[:n, :n] = A
        M[:n, n:] = B @ B.T
        M[n:, n:] = -A.T

        # Compute matrix exponential
        exp_MT = sp.linalg.expm(M * T)

        # Extract Gramian from upper-right block
        W = exp_MT[:n, n:]
        return W
    
    def _solve_finite_horizon_observability_gramian(self, A: np.ndarray, C: np.ndarray) -> np.ndarray:
        """
        Solve finite-horizon observability Gramian: W(T) = ∫₀ᵀ e^{A^T t} C^T C e^{At} dt.

        Uses the Van Loan method for numerical integration.
        """
        n = A.shape[0]
        T = self.config.horizon_T

        # Build augmented matrix for Van Loan method
        # M = [[A^T,    C^T C],
        #      [0,     -A]]
        M = np.zeros((2*n, 2*n))
        M[:n, :n] = A.T
        M[:n, n:] = C.T @ C
        M[n:, n:] = -A

        # Compute matrix exponential
        exp_MT = sp.linalg.expm(M * T)

        # Extract Gramian from upper-right block
        W = exp_MT[:n, n:]
        return W

    def _aggregate_total_gramians(self, Wc_port: Dict[Tuple[int, int], np.ndarray],
                                 blocks: List[int]) -> Dict[int, np.ndarray]:
        """Aggregate per-port Gramians into total controllability per block."""
        Wc_total = {}

        for r in blocks:
            # Sum all incoming port Gramians for block r
            total_W = None

            for (dest, src), W_port in Wc_port.items():
                if dest == r:  # Incoming port to block r
                    if total_W is None:
                        total_W = W_port.copy()
                    else:
                        total_W += W_port

            if total_W is not None:
                Wc_total[r] = total_W
            else:
                # No incoming ports - zero Gramian
                block_size = next(W.shape[0] for (dest, src), W in Wc_port.items() if dest == r)
                Wc_total[r] = np.zeros((block_size, block_size))

        return Wc_total

    def _compute_port_metrics(self, gramians: Dict[Union[Tuple[int, int], int], np.ndarray]
                            ) -> Dict[Union[Tuple[int, int], int], PortMetrics]:
        """Compute scalar metrics for Gramians."""
        metrics = {}

        for key, W in gramians.items():
            # Compute eigenvalues for metrics
            eigenvals = eigvals(W)
            eigenvals_real = np.real(eigenvals)
            eigenvals_real = np.maximum(eigenvals_real, 0)  # Ensure non-negative

            # Scalar metrics
            trace_val = np.trace(W)
            lambda_max = np.max(eigenvals_real)
            lambda_min = np.min(eigenvals_real)

            # Log determinant with regularization
            logdet_val = np.sum(np.log(1 + self.config.alpha_logdet * eigenvals_real))

            # Numerical rank
            rank_val = np.sum(eigenvals_real > self.config.rank_tolerance)

            # Condition number
            if lambda_max > self.config.rank_tolerance:
                cond_num = lambda_max / np.max([np.min(eigenvals_real[eigenvals_real > self.config.rank_tolerance]),
                                               self.config.rank_tolerance])
            else:
                cond_num = 1.0

            # Frobenius norm
            frob_norm = np.linalg.norm(W, 'fro')

            metrics[key] = PortMetrics(
                trace=trace_val,
                lambda_max=lambda_max,
                lambda_min=lambda_min,
                logdet=logdet_val,
                rank=rank_val,
                condition_number=cond_num,
                frobenius_norm=frob_norm
            )

        return metrics

    def _compute_port_modes(self, gramians: Dict[Union[Tuple[int, int], int], np.ndarray]
                          ) -> Dict[Union[Tuple[int, int], int], PortModes]:
        """Compute leading eigenmodes of Gramians."""
        modes = {}

        for key, W in gramians.items():
            try:
                # Compute eigendecomposition
                eigenvals, eigenvecs = eigh(W)

                # Sort by eigenvalue magnitude (descending)
                idx = np.argsort(eigenvals)[::-1]
                eigenvals = eigenvals[idx]
                eigenvecs = eigenvecs[:, idx]

                # Take top k modes
                k = min(self.config.top_k_modes, len(eigenvals))
                top_eigenvals = eigenvals[:k]
                top_eigenvecs = eigenvecs[:, :k]

                # Compute participation ratios
                # PR_i = (Σ_j |v_ij|²)² / Σ_j |v_ij|⁴
                participation_ratios = np.zeros(k)
                for i in range(k):
                    v = top_eigenvecs[:, i]
                    v_squared = v**2
                    participation_ratios[i] = np.sum(v_squared)**2 / np.sum(v_squared**2)

                modes[key] = PortModes(
                    eigenvalues=top_eigenvals,
                    eigenvectors=top_eigenvecs,
                    participation_ratios=participation_ratios
                )

            except Exception as e:
                warnings.warn(f"Failed to compute modes for {key}: {e}")
                # Fallback to empty modes
                modes[key] = PortModes(
                    eigenvalues=np.array([]),
                    eigenvectors=np.array([]).reshape(W.shape[0], 0),
                    participation_ratios=np.array([])
                )

        return modes

    def _rank_ports(self, port_metrics: Dict[Tuple[int, int], PortMetrics]
                   ) -> Dict[int, List[Tuple[int, float]]]:
        """Rank incoming ports for each block by chosen metric."""
        top_ports = {}

        # Group ports by destination block
        ports_by_dest = {}
        for (r, s), metrics in port_metrics.items():
            if r not in ports_by_dest:
                ports_by_dest[r] = []

            # Extract metric value
            metric_value = getattr(metrics, self.config.metric)
            ports_by_dest[r].append((s, metric_value))

        # Sort ports for each destination block
        for r, port_list in ports_by_dest.items():
            # Sort by metric value (descending)
            sorted_ports = sorted(port_list, key=lambda x: x[1], reverse=True)
            top_ports[r] = sorted_ports

        return top_ports


def validate_port_analysis(results: PortAnalysisResults) -> Dict[str, Any]:
    """
    Validate port analysis results and provide diagnostics.

    Returns
    -------
    validation_info : dict
        Validation results and diagnostics
    """
    validation = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'summary': {}
    }

    # Check Gramian properties
    for (r, s), W in results.Wc_port.items():
        # Check symmetry
        symmetry_error = np.max(np.abs(W - W.T))
        if symmetry_error > 1e-10:
            validation['warnings'].append(f"Port ({r},{s}) Gramian not symmetric (error: {symmetry_error:.2e})")

        # Check positive semidefiniteness
        min_eigenval = np.min(np.real(eigvals(W)))
        if min_eigenval < -1e-12:
            validation['warnings'].append(f"Port ({r},{s}) Gramian not PSD (min λ: {min_eigenval:.2e})")

    # Check additivity: Wc_total should equal sum of port Gramians
    for r, W_total in results.Wc_total.items():
        W_sum = np.zeros_like(W_total)
        for (dest, src), W_port in results.Wc_port.items():
            if dest == r:
                W_sum += W_port

        additivity_error = np.max(np.abs(W_total - W_sum))
        if additivity_error > 1e-12:
            validation['errors'].append(f"Block {r} total Gramian additivity error: {additivity_error:.2e}")
            validation['valid'] = False

    # Summary statistics
    num_ports = len(results.Wc_port)
    num_blocks = len(results.Wc_total)
    avg_rank = np.mean([metrics.rank for metrics in results.port_metrics.values()])

    validation['summary'] = {
        'num_ports': num_ports,
        'num_blocks': num_blocks,
        'avg_port_rank': avg_rank,
        'total_warnings': len(validation['warnings']),
        'total_errors': len(validation['errors'])
    }

    return validation


# Convenience functions for integration with existing CTRNN analysis

def analyze_ctrnn_ports(ctrnn_results, config: Optional[PortConfig] = None) -> PortAnalysisResults:
    """
    Convenience function to analyze ports from existing CTRNN results.

    Parameters
    ----------
    ctrnn_results : FixedPointAnalysis
        Results from CTRNNAnalyzer.analyze()
    config : PortConfig, optional
        Port analysis configuration

    Returns
    -------
    PortAnalysisResults
        Port controllability analysis results
    """
    analyzer = PortAnalyzer(config)
    return analyzer.analyze_ports(
        A_blocks=ctrnn_results.A_blocks,
        E_blocks=ctrnn_results.E_blocks
    )


def summarize_port_rankings(results: PortAnalysisResults, top_k: int = 5) -> Dict[int, Dict[str, Any]]:
    """
    Generate human-readable summary of port rankings.

    Parameters
    ----------
    results : PortAnalysisResults
        Port analysis results
    top_k : int
        Number of top ports to include in summary

    Returns
    -------
    summary : dict
        Summary of top ports for each block
    """
    summary = {}

    for r, ranked_ports in results.top_ports.items():
        block_summary = {
            'total_incoming_ports': len(ranked_ports),
            'total_controllability': results.total_metrics[r].trace,
            'top_ports': []
        }

        for i, (s, metric_value) in enumerate(ranked_ports[:top_k]):
            port_info = {
                'rank': i + 1,
                'source_block': s,
                'metric_value': metric_value,
                'relative_contribution': metric_value / results.total_metrics[r].trace if results.total_metrics[r].trace > 0 else 0,
                'rank_percentage': results.port_metrics[(r, s)].rank / results.block_sizes[r] * 100
            }
            block_summary['top_ports'].append(port_info)

        summary[r] = block_summary

    return summary