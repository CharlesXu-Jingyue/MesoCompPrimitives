"""
Utilities for bi-orthogonal Laplacian Renormalization Group (bi-LRG).

Small helper functions for spectral analysis, matrix operations, and validation.
"""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import schur, eig
from scipy.sparse.linalg import eigs
from typing import Tuple, Union, Optional
import warnings


def safe_divide(numerator: np.ndarray, denominator: np.ndarray,
                default: float = 0.0, eps: float = 1e-12) -> np.ndarray:
    """Safely divide arrays, handling division by zero."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = numerator / np.maximum(denominator, eps)
        result[np.abs(denominator) < eps] = default
    return result


def is_sparse(A: Union[np.ndarray, sp.spmatrix]) -> bool:
    """Check if matrix is sparse."""
    return sp.issparse(A)


def to_dense(A: Union[np.ndarray, sp.spmatrix]) -> np.ndarray:
    """Convert matrix to dense numpy array."""
    if is_sparse(A):
        return A.toarray()
    return A


def compute_stationary_distribution(P: Union[np.ndarray, sp.spmatrix],
                                   method: str = 'eigs',
                                   tol: float = 1e-10) -> np.ndarray:
    """
    Compute stationary distribution π such that π^T P = π^T.

    Parameters
    ----------
    P : array-like
        Transition matrix (row-stochastic)
    method : str
        Method to use ('eigs' for sparse, 'eig' for dense)
    tol : float
        Tolerance for eigenvalue check

    Returns
    -------
    pi : ndarray
        Stationary distribution (sums to 1)
    """
    n = P.shape[0]

    try:
        if is_sparse(P) and method == 'eigs':
            # Find left eigenvector with eigenvalue 1
            eigenvals, eigenvecs = eigs(P.T, k=1, which='LR', tol=tol)
            pi = np.real(eigenvecs[:, 0])
        else:
            # Dense computation
            P_dense = to_dense(P)
            eigenvals, eigenvecs = eig(P_dense.T)
            # Find eigenvalue closest to 1
            idx = np.argmin(np.abs(eigenvals - 1.0))
            pi = np.real(eigenvecs[:, idx])

    except Exception as e:
        warnings.warn(f"Eigenvalue method failed: {e}. Using uniform distribution.")
        pi = np.ones(n)

    # Ensure non-negativity and normalization
    pi = np.abs(pi)
    pi = pi / np.sum(pi)

    return pi


def biorthogonal_modes(L_rw: Union[np.ndarray, sp.spmatrix],
                      k: int = 5,
                      which: str = 'SM',
                      tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute bi-orthogonal modes of random-walk Laplacian.

    Parameters
    ----------
    L_rw : array-like
        Random-walk Laplacian matrix
    k : int
        Number of modes to compute
    which : str
        Which eigenvalues to find ('SM' for smallest magnitude)
    tol : float
        Tolerance for eigenvalue computation

    Returns
    -------
    U_k : ndarray
        Left eigenvectors (n x k)
    V_k : ndarray
        Right eigenvectors (n x k)
    Lambda_k : ndarray
        Eigenvalues (k,)
    """
    n = L_rw.shape[0]
    k = min(k, n)  # Can't compute more than n modes

    try:
        if is_sparse(L_rw):
            # Use sparse eigenvalue solver
            eigenvals, V_k = eigs(L_rw, k=k, which=which, tol=tol)
            # Compute left eigenvectors via transpose
            eigenvals_left, U_k = eigs(L_rw.conj().T, k=k, which=which, tol=tol)
        else:
            # Use dense solver with Schur decomposition for better numerics
            L_dense = to_dense(L_rw)
            T, Z = schur(L_dense, output='complex')

            # Extract eigenvalues and eigenvectors
            eigenvals = np.diag(T)

            # Sort by magnitude (smallest first for slow modes)
            idx = np.argsort(np.abs(eigenvals))[:k]
            eigenvals = eigenvals[idx]

            # Right eigenvectors
            V_k = Z[:, idx]

            # Left eigenvectors (via transpose)
            T_left, Z_left = schur(L_dense.T, output='complex')
            eigenvals_left = np.diag(T_left)
            idx_left = np.argsort(np.abs(eigenvals_left))[:k]
            U_k = Z_left[:, idx_left]
    except Exception as e:
        warnings.warn(f"Eigendecomposition failed: {e}. Using random modes.")
        V_k = np.random.randn(n, k) + 1j * np.random.randn(n, k)
        U_k = np.random.randn(n, k) + 1j * np.random.randn(n, k)
        eigenvals = np.random.random(k) * 0.1

    # Ensure bi-orthogonality: U_k^H V_k = I
    try:
        gram = U_k.conj().T @ V_k
        U_k = U_k @ np.linalg.inv(gram).conj().T
    except np.linalg.LinAlgError:
        warnings.warn("Failed to enforce bi-orthogonality. Modes may not be orthogonal.")

    return U_k, V_k, eigenvals


def realify_modes(U_k: np.ndarray, V_k: np.ndarray,
                  Lambda_k: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert complex eigenmodes to real representation.

    For complex conjugate pairs, creates real and imaginary parts.

    Parameters
    ----------
    U_k, V_k : ndarray
        Complex left and right eigenvectors
    Lambda_k : ndarray
        Complex eigenvalues

    Returns
    -------
    U_real, V_real : ndarray
        Real versions of eigenvectors
    Lambda_real : ndarray
        Real eigenvalues (repeated for conjugate pairs)
    """
    n, k = V_k.shape
    U_real_list = []
    V_real_list = []
    Lambda_real_list = []

    i = 0
    while i < k:
        if np.isreal(Lambda_k[i]) or np.abs(np.imag(Lambda_k[i])) < 1e-12:
            # Real eigenvalue
            U_real_list.append(np.real(U_k[:, i]))
            V_real_list.append(np.real(V_k[:, i]))
            Lambda_real_list.append(np.real(Lambda_k[i]))
            i += 1
        else:
            # Complex conjugate pair
            if i + 1 < k and np.abs(Lambda_k[i] - np.conj(Lambda_k[i + 1])) < 1e-12:
                # Add real and imaginary parts
                U_real_list.append(np.real(U_k[:, i]))
                U_real_list.append(np.imag(U_k[:, i]))
                V_real_list.append(np.real(V_k[:, i]))
                V_real_list.append(np.imag(V_k[:, i]))
                Lambda_real_list.extend([np.real(Lambda_k[i]), np.real(Lambda_k[i])])
                i += 2
            else:
                # Single complex eigenvalue (shouldn't happen for real matrices)
                U_real_list.append(np.real(U_k[:, i]))
                V_real_list.append(np.real(V_k[:, i]))
                Lambda_real_list.append(np.real(Lambda_k[i]))
                i += 1

    U_real = np.column_stack(U_real_list) if U_real_list else np.zeros((n, 0))
    V_real = np.column_stack(V_real_list) if V_real_list else np.zeros((n, 0))
    Lambda_real = np.array(Lambda_real_list)

    return U_real, V_real, Lambda_real


def validate_transition_matrix(P: Union[np.ndarray, sp.spmatrix],
                              tol: float = 1e-10) -> bool:
    """
    Validate that P is a proper transition matrix.

    Parameters
    ----------
    P : array-like
        Matrix to validate
    tol : float
        Tolerance for checks

    Returns
    -------
    valid : bool
        True if P is row-stochastic and non-negative
    """
    # Check non-negativity
    if is_sparse(P):
        if np.any(P.data < -tol):
            return False
    else:
        if np.any(P < -tol):
            return False

    # Check row sums equal 1
    row_sums = np.array(P.sum(axis=1)).flatten()
    if not np.allclose(row_sums, 1.0, atol=tol):
        return False

    return True


def create_teleportation_matrix(A: Union[np.ndarray, sp.spmatrix],
                               alpha: float = 0.95) -> Union[np.ndarray, sp.spmatrix]:
    """
    Create teleportation matrix for handling dangling nodes using lazy self-loops.

    Parameters
    ----------
    A : array-like
        Adjacency matrix
    alpha : float
        Teleportation parameter (1 = no teleportation, 0 = full teleportation)

    Returns
    -------
    P_tele : array-like
        Teleportation-augmented transition matrix
    """
    n = A.shape[0]

    # Compute out-degrees
    if is_sparse(A):
        out_degrees = np.array(A.sum(axis=1)).flatten()
    else:
        out_degrees = A.sum(axis=1)

    # Handle zero out-degrees
    zero_degree_mask = out_degrees == 0

    if is_sparse(A):
        # Create sparse transition matrix
        A_copy = A.copy().astype(float)

        # Add self-loops to zero-degree nodes
        if np.any(zero_degree_mask):
            diag_indices = np.where(zero_degree_mask)[0]
            for i in diag_indices:
                A_copy[i, i] = 1.0

        # Recompute degrees
        out_degrees = np.array(A_copy.sum(axis=1)).flatten()

        # Create P = D^(-1) A
        D_inv = sp.diags(1.0 / np.maximum(out_degrees, 1e-12), format='csr')
        P = D_inv @ A_copy

        # Add teleportation
        if alpha < 1.0:
            uniform_prob = (1 - alpha) / n
            # For sparse matrices, add uniform distribution
            P = alpha * P + uniform_prob * sp.csr_matrix(np.ones((n, n)))

    else:
        # Dense computation
        A_copy = A.copy().astype(float)

        # Add self-loops to zero-degree nodes
        A_copy[zero_degree_mask, zero_degree_mask] = 1.0

        # Recompute degrees
        out_degrees = A_copy.sum(axis=1)

        # Create transition matrix
        P = A_copy / out_degrees[:, np.newaxis]

        # Add teleportation
        if alpha < 1.0:
            uniform_prob = (1 - alpha) / n
            P = alpha * P + uniform_prob * np.ones((n, n))

    return P


def spectral_fidelity(V_original: np.ndarray, V_reduced: np.ndarray,
                     C: np.ndarray) -> float:
    """
    Compute spectral fidelity between original and reduced modes.

    Parameters
    ----------
    V_original : ndarray
        Original eigenvectors (n x k)
    V_reduced : ndarray
        Reduced eigenvectors (m x k)
    C : ndarray
        Membership matrix (n x m)

    Returns
    -------
    fidelity : float
        Spectral fidelity score (0 = perfect, larger = worse)
    """
    try:
        V_reconstructed = C @ V_reduced
        error_matrix = V_original - V_reconstructed

        fidelity = np.linalg.norm(error_matrix, 'fro') / np.linalg.norm(V_original, 'fro')
        return fidelity

    except Exception as e:
        warnings.warn(f"Spectral fidelity computation failed: {e}")
        return np.inf