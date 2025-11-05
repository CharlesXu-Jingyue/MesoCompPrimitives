"""
Degree-Corrected Stochastic Block Model (DC-SBM) implementation.

This module implements a weighted, directed DC-SBM using variational EM
with spectral initialization.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, coo_matrix
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from typing import Union, Tuple, List, Dict, Optional
import warnings


def safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute log with numerical stability."""
    return np.log(np.maximum(x, eps))


def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax with numerical stability."""
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))


def degrees(A: Union[csr_matrix, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute out-degrees and in-degrees of adjacency matrix.

    Parameters
    ----------
    A : csr_matrix or ndarray
        Adjacency matrix (n x n)

    Returns
    -------
    k_out : ndarray
        Out-degrees (n,)
    k_in : ndarray
        In-degrees (n,)
    """
    if sp.issparse(A):
        k_out = np.array(A.sum(axis=1)).flatten()
        k_in = np.array(A.sum(axis=0)).flatten()
    else:
        k_out = A.sum(axis=1)
        k_in = A.sum(axis=0)
    return k_out, k_in


def to_edge_list(A: Union[csr_matrix, np.ndarray]) -> List[Tuple[int, int, float]]:
    """
    Convert adjacency matrix to edge list.

    Parameters
    ----------
    A : csr_matrix or ndarray
        Adjacency matrix

    Returns
    -------
    edges : list of tuples
        Edge list as (source, target, weight)
    """
    if sp.issparse(A):
        A_coo = A.tocoo()
        return [(int(i), int(j), float(w)) for i, j, w in zip(A_coo.row, A_coo.col, A_coo.data)]
    else:
        rows, cols = np.nonzero(A)
        return [(int(i), int(j), float(A[i, j])) for i, j in zip(rows, cols)]


def spectral_init(A: Union[csr_matrix, np.ndarray], K: int, seed: Optional[int] = None,
                  d: Optional[int] = None) -> np.ndarray:
    """
    Spectral initialization for DC-SBM.

    Uses degree-regularized SVD embedding followed by k-means clustering
    and softmax smoothing.

    Parameters
    ----------
    A : csr_matrix or ndarray
        Adjacency matrix (n x n)
    K : int
        Number of blocks
    seed : int, optional
        Random seed
    d : int, optional
        Embedding dimension (default: min(2*K, 64))

    Returns
    -------
    Q : ndarray
        Soft membership matrix (n x K)
    """
    if seed is not None:
        np.random.seed(seed)

    n = A.shape[0]
    if d is None:
        d = min(2 * K, 64)

    try:
        # Degree regularization
        k_out, k_in = degrees(A)
        k_out_sqrt = np.sqrt(np.maximum(k_out, 1))
        k_in_sqrt = np.sqrt(np.maximum(k_in, 1))

        # Normalize by degree
        if sp.issparse(A):
            A_norm = A.copy().astype(float)
            A_coo = A_norm.tocoo()  # Convert to COO format to access row/col indices
            A_coo.data = A_coo.data / (k_out_sqrt[A_coo.row] * k_in_sqrt[A_coo.col])
            A_norm = A_coo.tocsr()  # Convert back to CSR
        else:
            A_norm = A / (k_out_sqrt[:, None] * k_in_sqrt[None, :])

        # SVD embedding
        if sp.issparse(A_norm):
            from scipy.sparse.linalg import svds
            U, s, Vt = svds(A_norm, k=min(d//2, n-1))
        else:
            U, s, Vt = np.linalg.svd(A_norm, full_matrices=False)
            U = U[:, :d//2]
            Vt = Vt[:d//2, :]

        # Concatenate embeddings
        embedding = np.hstack([U, Vt.T])

        # K-means clustering
        kmeans = KMeans(n_clusters=K, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(embedding)

        # Convert to one-hot then soften
        Q = np.zeros((n, K))
        Q[np.arange(n), labels] = 1.0

        # Soften with temperature
        tau = 0.1
        noise = np.random.normal(0, tau, (n, K))
        Q = stable_softmax(safe_log(Q) + noise, axis=1)

    except Exception as e:
        warnings.warn(f"Spectral initialization failed: {e}. Using random initialization.")
        # Fallback to random
        Q = np.random.dirichlet(np.ones(K), size=n)

    return Q


def heldout_split(A: Union[csr_matrix, np.ndarray], frac: float = 0.1,
                  stratify_degrees: bool = True, seed: Optional[int] = None) -> Tuple[csr_matrix, csr_matrix, csr_matrix]:
    """
    Split adjacency matrix into train/validation sets.

    Parameters
    ----------
    A : csr_matrix or ndarray
        Adjacency matrix
    frac : float
        Fraction of edges to hold out
    stratify_degrees : bool
        Whether to stratify by degree
    seed : int, optional
        Random seed

    Returns
    -------
    A_train : csr_matrix
        Training adjacency matrix
    A_val : csr_matrix
        Validation adjacency matrix (held-out entries only)
    mask_val : csr_matrix
        Binary mask indicating held-out entries
    """
    if seed is not None:
        np.random.seed(seed)

    A_csr = csr_matrix(A) if not sp.issparse(A) else A.tocsr()
    n = A_csr.shape[0]

    # Get non-zero entries
    A_coo = A_csr.tocoo()
    edges = list(zip(A_coo.row, A_coo.col, A_coo.data))
    n_edges = len(edges)
    n_holdout = int(frac * n_edges)

    if stratify_degrees:
        # Stratify by source node degree
        k_out, _ = degrees(A_csr)
        edge_degrees = [k_out[i] for i, j, w in edges]

        # Sort edges by degree and sample across quantiles
        sorted_indices = np.argsort(edge_degrees)
        chunk_size = n_edges // 10  # 10 quantiles
        holdout_indices = []

        for q in range(10):
            start = q * chunk_size
            end = (q + 1) * chunk_size if q < 9 else n_edges
            chunk_indices = sorted_indices[start:end]
            n_chunk_holdout = int(frac * len(chunk_indices))
            holdout_indices.extend(np.random.choice(chunk_indices, n_chunk_holdout, replace=False))
    else:
        holdout_indices = np.random.choice(n_edges, n_holdout, replace=False)

    # Create masks
    train_mask = np.ones(n_edges, dtype=bool)
    train_mask[holdout_indices] = False

    # Split data
    train_edges = [edges[i] for i in range(n_edges) if train_mask[i]]
    val_edges = [edges[i] for i in holdout_indices]

    # Create matrices
    def edges_to_csr(edge_list, shape):
        if not edge_list:
            return csr_matrix(shape)
        rows, cols, data = zip(*edge_list)
        return csr_matrix((data, (rows, cols)), shape=shape)

    A_train = edges_to_csr(train_edges, (n, n))
    A_val = edges_to_csr(val_edges, (n, n))

    # Create validation mask
    if val_edges:
        rows, cols, _ = zip(*val_edges)
        mask_val = csr_matrix((np.ones(len(val_edges)), (rows, cols)), shape=(n, n))
    else:
        mask_val = csr_matrix((n, n))

    return A_train, A_val, mask_val


class DCSBM:
    """
    Degree-Corrected Stochastic Block Model.

    Implements weighted, directed DC-SBM using variational EM with spectral initialization.

    Model specification:
    A_ij ~ Poisson(λ_ij)
    λ_ij = θ_out[i] * θ_in[j] * ω[g_i, g_j]

    With normalization constraints:
    sum(θ_out[i] for i in block r) = 1
    sum(θ_in[i] for i in block r) = 1

    Parameters
    ----------
    K : int
        Number of blocks
    max_iter : int, default=200
        Maximum number of EM iterations
    tol : float, default=1e-4
        Convergence tolerance for relative ELBO change
    seed : int, optional
        Random seed for reproducibility
    init : str, default="spectral"
        Initialization method ("spectral" or "random")
    zero_handling : str, default="ignore"
        How to handle zero entries ("ignore" or "add_epsilon")
    """

    def __init__(self, K: int, max_iter: int = 200, tol: float = 1e-4,
                 seed: Optional[int] = None, init: str = "spectral",
                 zero_handling: str = "ignore"):
        self.K = K
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.init = init
        self.zero_handling = zero_handling

        # Model parameters (set after fitting)
        self.Q = None
        self.labels_ = None
        self.theta_out_ = None
        self.theta_in_ = None
        self.Omega_ = None
        self.pi_ = None

        # Diagnostics
        self.elbo_ = []
        self.converged_ = False
        self.n_iter_ = 0

    def fit(self, A: Union[csr_matrix, np.ndarray]) -> "DCSBM":
        """
        Fit DC-SBM to adjacency matrix.

        Parameters
        ----------
        A : csr_matrix or ndarray
            Weighted, directed adjacency matrix (n x n)

        Returns
        -------
        self : DCSBM
            Fitted model
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        # Convert to CSR and validate
        A_csr = csr_matrix(A) if not sp.issparse(A) else A.tocsr()
        if A_csr.data.min() < 0:
            raise ValueError("Adjacency matrix must be non-negative")

        n = A_csr.shape[0]

        # Compute degrees
        k_out, k_in = degrees(A_csr)

        # Initialize parameters
        if self.init == "spectral":
            self.Q = spectral_init(A_csr, self.K, self.seed)
        else:
            self.Q = np.random.dirichlet(np.ones(self.K), size=n)

        self.pi_ = self.Q.mean(axis=0)

        # Initialize theta and Omega
        self._m_step(A_csr, k_out, k_in)

        # EM iterations
        self.elbo_ = []
        prev_elbo = -np.inf

        for iter_num in range(self.max_iter):
            # E-step
            self._e_step(A_csr, k_out, k_in)

            # M-step
            self._m_step(A_csr, k_out, k_in)

            # Compute ELBO
            elbo = self._compute_elbo(A_csr, k_out, k_in)
            self.elbo_.append(elbo)

            # Check convergence
            if len(self.elbo_) > 1:
                rel_change = abs(elbo - prev_elbo) / (abs(prev_elbo) + 1e-12)
                if rel_change < self.tol:
                    self.converged_ = True
                    break

            prev_elbo = elbo

        self.n_iter_ = iter_num + 1
        self.labels_ = self.Q.argmax(axis=1)

        return self

    def _e_step(self, A_csr: csr_matrix, k_out: np.ndarray, k_in: np.ndarray):
        """E-step: Update membership probabilities."""
        n, K = A_csr.shape[0], self.K
        log_q = np.zeros((n, K))

        # Prior term
        log_q += safe_log(self.pi_)[None, :]

        # Precompute block totals
        T_out = np.sum(self.Q * self.theta_out_[:, None], axis=0)  # (K,)
        T_in = np.sum(self.Q * self.theta_in_[:, None], axis=0)   # (K,)

        # Likelihood terms
        A_coo = A_csr.tocoo()

        for idx in range(len(A_coo.data)):
            i, j, w = A_coo.row[idx], A_coo.col[idx], A_coo.data[idx]

            # Outgoing edge from i to j
            for k in range(K):
                for l in range(K):
                    log_lambda = (safe_log(self.theta_out_[i]) +
                                 safe_log(self.theta_in_[j]) +
                                 safe_log(self.Omega_[k, l]))
                    log_q[i, k] += w * self.Q[j, l] * log_lambda
                    log_q[j, l] += w * self.Q[i, k] * log_lambda

        # Expected rate terms (negative part of Poisson likelihood)
        for i in range(n):
            for k in range(K):
                expected_out = self.theta_out_[i] * np.sum(self.Omega_[k, :] * T_in)
                expected_in = self.theta_in_[i] * np.sum(self.Omega_[:, k] * T_out)
                log_q[i, k] -= (expected_out + expected_in)

        # Normalize with damping
        Q_new = stable_softmax(log_q, axis=1)
        eta = 0.5  # Damping parameter
        self.Q = (1 - eta) * self.Q + eta * Q_new

        # Update prior
        self.pi_ = self.Q.mean(axis=0)

    def _m_step(self, A_csr: csr_matrix, k_out: np.ndarray, k_in: np.ndarray):
        """M-step: Update parameters."""
        n, K = A_csr.shape[0], self.K
        eps = 1e-12

        # Compute sufficient statistics
        S_out = np.sum(self.Q * k_out[:, None], axis=0)  # (K,)
        S_in = np.sum(self.Q * k_in[:, None], axis=0)    # (K,)

        # Block-pair totals
        m = np.zeros((K, K))
        A_coo = A_csr.tocoo()

        for idx in range(len(A_coo.data)):
            i, j, w = A_coo.row[idx], A_coo.col[idx], A_coo.data[idx]
            m += w * self.Q[i, :, None] * self.Q[j, None, :]

        # Update Omega
        self.Omega_ = m + eps

        # Update theta parameters
        self.theta_out_ = np.zeros(n)
        self.theta_in_ = np.zeros(n)

        for i in range(n):
            # Weighted average over blocks
            self.theta_out_[i] = np.sum(self.Q[i, :] * k_out[i] / (S_out + eps))
            self.theta_in_[i] = np.sum(self.Q[i, :] * k_in[i] / (S_in + eps))

        # Ensure positivity
        self.theta_out_ = np.maximum(self.theta_out_, eps)
        self.theta_in_ = np.maximum(self.theta_in_, eps)

    def _compute_elbo(self, A_csr: csr_matrix, k_out: np.ndarray, k_in: np.ndarray) -> float:
        """Compute Evidence Lower BOund (ELBO)."""
        n, K = A_csr.shape[0], self.K
        elbo = 0.0

        # Expected log-likelihood
        A_coo = A_csr.tocoo()
        for idx in range(len(A_coo.data)):
            i, j, w = A_coo.row[idx], A_coo.col[idx], A_coo.data[idx]
            for k in range(K):
                for l in range(K):
                    lambda_ij = self.theta_out_[i] * self.theta_in_[j] * self.Omega_[k, l]
                    log_prob = w * np.log(lambda_ij) - lambda_ij
                    elbo += self.Q[i, k] * self.Q[j, l] * log_prob

        # Entropy of Q
        elbo += -np.sum(self.Q * safe_log(self.Q))

        # Prior term
        elbo += np.sum(self.Q * safe_log(self.pi_)[None, :])

        return elbo

    def fit_transform(self, A: Union[csr_matrix, np.ndarray]) -> np.ndarray:
        """Fit model and return soft membership matrix."""
        self.fit(A)
        return self.Q

    def predict(self) -> np.ndarray:
        """Get hard block assignments."""
        if self.Q is None:
            raise ValueError("Model must be fitted before prediction")
        return self.labels_

    def score(self, A: Union[csr_matrix, np.ndarray], mask: Optional[csr_matrix] = None) -> float:
        """
        Compute average predictive log-likelihood.

        Parameters
        ----------
        A : csr_matrix or ndarray
            Adjacency matrix
        mask : csr_matrix, optional
            Binary mask for held-out entries

        Returns
        -------
        score : float
            Average log-likelihood
        """
        if self.theta_out_ is None:
            raise ValueError("Model must be fitted before scoring")

        A_csr = csr_matrix(A) if not sp.issparse(A) else A.tocsr()

        if mask is not None:
            # Score only on masked entries
            mask_coo = mask.tocoo()
            total_ll = 0.0
            count = 0

            for idx in range(len(mask_coo.data)):
                i, j = mask_coo.row[idx], mask_coo.col[idx]
                if mask_coo.data[idx] > 0:  # Only score masked entries
                    a_ij = A_csr[i, j]

                    # Expected rate
                    lambda_ij = 0.0
                    for k in range(self.K):
                        for l in range(self.K):
                            lambda_ij += (self.Q[i, k] * self.Q[j, l] *
                                        self.theta_out_[i] * self.theta_in_[j] *
                                        self.Omega_[k, l])

                    # Poisson log-likelihood
                    if a_ij == 0:
                        ll = -lambda_ij
                    else:
                        ll = a_ij * np.log(lambda_ij) - lambda_ij

                    total_ll += ll
                    count += 1

            return total_ll / max(count, 1)

        else:
            # Score all entries
            A_coo = A_csr.tocoo()
            total_ll = 0.0
            count = len(A_coo.data)

            for idx in range(count):
                i, j, a_ij = A_coo.row[idx], A_coo.col[idx], A_coo.data[idx]

                # Expected rate
                lambda_ij = 0.0
                for k in range(self.K):
                    for l in range(self.K):
                        lambda_ij += (self.Q[i, k] * self.Q[j, l] *
                                    self.theta_out_[i] * self.theta_in_[j] *
                                    self.Omega_[k, l])

                # Poisson log-likelihood
                if a_ij == 0:
                    ll = -lambda_ij
                else:
                    ll = a_ij * np.log(lambda_ij) - lambda_ij

                total_ll += ll

            return total_ll / max(count, 1)

    def get_params(self) -> Dict[str, np.ndarray]:
        """Get model parameters."""
        if self.theta_out_ is None:
            raise ValueError("Model must be fitted before getting parameters")

        return {
            'theta_out': self.theta_out_.copy(),
            'theta_in': self.theta_in_.copy(),
            'Omega': self.Omega_.copy(),
            'Q': self.Q.copy(),
            'pi': self.pi_.copy()
        }

    def diagnostics(self) -> Dict:
        """Get diagnostic information."""
        return {
            'elbo_trace': self.elbo_.copy(),
            'converged': self.converged_,
            'n_iter': self.n_iter_,
            'final_elbo': self.elbo_[-1] if self.elbo_ else None
        }