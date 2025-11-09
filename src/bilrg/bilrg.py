"""
Bi-orthogonal Laplacian Renormalization Group (bi-LRG) implementation.

This module implements the complete bi-LRG pipeline for hierarchical
coarse-graining of directed weighted networks.
"""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eig, inv
from sklearn.cluster import KMeans
from typing import Union, Tuple, Optional, Dict, List
import warnings

from .utils_bilrg import (
    is_sparse, to_dense, safe_divide, compute_stationary_distribution,
    biorthogonal_modes, realify_modes, validate_transition_matrix,
    create_teleportation_matrix, spectral_fidelity
)


class BiLRG:
    """
    Bi-orthogonal Laplacian Renormalization Group for directed networks.

    Implements the complete pipeline from adjacency matrix to coarse-grained
    network representation via spectral analysis and mesoscale grouping.

    Parameters
    ----------
    k : int, default=5
        Number of slow modes to retain
    alpha : float, default=0.95
        Teleportation parameter for handling dangling nodes
    cluster_method : str, default='kmeans'
        Clustering method ('kmeans' or 'soft')
    realify : bool, default=True
        Whether to convert complex modes to real representation
    spectral_matrix : str, default='L'
        Matrix to use for spectral decomposition ('L' for Laplacian, 'P' for transition matrix)
    random_state : int, optional
        Random seed for reproducibility
    """

    def __init__(self, k: int = 5, alpha: float = 0.95,
                 cluster_method: str = 'kmeans', realify: bool = True,
                 spectral_matrix: str = 'L', random_state: Optional[int] = None):
        self.k = k
        self.alpha = alpha
        self.cluster_method = cluster_method
        self.realify = realify
        self.spectral_matrix = spectral_matrix
        self.random_state = random_state

        # Results storage
        self.A_ = None
        self.P_ = None
        self.L_ = None
        self.U_k_ = None
        self.V_k_ = None
        self.Lambda_k_ = None
        self.X_ = None  # Bi-embedding
        self.C_ = None  # Membership matrix
        self.P_group_ = None
        self.L_group_ = None
        self.A_group_ = None
        self.pi_ = None  # Stationary distribution
        self.groups_ = None  # Group labels
        self.fidelity_ = None

        # Bi-Galerkin projection results
        self.P_galerkin_ = None  # Bi-Galerkin projected transition matrix
        self.L_galerkin_ = None  # Bi-Galerkin projected Laplacian
        self.A_galerkin_ = None  # Bi-Galerkin projected adjacency

    def fit(self, A: Union[np.ndarray, sp.spmatrix],
            L: Optional[Union[np.ndarray, sp.spmatrix]] = None) -> "BiLRG":
        """
        Fit bi-LRG to adjacency matrix.

        Parameters
        ----------
        A : array-like, shape (n, n)
            Weighted directed adjacency matrix
        L : array-like, shape (n, n), optional
            Custom Laplacian matrix. If provided, bypasses internal Laplacian computation

        Returns
        -------
        self : BiLRG
            Fitted model
        """
        # Store original adjacency
        self.A_ = A

        # Step 1: Create transition matrix
        self.P_ = self._create_transition_matrix(A)

        # Step 2: Use provided Laplacian or create random-walk Laplacian
        if L is not None:
            # Use user-provided Laplacian
            self.L_ = L
        else:
            # Create random-walk Laplacian: L_rw = I - P
            self.L_ = self._create_laplacian(self.P_)

        # Step 3: Compute bi-orthogonal modes
        if self.spectral_matrix == 'P':
            # Use transition matrix for spectral decomposition
            spectral_op = self.P_
        elif self.spectral_matrix == 'L':
            # Use Laplacian for spectral decomposition (default)
            spectral_op = self.L_
        else:
            raise ValueError(f"spectral_matrix must be 'P' or 'L', got {self.spectral_matrix}")

        self.U_k_, self.V_k_, self.Lambda_k_ = self._compute_modes(spectral_op)

        # Step 4: Create bi-embedding
        self.X_ = self._create_biembedding(self.U_k_, self.V_k_)

        # Step 5: Discover groups
        self.C_, self.groups_ = self._discover_groups(self.X_)

        # Step 6: Create coarse operators via Markov lumping
        self.P_group_, self.L_group_, self.A_group_ = self._create_coarse_operators(
            self.P_, self.C_
        )

        # Step 7: Compute bi-Galerkin projection
        self.P_galerkin_, self.L_galerkin_, self.A_galerkin_ = self._compute_bi_galerkin_projection()

        # Step 8: Compute quality metrics
        self.fidelity_ = self._compute_fidelity()

        return self

    def _create_transition_matrix(self, A: Union[np.ndarray, sp.spmatrix]
                                 ) -> Union[np.ndarray, sp.spmatrix]:
        """Create row-stochastic transition matrix with teleportation."""
        # Handle teleportation and dangling nodes
        P = create_teleportation_matrix(A, alpha=self.alpha)

        # Validate result
        if not validate_transition_matrix(P):
            warnings.warn("Created transition matrix is not properly row-stochastic")

        return P

    def _create_laplacian(self, P: Union[np.ndarray, sp.spmatrix]
                         ) -> Union[np.ndarray, sp.spmatrix]:
        """Create random-walk Laplacian L = I - P."""
        n = P.shape[0]

        if is_sparse(P):
            I = sp.eye(n, format='csr')
            L = I - P
        else:
            I = np.eye(n)
            L = I - P

        return L

    def _compute_modes(self, spectral_op: Union[np.ndarray, sp.spmatrix]
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute bi-orthogonal modes from spectral operator (P or L)."""
        U_k, V_k, Lambda_k = biorthogonal_modes(spectral_op, k=self.k)

        # Realify if requested
        if self.realify:
            U_k, V_k, Lambda_k = realify_modes(U_k, V_k, Lambda_k)

        return U_k, V_k, Lambda_k

    def _create_biembedding(self, U_k: np.ndarray, V_k: np.ndarray) -> np.ndarray:
        """
        Create bi-embedding of nodes.

        Each node i gets embedding X_i = [Re(V_k[i, :]), Re(U_k[i, :])].
        """
        # Take real parts for embedding
        psi_R = np.real(V_k)  # Right modes (forward)
        psi_L = np.real(U_k)  # Left modes (backward)

        # Concatenate for joint embedding
        X = np.hstack([psi_R, psi_L])

        return X

    def _discover_groups(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Discover groups via clustering in embedding space.

        Parameters
        ----------
        X : ndarray, shape (n, 2k)
            Bi-embedding of nodes

        Returns
        -------
        C : ndarray, shape (n, m)
            Membership matrix
        labels : ndarray, shape (n,)
            Group labels
        """
        n = X.shape[0]

        # Determine number of groups (default to k if not specified)
        m = min(self.k, n)

        if self.cluster_method == 'kmeans':
            # Hard assignment via k-means
            kmeans = KMeans(n_clusters=m, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)

            # Create hard membership matrix
            C = np.zeros((n, m))
            C[np.arange(n), labels] = 1.0

        elif self.cluster_method == 'soft':
            # Soft assignment (simplified version)
            # In practice, would use PCCA+ or similar sophisticated method (20251107)
            kmeans = KMeans(n_clusters=m, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)

            # Create soft membership based on distances to centroids
            distances = kmeans.transform(X)
            
            # Convert distances to probabilities via softmax
            C = np.exp(-distances / np.std(distances))
            C = C / C.sum(axis=1, keepdims=True)

        else:
            raise ValueError(f"Unknown cluster_method: {self.cluster_method}")

        return C, labels

    def _create_coarse_operators(self, P: Union[np.ndarray, sp.spmatrix],
                               C: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create coarse-grained operators via mass-preserving lumping.

        Parameters
        ----------
        P : array-like
            Original transition matrix
        C : ndarray
            Membership matrix

        Returns
        -------
        P_group : ndarray
            Coarse transition matrix
        L_group : ndarray
            Coarse Laplacian
        A_group : ndarray
            Coarse adjacency matrix
        """
        # Compute stationary distribution
        pi = compute_stationary_distribution(P)
        self.pi_ = pi

        # Create diagonal matrix of stationary probabilities
        Pi = np.diag(pi)

        # Mass-preserving aggregation and disaggregation operators
        # R_C = (C^T Pi C)^(-1) C^T Pi
        # P_C = C

        CTPiC = C.T @ Pi @ C
        try:
            CTPiC_inv = inv(CTPiC)
        except np.linalg.LinAlgError:
            # Handle singular matrix
            CTPiC_inv = np.linalg.pinv(CTPiC)
            warnings.warn("Using pseudo-inverse for singular CTPiC matrix")

        R_C = CTPiC_inv @ C.T @ Pi

        # Coarse transition matrix: P_group = R_C P C
        if is_sparse(P):
            P_dense = to_dense(P)
            P_group = R_C @ P_dense @ C
        else:
            P_group = R_C @ P @ C

        # Coarse Laplacian: L_group = I - P_group
        m = P_group.shape[0]
        L_group = np.eye(m) - P_group

        # Coarse adjacency: A_group = D_out_group @ P_group
        # Use aggregated out-degrees or identity
        if is_sparse(self.A_):
            out_degrees = np.array(self.A_.sum(axis=1)).flatten()
        else:
            out_degrees = self.A_.sum(axis=1)

        # Aggregate out-degrees
        out_degrees_group = C.T @ out_degrees
        D_out_group = np.diag(out_degrees_group)

        A_group = D_out_group @ P_group

        return P_group, L_group, A_group

    def _compute_bi_galerkin_projection(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute bi-Galerkin projection of operators into spectral subspace.

        This projects the operators using the bi-orthogonal modes:
        P_galerkin = U_k^H @ P @ V_k
        L_galerkin = U_k^H @ L @ V_k
        A_galerkin = D_out_galerkin @ P_galerkin
        where D_out_galerkin = diag(V_k^H @ out_degrees)

        Returns
        -------
        P_galerkin : ndarray, shape (k, k)
            Bi-Galerkin projected transition matrix
        L_galerkin : ndarray, shape (k, k)
            Bi-Galerkin projected Laplacian
        A_galerkin : ndarray, shape (k, k)
            Bi-Galerkin projected adjacency matrix
        """
        if self.U_k_ is None or self.V_k_ is None:
            raise ValueError("Bi-orthogonal modes must be computed before Galerkin projection")

        # Convert to dense for projection if needed
        P_dense = to_dense(self.P_) if is_sparse(self.P_) else self.P_

        # Bi-Galerkin projection: U_k^H @ Operator @ V_k
        # Note: For real modes, U_k^H = U_k^T
        U_k_H = self.U_k_.conj().T  # Hermitian transpose

        try:
            # Project transition matrix: P_galerkin = U_k^H @ P @ V_k
            P_galerkin = U_k_H @ P_dense @ self.V_k_

            # Project the spectral operator used for mode computation
            if self.spectral_matrix == 'P':
                # If modes computed from P, project P in spectral space
                spectral_op_galerkin = U_k_H @ P_dense @ self.V_k_
            else:
                # If modes computed from L, project L in spectral space
                L_dense = to_dense(self.L_) if is_sparse(self.L_) else self.L_
                spectral_op_galerkin = U_k_H @ L_dense @ self.V_k_

            # For consistency, always provide both P_galerkin and L_galerkin
            L_dense = to_dense(self.L_) if is_sparse(self.L_) else self.L_
            L_galerkin = U_k_H @ L_dense @ self.V_k_

            # Compute coarse adjacency properly via out-degrees
            # A_galerkin = D_out_galerkin @ P_galerkin
            # where D_out_galerkin = diag(V_k^H @ D_out @ 1)

            # Get out-degrees from original adjacency matrix
            if is_sparse(self.A_):
                out_degrees = np.array(self.A_.sum(axis=1)).flatten()
            else:
                out_degrees = self.A_.sum(axis=1)

            # Project out-degrees: V_k^H @ D_out @ 1 = V_k^H @ out_degrees
            out_degrees_galerkin = self.V_k_.conj().T @ out_degrees

            # Create diagonal matrix of projected out-degrees
            D_out_galerkin = np.diag(out_degrees_galerkin)

            # Compute coarse adjacency: A_galerkin = D_out_galerkin @ P_galerkin
            A_galerkin = D_out_galerkin @ P_galerkin

            # Take real part if modes were realified
            if self.realify:
                P_galerkin = np.real(P_galerkin)
                L_galerkin = np.real(L_galerkin)
                out_degrees_galerkin = np.real(out_degrees_galerkin)
                D_out_galerkin = np.real(D_out_galerkin)
                A_galerkin = np.real(A_galerkin)

        except Exception as e:
            warnings.warn(f"Bi-Galerkin projection failed: {e}. Using zero matrices.")
            k = self.U_k_.shape[1]
            P_galerkin = np.zeros((k, k))
            L_galerkin = np.zeros((k, k))
            A_galerkin = np.zeros((k, k))

        return P_galerkin, L_galerkin, A_galerkin

    def _compute_fidelity(self) -> float:
        """Compute spectral fidelity of coarse-graining."""
        if self.V_k_ is None or self.C_ is None:
            return np.inf

        # Get reduced modes (assuming P_group has same eigenvectors structure)
        try:
            _, V_group = eig(self.P_group_.T)
            V_group = np.real(V_group[:, :self.k])

            fidelity = spectral_fidelity(self.V_k_, V_group, self.C_)
            return fidelity

        except Exception as e:
            warnings.warn(f"Fidelity computation failed: {e}")
            return np.inf

    def get_coarse_graph(self) -> Dict[str, np.ndarray]:
        """
        Get coarse-grained graph representation.

        Returns
        -------
        coarse_graph : dict
            Dictionary containing coarse operators and metadata
        """
        if self.P_group_ is None:
            raise ValueError("Model must be fitted before getting coarse graph")

        return {
            # Markov lumping results
            'P_group': self.P_group_,
            'L_group': self.L_group_,
            'A_group': self.A_group_,
            'membership': self.C_,
            'groups': self.groups_,

            # Bi-Galerkin projection results
            'P_galerkin': self.P_galerkin_,
            'L_galerkin': self.L_galerkin_,
            'A_galerkin': self.A_galerkin_,

            # Spectral and quality metrics
            'eigenvalues': self.Lambda_k_,
            'fidelity': self.fidelity_,
            'spectral_matrix': self.spectral_matrix,
            'n_original': self.A_.shape[0],
            'n_coarse': self.P_group_.shape[0],
            'n_galerkin': self.P_galerkin_.shape[0] if self.P_galerkin_ is not None else None
        }

    def get_embedding(self) -> np.ndarray:
        """Get bi-embedding of nodes."""
        if self.X_ is None:
            raise ValueError("Model must be fitted before getting embedding")
        return self.X_.copy()

    def get_modes(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get bi-orthogonal modes."""
        if self.U_k_ is None:
            raise ValueError("Model must be fitted before getting modes")
        return self.U_k_.copy(), self.V_k_.copy(), self.Lambda_k_.copy()

    def get_bi_galerkin_operators(self) -> Dict[str, np.ndarray]:
        """
        Get bi-Galerkin projected operators.

        Returns
        -------
        galerkin_ops : dict
            Dictionary containing bi-Galerkin projected operators
        """
        if self.P_galerkin_ is None:
            raise ValueError("Model must be fitted before getting bi-Galerkin operators")

        return {
            'P_galerkin': self.P_galerkin_.copy(),
            'L_galerkin': self.L_galerkin_.copy(),
            'A_galerkin': self.A_galerkin_.copy(),
            'eigenvalues': self.Lambda_k_.copy(),
            'spectral_matrix': self.spectral_matrix,
            'shape': self.P_galerkin_.shape
        }

    def transform(self, steps: int = 1) -> np.ndarray:
        """
        Apply coarse dynamics for given number of steps.

        Parameters
        ----------
        steps : int
            Number of time steps to evolve

        Returns
        -------
        P_evolved : ndarray
            Coarse transition matrix raised to power 'steps'
        """
        if self.P_group_ is None:
            raise ValueError("Model must be fitted before transformation")

        return np.linalg.matrix_power(self.P_group_, steps)

    def fit_transform(self, A: Union[np.ndarray, sp.spmatrix],
                      L: Optional[Union[np.ndarray, sp.spmatrix]] = None) -> np.ndarray:
        """Fit model and return bi-embedding."""
        self.fit(A, L)
        return self.get_embedding()


class HierarchicalBiLRG:
    """
    Hierarchical (multilevel) bi-LRG for recursive coarse-graining.

    Parameters
    ----------
    k : int, default=5
        Number of slow modes per level
    max_levels : int, default=3
        Maximum number of hierarchy levels
    min_nodes : int, default=10
        Minimum nodes to continue coarse-graining
    fidelity_threshold : float, default=0.5
        Stop if fidelity exceeds this threshold
    **kwargs
        Additional parameters passed to BiLRG
    """

    def __init__(self, k: int = 5, max_levels: int = 3, min_nodes: int = 10,
                 fidelity_threshold: float = 0.5, **kwargs):
        self.k = k
        self.max_levels = max_levels
        self.min_nodes = min_nodes
        self.fidelity_threshold = fidelity_threshold
        self.bilrg_kwargs = kwargs

        # Results storage
        self.levels_ = []  # List of BiLRG instances
        self.hierarchy_ = []  # List of coarse graphs

    def fit(self, A: Union[np.ndarray, sp.spmatrix]) -> "HierarchicalBiLRG":
        """
        Fit hierarchical bi-LRG.

        Parameters
        ----------
        A : array-like
            Initial adjacency matrix

        Returns
        -------
        self : HierarchicalBiLRG
            Fitted hierarchical model
        """
        self.levels_ = []
        self.hierarchy_ = []

        current_A = A

        for level in range(self.max_levels):
            # Check stopping criteria
            n_nodes = current_A.shape[0]
            if n_nodes <= self.min_nodes:
                break

            # Fit BiLRG at current level
            bilrg = BiLRG(k=self.k, **self.bilrg_kwargs)
            bilrg.fit(current_A)

            # Store level results
            self.levels_.append(bilrg)
            coarse_graph = bilrg.get_coarse_graph()
            self.hierarchy_.append(coarse_graph)

            # Check fidelity stopping criterion
            if bilrg.fidelity_ > self.fidelity_threshold:
                warnings.warn(f"Stopping at level {level}: fidelity {bilrg.fidelity_:.3f} "
                             f"exceeds threshold {self.fidelity_threshold}")
                break

            # Prepare next level
            current_A = coarse_graph['A_group']

        return self

    def get_hierarchy(self) -> List[Dict]:
        """Get complete hierarchy of coarse graphs."""
        return self.hierarchy_

    def get_level(self, level: int) -> BiLRG:
        """Get BiLRG instance at specific level."""
        if level >= len(self.levels_):
            raise IndexError(f"Level {level} not available. Only {len(self.levels_)} levels computed.")
        return self.levels_[level]

    def project_to_level(self, x: np.ndarray, target_level: int) -> np.ndarray:
        """
        Project vector from original space to specified level.

        Parameters
        ----------
        x : ndarray
            Vector in original space
        target_level : int
            Target hierarchy level

        Returns
        -------
        x_projected : ndarray
            Vector projected to target level
        """
        current_x = x.copy()

        for level in range(target_level + 1):
            if level >= len(self.levels_):
                break

            # Project using membership matrix
            C = self.levels_[level].C_
            current_x = C.T @ current_x

        return current_x