"""
Test suite for bi-orthogonal Laplacian Renormalization Group (bi-LRG).
"""

import numpy as np
import scipy.sparse as sp
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bilrg import BiLRG, HierarchicalBiLRG
from bilrg.utils_bilrg import (
    compute_stationary_distribution, biorthogonal_modes, realify_modes,
    validate_transition_matrix, create_teleportation_matrix, spectral_fidelity
)


def create_test_adjacency(n=20, density=0.3, seed=42):
    """Create a test directed adjacency matrix."""
    np.random.seed(seed)
    A = np.random.random((n, n))
    A = (A < density).astype(float)

    # Add some weights
    A[A > 0] = np.random.random(np.sum(A > 0)) * 5

    # Ensure some connectivity
    np.fill_diagonal(A, 1.0)

    return A


def create_sparse_test_adjacency(n=50, density=0.1, seed=42):
    """Create a sparse test adjacency matrix."""
    np.random.seed(seed)
    A = sp.random(n, n, density=density, format='csr', random_state=seed)
    A.data = np.abs(A.data) * 5  # Make positive weights
    A.setdiag(1.0)  # Add self-loops
    return A


class TestUtilities:
    """Test utility functions."""

    def test_compute_stationary_distribution(self):
        """Test stationary distribution computation."""
        # Create simple transition matrix
        P = np.array([[0.7, 0.3],
                      [0.4, 0.6]])

        pi = compute_stationary_distribution(P)

        # Check properties
        assert len(pi) == 2
        assert np.abs(pi.sum() - 1.0) < 1e-10
        assert np.all(pi >= 0)

        # Check stationarity
        assert np.allclose(pi @ P, pi, atol=1e-10)

    def test_biorthogonal_modes(self):
        """Test bi-orthogonal mode computation."""
        # Create test Laplacian
        A = create_test_adjacency(10)
        P = A / A.sum(axis=1, keepdims=True)
        L_rw = np.eye(10) - P

        U_k, V_k, Lambda_k = biorthogonal_modes(L_rw, k=3)

        # Check shapes
        assert U_k.shape == (10, 3)
        assert V_k.shape == (10, 3)
        assert len(Lambda_k) == 3

    def test_realify_modes(self):
        """Test complex mode realification."""
        n, k = 10, 4

        # Create complex modes
        U_k = np.random.randn(n, k) + 1j * np.random.randn(n, k)
        V_k = np.random.randn(n, k) + 1j * np.random.randn(n, k)
        Lambda_k = np.random.randn(k) + 1j * np.random.randn(k)

        U_real, V_real, Lambda_real = realify_modes(U_k, V_k, Lambda_k)

        # Check that output is real
        assert np.all(np.isreal(U_real))
        assert np.all(np.isreal(V_real))
        assert np.all(np.isreal(Lambda_real))

    def test_validate_transition_matrix(self):
        """Test transition matrix validation."""
        # Valid transition matrix
        P_valid = np.array([[0.6, 0.4],
                           [0.3, 0.7]])
        assert validate_transition_matrix(P_valid)

        # Invalid (negative entries)
        P_invalid = np.array([[0.6, -0.4],
                             [0.3, 0.7]])
        assert not validate_transition_matrix(P_invalid)

        # Invalid (row sums != 1)
        P_invalid2 = np.array([[0.5, 0.4],
                              [0.3, 0.7]])
        assert not validate_transition_matrix(P_invalid2)

    def test_create_teleportation_matrix(self):
        """Test teleportation matrix creation."""
        # Adjacency with dangling node
        A = np.array([[1, 1, 0],
                      [0, 1, 1],
                      [0, 0, 0]])  # Last row has zero out-degree

        P = create_teleportation_matrix(A, alpha=0.9)

        # Check that result is valid transition matrix
        assert validate_transition_matrix(P)

        # Check that dangling node was handled
        assert P[2, 2] > 0  # Should have self-loop or teleportation

    def test_spectral_fidelity(self):
        """Test spectral fidelity computation."""
        n, k, m = 10, 3, 2

        # Create test data
        V_original = np.random.randn(n, k)
        V_reduced = np.random.randn(m, k)
        C = np.random.randn(n, m)
        C = C / C.sum(axis=1, keepdims=True)  # Make rows sum to 1

        fidelity = spectral_fidelity(V_original, V_reduced, C)

        assert isinstance(fidelity, float)
        assert fidelity >= 0


class TestBiLRG:
    """Test BiLRG class."""

    def test_init(self):
        """Test BiLRG initialization."""
        bilrg = BiLRG(k=5, alpha=0.95)
        assert bilrg.k == 5
        assert bilrg.alpha == 0.95
        assert bilrg.A_ is None

    def test_fit_dense(self):
        """Test BiLRG fitting with dense matrix."""
        A = create_test_adjacency(15, seed=42)
        bilrg = BiLRG(k=3, random_state=42)

        bilrg.fit(A)

        # Check that all components were computed
        assert bilrg.A_ is not None
        assert bilrg.P_ is not None
        assert bilrg.L_rw_ is not None
        assert bilrg.U_k_ is not None
        assert bilrg.V_k_ is not None
        assert bilrg.Lambda_k_ is not None
        assert bilrg.X_ is not None
        assert bilrg.C_ is not None
        assert bilrg.groups_ is not None

        # Check shapes
        n = A.shape[0]
        assert bilrg.U_k_.shape[0] == n
        assert bilrg.V_k_.shape[0] == n
        assert bilrg.X_.shape[0] == n
        assert bilrg.C_.shape[0] == n

    def test_fit_sparse(self):
        """Test BiLRG fitting with sparse matrix."""
        A = create_sparse_test_adjacency(25, seed=42)
        bilrg = BiLRG(k=3, random_state=42)

        bilrg.fit(A)

        # Check that fitting completed
        assert bilrg.P_group_ is not None
        assert bilrg.L_group_ is not None
        assert bilrg.A_group_ is not None

    def test_get_coarse_graph(self):
        """Test coarse graph retrieval."""
        A = create_test_adjacency(12, seed=42)
        bilrg = BiLRG(k=3, random_state=42)
        bilrg.fit(A)

        coarse_graph = bilrg.get_coarse_graph()

        # Check required keys
        required_keys = ['P_group', 'L_group', 'A_group', 'membership',
                        'groups', 'P_galerkin', 'L_galerkin', 'A_galerkin',
                        'eigenvalues', 'fidelity', 'n_original', 'n_coarse', 'n_galerkin']
        for key in required_keys:
            assert key in coarse_graph

    def test_get_embedding(self):
        """Test embedding retrieval."""
        A = create_test_adjacency(10, seed=42)
        bilrg = BiLRG(k=3, random_state=42)
        bilrg.fit(A)

        X = bilrg.get_embedding()

        assert X.shape[0] == 10
        assert X.shape[1] == 6  # 2 * k

    def test_get_modes(self):
        """Test mode retrieval."""
        A = create_test_adjacency(10, seed=42)
        bilrg = BiLRG(k=3, random_state=42)
        bilrg.fit(A)

        U_k, V_k, Lambda_k = bilrg.get_modes()

        assert U_k.shape == (10, 3)
        assert V_k.shape == (10, 3)
        assert len(Lambda_k) == 3

    def test_transform(self):
        """Test dynamics transformation."""
        A = create_test_adjacency(8, seed=42)
        bilrg = BiLRG(k=2, random_state=42)
        bilrg.fit(A)

        P_evolved = bilrg.transform(steps=2)

        # Check that it's a valid transition matrix
        assert validate_transition_matrix(P_evolved)

    def test_fit_transform(self):
        """Test fit_transform method."""
        A = create_test_adjacency(10, seed=42)
        bilrg = BiLRG(k=3, random_state=42)

        X = bilrg.fit_transform(A)

        assert X.shape == (10, 6)  # n x 2k

    def test_bi_galerkin_projection(self):
        """Test bi-Galerkin projection computation."""
        A = create_test_adjacency(15, seed=42)
        bilrg = BiLRG(k=4, random_state=42)
        bilrg.fit(A)

        # Check that bi-Galerkin operators were computed
        assert bilrg.P_galerkin_ is not None
        assert bilrg.L_galerkin_ is not None
        assert bilrg.A_galerkin_ is not None

        # Check dimensions (should be k x k)
        assert bilrg.P_galerkin_.shape == (4, 4)
        assert bilrg.L_galerkin_.shape == (4, 4)
        assert bilrg.A_galerkin_.shape == (4, 4)

        # Check that results are real (since realify=True by default)
        assert np.all(np.isreal(bilrg.P_galerkin_))
        assert np.all(np.isreal(bilrg.L_galerkin_))
        assert np.all(np.isreal(bilrg.A_galerkin_))

    def test_get_bi_galerkin_operators(self):
        """Test bi-Galerkin operators retrieval."""
        A = create_test_adjacency(12, seed=42)
        bilrg = BiLRG(k=3, random_state=42)
        bilrg.fit(A)

        galerkin_ops = bilrg.get_bi_galerkin_operators()

        # Check required keys
        required_keys = ['P_galerkin', 'L_galerkin', 'A_galerkin', 'eigenvalues', 'shape']
        for key in required_keys:
            assert key in galerkin_ops

        # Check shapes
        assert galerkin_ops['P_galerkin'].shape == (3, 3)
        assert galerkin_ops['L_galerkin'].shape == (3, 3)
        assert galerkin_ops['A_galerkin'].shape == (3, 3)

    def test_error_handling(self):
        """Test error handling."""
        bilrg = BiLRG(k=3)

        # Should raise error before fitting
        with pytest.raises(ValueError):
            bilrg.get_coarse_graph()

        with pytest.raises(ValueError):
            bilrg.get_embedding()

        with pytest.raises(ValueError):
            bilrg.get_modes()


class TestHierarchicalBiLRG:
    """Test HierarchicalBiLRG class."""

    def test_init(self):
        """Test hierarchical initialization."""
        hbilrg = HierarchicalBiLRG(k=3, max_levels=2)
        assert hbilrg.k == 3
        assert hbilrg.max_levels == 2

    def test_fit(self):
        """Test hierarchical fitting."""
        A = create_test_adjacency(30, seed=42)
        hbilrg = HierarchicalBiLRG(k=3, max_levels=2, min_nodes=5, random_state=42)

        hbilrg.fit(A)

        # Check that levels were created
        assert len(hbilrg.levels_) > 0
        assert len(hbilrg.hierarchy_) > 0

    def test_get_hierarchy(self):
        """Test hierarchy retrieval."""
        A = create_test_adjacency(20, seed=42)
        hbilrg = HierarchicalBiLRG(k=2, max_levels=2, min_nodes=3, random_state=42)
        hbilrg.fit(A)

        hierarchy = hbilrg.get_hierarchy()

        assert isinstance(hierarchy, list)
        assert len(hierarchy) > 0

    def test_get_level(self):
        """Test level retrieval."""
        A = create_test_adjacency(20, seed=42)
        hbilrg = HierarchicalBiLRG(k=2, max_levels=2, min_nodes=3, random_state=42)
        hbilrg.fit(A)

        level_0 = hbilrg.get_level(0)
        assert isinstance(level_0, BiLRG)

        # Should raise error for non-existent level
        with pytest.raises(IndexError):
            hbilrg.get_level(10)

    def test_project_to_level(self):
        """Test vector projection to hierarchy level."""
        A = create_test_adjacency(15, seed=42)
        hbilrg = HierarchicalBiLRG(k=2, max_levels=2, min_nodes=3, random_state=42)
        hbilrg.fit(A)

        # Create test vector
        x = np.random.randn(15)

        x_projected = hbilrg.project_to_level(x, target_level=0)

        # Should be smaller dimension
        assert len(x_projected) <= len(x)


def test_edge_cases():
    """Test edge cases and robustness."""
    # Very small matrix
    A_small = np.array([[1, 1], [1, 1]])
    bilrg = BiLRG(k=1, random_state=42)
    bilrg.fit(A_small)
    assert bilrg.P_group_ is not None

    # Matrix with zeros
    A_sparse = np.zeros((5, 5))
    A_sparse[0, 1] = 1
    A_sparse[1, 2] = 1
    A_sparse[2, 0] = 1

    bilrg = BiLRG(k=2, alpha=0.8, random_state=42)
    bilrg.fit(A_sparse)
    assert bilrg.P_ is not None


if __name__ == "__main__":
    # Run basic tests
    print("Running basic bi-LRG tests...")

    # Test utility functions
    test_utils = TestUtilities()
    test_utils.test_compute_stationary_distribution()
    test_utils.test_validate_transition_matrix()
    print("✓ Utility tests passed")

    # Test BiLRG
    test_bilrg = TestBiLRG()
    test_bilrg.test_fit_dense()
    test_bilrg.test_get_coarse_graph()
    test_bilrg.test_bi_galerkin_projection()
    test_bilrg.test_get_bi_galerkin_operators()
    print("✓ BiLRG tests passed")

    # Test edge cases
    test_edge_cases()
    print("✓ Edge case tests passed")

    print("All basic tests completed successfully!")