"""
Test suite for DC-SBM implementation.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dcsbm import DCSBM, spectral_init, degrees, heldout_split, to_edge_list


def generate_synthetic_dcsbm(n=100, K=3, seed=42):
    """Generate synthetic DC-SBM data for testing."""
    np.random.seed(seed)

    # Random block assignments
    labels = np.random.choice(K, size=n)

    # Random degree parameters
    theta_out = np.random.gamma(2, 0.5, size=n)
    theta_in = np.random.gamma(2, 0.5, size=n)

    # Normalize within blocks
    for k in range(K):
        mask = labels == k
        if mask.sum() > 0:
            theta_out[mask] /= theta_out[mask].sum()
            theta_in[mask] /= theta_in[mask].sum()

    # Block-block rates
    Omega = np.random.gamma(2, 1, size=(K, K))

    # Generate adjacency matrix
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            rate = theta_out[i] * theta_in[j] * Omega[labels[i], labels[j]]
            A[i, j] = np.random.poisson(rate)

    return A, labels, theta_out, theta_in, Omega


class TestDCSBM:
    """Test cases for DCSBM class."""

    def test_init(self):
        """Test model initialization."""
        model = DCSBM(K=3, max_iter=100, tol=1e-5, seed=42)
        assert model.K == 3
        assert model.max_iter == 100
        assert model.tol == 1e-5
        assert model.seed == 42
        assert model.Q is None
        assert model.labels_ is None

    def test_fit_basic(self):
        """Test basic fitting functionality."""
        A, true_labels, _, _, _ = generate_synthetic_dcsbm(n=50, K=2, seed=42)

        model = DCSBM(K=2, max_iter=50, seed=42)
        model.fit(A)

        # Check that model was fitted
        assert model.Q is not None
        assert model.labels_ is not None
        assert model.theta_out_ is not None
        assert model.theta_in_ is not None
        assert model.Omega_ is not None

        # Check shapes
        n = A.shape[0]
        assert model.Q.shape == (n, 2)
        assert model.labels_.shape == (n,)
        assert model.theta_out_.shape == (n,)
        assert model.theta_in_.shape == (n,)
        assert model.Omega_.shape == (2, 2)

        # Check properties
        assert np.allclose(model.Q.sum(axis=1), 1.0)  # Q rows sum to 1
        assert np.all(model.theta_out_ > 0)  # Positive parameters
        assert np.all(model.theta_in_ > 0)
        assert np.all(model.Omega_ >= 0)

    def test_elbo_monotonic(self):
        """Test that ELBO increases monotonically."""
        A, _, _, _, _ = generate_synthetic_dcsbm(n=30, K=2, seed=42)

        model = DCSBM(K=2, max_iter=20, seed=42)
        model.fit(A)

        # Check ELBO is monotonically increasing (with small tolerance)
        elbo_trace = model.elbo_
        assert len(elbo_trace) > 1

        for i in range(1, len(elbo_trace)):
            assert elbo_trace[i] >= elbo_trace[i-1] - 1e-6

    def test_sparse_input(self):
        """Test fitting with sparse matrices."""
        A, _, _, _, _ = generate_synthetic_dcsbm(n=30, K=2, seed=42)
        A_sparse = csr_matrix(A)

        model = DCSBM(K=2, max_iter=20, seed=42)
        model.fit(A_sparse)

        assert model.Q is not None
        assert model.converged_ is not None

    def test_predict(self):
        """Test prediction functionality."""
        A, _, _, _, _ = generate_synthetic_dcsbm(n=30, K=2, seed=42)

        model = DCSBM(K=2, max_iter=20, seed=42)
        model.fit(A)

        labels = model.predict()
        assert labels.shape == (30,)
        assert np.all(np.isin(labels, [0, 1]))

    def test_fit_transform(self):
        """Test fit_transform method."""
        A, _, _, _, _ = generate_synthetic_dcsbm(n=30, K=2, seed=42)

        model = DCSBM(K=2, max_iter=20, seed=42)
        Q = model.fit_transform(A)

        assert Q.shape == (30, 2)
        assert np.allclose(Q.sum(axis=1), 1.0)
        assert np.array_equal(Q, model.Q)

    def test_score(self):
        """Test scoring functionality."""
        A, _, _, _, _ = generate_synthetic_dcsbm(n=30, K=2, seed=42)

        model = DCSBM(K=2, max_iter=20, seed=42)
        model.fit(A)

        score = model.score(A)
        assert isinstance(score, float)
        assert not np.isnan(score)

    def test_get_params(self):
        """Test parameter retrieval."""
        A, _, _, _, _ = generate_synthetic_dcsbm(n=30, K=2, seed=42)

        model = DCSBM(K=2, max_iter=20, seed=42)
        model.fit(A)

        params = model.get_params()
        required_keys = ['theta_out', 'theta_in', 'Omega', 'Q', 'pi']
        assert all(key in params for key in required_keys)

        # Check that returned parameters are copies
        params['Q'][0, 0] = -999
        assert model.Q[0, 0] != -999

    def test_diagnostics(self):
        """Test diagnostic information."""
        A, _, _, _, _ = generate_synthetic_dcsbm(n=30, K=2, seed=42)

        model = DCSBM(K=2, max_iter=20, seed=42)
        model.fit(A)

        diag = model.diagnostics()
        required_keys = ['elbo_trace', 'converged', 'n_iter', 'final_elbo']
        assert all(key in diag for key in required_keys)

        assert isinstance(diag['converged'], bool)
        assert isinstance(diag['n_iter'], int)
        assert diag['n_iter'] > 0


class TestUtilities:
    """Test utility functions."""

    def test_degrees(self):
        """Test degree computation."""
        A = np.array([[0, 2, 1],
                      [1, 0, 3],
                      [2, 1, 0]])

        k_out, k_in = degrees(A)

        expected_out = np.array([3, 4, 3])
        expected_in = np.array([3, 3, 4])

        assert np.array_equal(k_out, expected_out)
        assert np.array_equal(k_in, expected_in)

    def test_degrees_sparse(self):
        """Test degree computation with sparse matrices."""
        A = csr_matrix([[0, 2, 1],
                        [1, 0, 3],
                        [2, 1, 0]])

        k_out, k_in = degrees(A)

        expected_out = np.array([3, 4, 3])
        expected_in = np.array([3, 3, 4])

        assert np.array_equal(k_out, expected_out)
        assert np.array_equal(k_in, expected_in)

    def test_to_edge_list(self):
        """Test edge list conversion."""
        A = np.array([[0, 2, 0],
                      [1, 0, 3],
                      [0, 1, 0]])

        edges = to_edge_list(A)
        expected = [(0, 1, 2.0), (1, 0, 1.0), (1, 2, 3.0), (2, 1, 1.0)]

        assert len(edges) == len(expected)
        for edge in expected:
            assert edge in edges

    def test_spectral_init(self):
        """Test spectral initialization."""
        A, _, _, _, _ = generate_synthetic_dcsbm(n=50, K=3, seed=42)

        Q = spectral_init(A, K=3, seed=42)

        assert Q.shape == (50, 3)
        assert np.allclose(Q.sum(axis=1), 1.0)
        assert np.all(Q >= 0)

    def test_heldout_split(self):
        """Test held-out data splitting."""
        A, _, _, _, _ = generate_synthetic_dcsbm(n=30, K=2, seed=42)
        A = csr_matrix(A)

        A_train, A_val, mask_val = heldout_split(A, frac=0.2, seed=42)

        # Check shapes
        assert A_train.shape == A.shape
        assert A_val.shape == A.shape
        assert mask_val.shape == A.shape

        # Check that total edges are preserved
        total_orig = A.nnz
        total_train = A_train.nnz
        total_val = A_val.nnz

        assert total_train + total_val == total_orig

        # Check held-out fraction is approximately correct
        frac_actual = total_val / total_orig
        assert abs(frac_actual - 0.2) < 0.05  # Allow 5% tolerance


def test_edge_cases():
    """Test edge cases and error handling."""
    # Test with negative values (should raise error)
    A_neg = np.array([[-1, 2], [1, 0]])
    model = DCSBM(K=2)

    with pytest.raises(ValueError, match="non-negative"):
        model.fit(A_neg)

    # Test prediction before fitting
    model = DCSBM(K=2)
    with pytest.raises(ValueError, match="must be fitted"):
        model.predict()

    # Test scoring before fitting
    A = np.random.poisson(2, size=(10, 10))
    with pytest.raises(ValueError, match="must be fitted"):
        model.score(A)

    # Test get_params before fitting
    with pytest.raises(ValueError, match="must be fitted"):
        model.get_params()


if __name__ == "__main__":
    # Run basic tests
    print("Running basic DC-SBM tests...")

    # Test synthetic data generation
    A, labels, theta_out, theta_in, Omega = generate_synthetic_dcsbm(n=50, K=3, seed=42)
    print(f"Generated synthetic data: {A.shape}, {len(np.unique(labels))} blocks")

    # Test model fitting
    model = DCSBM(K=3, max_iter=50, seed=42)
    model.fit(A)
    print(f"Model fitted: converged={model.converged_}, iterations={model.n_iter_}")

    # Test recovery quality (basic check)
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(labels, model.predict())
    print(f"Recovery ARI: {ari:.3f}")

    print("Basic tests completed successfully!")