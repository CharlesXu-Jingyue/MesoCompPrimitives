"""
Test suite for CTRNN (Continuous-Time RNN) Analysis.
"""

import numpy as np
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sysid import CTRNNAnalyzer, FixedPointAnalysis


def create_test_weight_matrix(n=10, density=0.5, seed=42):
    """Create a test signed weight matrix."""
    np.random.seed(seed)
    W = np.random.randn(n, n) * 2  # Random signed weights
    mask = np.random.random((n, n)) < density
    W = W * mask
    return W


def create_block_structure(n=10, num_blocks=2, seed=42):
    """Create test block labels."""
    np.random.seed(seed)
    block_labels = np.random.randint(0, num_blocks, n)
    return block_labels


class TestCTRNNAnalyzer:
    """Test CTRNNAnalyzer class."""

    def test_init(self):
        """Test CTRNNAnalyzer initialization."""
        analyzer = CTRNNAnalyzer()
        assert analyzer.safety_margin == 0.9
        assert analyzer.tolerance == 1e-6
        assert analyzer.damping == 0.5
        assert analyzer.max_iterations == 1000

        # Test custom parameters
        analyzer2 = CTRNNAnalyzer(safety_margin=0.8, tolerance=1e-8, damping=0.3, max_iterations=500)
        assert analyzer2.safety_margin == 0.8
        assert analyzer2.tolerance == 1e-8
        assert analyzer2.damping == 0.3
        assert analyzer2.max_iterations == 500

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Invalid safety margin
        with pytest.raises(ValueError):
            CTRNNAnalyzer(safety_margin=1.1)

        with pytest.raises(ValueError):
            CTRNNAnalyzer(safety_margin=0.0)

        # Invalid damping
        with pytest.raises(ValueError):
            CTRNNAnalyzer(damping=0.0)

        with pytest.raises(ValueError):
            CTRNNAnalyzer(damping=1.5)

    def test_global_normalization(self):
        """Test global row-sum normalization."""
        analyzer = CTRNNAnalyzer(safety_margin=0.9)

        # Test matrix
        W = np.array([[2, -1], [1, 3]])  # Row sums: [3, 4], max = 4
        W_norm, alpha, s_inf = analyzer._global_normalization(W)

        assert s_inf == 4.0
        expected_alpha = 4 * 0.9 / 4.0  # = 0.9
        assert np.isclose(alpha, expected_alpha)
        assert np.allclose(W_norm, alpha * W)

        # Check normalized matrix satisfies contraction property
        normalized_max_row_sum = np.max(np.sum(np.abs(W_norm), axis=1))
        assert normalized_max_row_sum <= 4 * analyzer.safety_margin

    def test_sigmoid_functions(self):
        """Test sigmoid and sigmoid derivative functions."""
        analyzer = CTRNNAnalyzer()

        # Test sigmoid
        x = np.array([-10, -1, 0, 1, 10])
        sigma = analyzer._sigmoid(x)

        assert np.all(sigma >= 0)
        assert np.all(sigma <= 1)
        assert np.isclose(sigma[2], 0.5)  # σ(0) = 0.5

        # Test sigmoid derivative
        sigma_prime = analyzer._sigmoid_derivative(x)

        assert np.all(sigma_prime >= 0)
        assert np.all(sigma_prime <= 0.25)  # Max value is 1/4
        assert np.isclose(sigma_prime[2], 0.25)  # σ'(0) = 0.25

    def test_get_block_info(self):
        """Test block information extraction."""
        analyzer = CTRNNAnalyzer()
        block_labels = np.array([0, 1, 0, 2, 1, 2])
        block_info = analyzer._get_block_info(block_labels)

        assert len(block_info) == 3  # Three unique blocks
        assert np.array_equal(block_info[0], [0, 2])
        assert np.array_equal(block_info[1], [1, 4])
        assert np.array_equal(block_info[2], [3, 5])

    def test_basic_analysis(self):
        """Test basic CTRNN analysis pipeline."""
        # Create test data
        W = create_test_weight_matrix(n=8, density=0.6, seed=42)
        block_labels = create_block_structure(n=8, num_blocks=2, seed=42)

        analyzer = CTRNNAnalyzer(safety_margin=0.9, tolerance=1e-6, max_iterations=100)

        # Run analysis
        results = analyzer.analyze(
            W=W,
            block_labels=block_labels,
            perform_optional_analyses=False  # Skip optional analyses for speed
        )

        # Check results structure
        assert isinstance(results, FixedPointAnalysis)
        assert results.W_normalized is not None
        assert results.normalization_factor > 0
        assert results.original_norm > 0
        assert results.safety_margin == 0.9

        # Check fixed points
        assert len(results.fixed_points) == len(np.unique(block_labels))
        assert len(results.sigmoid_gains) == len(np.unique(block_labels))

        # Check linearized dynamics
        assert results.A_global.shape == (8, 8)
        assert len(results.A_blocks) == len(np.unique(block_labels))

    def test_convergence_info(self):
        """Test convergence information tracking."""
        # Create simple test case
        W = np.array([[0.1, 0.2], [0.1, 0.1]])  # Small weights for easy convergence
        block_labels = np.array([0, 1])  # Two single-neuron blocks

        analyzer = CTRNNAnalyzer(tolerance=1e-8, max_iterations=50)
        results = analyzer.analyze(W, block_labels, perform_optional_analyses=False)

        # Check convergence info
        for block_id, conv_info in results.convergence_info.items():
            assert 'converged' in conv_info
            assert 'iterations' in conv_info
            assert 'final_error' in conv_info
            assert 'error_history' in conv_info

            # For this simple case, should converge
            assert conv_info['converged'] == True
            assert conv_info['final_error'] < analyzer.tolerance

    def test_stability_check(self):
        """Test global stability analysis."""
        # Create stable system (small weights)
        W = np.array([[0.1, 0.05], [0.05, 0.1]])
        block_labels = np.array([0, 0])  # Single block

        analyzer = CTRNNAnalyzer()
        results = analyzer.analyze(W, block_labels, perform_optional_analyses=False)

        # Check stability
        stability_info = analyzer.check_global_stability(results.A_global)
        assert 'is_stable' in stability_info
        assert 'max_real_eigenvalue' in stability_info
        assert 'stability_margin' in stability_info
        assert 'num_unstable_modes' in stability_info

    def test_with_bias_and_time_constants(self):
        """Test analysis with bias and time constants."""
        W = create_test_weight_matrix(n=6, seed=123)
        block_labels = np.array([0, 0, 1, 1, 2, 2])  # Three blocks
        bias = np.random.randn(6) * 0.1
        time_constants = np.array([1.0, 2.0, 0.5])  # Different time constants per block

        analyzer = CTRNNAnalyzer()
        results = analyzer.analyze(
            W=W,
            block_labels=block_labels,
            bias=bias,
            time_constants=time_constants,
            perform_optional_analyses=False
        )

        assert results is not None
        assert len(results.fixed_points) == 3
        assert results.A_global.shape == (6, 6)

    def test_input_weights_and_balanced_truncation(self):
        """Test analysis with input weights and balanced truncation."""
        W = create_test_weight_matrix(n=4, seed=456)
        block_labels = np.array([0, 0, 1, 1])  # Two blocks
        input_weights = np.random.randn(4, 2) * 0.1  # 2 inputs

        analyzer = CTRNNAnalyzer()
        results = analyzer.analyze(
            W=W,
            block_labels=block_labels,
            input_weights=input_weights,
            perform_optional_analyses=True
        )

        assert results.B_linear is not None
        assert results.B_linear.shape == (4, 2)
        assert results.eigenvalues is not None
        assert results.schur_decomp is not None

    def test_zero_weight_matrix(self):
        """Test handling of zero weight matrix."""
        W = np.zeros((3, 3))
        block_labels = np.array([0, 1, 2])

        analyzer = CTRNNAnalyzer()
        results = analyzer.analyze(W, block_labels, perform_optional_analyses=False)

        # Should handle zero matrix gracefully
        assert results.normalization_factor == 1.0
        assert results.original_norm == 0.0


def test_edge_cases():
    """Test edge cases and robustness."""
    analyzer = CTRNNAnalyzer()

    # Single neuron
    W_single = np.array([[0.5]])
    block_labels_single = np.array([0])

    results = analyzer.analyze(W_single, block_labels_single, perform_optional_analyses=False)
    assert results.A_global.shape == (1, 1)
    assert len(results.fixed_points) == 1

    # Identical blocks
    W_identical = np.random.randn(4, 4) * 0.3
    block_labels_identical = np.array([0, 0, 0, 0])  # All same block

    results = analyzer.analyze(W_identical, block_labels_identical, perform_optional_analyses=False)
    assert len(results.fixed_points) == 1
    assert results.A_global.shape == (4, 4)


if __name__ == "__main__":
    # Run basic tests
    print("Running CTRNN analyzer tests...")

    # Test initialization
    test_analyzer = TestCTRNNAnalyzer()
    test_analyzer.test_init()
    test_analyzer.test_parameter_validation()
    print("✓ Initialization tests passed")

    # Test normalization
    test_analyzer.test_global_normalization()
    test_analyzer.test_sigmoid_functions()
    test_analyzer.test_get_block_info()
    print("✓ Utility function tests passed")

    # Test analysis pipeline
    test_analyzer.test_basic_analysis()
    test_analyzer.test_convergence_info()
    test_analyzer.test_stability_check()
    print("✓ Basic analysis tests passed")

    # Test edge cases
    test_edge_cases()
    print("✓ Edge case tests passed")

    print("All CTRNN tests completed successfully!")