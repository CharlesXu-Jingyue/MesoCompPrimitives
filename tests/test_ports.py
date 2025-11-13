"""
Test suite for Inter-Block Control Ports & Per-Port Controllability Analysis.
"""

import numpy as np
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sysid.ports import (PortAnalyzer, PortAnalysisResults, PortConfig, PortMetrics,
                        GramianMode, CovarianceWeighting, analyze_ctrnn_ports,
                        validate_port_analysis, summarize_port_rankings)
from sysid.ctrnn import CTRNNAnalyzer, FixedPointAnalysis


def create_stable_system():
    """Create a simple stable 2-block system for testing."""
    # Block 0: 2x2, Block 1: 3x3
    A_blocks = {
        0: np.array([[-1.0, 0.2], [0.1, -1.5]]),  # Stable
        1: np.array([[-2.0, 0.3, 0.1], [0.2, -1.8, 0.4], [0.1, 0.2, -2.2]])  # Stable
    }

    # Inter-block coupling: E_rs
    E_blocks = {
        (0, 1): np.array([[0.3, 0.1, 0.2], [0.4, 0.2, 0.1]]),  # 1 → 0
        (1, 0): np.array([[0.2, 0.3], [0.1, 0.4], [0.3, 0.1]])   # 0 → 1
    }

    return A_blocks, E_blocks


def create_unstable_system():
    """Create a system with one unstable block."""
    A_blocks = {
        0: np.array([[0.5, 0.2], [0.1, -1.0]]),  # Unstable (eigenvalue > 0)
        1: np.array([[-2.0, 0.3], [0.2, -1.5]])   # Stable
    }

    E_blocks = {
        (0, 1): np.array([[0.2, 0.3], [0.1, 0.2]]),
        (1, 0): np.array([[0.3, 0.1], [0.2, 0.4]])
    }

    return A_blocks, E_blocks


class TestPortConfig:
    """Test PortConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = PortConfig()
        assert config.mode == GramianMode.INFINITE_HORIZON
        assert config.discount_lambda == 0.1
        assert config.horizon_T == 2.0
        assert config.covariance_weighting == CovarianceWeighting.NONE
        assert config.alpha_logdet == 1e-2
        assert config.top_k_modes == 5
        assert config.metric == "trace"
        assert config.stability_check == True
        assert config.rank_tolerance == 1e-12

    def test_custom_config(self):
        """Test custom configuration."""
        config = PortConfig(
            mode=GramianMode.DISCOUNTED,
            discount_lambda=0.2,
            horizon_T=5.0,
            covariance_weighting=CovarianceWeighting.STATE,
            metric="lambda_max",
            stability_check=False
        )

        assert config.mode == GramianMode.DISCOUNTED
        assert config.discount_lambda == 0.2
        assert config.horizon_T == 5.0
        assert config.covariance_weighting == CovarianceWeighting.STATE
        assert config.metric == "lambda_max"
        assert config.stability_check == False


class TestPortAnalyzer:
    """Test PortAnalyzer class."""

    def test_init(self):
        """Test PortAnalyzer initialization."""
        analyzer = PortAnalyzer()
        assert analyzer.config is not None
        assert analyzer.config.mode == GramianMode.INFINITE_HORIZON

        custom_config = PortConfig(mode=GramianMode.DISCOUNTED)
        analyzer2 = PortAnalyzer(custom_config)
        assert analyzer2.config.mode == GramianMode.DISCOUNTED

    def test_build_state_ports(self):
        """Test state port construction."""
        analyzer = PortAnalyzer()
        A_blocks, E_blocks = create_stable_system()

        port_map = analyzer._build_state_ports(E_blocks)

        # Should have same keys as E_blocks
        assert set(port_map.keys()) == set(E_blocks.keys())

        # Port matrices should be identical to E_blocks
        for key in E_blocks:
            assert np.array_equal(port_map[key], E_blocks[key])

    def test_stability_check(self):
        """Test stability checking."""
        analyzer = PortAnalyzer()

        # Test stable system
        A_blocks_stable, _ = create_stable_system()
        stability_info = analyzer._check_stability(A_blocks_stable)

        for r in A_blocks_stable:
            assert stability_info[r]['is_stable'] == True
            assert stability_info[r]['max_real_eigenvalue'] < 0

        # Test unstable system
        A_blocks_unstable, _ = create_unstable_system()
        stability_info = analyzer._check_stability(A_blocks_unstable)

        assert stability_info[0]['is_stable'] == False  # Block 0 is unstable
        assert stability_info[1]['is_stable'] == True   # Block 1 is stable

    def test_infinite_horizon_gramian(self):
        """Test infinite-horizon Gramian computation."""
        analyzer = PortAnalyzer()

        # Simple stable system
        A = np.array([[-2.0, 0.1], [0.2, -1.5]])
        B = np.array([[1.0, 0.0], [0.0, 1.0]])

        W = analyzer._solve_infinite_horizon_gramian(A, B)

        # Check properties
        assert W.shape == (2, 2)
        assert np.allclose(W, W.T)  # Symmetric

        # Check Lyapunov equation: AW + WA^T + BB^T = 0
        residual = A @ W + W @ A.T + B @ B.T
        assert np.max(np.abs(residual)) < 1e-10

        # Check positive semidefiniteness
        eigenvals = np.linalg.eigvals(W)
        assert np.all(np.real(eigenvals) >= -1e-12)

    def test_discounted_gramian(self):
        """Test discounted Gramian computation."""
        config = PortConfig(mode=GramianMode.DISCOUNTED, discount_lambda=0.5)
        analyzer = PortAnalyzer(config)

        # Use potentially unstable system
        A = np.array([[0.1, 0.2], [0.3, -0.5]])  # Mixed eigenvalues
        B = np.array([[1.0], [1.0]])

        W = analyzer._solve_discounted_gramian(A, B)

        # Check properties
        assert W.shape == (2, 2)
        assert np.allclose(W, W.T)  # Symmetric

        # Check discounted Lyapunov equation
        A_disc = A + config.discount_lambda * np.eye(2)
        residual = A_disc @ W + W @ A_disc.T + B @ B.T
        assert np.max(np.abs(residual)) < 1e-10

    def test_basic_port_analysis(self):
        """Test basic port analysis pipeline."""
        analyzer = PortAnalyzer()
        A_blocks, E_blocks = create_stable_system()

        results = analyzer.analyze_ports(A_blocks, E_blocks)

        # Check results structure
        assert isinstance(results, PortAnalysisResults)
        assert len(results.port_map) == len(E_blocks)
        assert len(results.Wc_port) == len(E_blocks)
        assert len(results.Wc_total) == len(A_blocks)

        # Check block sizes
        assert results.block_sizes[0] == 2
        assert results.block_sizes[1] == 3

        # Check Gramian shapes
        for (r, s), W in results.Wc_port.items():
            expected_size = A_blocks[r].shape[0]
            assert W.shape == (expected_size, expected_size)

        # Check total Gramians
        for r, W_total in results.Wc_total.items():
            expected_size = A_blocks[r].shape[0]
            assert W_total.shape == (expected_size, expected_size)

    def test_gramian_additivity(self):
        """Test that total Gramians equal sum of port Gramians."""
        analyzer = PortAnalyzer()
        A_blocks, E_blocks = create_stable_system()

        results = analyzer.analyze_ports(A_blocks, E_blocks)

        # Check additivity for each block
        for r in A_blocks:
            # Manually sum port Gramians
            W_sum = np.zeros_like(results.Wc_total[r])
            for (dest, src), W_port in results.Wc_port.items():
                if dest == r:
                    W_sum += W_port

            # Should equal total Gramian
            assert np.allclose(results.Wc_total[r], W_sum, atol=1e-12)

    def test_port_metrics(self):
        """Test port metrics computation."""
        analyzer = PortAnalyzer()
        A_blocks, E_blocks = create_stable_system()

        results = analyzer.analyze_ports(A_blocks, E_blocks)

        # Check metrics for each port
        for (r, s), metrics in results.port_metrics.items():
            assert isinstance(metrics, PortMetrics)
            assert metrics.trace >= 0
            assert metrics.lambda_max >= 0
            assert metrics.rank >= 0
            assert metrics.condition_number >= 1.0
            assert metrics.frobenius_norm >= 0

        # Check total metrics
        for r, metrics in results.total_metrics.items():
            assert isinstance(metrics, PortMetrics)
            assert metrics.trace >= 0

    def test_port_rankings(self):
        """Test port ranking by metrics."""
        analyzer = PortAnalyzer(PortConfig(metric="trace"))
        A_blocks, E_blocks = create_stable_system()

        results = analyzer.analyze_ports(A_blocks, E_blocks)

        # Check rankings structure
        for r, rankings in results.top_ports.items():
            assert isinstance(rankings, list)
            for src, metric_value in rankings:
                assert isinstance(src, int)
                assert isinstance(metric_value, (int, float))

            # Check that rankings are sorted (descending)
            if len(rankings) > 1:
                for i in range(len(rankings) - 1):
                    assert rankings[i][1] >= rankings[i+1][1]

    def test_unstable_system_handling(self):
        """Test handling of unstable systems."""
        # Test with stability check enabled
        config = PortConfig(mode=GramianMode.INFINITE_HORIZON, stability_check=True)
        analyzer = PortAnalyzer(config)

        A_blocks, E_blocks = create_unstable_system()

        # Should issue warnings but complete analysis
        with pytest.warns(UserWarning):
            results = analyzer.analyze_ports(A_blocks, E_blocks)

        assert isinstance(results, PortAnalysisResults)
        assert len(results.Wc_port) == len(E_blocks)

    def test_discounted_mode(self):
        """Test discounted mode for unstable systems."""
        config = PortConfig(mode=GramianMode.DISCOUNTED, discount_lambda=0.5)
        analyzer = PortAnalyzer(config)

        A_blocks, E_blocks = create_unstable_system()

        # Should complete without warnings
        results = analyzer.analyze_ports(A_blocks, E_blocks)

        assert isinstance(results, PortAnalysisResults)
        assert len(results.Wc_port) == len(E_blocks)

        # Check that Gramians are computed
        for W in results.Wc_port.values():
            assert not np.allclose(W, 0)

    def test_covariance_weighting(self):
        """Test covariance weighting functionality."""
        config = PortConfig(covariance_weighting=CovarianceWeighting.STATE)
        analyzer = PortAnalyzer(config)

        A_blocks, E_blocks = create_stable_system()

        # Create synthetic covariance matrices
        covariance_matrices = {
            0: np.array([[1.0, 0.1], [0.1, 0.8]]),
            1: np.array([[1.2, 0.05, 0.1], [0.05, 0.9, 0.15], [0.1, 0.15, 1.1]])
        }

        results = analyzer.analyze_ports(A_blocks, E_blocks,
                                       covariance_matrices=covariance_matrices)

        assert isinstance(results, PortAnalysisResults)

        # Weighted results should be different from unweighted
        results_unweighted = PortAnalyzer().analyze_ports(A_blocks, E_blocks)

        for key in results.Wc_port:
            assert not np.allclose(results.Wc_port[key], results_unweighted.Wc_port[key])


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_validate_port_analysis(self):
        """Test port analysis validation."""
        analyzer = PortAnalyzer()
        A_blocks, E_blocks = create_stable_system()
        results = analyzer.analyze_ports(A_blocks, E_blocks)

        validation = validate_port_analysis(results)

        assert 'valid' in validation
        assert 'warnings' in validation
        assert 'errors' in validation
        assert 'summary' in validation

        # Should be valid for well-conditioned stable system
        assert validation['valid'] == True
        assert validation['summary']['num_ports'] == len(E_blocks)
        assert validation['summary']['num_blocks'] == len(A_blocks)

    def test_summarize_port_rankings(self):
        """Test port ranking summarization."""
        analyzer = PortAnalyzer()
        A_blocks, E_blocks = create_stable_system()
        results = analyzer.analyze_ports(A_blocks, E_blocks)

        summary = summarize_port_rankings(results, top_k=2)

        # Check summary structure
        for r in A_blocks:
            assert r in summary
            block_summary = summary[r]
            assert 'total_incoming_ports' in block_summary
            assert 'total_controllability' in block_summary
            assert 'top_ports' in block_summary

            for port_info in block_summary['top_ports']:
                assert 'rank' in port_info
                assert 'source_block' in port_info
                assert 'metric_value' in port_info
                assert 'relative_contribution' in port_info


def test_integration_with_ctrnn():
    """Test integration with existing CTRNN analysis."""
    # Create a mock CTRNN result-like object
    class MockCTRNNResults:
        def __init__(self):
            A_blocks, E_blocks = create_stable_system()
            self.A_blocks = A_blocks
            self.E_blocks = E_blocks

    mock_results = MockCTRNNResults()

    # Test convenience function
    port_results = analyze_ctrnn_ports(mock_results)

    assert isinstance(port_results, PortAnalysisResults)
    assert len(port_results.port_map) == len(mock_results.E_blocks)
    assert len(port_results.Wc_total) == len(mock_results.A_blocks)


def test_edge_cases():
    """Test edge cases and error handling."""
    analyzer = PortAnalyzer()

    # Test with no inter-block connections
    A_blocks = {0: np.array([[-1.0]])}
    E_blocks = {}

    results = analyzer.analyze_ports(A_blocks, E_blocks)

    assert len(results.port_map) == 0
    assert len(results.Wc_port) == 0
    assert len(results.Wc_total) == 1
    assert np.allclose(results.Wc_total[0], 0)  # No input ports

    # Test single connection
    A_blocks = {0: np.array([[-1.0]]), 1: np.array([[-2.0]])}
    E_blocks = {(0, 1): np.array([[0.5]])}

    results = analyzer.analyze_ports(A_blocks, E_blocks)

    assert len(results.port_map) == 1
    assert len(results.Wc_port) == 1
    assert (0, 1) in results.Wc_port


if __name__ == "__main__":
    # Run basic tests
    print("Running port analysis tests...")

    # Test configuration
    test_config = TestPortConfig()
    test_config.test_default_config()
    test_config.test_custom_config()
    print("✓ Configuration tests passed")

    # Test analyzer
    test_analyzer = TestPortAnalyzer()
    test_analyzer.test_init()
    test_analyzer.test_build_state_ports()
    test_analyzer.test_stability_check()
    print("✓ Basic analyzer tests passed")

    # Test Gramian computation
    test_analyzer.test_infinite_horizon_gramian()
    test_analyzer.test_discounted_gramian()
    print("✓ Gramian computation tests passed")

    # Test full analysis
    test_analyzer.test_basic_port_analysis()
    test_analyzer.test_gramian_additivity()
    test_analyzer.test_port_metrics()
    test_analyzer.test_port_rankings()
    print("✓ Full analysis tests passed")

    # Test convenience functions
    test_convenience = TestConvenienceFunctions()
    test_convenience.test_validate_port_analysis()
    test_convenience.test_summarize_port_rankings()
    print("✓ Convenience function tests passed")

    # Test edge cases
    test_edge_cases()
    print("✓ Edge case tests passed")

    print("All port analysis tests completed successfully!")