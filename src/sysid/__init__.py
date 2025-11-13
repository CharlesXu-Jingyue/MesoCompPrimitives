"""
System Identification package for MesoCompPrimitives.

This package provides tools for identifying and analyzing dynamical systems,
including continuous-time recurrent neural networks (CTRNNs), linearization
techniques, and inter-block port controllability analysis.
"""

from .ctrnn import CTRNNAnalyzer, FixedPointAnalysis
from .ports import (PortAnalyzer, PortAnalysisResults, PortConfig, PortMetrics,
                   PortModes, GramianMode, CovarianceWeighting,
                   analyze_ctrnn_ports, validate_port_analysis,
                   summarize_port_rankings)

__all__ = [
    'CTRNNAnalyzer', 'FixedPointAnalysis',
    'PortAnalyzer', 'PortAnalysisResults', 'PortConfig', 'PortMetrics', 'PortModes',
    'GramianMode', 'CovarianceWeighting',
    'analyze_ctrnn_ports', 'validate_port_analysis', 'summarize_port_rankings'
]