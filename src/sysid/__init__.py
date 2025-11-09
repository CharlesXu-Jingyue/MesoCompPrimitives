"""
System Identification package for MesoCompPrimitives.

This package provides tools for identifying and analyzing dynamical systems,
including continuous-time recurrent neural networks (CTRNNs) and linearization
techniques.
"""

from .ctrnn import CTRNNAnalyzer, FixedPointAnalysis

__all__ = ['CTRNNAnalyzer', 'FixedPointAnalysis']