"""
Degree-Corrected Stochastic Block Model (DC-SBM) implementation.

This package provides a clean, efficient implementation of weighted, directed
DC-SBM using variational EM with spectral initialization.
"""

from .dcsbm import DCSBM, spectral_init, to_edge_list, degrees, heldout_split

__version__ = "0.1.0"
__all__ = ["DCSBM", "spectral_init", "to_edge_list", "degrees", "heldout_split"]