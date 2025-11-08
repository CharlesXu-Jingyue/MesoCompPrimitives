"""
MesoCompPrimitives: Mesoscale computational primitives analysis.

This package provides tools for analyzing neural connectivity and extracting
computational primitives using various methods including DC-SBM and bi-LRG.
"""

from .dcsbm import DCSBM, spectral_init, to_edge_list, degrees, heldout_split
from .bilrg import BiLRG, HierarchicalBiLRG

__version__ = "0.1.0"

__all__ = [
    'DCSBM', 'spectral_init', 'to_edge_list', 'degrees', 'heldout_split',
    'BiLRG', 'HierarchicalBiLRG'
]