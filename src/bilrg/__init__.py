"""
Bi-orthogonal Laplacian Renormalization Group (bi-LRG) package.

This module implements hierarchical coarse-graining of directed weighted networks
using spectral analysis and mesoscale grouping.
"""

from .bilrg import BiLRG, HierarchicalBiLRG
from .utils_bilrg import (
    compute_stationary_distribution, biorthogonal_modes, realify_modes,
    validate_transition_matrix, create_teleportation_matrix, spectral_fidelity,
    safe_divide, is_sparse, to_dense
)

__all__ = [
    'BiLRG', 'HierarchicalBiLRG',
    'compute_stationary_distribution', 'biorthogonal_modes', 'realify_modes',
    'validate_transition_matrix', 'create_teleportation_matrix', 'spectral_fidelity',
    'safe_divide', 'is_sparse', 'to_dense'
]