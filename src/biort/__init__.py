"""
Bi-orthogonal Embedding package.

This module implements hierarchical coarse-graining of directed weighted networks
using spectral analysis and mesoscale grouping.
"""

from .biort import BiorthEmbedding, HierarchicalBiorthEmbedding
from .utils_biort import (
    compute_stationary_distribution, biorthogonal_modes, realify_modes,
    validate_transition_matrix, create_teleportation_matrix, spectral_fidelity,
    safe_divide, is_sparse, to_dense
)

__all__ = [
    'BiorthEmbedding', 'HierarchicalBiorthEmbedding',
    'compute_stationary_distribution', 'biorthogonal_modes', 'realify_modes',
    'validate_transition_matrix', 'create_teleportation_matrix', 'spectral_fidelity',
    'safe_divide', 'is_sparse', 'to_dense'
]