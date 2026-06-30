"""
Ring connectome preprocessing for the bi-orthogonal embedding exploration.

Task-specific, exploratory helper (not a general-purpose ``src`` package). It
distills the essential preprocessing of ``poc_hemibrain_epg_working.ipynb`` —
load the signed connectivity matrix and metadata, group by cell type, drop
unconnected nodes, select cell types, and (optionally) reorder nodes — into a
single ``load_ring_connectome`` call that returns the unsigned adjacency ``A``
together with labels aligned to it. The downstream embedding operator is built
internally by ``biort.BiorthEmbedding``, so the Laplacian/diffusion diagnostics
from the notebook are intentionally omitted here.

Dataset: Drosophila hemibrain central-complex "ring" connectome
(Delta7 + EPG), stored as CSVs under ``~/local/data/mcp/hemibrain/ring``.
"""

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_DATA_PATH = "~/local/data/mcp/hemibrain/ring"

# Trailing protocerebral-bridge wedge label, e.g. the "L3" in "EPG(PB08)_L3".
_WEDGE_RE = re.compile(r"^([LR])(\d+)$")


def spectral_seriation(A: np.ndarray, symmetrize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Order nodes by the Fiedler vector of the (symmetrized) graph Laplacian.

    Copied from the working notebook so the ordering matches exactly.

    Parameters
    ----------
    A : ndarray, shape (N, N)
        Square adjacency/weight matrix.
    symmetrize : bool
        If True, build the Laplacian from W = (|A| + |A.T|) / 2.

    Returns
    -------
    perm : ndarray, shape (N,)
        Permutation such that ``A[perm][:, perm]`` is the reordered matrix.
    A_reordered : ndarray, shape (N, N)
    """
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D array")

    n = A.shape[0]
    if n <= 1:
        return np.arange(n), A.copy()

    W = 0.5 * (np.abs(A) + np.abs(A.T)) if symmetrize else np.asarray(A, dtype=float)
    L = np.diag(W.sum(axis=1)) - W
    _, evecs = np.linalg.eigh(L)
    fiedler = evecs[:, 1] if n > 1 else evecs[:, 0]
    perm = np.argsort(fiedler)
    return perm, A[np.ix_(perm, perm)]


def _wedge_label(cioi_entry: str) -> Optional[str]:
    """Return the trailing wedge token (e.g. 'L3') of a cell-instance label.

    EPG instances look like ``EPG(PB08)_L3``; the token after the first
    underscore is a clean ``L#``/``R#`` wedge. Delta7 instances such as
    ``Delta7(PB15)_L6R3_L`` do not match and yield ``None``.
    """
    parts = str(cioi_entry).split("_")
    if len(parts) < 2:
        return None
    token = parts[1]
    return token if _WEDGE_RE.match(token) else None


def _ring_layout(n_compartments: int = 8) -> List[str]:
    """Interleaved ring layout L1, R8, L2, R7, ..., L8, R1 (notebook cell 30)."""
    order = []
    for i in range(1, n_compartments + 1):
        order.append(f"L{i}")
        order.append(f"R{n_compartments + 1 - i}")
    return order


def epg_ring_position(cioi: Sequence[str], n_compartments: int = 8) -> np.ndarray:
    """Ring index (position in the interleaved L/R layout) for each node.

    EPG nodes map to an integer position in ``[0, 2*n_compartments)``;
    non-matching labels (e.g. Delta7) get ``np.nan``.
    """
    rank = {label: idx for idx, label in enumerate(_ring_layout(n_compartments))}
    pos = np.full(len(cioi), np.nan, dtype=float)
    for i, entry in enumerate(cioi):
        label = _wedge_label(entry)
        if label is not None and label in rank:
            pos[i] = rank[label]
    return pos


def spatial_ring_order(cioi: Sequence[str], n_compartments: int = 8) -> np.ndarray:
    """Permutation that lays nodes out along the EPG ring (notebook cell 30).

    Nodes whose wedge label is unknown (Delta7) are pushed to the end while
    preserving their relative order (stable sort).
    """
    rank = {label: idx for idx, label in enumerate(_ring_layout(n_compartments))}
    keys = [rank.get(_wedge_label(entry), np.inf) for entry in cioi]
    return np.array(sorted(range(len(cioi)), key=lambda i: (keys[i], i)))


@dataclass
class RingConnectome:
    """Preprocessed ring connectome with labels aligned to ``A``."""

    A: np.ndarray                      # unsigned, reordered adjacency (n, n)
    C_signed: np.ndarray               # signed, reordered adjacency (n, n)
    ctoi: List[str]                    # cell type per node
    noi: list                          # neuron id per node
    cioi: List[str]                    # cell instance per node
    unique_cell_types: List[str]
    cell_type_counts: dict
    ring_pos: np.ndarray               # EPG ring index, NaN otherwise (n,)
    perm: np.ndarray                   # reorder applied to the selected nodes
    n: int = field(init=False)

    def __post_init__(self):
        self.n = self.A.shape[0]


def _read_signed_matrix(data_path: str) -> np.ndarray:
    df = pd.read_csv(os.path.join(data_path, "connectivity_matrix_roi.csv"))
    # Drop a leading row-label column if present (square + 1 columns).
    if df.shape[1] == df.shape[0] + 1:
        values = df.iloc[:, 1:].values
    else:
        values = df.values
    return np.asarray(values, dtype=float)


def load_ring_connectome(
    select_cell_types: Sequence[str] = ("Delta7", "EPG"),
    data_path: str = DEFAULT_DATA_PATH,
    reorder: Optional[str] = "seriation",
    drop_unconnected: bool = True,
) -> RingConnectome:
    """Load and preprocess the ring connectome into a ``RingConnectome``.

    Pipeline (mirrors the working notebook): load signed matrix + metadata →
    group by cell type (alphabetical) → ``C_unsigned = |C_signed|`` → drop
    zero-degree nodes → select cell types → reorder.

    Parameters
    ----------
    select_cell_types : sequence of str
        Cell types to keep (e.g. ``['EPG']`` or ``['Delta7', 'EPG']``).
    data_path : str
        Directory holding ``connectivity_matrix_roi.csv`` and the
        ``ctoi_list.csv`` / ``noi_list.csv`` / ``cioi_list.csv`` metadata.
    reorder : {'seriation', 'spatial', None}
        Node ordering of the returned matrices. The embedding itself is
        permutation-equivariant; ordering only affects visualization and the
        ring coordinate. ``'spatial'`` uses the EPG L/R-wedge ring layout.
    drop_unconnected : bool
        Remove nodes with zero in- and out-degree before selection.
    """
    data_path = os.path.expanduser(data_path)

    C_signed = _read_signed_matrix(data_path)
    ctoi = pd.read_csv(os.path.join(data_path, "ctoi_list.csv"))["Cell Type"].tolist()
    noi = pd.read_csv(os.path.join(data_path, "noi_list.csv"))["Neuron ID"].tolist()
    cioi = pd.read_csv(os.path.join(data_path, "cioi_list.csv"))["Cell Instance"].tolist()

    if not (C_signed.shape[0] == C_signed.shape[1] == len(ctoi) == len(noi) == len(cioi)):
        raise ValueError(
            f"Size mismatch: matrix {C_signed.shape}, "
            f"ctoi {len(ctoi)}, noi {len(noi)}, cioi {len(cioi)}"
        )

    # Group by cell type (alphabetical), reordering matrix and all label lists.
    grp_order: List[int] = []
    for ct in sorted(set(ctoi)):
        grp_order.extend(i for i, c in enumerate(ctoi) if c == ct)
    C_signed, ctoi, noi, cioi = _apply_index(C_signed, ctoi, noi, cioi, grp_order)
    C_unsigned = np.abs(C_signed)

    # Drop nodes with zero in- and out-degree.
    if drop_unconnected:
        deg = C_unsigned.sum(axis=1) + C_unsigned.sum(axis=0)
        keep = np.where(deg > 0)[0]
        C_signed, ctoi, noi, cioi = _apply_index(C_signed, ctoi, noi, cioi, keep)
        C_unsigned = np.abs(C_signed)

    # Select requested cell types.
    available = set(ctoi)
    missing = [ct for ct in select_cell_types if ct not in available]
    if missing:
        raise ValueError(f"Requested cell types not in data: {missing}. Available: {sorted(available)}")
    sel = [i for i, c in enumerate(ctoi) if c in select_cell_types]
    C_signed, ctoi, noi, cioi = _apply_index(C_signed, ctoi, noi, cioi, sel)
    C_unsigned = np.abs(C_signed)

    A = C_unsigned

    # Reorder nodes.
    if reorder == "seriation":
        perm, _ = spectral_seriation(A)
    elif reorder == "spatial":
        perm = spatial_ring_order(cioi)
    elif reorder is None:
        perm = np.arange(A.shape[0])
    else:
        raise ValueError(f"Unknown reorder mode: {reorder!r}")
    A, ctoi, noi, cioi = _apply_index(A, ctoi, noi, cioi, perm)
    C_signed = C_signed[np.ix_(perm, perm)]

    ring_pos = epg_ring_position(cioi)
    unique_cell_types = sorted(set(ctoi))
    cell_type_counts = dict(Counter(ctoi))

    return RingConnectome(
        A=A,
        C_signed=C_signed,
        ctoi=ctoi,
        noi=noi,
        cioi=cioi,
        unique_cell_types=unique_cell_types,
        cell_type_counts=cell_type_counts,
        ring_pos=ring_pos,
        perm=np.asarray(perm),
    )


def _apply_index(C, ctoi, noi, cioi, idx):
    """Apply an index/permutation to the matrix (both axes) and label lists."""
    idx = list(idx)
    C = C[np.ix_(idx, idx)]
    ctoi = [ctoi[i] for i in idx]
    noi = [noi[i] for i in idx]
    cioi = [cioi[i] for i in idx]
    return C, ctoi, noi, cioi
