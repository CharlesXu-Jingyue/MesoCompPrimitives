# Bi-orthogonal Embedding

> Implemented in the `biort` module (class `BiorthEmbedding`); "bi-orthogonal embedding" is the method's descriptive name.

A Python implementation of spectral coarse-graining for **directed weighted networks**, based on the bi-orthogonal decomposition of the random-walk Laplacian. Each node is embedded jointly in its forward (right-eigenvector) and backward (left-eigenvector) propagation modes — the *bi-orthogonal embedding* — and grouped by dynamical role.

## What it's for

Neural circuits — like the EPG ring-attractor in the *Drosophila* central complex — can be directed: synaptic weight matrices are not symmetric. Standard spectral graph methods (e.g., symmetric Laplacian eigenmaps) assume undirected graphs and break down when applied to directed connectomes.

The bi-orthogonal embedding addresses this by working directly with the **non-symmetric random-walk Laplacian** of a directed graph. It extracts the $k$ "slowest" dynamical modes (those that decay most slowly under diffusion), embeds each neuron jointly in the space of *forward* and *backward* propagation modes, groups neurons by their dynamical role, and produces a small coarse network that preserves the low-frequency spectral properties of the original.

The output is a hierarchy of mesoscale groups — a compressed wiring diagram — where each group corresponds to a coherent dynamical unit.

---

## Intuition

### Directed diffusion and the random-walk Laplacian

Think of the connectivity matrix $A$ as defining a random walk: a signal at neuron $i$ spreads to neuron $j$ with probability proportional to the synapse weight $A_{ij}$. Normalizing row-wise gives the **transition matrix**

$$P_{ij} = \frac{A_{ij}}{\sum_j A_{ij}}$$

The **random-walk Laplacian** $L = I - P$ measures how much a node's value differs from the average of its downstream neighbors. Eigenmodes of $L$ with small eigenvalues (near zero) change slowly under iteration — they represent the long-lived, large-scale patterns of activity.

For symmetric $A$, $L$ is symmetric and its eigenvectors form an orthonormal basis. For directed $A$, $L$ is not symmetric: its left and right eigenvectors are **different functions** of the network, encoding backward (who influences me?) and forward (who do I influence?) dynamics respectively.

### Why bi-orthogonality matters

For a non-symmetric operator $L$, the right eigenvectors $\{v_i\}$ (satisfying $L v_i = \lambda_i v_i$) describe *modes of forward propagation*, while the left eigenvectors $\{u_i\}$ (satisfying $u_i^T L = \lambda_i u_i^T$) describe *observation modes* — the dual basis needed to decompose any initial state. Together they satisfy the bi-orthogonality relation

$$u_i^H v_j = \delta_{ij}$$

This is the directed generalization of the familiar identity $v_i^T v_j = \delta_{ij}$ for symmetric matrices. Without tracking both $U$ and $V$, projections onto the slow subspace are not well-defined.

### Coarse-graining as Markov lumping

Once neurons are grouped into $m$ clusters (indexed by cluster membership matrix $C \in \{0,1\}^{n \times m}$), the coarse transition matrix is defined by **Markov lumping** weighted by the stationary distribution $\pi$:

$$P_\text{group} = \underbrace{(C^T \Pi C)^{-1} C^T \Pi}_{R_C} \; P \; C$$

where $\Pi = \text{diag}(\pi)$. The operator $R_C$ is the $\pi$-weighted restriction (coarse-graining map) and $C$ is the prolongation (interpolation). This construction ensures that the coarse walk is itself a valid Markov chain.

---

## Mathematics

### Step 1: Transition matrix with teleportation

Given adjacency matrix $A \in \mathbb{R}^{n \times n}_{\geq 0}$, handle dangling nodes (zero out-degree) by adding self-loops, then form

$$P = \alpha D_\text{out}^{-1} A + \frac{1-\alpha}{n} \mathbf{1}\mathbf{1}^T$$

where $D_\text{out} = \text{diag}(A\mathbf{1})$ and $\alpha \in (0,1]$ is the teleportation parameter (default $\alpha = 0.95$). The teleportation term ensures $P$ is irreducible and aperiodic, guaranteeing a unique stationary distribution.

### Step 2: Random-walk Laplacian

$$L = I - P$$

Eigenvalues of $L$ lie in $[0, 2]$ (or on the complex plane for directed graphs). The zero eigenvalue corresponds to the stationary distribution; small eigenvalues correspond to slow relaxation modes.

### Step 3: Bi-orthogonal eigendecomposition

Compute the $k$ eigenpairs with smallest-magnitude eigenvalues:

$$L V_k = V_k \Lambda_k, \qquad L^T U_k = U_k \Lambda_k$$

where $V_k, U_k \in \mathbb{C}^{n \times k}$ and $\Lambda_k = \text{diag}(\lambda_1, \ldots, \lambda_k)$.

Enforce bi-orthogonality by applying the correction $U_k \leftarrow U_k (U_k^H V_k)^{-H}$, so that $U_k^H V_k = I_k$.

For real networks $A$, complex eigenvalues come in conjugate pairs $\lambda, \bar\lambda$. The optional `realify` step converts each conjugate pair $\{v, \bar v\}$ into the real-valued pair $\{\text{Re}(v), \text{Im}(v)\}$, yielding a real embedding.

### Step 4: Bi-embedding

Each node $i$ receives a $2k$-dimensional coordinate vector by stacking its right (forward) and left (backward) mode coefficients:

$$\mathbf{x}_i = \bigl[\underbrace{\text{Re}(V_k)_{i,:}}_{\text{forward modes}},\; \underbrace{\text{Re}(U_k)_{i,:}}_{\text{backward modes}}\bigr] \in \mathbb{R}^{2k}$$

Nodes that propagate signals similarly and receive signals similarly will cluster together in this space.

### Step 5: Group discovery

Apply $k$-means clustering in the $2k$-dimensional bi-embedding to assign each node to one of $m = k$ groups, producing a hard membership matrix $C \in \{0,1\}^{n \times m}$.

### Step 6: Markov lumping (coarse operators)

Compute the stationary distribution $\pi$ satisfying $\pi^T P = \pi^T$, $\pi_i \geq 0$, $\sum_i \pi_i = 1$.

Define the $\pi$-weighted coarse-graining and prolongation operators:

$$R_C = (C^T \Pi C)^{-1} C^T \Pi, \qquad P_C = C$$

Then:

$$P_\text{group} = R_C \, P \, C \in \mathbb{R}^{m \times m}$$
$$L_\text{group} = I_m - P_\text{group}$$
$$A_\text{group} = D_\text{out,group} \, P_\text{group}, \qquad D_\text{out,group} = \text{diag}(C^T d_\text{out})$$

$P_\text{group}$ is again row-stochastic and can be interpreted as the transition matrix of a random walk on the $m$ coarse groups.

### Step 7: Bi-Galerkin projection

An alternative (spectral) coarse-graining projects the operator directly onto the slow subspace:

$$P_\text{Galerkin} = U_k^H P V_k \in \mathbb{C}^{k \times k}$$
$$L_\text{Galerkin} = U_k^H L V_k = \Lambda_k \quad \text{(diagonal)}$$

By bi-orthogonality, $L_\text{Galerkin}$ is exactly diagonal with the $k$ slow eigenvalues on the diagonal — a verification that the mode computation is correct. $P_\text{Galerkin}$ is the reduced-order model of the transition dynamics in the slow subspace.

### Step 8: Spectral fidelity

The quality of the lumping is measured by how well the slow modes are preserved after restriction and prolongation:

$$\text{fidelity} = \frac{\|V_k - C V_k^\text{group}\|_F}{\|V_k\|_F}$$

where $V_k^\text{group}$ are the eigenvectors of $P_\text{group}$. A value of 0 means perfect preservation; smaller is better.

---

## Installation

```python
import sys
sys.path.append('path/to/MesoCompPrimitives/src')

from biort import BiorthEmbedding, HierarchicalBiorthEmbedding
```

---

## Quick Start

```python
import numpy as np
from biort import BiorthEmbedding

# A: weighted directed adjacency matrix (n x n), non-negative
A = ...  # e.g., synaptic weight matrix

# Fit with k=8 slow modes
biort = BiorthEmbedding(k=8, alpha=0.95, cluster_method='kmeans',
              realify=False, spectral_matrix='L', random_state=42)
biort.fit(A)

# Coarse network
coarse = biort.get_coarse_graph()
print(f"Compressed {coarse['n_original']} → {coarse['n_coarse']} groups")
print(f"Spectral fidelity: {coarse['fidelity']:.4f}")

# Group labels for each neuron
labels = coarse['groups']           # shape (n,)
P_group = coarse['P_group']         # (m, m) coarse transition matrix
L_galerkin = coarse['L_galerkin']   # (k, k) — should be ~diagonal

# Bi-embedding for visualization
X = biort.get_embedding()           # (n, 2k)

# Left/right modes
U_k, V_k, Lambda_k = biort.get_modes()
```

---

## API Reference

### `BiorthEmbedding`

| Parameter | Default | Description |
|---|---|---|
| `k` | 5 | Number of slow modes to retain |
| `alpha` | 0.95 | Teleportation parameter (1 = no teleportation) |
| `cluster_method` | `'kmeans'` | `'kmeans'` (hard) or `'soft'` (distance-weighted) |
| `realify` | `True` | Convert complex conjugate mode pairs to real representation |
| `spectral_matrix` | `'L'` | `'L'` (Laplacian) or `'P'` (transition matrix) for eigendecomposition |
| `random_state` | `None` | Reproducibility seed |

**Methods:**

- `fit(A, L=None)` — fit model; optionally supply a custom Laplacian in place of $I - P$
- `get_coarse_graph()` — returns dict with `P_group`, `L_group`, `A_group`, `P_galerkin`, `L_galerkin`, `A_galerkin`, `groups`, `fidelity`, etc.
- `get_embedding()` — returns bi-embedding $X \in \mathbb{R}^{n \times 2k}$
- `get_modes()` — returns `(U_k, V_k, Lambda_k)`
- `get_bi_galerkin_operators()` — returns dict with Galerkin-projected operators
- `transform(steps)` — returns $P_\text{group}^t$ (coarse dynamics after `steps` steps)

**Fitted attributes:**

| Attribute | Shape | Description |
|---|---|---|
| `P_` | $(n,n)$ | Teleportation-augmented transition matrix |
| `L_` | $(n,n)$ | Random-walk Laplacian used for spectral decomp |
| `U_k_` | $(n,k)$ | Left eigenvectors (backward modes) |
| `V_k_` | $(n,k)$ | Right eigenvectors (forward modes) |
| `Lambda_k_` | $(k,)$ | Slow eigenvalues |
| `X_` | $(n,2k)$ | Bi-embedding |
| `C_` | $(n,m)$ | Membership matrix |
| `groups_` | $(n,)$ | Integer group labels |
| `pi_` | $(n,)$ | Stationary distribution |
| `P_group_` | $(m,m)$ | Coarse transition matrix |
| `L_galerkin_` | $(k,k)$ | Bi-Galerkin Laplacian (should be $\approx\Lambda_k$) |

---

### `HierarchicalBiorthEmbedding`

Applies the bi-orthogonal embedding recursively: the coarse adjacency $A_\text{group}$ at each level becomes the input to the next. Stops when nodes $\leq$ `min_nodes` or fidelity exceeds `fidelity_threshold`.

```python
from biort import HierarchicalBiorthEmbedding

hbiort = HierarchicalBiorthEmbedding(k=5, max_levels=3, min_nodes=10, fidelity_threshold=0.5)
hbiort.fit(A)

# Access each level
level0 = hbiort.get_level(0)   # BiorthEmbedding instance at level 0
hierarchy = hbiort.get_hierarchy()  # list of coarse_graph dicts

# Project a node-level vector to coarse level 1
x_coarse = hbiort.project_to_level(x, target_level=1)
```

---

## Usage Notes

- **`realify=False`** keeps complex modes, which is useful for inspecting the eigenspectrum on the complex plane (oscillatory vs. purely decaying modes). Set to `True` for real-valued clustering and visualization.
- **Custom Laplacian**: pass `biort.fit(A, L=L_bal)` to use a balanced or symmetrized Laplacian instead of $I - P$. The spectral analysis uses your $L$ but the transition matrix $P$ (built from $A$) is still used for Markov lumping.
- **Verification**: after fitting, check `U_k^H @ biort.L_ @ V_k` — it should be diagonal with `Lambda_k` on the diagonal. Max off-diagonal magnitude $< 10^{-10}$ confirms correct bi-orthogonal decomposition.
- **Fidelity interpretation**: values $< 0.3$ are generally good; values $> 0.7$ suggest the chosen $k$ is too small to capture the dominant structure.

---

## Connection to the Broader Pipeline

The bi-orthogonal embedding is the **partition and role-discovery** step of MesoCompPrimitives. Its outputs feed directly into:

- **DC-SBM** (`src/dcsbm`): group labels from the bi-orthogonal embedding can initialize block assignments for degree-corrected stochastic block model fitting, combining spectral and generative model perspectives.
- **System ID / CTRNN Analysis** (`src/sysid`): the coarse groups define the blocks $C_k$ for which per-block fixed points are computed and local linearizations $A_k^\text{red}$, $B_k^\text{red}$ are extracted.
- **Port Analysis**: the bi-embedding's left modes indicate which neurons are strongly driven by inputs (high $|U_k|$), informing port definitions for controllability analysis.
