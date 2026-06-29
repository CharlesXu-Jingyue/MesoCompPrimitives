# sysid: System Identification for Block-Structured CTRNNs

A Python implementation of fixed-point, linearization, and controllability analysis for **continuous-time recurrent neural networks (CTRNNs)** whose units have been partitioned into blocks (e.g., the groups discovered by `biort` or `dcsbm`).

## What it's for

Once a connectome has been partitioned into mesoscale groups, we want to know what each group *computes* and how groups drive one another. `sysid` treats the signed weight matrix as a rate-based CTRNN, finds the operating point (fixed point) of each block, linearizes the dynamics around it, and characterizes the resulting block-structured linear system with control-theoretic tools.

It answers two questions:

1. **Local dynamics** (`ctrnn`): around its fixed point, what is each block's linear "skeleton" $A_{rr}$ — its eigenvalues, time scales, and reduced (balanced) order?
2. **Inter-block control** (`ports`): how strongly, and in how many independent directions, does each block *drive* its neighbors? Which incoming connections dominate a block's controllable subspace?

Together these turn a partitioned connectome into a set of local linear models plus a quantified inter-block coupling structure — the input to canonical-primitive labeling.

---

## Intuition

### The CTRNN and its fixed points

We interpret the (normalized) weight matrix $\tilde W$ as a rate-based recurrent network with sigmoid activation:

$$\tau_r \, \dot{x}_i = -x_i + \sum_j \tilde{W}_{ij}\,\sigma(x_j) + b_i, \qquad \sigma(x) = \frac{1}{1+e^{-x}}$$

A steady state satisfies $x = \tilde W \sigma(x) + b$. Rather than solving the full $N$-dimensional system at once, we solve it **block by block**: for block $r$ we treat only the within-block recurrence $\tilde W_{rr}$ and find $x_r^* = \tilde W_{rr}\,\sigma(x_r^*) + b_r$. This is the local operating point each group settles into in isolation.

### Why row-sum normalization guarantees a fixed point

The fixed-point map $T(x) = \tilde W_{rr}\,\sigma(x) + b_r$ is a contraction whenever its Lipschitz constant is below 1. Since $\max_x |\sigma'(x)| = \tfrac14$ and the induced $\infty$-norm of a product is bounded by the product of norms,

$$\mathrm{Lip}(T) \le \|\tilde W_{rr}\|_\infty \cdot \tfrac14.$$

Scaling the whole matrix by $\alpha = 4c / \|W\|_\infty$ (so $\tilde W = \alpha W$) forces $\|\tilde W\|_\infty = 4c$, hence $\mathrm{Lip}(T) \le c < 1$. The **safety margin** $c$ (default 0.9) thus guarantees a unique fixed point reachable by simple iteration, with contraction factor $c$.

### Linearization and the port factorization

Linearizing the CTRNN about the fixed points yields a block-structured Jacobian. The diagonal blocks are each block's intrinsic dynamics; the off-diagonal blocks are inter-block coupling:

$$A_{rr} = \tfrac{1}{\tau_r}\bigl(-I + \tilde W_{rr}\Gamma_r\bigr), \qquad E_{rs} = \tfrac{1}{\tau_r}\,\tilde W_{rs}\,\Gamma_s,$$

where $\Gamma_r = \mathrm{diag}(\sigma'(x_r^*))$ are the local **sigmoid gains**. The key observation for control analysis is that the coupling factors as

$$E_{rs} = B_{rs}\,C_{ss}, \qquad B_{rs} = \tfrac{1}{\tau_r}\tilde W_{rs}, \quad C_{ss} = \Gamma_s.$$

This lets us read block $s$'s state as an **input** entering block $r$ through a "port" $B_{rs}$, with $s$'s gain $C_{ss}$ acting as its output map. The dynamics of block $r$ become

$$\delta\dot{x}_r = A_{rr}\,\delta x_r + \sum_{s\neq r} B_{rs}\,\delta x_s,$$

a local linear system driven by its incoming ports — exactly the form control theory needs.

### Controllability as drive

For a stable block $A_{rr}$ driven through port $B_{rs}$, the **controllability Gramian** $W_c$ measures how much, and in which directions, the input can move the block's state. Its trace is total drive energy; its eigenvectors are the controllable directions; its numerical rank is how many independent directions the port actually excites. Because the Gramian is linear in $BB^T$ and all incoming ports share the same $A_{rr}$, the per-port Gramians **add up** to the block's total controllability — so we can attribute a block's controllable subspace to individual source blocks and rank them.

---

## Mathematics

### Step 1 — Global row-sum normalization

Given a signed weight matrix $W \in \mathbb{R}^{N\times N}$, compute the induced infinity norm $s_\infty = \|W\|_\infty = \max_i \sum_j |W_{ij}|$ and rescale:

$$\alpha = \frac{4c}{s_\infty}, \qquad \tilde W = \alpha W,$$

so the iteration map is a contraction with factor $c$ (the `safety_margin`). If $s_\infty = 0$ the matrix is returned unchanged.

### Step 2 — Per-block fixed points (damped Picard)

For each block $r$ with index set $I_r$, extract $\tilde W_{rr} = \tilde W[I_r, I_r]$ and $b_r = b[I_r]$, then iterate from $x_r^{(0)} = b_r$:

$$x_r^{(t+1)} = (1-\eta)\,x_r^{(t)} + \eta\bigl(\tilde W_{rr}\,\sigma(x_r^{(t)}) + b_r\bigr),$$

with damping $\eta$ (default 0.5). Iteration stops when $\|x_r^{(t+1)} - x_r^{(t)}\|_\infty < \text{tol}$. A global fixed point over all $N$ units (single "block") is computed the same way for whole-system linearization. Convergence (status, iteration count, error history) is recorded per block.

### Step 3 — Sigmoid gains

$$\Gamma_r = \mathrm{diag}\bigl(\sigma'(x_r^*)\bigr), \qquad \sigma'(x) = \sigma(x)\,(1-\sigma(x)).$$

### Step 4 — Block-structured linearization

$$A_{rr} = -\tfrac{1}{\tau_r} I + \tfrac{1}{\tau_r}\tilde W_{rr}\Gamma_r, \qquad
C_{rr} = \Gamma_r,$$
$$B_{rs} = \tfrac{1}{\tau_r}\tilde W_{rs}, \qquad
E_{rs} = B_{rs}\,\Gamma_s = \tfrac{1}{\tau_r}\tilde W_{rs}\Gamma_s \quad (s \neq r),$$
$$B^{\text{lin}}_r = \tfrac{1}{\tau_r} B_r \quad (\text{if external input weights supplied}).$$

The full Jacobian $A_{\text{global}} \in \mathbb{R}^{N\times N}$ is assembled with $A_{rr}$ on the diagonal blocks and $E_{rs}$ on the off-diagonal blocks. (In the single-block case $C$, $B$, $E$ are returned as zeros — there is no inter-block coupling to factor.)

### Step 5 — Optional local analyses

- **Eigenvalues** of each $A_{rr}$ (time scales, oscillation).
- **Schur decomposition** $A_{rr} = Q T Q^H$ (ordered triangularization).
- **Balanced truncation** (only when input weights are given and the block is stable): solve the Lyapunov equations
  $$A_{rr} W_c + W_c A_{rr}^T + B_r B_r^T = 0, \qquad A_{rr}^T W_o + W_o A_{rr} + C_r^T C_r = 0 \;(C_r = I),$$
  and report the **Hankel singular values** $\sigma_i = \sqrt{\lambda_i(W_c W_o)}$ in descending order. Large $\sigma_i$ mark the modes that carry the block's input–output behavior.

### Step 6 — Global stability

$A_{\text{global}}$ is stable iff $\max_i \mathrm{Re}(\lambda_i) < 0$; the **stability margin** is $-\max_i \mathrm{Re}(\lambda_i)$, and the number of non-negative-real eigenvalues counts unstable modes.

### Inter-block port controllability (`ports`)

Given the block matrices from Step 4, define each state-port as $B_{rs} = E_{rs}$ and, for each, solve a Lyapunov equation for the per-port **controllability Gramian** (mode chosen by config):

| Mode | Equation | Use |
|---|---|---|
| Infinite-horizon | $A_{rr} W + W A_{rr}^T + \hat B_{rs}\hat B_{rs}^T = 0$ | stable blocks |
| Discounted | $(A_{rr}+\lambda I) W + W (A_{rr}+\lambda I)^T + \hat B_{rs}\hat B_{rs}^T = 0$ | unstable blocks (shift by $\lambda$) |
| Finite-horizon | $W(T)=\int_0^T e^{A_{rr}t}\hat B_{rs}\hat B_{rs}^T e^{A_{rr}^T t}\,dt$ via Van Loan | bounded time window |

with optional **covariance weighting** $\hat B_{rs} = B_{rs}\,\Sigma_s^{1/2}$ (using the PSD square root of a supplied state covariance). Observability Gramians are computed analogously from $A_{ss}^T W + W A_{ss} + \hat C^T \hat C = 0$.

The block's **total** controllability is the sum of its incoming ports, $W_c^{(r)} = \sum_s W_c^{(r,s)}$, and `validate_port_analysis` checks this additivity to machine precision.

### Port metrics and modes

For each Gramian $W$ (eigenvalues clamped to $\ge 0$):

| Metric | Definition | Reads as |
|---|---|---|
| `trace` | $\mathrm{tr}(W)$ | total drive energy |
| `lambda_max` / `lambda_min` | $\lambda_{\max}, \lambda_{\min}$ | strongest / weakest direction |
| `logdet` | $\sum_i \log(1 + \alpha\,\lambda_i)$ | regularized controllable volume |
| `rank` | $\#\{\lambda_i > \text{tol}\}$ | independent driven directions |
| `condition_number` | $\lambda_{\max}/\lambda_{\min}^{>\text{tol}}$ | anisotropy of drive |
| `frobenius_norm` | $\|W\|_F$ | overall magnitude |

The leading $k$ eigenvectors are the dominant controllable directions; each carries a **participation ratio** $\mathrm{PR} = (\sum_j v_j^2)^2 / \sum_j v_j^4$ measuring how distributed it is across the block. Incoming ports are then ranked per destination block by the chosen metric.

---

## Installation

```python
import sys
sys.path.append('path/to/MesoCompPrimitives/src')

from sysid import CTRNNAnalyzer, PortAnalyzer, PortConfig, GramianMode
```

---

## Quick Start

```python
import numpy as np
from sysid import CTRNNAnalyzer, PortAnalyzer, PortConfig, GramianMode

# W: signed weight matrix (N x N); block_labels: group id per unit (e.g. from biort)
W = ...                       # (N, N)
block_labels = ...            # (N,) integer labels

# --- Local dynamics: fixed points + blockwise linearization ---
analyzer = CTRNNAnalyzer(safety_margin=0.9, tolerance=1e-6, damping=0.5)
res = analyzer.analyze(
    W, block_labels,
    bias=None,                # defaults to zeros
    time_constants=1.0,       # scalar or (k,) per-block taus
    input_weights=None,       # (N, M) to enable balanced truncation
    perform_optional_analyses=True,
)

print("normalization alpha:", res.normalization_factor)
print("blocks:", list(res.A_blocks))                 # diagonal A_rr per block
print("block 0 eigenvalues:", res.eigenvalues[0])    # time scales of block 0

# Global stability of the assembled Jacobian
stab = analyzer.check_global_stability(res.A_global)
print("stable:", stab['is_stable'], "margin:", stab['stability_margin'])

# --- Inter-block control: per-port controllability ---
cfg = PortConfig(mode=GramianMode.INFINITE_HORIZON, metric='trace', top_k_modes=5)
ports = PortAnalyzer(cfg).analyze_ports(
    res.A_blocks, res.B_blocks, res.C_blocks, res.E_blocks
)

# Strongest incoming drivers of block r, ranked by the chosen metric
for r, ranked in ports.top_ports.items():
    print(f"block {r} driven most by:", ranked[:3])   # [(source_block, metric), ...]

print("block 0 total controllability (trace):", ports.total_metrics[0].trace)
```

For unstable blocks, switch to `GramianMode.DISCOUNTED` (with `discount_lambda`) or `GramianMode.FINITE_HORIZON` (with `horizon_T`).

---

## API Reference

### `CTRNNAnalyzer`

```python
CTRNNAnalyzer(safety_margin=0.9, tolerance=1e-6, damping=0.5, max_iterations=1000)
```

| Parameter | Default | Description |
|---|---|---|
| `safety_margin` | 0.9 | Contraction margin $c \in (0,1)$; sets $\alpha = 4c/\|W\|_\infty$ |
| `tolerance` | 1e-6 | $\infty$-norm convergence tolerance for Picard iteration |
| `damping` | 0.5 | Picard damping $\eta \in (0,1]$ |
| `max_iterations` | 1000 | Iteration cap per block |

**Methods**

- `analyze(W, block_labels, bias=None, time_constants=None, input_weights=None, perform_optional_analyses=True)` — run the full pipeline; returns a `FixedPointAnalysis`.
- `check_global_stability(A_global)` — returns `{eigenvalues, is_stable, max_real_eigenvalue, stability_margin, num_unstable_modes}`.

Module-level helpers `sigmoid(x)` and `sigmoid_derivative(x)` are also exported.

### `FixedPointAnalysis` (dataclass)

| Attribute | Description |
|---|---|
| `W_normalized`, `normalization_factor`, `original_norm`, `safety_margin` | normalization outputs ($\tilde W$, $\alpha$, $s_\infty$, $c$) |
| `fixed_points`, `fixed_points_global` | per-block and whole-system $x_r^*$ |
| `sigmoid_gains` | $\Gamma_r = \mathrm{diag}(\sigma'(x_r^*))$ per block |
| `convergence_info` | per-block `{converged, iterations, final_error, error_history}` |
| `A_blocks`, `B_blocks`, `C_blocks`, `E_blocks` | $A_{rr}$, $B_{rs}$, $C_{rr}$, $E_{rs}$ |
| `A_assembled`, `A_global` | assembled per-partition Jacobian / whole-system Jacobian |
| `B_linear` | $\tfrac1{\tau_r}B_r$ if `input_weights` given |
| `eigenvalues`, `schur_decomp`, `balanced_truncation` | optional analyses (Hankel SVs live in `balanced_truncation[r]['hankel_svs']`) |

### `PortAnalyzer`

```python
PortAnalyzer(config: PortConfig = PortConfig())
```

**Method**

- `analyze_ports(A_blocks, B_blocks, C_blocks, E_blocks, time_constants=None, covariance_matrices=None)` — returns a `PortAnalysisResults`.

### `PortConfig` (dataclass)

| Parameter | Default | Description |
|---|---|---|
| `mode` | `INFINITE_HORIZON` | `GramianMode`: `INFINITE_HORIZON`, `DISCOUNTED`, `FINITE_HORIZON` |
| `discount_lambda` | 0.1 | shift $\lambda$ for discounted mode |
| `horizon_T` | 2.0 | window $T$ for finite-horizon mode |
| `covariance_weighting` | `NONE` | `CovarianceWeighting`: `NONE`, `STATE`, `RATE` |
| `alpha_logdet` | 1e-2 | regularizer in the `logdet` metric |
| `top_k_modes` | 5 | number of leading Gramian modes to keep |
| `metric` | `"trace"` | metric used for ranking ports |
| `stability_check` | True | warn on unstable blocks in infinite-horizon mode |
| `rank_tolerance` | 1e-12 | eigenvalue threshold for numerical rank / PSD checks |

### `PortAnalysisResults` (dataclass)

| Attribute | Description |
|---|---|
| `port_map` | $B_{rs}=E_{rs}$ per port $(r,s)$ |
| `block_sizes` | size of each block |
| `Wc_port`, `Wo_port` | per-port controllability / observability Gramians |
| `Wc_total` | per-block total controllability ($\sum_s W_c^{(r,s)}$) |
| `port_metrics`, `total_metrics` | `PortMetrics` per port / per block |
| `port_modes`, `total_modes` | `PortModes` (top-$k$ eigenpairs + participation ratios) |
| `top_ports` | per destination block, sources ranked `[(source, metric), ...]` |
| `stability_info` | per-block eigenvalues, spectral abscissa, stability flag |
| `config` | the `PortConfig` used |

### Utility functions

- `validate_port_analysis(results)` — checks Gramian symmetry, PSD-ness, and total-Gramian additivity; returns `{valid, warnings, errors, summary}`.
- `summarize_port_rankings(results, top_k=5)` — human-readable per-block ranking with each port's relative contribution and rank percentage.
- `analyze_ctrnn_ports(ctrnn_results, config=None)` — convenience wrapper that constructs a `PortAnalyzer` from a `FixedPointAnalysis`.

---

## Usage Notes

- **Block labels are the bridge.** `block_labels` are typically the group assignments from `biort` (`coarse['groups']`) or `dcsbm` (`model.predict()`), so `sysid` operates on exactly the mesoscale partition discovered upstream.
- **Per-block vs. global.** Per-block fixed points isolate each group's intrinsic operating point; the global fixed point and `A_global` capture the coupled system. Compare the two to see how much inter-block coupling shifts the operating point.
- **Stability gates the Gramians.** Infinite-horizon controllability requires stable blocks ($\mathrm{Re}(\lambda) < 0$). For blocks with non-negative eigenvalues, use `DISCOUNTED` (adds $\lambda I$ to stabilize) or `FINITE_HORIZON` (integrates over $[0,T]$), or the balanced-truncation step will be skipped with a warning.
- **Reading the rankings.** A high-`trace`, high-`rank` incoming port means a source block drives its target strongly and in many independent directions — a candidate "wide" coupling. Low rank but high `lambda_max` indicates drive concentrated in one direction.
- **Covariance weighting** scales each port input by $\Sigma_s^{1/2}$, so controllability reflects the source block's actual fluctuation statistics rather than unit-variance inputs.

---

## Connection to the Broader Pipeline

`sysid` is the **local response modeling** and **control-theoretic reduction** stage of MesoCompPrimitives. It consumes the partition produced upstream and feeds the canonical-labeling stage:

- **From `biort` / `dcsbm`**: group labels define the blocks. Each block's $A_{rr}$ is its local dynamical model.
- **`ctrnn`**: per-block fixed points and linearizations give each primitive's linear skeleton; eigenvalues and Hankel singular values (balanced truncation) provide the reduced order $A_k^{\text{red}}$.
- **`ports`**: per-port controllability quantifies inter-block coupling and identifies the dominant input/output subspaces — the **ports** through which primitives compose.
- **Downstream (role discovery / synthesis)**: the reduced models + port structure are matched against templates (integrator, WTA, gate, ring attractor) and recomposed into a coarse network for spectral- and task-level validation.

---

## References

### Control Theory
- Antoulas (2005) "Approximation of Large-Scale Dynamical Systems" (balanced truncation, Gramians)
- Van Loan (1978) "Computing integrals involving the matrix exponential" (finite-horizon Gramian)
- Moore (1981) "Principal component analysis in linear systems: controllability, observability, and model reduction"

### Recurrent Network Dynamics
- Beer (1995) "On the dynamics of small continuous-time recurrent neural networks"
- Sussillo & Barak (2013) "Opening the black box: low-dimensional dynamics in high-dimensional recurrent neural networks"
