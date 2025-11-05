# DC-SBM: Degree-Corrected Stochastic Block Model

A clean, efficient Python implementation of weighted, directed Degree-Corrected Stochastic Block Model (DC-SBM) using variational EM with spectral initialization.

## Features

- **Weighted, directed networks**: Handles Poisson-distributed edge weights
- **Degree correction**: Accounts for heterogeneous node degrees within blocks
- **Variational EM**: Efficient parameter estimation with monotonic ELBO convergence
- **Spectral initialization**: Robust initialization using degree-regularized SVD
- **Sparse matrix support**: Optimized for large, sparse networks
- **Cross-validation**: Built-in train/validation splitting with stratification

## Installation

```bash
# Add the src directory to your Python path
import sys
sys.path.append('path/to/MesoCompPrimitives/src')

from dcsbm import DCSBM, heldout_split
```

## Quick Start

```python
from dcsbm import DCSBM, heldout_split
from scipy.sparse import csr_matrix
import numpy as np

# Load your adjacency matrix
A = csr_matrix(...)  # weighted, directed CSR matrix (n x n)

# Split data for validation
A_train, A_val, mask = heldout_split(A, frac=0.1, stratify_degrees=True, seed=42)

# Fit DC-SBM model
model = DCSBM(K=6, max_iter=300, tol=1e-5, seed=42)
model.fit(A_train)

# Get results
print("Converged:", model.converged_, "ELBO iters:", model.n_iter_)
labels = model.predict()                    # Hard block assignments
Q = model.fit_transform(A_train)           # Soft membership matrix
params = model.get_params()                # Model parameters

# Evaluate on held-out data
val_ll = model.score(A_val, mask=mask)
print("Held-out log-likelihood:", val_ll)
```

## Model Specification

The DC-SBM models weighted, directed networks where edge weights follow:

```
A_ij ~ Poisson(λ_ij)
λ_ij = θ_out[i] * θ_in[j] * ω[g_i, g_j]
```

Where:
- `θ_out[i]`, `θ_in[j]`: Degree correction parameters for nodes i, j
- `ω[k,l]`: Block-pair interaction rates
- `g_i`: Block assignment for node i

**Normalization constraints** (for identifiability):
```
Σ(θ_out[i] : g_i = k) = 1  ∀k
Σ(θ_in[i]  : g_i = k) = 1  ∀k
```

## API Reference

### DCSBM Class

```python
DCSBM(K, max_iter=200, tol=1e-4, seed=None, init="spectral", zero_handling="ignore")
```

**Parameters:**
- `K`: Number of blocks
- `max_iter`: Maximum EM iterations
- `tol`: Convergence tolerance for relative ELBO change
- `seed`: Random seed for reproducibility
- `init`: Initialization method ("spectral" or "random")
- `zero_handling`: How to handle zero entries ("ignore" or "add_epsilon")

**Methods:**
- `fit(A)`: Fit model to adjacency matrix
- `fit_transform(A)`: Fit and return soft membership matrix Q
- `predict()`: Get hard block assignments
- `score(A, mask=None)`: Compute average predictive log-likelihood
- `get_params()`: Get model parameters (θ_out, θ_in, Ω, Q, π)
- `diagnostics()`: Get convergence diagnostics

**Attributes after fitting:**
- `Q`: Soft membership matrix (n × K)
- `labels_`: Hard block assignments (n,)
- `theta_out_`, `theta_in_`: Degree parameters (n,)
- `Omega_`: Block-block rates (K × K)
- `elbo_`: ELBO trace
- `converged_`: Convergence flag
- `n_iter_`: Number of iterations

### Utility Functions

```python
# Data splitting
heldout_split(A, frac=0.1, stratify_degrees=True, seed=None)

# Network properties
degrees(A)  # Returns (k_out, k_in)
to_edge_list(A)  # Convert matrix to edge list

# Initialization
spectral_init(A, K, seed=None, d=None)  # Spectral initialization
```

## Algorithm Details

### Variational EM

**E-step**: Update soft memberships Q using block-wise posterior:
```
log q_ik ∝ log π_k + Σ_edges w * q_jl * [log θ_out[i] + log θ_in[j] + log ω[kl]]
            - θ_out[i] * Σ_l ω[kl] * T_in[l] - θ_in[i] * Σ_l ω[lk] * T_out[l]
```

**M-step**: Update parameters using sufficient statistics:
```
ω[kl] = m[kl]  (soft block-pair totals)
θ_out[i] = Σ_k q_ik * k_out[i] / S_out[k]
θ_in[i]  = Σ_k q_ik * k_in[i] / S_in[k]
```

### Spectral Initialization

1. Degree-regularized adjacency: `A_norm[ij] = A[ij] / (√k_out[i] * √k_in[j])`
2. SVD embedding: concatenate left and right singular vectors
3. K-means clustering on embedding
4. Softmax smoothing with temperature τ = 0.1

### Numerical Stability

- All log operations use `safe_log()` with ε = 1e-12
- Softmax computed using `logsumexp` for numerical stability
- Damped membership updates: `Q ← (1-η)Q + η * Q_new`
- Small regularization added to Ω diagonal

## Performance

**Time Complexity**: O(|E|K) per EM iteration
**Space Complexity**: O(nK + K²)

Optimized for sparse matrices using CSR format with efficient edge iteration.

## Testing

```bash
# Run test suite
cd tests
python test_dcsbm.py

# Or with pytest
pytest test_dcsbm.py -v
```

Tests include:
- Synthetic data recovery (ARI > 0.7 typical)
- ELBO monotonicity
- Parameter constraint satisfaction
- Sparse matrix compatibility
- Edge case handling

## Example: Neural Connectivity Analysis

```python
# Example with FlyWire data (from poc.ipynb)
import numpy as np
from dcsbm import DCSBM, heldout_split

# Load Delta7 + EPG connectivity (87 neurons)
A = connectivity_matrix  # 87 x 87 unsigned adjacency matrix

# Split data
A_train, A_val, mask = heldout_split(A, frac=0.1, seed=42)

# Fit DC-SBM with K=2 blocks
model = DCSBM(K=2, max_iter=200, seed=42)
model.fit(A_train)

# Analyze results
labels = model.predict()
params = model.get_params()

print(f"Block 0: {np.sum(labels == 0)} neurons")
print(f"Block 1: {np.sum(labels == 1)} neurons")
print(f"Block-block rates:\n{params['Omega']}")

# Validation
val_score = model.score(A_val, mask=mask)
print(f"Held-out log-likelihood: {val_score:.4f}")
```

## Mathematical Background

The DC-SBM extends the standard Stochastic Block Model by incorporating node-specific degree parameters, allowing for realistic modeling of networks with heterogeneous degree distributions within communities.

**Key References:**
- Karrer & Newman (2011). "Stochastic blockmodels and community structure in networks"
- Aicher et al. (2014). "Learning latent block structure in weighted networks"

## License

This implementation is part of the MesoCompPrimitives project for analyzing neural connectivity and extracting computational primitives.