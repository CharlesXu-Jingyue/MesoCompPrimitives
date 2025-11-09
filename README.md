# MesoCompPrimitives

By Charles Xu @ Caltech

Distilling mesoscale primitives that can be (re)assembled to match function.

## Framing

Goal: compress a dense, directed, signed connectome $W$ into a small library of I/O primitives (e.g., integrator, WTA/normalizer, gate/switch, ring line-attractor, relay), each with:
- an effective transfer operator (local linear model or low-order nonlinear normal form),
- a role (hub/connector, boundary, feedforward relay, recurrent core),
- ports (well-defined input/output subspaces),
- and composition rules (how primitives wire to preserve global behavior).

## Pipeline

### Partition & role discovery

Use a directed, degree-corrected block model (or spectral clustering on a Hermitian lift / random-walk operator).

Identify roles with stochastic blockmodel “image graphs” (core↔periphery, bow-tie feedforward, reciprocal cores).

### Local response modeling (primitive identification)

For each community $C_k$, define in/out port sets $P_k^{in}$, $P_k^{out}$. Estimate a reduced map
$$\dot{z}_k = A_k z_k + \sum_j G_{kj} z_{j} + B_k u_k + \text{nonlinear terms,}$$
where $z_k$ is the first few principal/singular response modes of $C_k$.

Practically: impulse/step probe within the subgraph, fit a low-order model (LDS, SLDS, or a normal form like pitchfork/saddle-node near operating point).

### Control-theoretic reduction

Inside each $C_k$, do balanced truncation: compute controllability/observability Gramians $W_c$, $W_o$, take Hankel singular values $\sigma_i$, keep modes with large $\sigma_i$. This yields a minimal $A_k^{red}$, $B_k^{red}$, $C_k^{red}$ as the primitives' linear “skeleton.”

If clearly nonlinear, keep a low-order polynomial (SINDy/Koopman features) on top of the reduced linear part.

### Canonical labeling (library)

Compare each primitive’s step/impulse & frequency response to templates: integrator (near-unit eigenvalue), WTA (competitive inhibition + saturating nonlinearity), normalizer (divisive), ring attractor (approximate rotational symmetry), gate (multiplicative gain control).

### Reconstruction / synthesis test

Compose primitives via their ports to build a coarse network $\tilde{W}$. Two validation tiers:
- Spectral & response invariants: match leading singular vectors/eigenvalues and step responses to those of the full system.
- Task-level equivalence: on working-memory persistence, routing/gating, or path integration, the coarse model replicates performance within tolerance while being orders smaller.

### Ablation-based falsification

Swap a primitive with a functionally similar but topologically different one (e.g., WTA implemented via pooled inhibition vs. subtractive feedforward) and show which task metrics shift—this demonstrates the necessity of the identified computational primitive.

## Implementation Progress

### Current Status: Proof-of-Concept (notebooks/poc.ipynb)

**Completed:**
- **Data Loading & Preprocessing**: FlyWire ring extend connectivity data (855 neurons, 68 cell types)
- **Cell Type Grouping**: Neurons organized by cell type with alphabetical ordering
- **Matrix Visualization**: Signed and unsigned connectivity matrices with cell type boundaries
- **Data Structure Setup**: Grouped connectivity matrices, neuron IDs, and cell type mappings
- **Unconnected Node Detection**: Analysis shows all 855 neurons are connected (no isolated nodes)
- **Cell Type Selection**: Focused analysis on Delta7 (40 neurons) and EPG (47 neurons) subset (87 total)
- **Graph Laplacian Analysis**: Complete implementation of multiple Laplacian variants:
  - Out-degree Laplacian (L_out = D_out - A)
  - In-degree Laplacian (L_in = D_in - A^T)
  - Random walk Laplacian (L_rw = I - D_out^(-1) * A)
  - Symmetric Laplacian via symmetrize-then-normalize
  - Direct symmetrization of out-degree Laplacian
- **Community Detection**: Implemented and compared Louvain and Leiden algorithms:
  - Louvain: 4 communities (sizes: 24, 24, 20, 19), modularity: 0.2123
  - Leiden: 6 communities (sizes: 23, 20, 13, 13, 10, 8), modularity: 0.1853
  - Cell type distribution analysis across detected communities

**Recently Added:**
- **Bi-orthogonal Laplacian Renormalization Group (bi-LRG)**: Complete implementation of hierarchical coarse-graining
  - Spectral analysis of directed random-walk dynamics with bi-orthogonal decomposition
  - Left and right eigenvector computation for non-symmetric operators
  - Bi-orthogonality verification (δ_ij inner products)
  - Mass-preserving Markov lumping with group assignment
  - Coarse-grained network operators (P_group, L_group, A_group)
  - Spectral fidelity metrics and hierarchical reduction capability
- **Bi-Galerkin Projection Analysis**: Advanced spectral projection method
  - Projection of operators into subspaces spanned by slowest modes
  - Bi-Galerkin projected operators: P_galerkin, L_galerkin, A_galerkin
  - Mathematical framework: U_k^H @ Operator @ V_k projections
  - Comparative analysis with Markov lumping approaches
  - Visualization of projected operators with diverging colormaps
- **Enhanced Eigenspectrum Analysis**: Comprehensive spectral characterization
  - Multiple Laplacian variants: L_out, L_in, L_rw, L_sym, L_bal
  - Balanced symmetrization implementation
  - Eigenvalue statistics and zero eigenvalue detection
  - Spectral plotting with eigenvalue distributions
- **System Identification - CTRNN Analysis**: Row-sum normalized fixed-point and blockwise linearization pipeline
  - Global row-sum normalization ensuring contraction with safety margin c < 1
  - Per-block fixed-point computation using damped Picard iteration
  - Local sigmoid gains computation: Γ_r = diag(σ'(x_r*))
  - Linearized dynamics construction: δẋ = Aδx + Bδu with block structure
  - Optional eigenvalue, Schur decomposition, and balanced truncation analyses
  - Global stability checking and convergence diagnostics

**In Progress:**
- **Role Discovery**: Computational library development and canonical labeling
- **System Synthesis**: Primitive composition and reconstruction testing
- **Koopman Mode Decomposition**: Advanced dynamical analysis methods

**Mathematical Framework Implemented:**
- **Bi-orthogonal Spectral Analysis**: For directed networks with non-symmetric Laplacians
  - Left/right eigenvector decomposition with bi-orthogonality constraints
  - Spectral fidelity preservation through hierarchical coarse-graining
  - Multi-scale network dynamics via eigenmode projection
- **Galerkin Projection Methods**: Dimensionality reduction preserving spectral properties
  - Bi-Galerkin projection: U_k^H @ L @ V_k for reduced-order modeling
  - Comparison between spectral projection and Markov lumping
  - Visualization and analysis of projected operator structure
- **Graph Laplacian Variants**: Comprehensive treatment of directed network operators
  - Out/in-degree Laplacians for directed flow analysis
  - Random walk and symmetric normalization schemes
  - Balanced symmetrization for numerical stability
- **Continuous-Time RNN Linearization**: Fixed-point and local linearization framework
  - Row-sum normalization: W̃ = αW with contraction factor α = 4c/||W||∞
  - Block-wise fixed-point iteration: x_r* = W̃_rr σ(x_r*) + b_r
  - Local gain matrices: Γ_r = diag(σ'(x_r*)) for sigmoid nonlinearity
  - LTI dynamics: A_rr = -I/τ + W̃_rr Γ_r/τ, E_rs = U_rs Γ_s/τ
  - Stability analysis and balanced realization theory

**Dataset Overview:**
- Original: 855 neurons across 68 cell types
- Current focus: 87 neurons (Delta7 + EPG cell types)
- Signed connectivity matrix with both excitatory and inhibitory connections
- Dense connectivity: 1,729 edges in 87-neuron subnetwork

## Deliverables

Deliverable for the 20-min talk: one end-to-end example (e.g., a reciprocal core + relay shell distilled to "integrator + gate").

Exercise for the 45-min tutorial: give a 3-block toy $W$, ask them to (i) find ports via SVD on inter-block submatrices, (ii) keep 1–2 Hankel modes/block, (iii) classify the primitive from its $A_k^{red}$ spectrum.