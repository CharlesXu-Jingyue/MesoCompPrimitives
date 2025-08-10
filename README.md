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

## Deliverables

Deliverable for the 20-min talk: one end-to-end example (e.g., a reciprocal core + relay shell distilled to “integrator + gate”).

Exercise for the 45-min tutorial: give a 3-block toy $W$, ask them to (i) find ports via SVD on inter-block submatrices, (ii) keep 1–2 Hankel modes/block, (iii) classify the primitive from its $A_k^{red}$ spectrum.