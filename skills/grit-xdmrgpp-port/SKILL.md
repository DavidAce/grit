---
name: grit-xdmrgpp-port
description: Use when porting xDMRGpp eigensolver logic or changing orthogonalization, Ritz extraction, restart, Jacobi-Davidson, or convergence logic.
---

# xDMRGpp Porting

Use `/home/cluster_users/x_aceituno/xDMRG` as the read-only reference implementation.

Porting rules:

- Keep copied or ported solver code recognizable.
- Preserve naming, function order, control flow, and assertions where practical.
- Do not add helper functions or simplifications unless there is a clear reason.
- Do not port DMRG-specific user-space logic into GRIT.
- Treat orthogonalization, Ritz extraction, restart handling, residual correction, and status checks as delicate code.

Notation:

- `l2`: Euclidean inner-product logic. For a basis `V`, this corresponds to `V.adjoint() * V = I`.
- `bm`: B-metric inner-product logic for generalized eigenvalue problems. For a basis `V`, this corresponds to `V.adjoint() * B * V = I`.
- The `m` in `bm` means metric. It replaces the old xDMRGpp-specific `h2` name in GRIT.
- Do not introduce `h2` into GRIT APIs or user-facing options.

Solver preferences:

- Keep standard and generalized paths distinct where the B operator matters.
- B-metric logic belongs in generalized problems.
- `use_jd_b_only` should work in both L2 and B-metric projector modes for generalized Jacobi-Davidson correction.
- Preconditioner callbacks should be user-settable. The default preconditioner should be the identity operation.
- `Factorization` and block-Jacobi details belong in user space, not in GRIT.
- Negative iteration limits mean unlimited iterations where the option supports it.
- `ncv` is the total maximum number of basis columns and should be adjusted to the problem size.
- Preserve xDMRGpp-style standard deviation tests for stalling detection.
- Coarse correction and Chebyshev-related logic are not priorities now.
