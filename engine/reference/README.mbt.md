# LunaFlux reference executor

This package interprets an immutable `ModelPlan` using the correctness-only
`kernels/reference` implementation. Production workers must not use it as a
fallback: it deliberately favors explicit semantics and deterministic fixtures
over throughput, retains intermediate activations, and performs bounded host
allocation.

The production package has no dependency on a model-family builder or a device
backend. Model-specific behavior ends in the validated plan; tensor identity,
operation order, input roles, dimensions, and caller limits are checked before
or during interpretation. `prepare` decodes immutable BF16 tensors once for
reuse across full-recomputation steps. Greedy selection uses only the last
logits row; `generate` repeats that path under explicit token and context
bounds.
