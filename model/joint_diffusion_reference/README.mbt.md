# Joint diffusion correctness reference

This family-neutral package provides allocation-bounded correctness references
for a validated `JointDiffusionPlan`: shifted Rectified Flow schedules,
counter-based latent noise streams, packed projected-row layout, and pipeline
stage transitions.

It intentionally owns no model weights, transformer or VAE numerics, device
resources, kernels, scheduler queues, HTTP types, or media codecs. The routines
are executable specifications and small-fixture oracles, not production tensor
kernels.
