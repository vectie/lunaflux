# Generic tensor-parallel plan

This package owns architecture-neutral, authority-free rank-local sharding
vocabulary: topology, row/column placement, matrix extents, aligned arena
regions, direct transfer recipes, exact operation-local geometry, and ordered
collective sites. Model-family builders emit this generic contract, while its
constructor reauthenticates operation coverage and geometry against the
immutable semantic model plan and equal-sharding v1 topology.

The package does not import safetensors, Llama, devices, kernels, schedulers,
workers, or native backends. Its constructors revalidate physical scalar
relationships and make an ordered rank plan immutable; they do not allocate
device memory or open a resource.
