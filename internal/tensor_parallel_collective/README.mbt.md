# Authenticated tensor-parallel collective projection

This private logical package is the sole production constructor of a generic
device collective contract. It authenticates an independently supplied model
identity against one immutable generic tensor-parallel rank plan, copies its
rank/world and exact one-based generic collective site table, binds an explicit
maximum live query-token envelope, and derives a canonical SHA-256 contract
identity over those claims. Model-family builders, currently including Llama,
choose semantic sites before this boundary; this package contains no family
graph or operation-order branch.

The package does not import NCCL or CUDA and owns no device, communicator,
scheduler, rank-group, configuration, or filesystem authority. Execution and
deterministic cleanup belong to the model-neutral `device` collective owner.
