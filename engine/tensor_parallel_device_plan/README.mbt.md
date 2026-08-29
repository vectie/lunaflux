# Tensor-parallel rank device plan

`engine/tensor_parallel_device_plan` is the authority-free execution bridge
described by ADR-0010. It joins five independently admitted startup facts: the
dense-Llama semantic plan, its immutable tensor-parallel rank plan, the exact
file-bound sharded weight inspection, one rank-to-device assignment, and the
expected homogeneous device target.

Construction fails before publishing a view if any model identity, complete
rank plan, rank, world size, device target, tensor reference, placement,
extent, local arena region, operation mapping, or collective site differs. A
multi-rank plan additionally proves that every row- or column-sharded local
region is strictly smaller than the full BF16 tensor. Replicated normalization
vectors remain explicit and bounded.

The result contains scalar shapes, local byte regions, and stable semantic
collective sites only. It owns no allocation, context, stream, module,
communicator, scheduler state, schedule plan, or backend handle. World size one
uses an explicit single-device assignment, retains full local tensor extents,
and has no collective sites, making it the deterministic oracle view.

The initial multi-rank assignment constructor consumes the already admitted
same-host homogeneous full-mesh topology. It preserves topology rank order and
process-visible ordinals but does not confer peer-access or collective
authority. A later rank-group execution owner must still authenticate group
generation and schedule-plan sequence before dispatching these collective
sites.
