# Luna attention tile schedule

This package lowers the functional attention dialect into a backend-neutral
loop schedule. Query rows and heads are pure parallel maps, paged key/value
traversal is an explicit ordered fold, and head components are a vector map.

The schedule retains the selected arithmetic and memory classes while avoiding
CUDA blocks, warps, PTX, HIP workgroups, MFMA, XMX, or CPU thread vocabulary.
Each backend refines the same immutable schedule into its native execution
model. Canonical schedule identity makes pure compilation results cacheable and
allows independent backend lowering and autotune.

Optimizer-selected page-lookup hoisting is also explicit schedule data. It
means one logical K/V row computes its paged address once for all vector
fragments; it does not prescribe whether a backend realizes reuse with a
subgroup broadcast, local memory, or scalar register sharing.

Online-softmax tile-storage reuse is likewise explicit schedule data. The
schedule shortens the portable shared working-set bound by overlapping
disjoint key/probability lifetimes and retaining fold state with its query
owner. Concrete address-space placement remains backend-specific.

For matrix QK/PV, `matrix_storage_layout()` exposes typed byte spans and
half-open live phases. Disjoint phases share a region whose capacity is the
maximum of their requirements; values live together add their requirements.
The key overlay therefore reserves the maximum of staged K, probabilities
**plus** previous scales, and terminal maximum/denominator. This avoids
overwriting V when the query tile is as wide as the head dimension. Query
bounds and validation scratch occupy disjoint score slots before QK.

The immutable layout owns offsets, lifetimes, and peak storage, all bound into
the schedule identity. CUDA consumes those offsets rather than recalculating
them. This is storage extraction for the existing online-softmax rewrite,
not yet a general arbitrary-IR allocator or a cross-iteration alias proof.
