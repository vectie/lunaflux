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
