# Luna attention tile schedule

This package lowers the functional attention dialect into a backend-neutral
loop schedule. Query rows and heads are pure parallel maps, paged key/value
traversal is an explicit ordered fold, and head components are a vector map.

The schedule retains the selected arithmetic and memory classes while avoiding
CUDA blocks, warps, PTX, HIP workgroups, MFMA, XMX, or CPU thread vocabulary.
Each backend refines the same immutable schedule into its native execution
model. Canonical schedule identity makes pure compilation results cacheable and
allows independent backend lowering and autotune.
