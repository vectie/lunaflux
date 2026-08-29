# Tensor-parallel execution plan

This package owns the inert physical blueprint for one authenticated local
tensor-parallel rank. It joins an admitted model operation graph, one exact
selected rank-local AOT launch profile, a file-authenticated sharded device
plan, and its content-addressed artifact bundle.

Admission derives five disjoint scalar address spaces: descriptors, rank
constants, sharded weights, persistent KV, and activation/workspace. Semantic
activations are assigned to startup-planned slots by exact producer/use
liveness; simultaneously live values never overlap. All operations share the
largest aligned workspace. Sum reductions use the producer output in place.
The vocabulary all-gather reserves the complete vocabulary result and places
rank `r`'s local send region at `r * local_bytes` inside that result.

The output is authority-free. It contains no device pointer, context, module
handle, function handle, communicator, stream, filesystem authority, compiler,
or JIT path. Its opaque SHA-256 digest covers the authenticated identities,
rank/device/catalog/profile scalars, launch contracts, rank constants, device
and KV layouts, every module and entry-point binding, every physical region,
operation, operand, and collective.
