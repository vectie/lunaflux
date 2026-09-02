# Luna attention tile IR

This package is the functional, backend-neutral attention dialect between the
serving graph and device lowering. A program is an immutable sequence of typed
value bindings. Every binding is pure and topologically ordered; paged K/V
iteration is represented by an explicit online-softmax fold rather than hidden
mutation in a CUDA source template.

The compiler fixes semantic dataflow and selected tile geometry here. Later
passes may schedule asynchronous copies, subgroup work, and matrix operations
for a target backend without changing the model graph or the online-softmax
meaning. Canonical bytes and a stable digest make pure-pass output reusable as
a content-addressed AOT compilation key.
