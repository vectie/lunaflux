# Luna execution graph strategy

This backend-neutral package builds phase-aware power-of-two graph buckets at
startup. Runtime selection maps `(phase, batch rows, query tokens, context)` to
one prebuilt slot without allocation or a bucket scan. CUDA Graph, HIP Graph,
and command-graph ownership remain private to their device backends.
