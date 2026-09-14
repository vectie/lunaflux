# Luna execution graph strategy

This backend-neutral package builds phase-aware power-of-two graph buckets at
startup. Runtime selection maps `(phase, batch rows, query tokens, context)` to
one prebuilt slot without allocation or a bucket scan. CUDA Graph, HIP Graph,
and command-graph ownership remain private to their device backends.

`ExecutionGraphMeasuredRoutes` is a pure startup transform from matched offline
measurements to immutable slot choices. Phase, batch, query and context remain
separate axes. Baseline measurements are mandatory for observed slots; absent
slots do not inherit another shape's winner. Dispatch uses bounded shape mapping
and a single array lookup, without measurement collection or a candidate scan.
Source/device identity and artifact ownership remain responsibilities of the
startup importer and backend, respectively.
