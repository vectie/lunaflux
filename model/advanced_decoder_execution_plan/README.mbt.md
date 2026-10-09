# Advanced decoder execution plan

This package owns an immutable, architecture-neutral semantic plan for
decoder-only models that combine low-rank attention, hybrid compressed
attention, multi-stream hyper-connections, routed/shared experts, and auxiliary
prediction heads.

The plan is built once during startup, validates every layer in exact order,
and mints a canonical SHA-256 identity over all execution-relevant fields. It
does not select a device, kernel artifact, scheduling policy, KV allocator, or
weight materializer. Those later joins must bind this exact identity.
