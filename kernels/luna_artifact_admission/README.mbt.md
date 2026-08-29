# Specialized AOT artifact admission

This package is the production-facing inert admission boundary for one signed,
externally approved Luna kernel capability. It requires exact agreement among
the admitted capability manifest, deterministic specialization record,
digest-verified artifact bundle, and full-graph paged launch contracts.

Admission checks model identity, target, catalog and semantic versions,
operation capability, stable profile entry point, module digest, KV layout,
workspace, artifact linkage, and the explicit graph disposition. The result
retains only immutable scalar identities, approval/source digests, graph
metadata, and stable launch metadata. It
exposes no manifest source, detached signature, compiler, model loader, device
context, native module handle, filesystem root, or request-path JIT channel.

Until physical graph capture and CUDA evidence land, the sole admitted
execution disposition is the validated eager fallback named by the capability
manifest. This package does not claim physical kernel correctness, graph
capture success, sanitizer/race coverage, or benchmark promotion.
