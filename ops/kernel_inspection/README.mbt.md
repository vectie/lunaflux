# Kernel inspection

This package renders bounded operator diagnostics from one already-admitted
Phase 5 `LunaAotKernelAdmission`. It reads only immutable identity, target,
operation, artifact, workspace, provenance-receipt, and graph metadata.

It does not parse a manifest, verify a signature, accept generated source,
load a compiler or device, select a runtime capability, or expose JIT. The CLI
may call it only after a real digest-pinned admission chain exists for the
selected model.

Graph memory is labelled as an authenticated declared startup upper bound,
never as an observed driver allocation. Metadata that rejects capture renders
an explicit unsupported state and no fabricated zero-byte value.
