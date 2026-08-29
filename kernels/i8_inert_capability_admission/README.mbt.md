# Symmetric-I8 inert capability admission

This package performs one catalog-only Phase-8 join. It derives a canonical,
opaque admission from an exact model numeric schema, the symmetric-I8 device
feature, and catalog-v4 resolution. There are no caller-supplied claims.

Every selected operation is closed to reviewed dense projection families and
is operation-atomic: all tensor operands are symmetric I8. Each weight binds
its rank-two shape and storage digest to one globally unique plain-F32
rank-one scale whose length equals the output-channel count. Each catalog
binding is numeric-v4, content-addressed AOT, and retains the exact module,
family, and entry-point identity together with its workspace envelope.

The result is deliberately inert and catalog-only. It owns no model plan,
resolved catalog, artifact bytes, module, symbol, launch contract, device
plan, context, executor, or readiness state. Content-addressed AOT identities
are values, not an artifact owner. A later physical slice must independently
admit artifacts and launches, then reauthenticate this digest before acquiring
any execution resource.

Admission is two-pass. The first pass validates and counts without retaining
operation or weight evidence, authenticates the resolved catalog's aggregate
workspace envelope, and proves the exact canonical-v1 byte length with checked
arithmetic. Only then may bounded evidence arrays and the limit-aware canonical
writer allocate; every write preserves the preflight envelope and final length.
