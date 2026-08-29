# FP8 externally approved release authority

This package is the sole executable bridge for FP8 ABI v2. It consumes an
externally approved Luna FP8 release manifest, reopens that exact manifest and
every canonical `sha256/<digest>.cubin` beneath a pinned `ApprovedRoot` using
no-follow immutable snapshots, and joins model content/plan identity, target,
runtime-recipe digest, operation order, entry point, symbol, complete raw ABI,
workspace, source/recipe identities, and actual CUBIN bytes.

Caller-constructible compile receipts and compile-only evidence are not inputs.
The opaque result owns immutable bytes and release approval only; device
context, module loading, execution, numerical success, and readiness remain
owned by the consuming device executor and its scale-cell validation.
