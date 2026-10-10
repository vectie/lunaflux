# FP8 externally approved release authority

This package is the sole executable bridge for FP8 ABI v2. It consumes an
Luna FP8 release manifest and reads each unique declared
`sha256/<label>.cubin` beneath an `ApprovedRoot` once using bounded snapshots.
It does not reopen the retained manifest, hash CUBINs per operation, or compare
whole CUBIN payloads against a second copy. Manifest/module labels associate
metadata; they do not authenticate loaded bytes. It joins model/plan identity, target,
runtime-recipe digest, operation order, entry point, symbol, complete raw ABI,
workspace, source/recipe labels, and module lengths. The legacy manifest-locator
argument is retained for API compatibility but is not read. Unique-module and
aggregate byte budgets remain required.

Caller-constructible compile receipts and compile-only evidence are not inputs.
The opaque result owns immutable bytes and release approval only; device
context, module loading, execution, numerical success, and readiness remain
owned by the consuming device executor and its scale-cell validation.
