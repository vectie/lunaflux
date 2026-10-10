# DeepSeek V4 host materialization

This startup-only package joins bounded streaming safetensors metadata to
the exact DeepSeek V4 semantic weight binding and numeric-plan digests. Complete
raw tensor payloads are copied directly into bounded segmented final arenas and
owned by an explicitly invalidating release type. Release drops owner references
without a security-only payload scrub.

Loading consumes the construction-time immutable binding, numeric plan and
manifest label. It does not repeat the numeric-manifest scan or reserialize and
hash the tensor/layout manifest. Model/plan association, range and copy checks
remain; supplied source labels are not runtime payload-authentication claims.

The admitted header tags are exactly `BF16`, `F32`, `I64`, `I8`, `F8_E4M3`,
and `F8_E8M0`. Quantized tensors can be projected to the separate bounded
payload contract when their parameter/scale geometry is exact. UE8M0 scalar
interpretation, FP4 nibble/codebook ordering, device materialization, and
kernel execution remain typed unavailable. Raw host admission is therefore not
an executable-weight claim.

The first three official I64 `tid2eid` tensors can additionally be converted
directly from their final authenticated host arenas into explicitly releasable
I32 sidecars. Conversion is startup-only, binds the exact model and manifest,
checks every expert index and per-row uniqueness, and never creates an
intermediate I64 table copy. The sidecars remain device- and kernel-neutral.
