# Read-only paged-attention qualification candidate

This package derives a distinct CUDA AOT candidate from the authenticated
write-plus-attend reference lowering. The candidate accepts already-positioned
query values and already-written paged key/value storage. Its raw ABI declares
both cache pointers `const`, and generated source has no KV mutation path.

The candidate and its deterministic compiled binding are qualification-only:
`manifest_bindable=false` and promotion authority is absent. Runtime dispatch
must continue using the existing write-plus-attend family until a future
grouped manifest admission binds a standalone positioned-RoPE/KV-write
operation and this read-only attention operation as one ordered selection.

That future admission must bind the exact model generation, operation and
activation identities, KV layer/layout, profile, target, writer candidate and
compiled artifact, read-only candidate and compiled artifact, raw ABI, source
and recipe digests, and same-stream writer-before-reader ordering. Any missing
or mismatched identity selects the existing write-plus-attend fallback.

Future physical evidence must additionally bind device/compiler/driver
identity, deterministic CUBIN bytes, numerical parity with the fallback over
scalar, page-boundary, mixed-row, and long-context cases, identical cache
digests before and after the read-only launch, cache canaries, sanitizer
results, resource bounds, cleanup balance, and non-circular evidence seals.
No source-only campaign result is physical CUDA evidence.
