# FP8 v2 device executor

This package is the first private physical-resource owner for the finite-E4M3
projection ABI v2. Preparation consumes one already authenticated
`Fp8ReleaseAuthorityV2`, one authenticated numeric-weight owner, and one
caller-owned device context. That opaque release owner already joined the
exact `Fp8RuntimeRecipeV2` to an externally approved Luna release manifest and
no-follow, digest-inspected CUBIN files. Caller-created compile receipts or raw
module bytes cannot reach preparation.

All module, function, stream, allocation, argument, and close capabilities stay
inside the opaque executor. Raw PagedV4 operands are translated in their exact
admitted order. Weight and scale operands borrow the authenticated weight
allocation; every runtime/activation identity receives one deterministic
startup allocation; all launches share the recipe's single bounded workspace.

Execution is serialized and synchronous. After each kernel completes, the
owner copies the operation's complete four- or eight-byte activation-scale
workspace into fixed host scratch. It publishes an executed capability only
when every little-endian F32 cell is finite and strictly positive. The
canonical quiet-NaN failure sentinel, infinities, negative values, and either
zero sign therefore fail-stop before output access.

This is an inert engine seam. It is not imported by runtime descriptors,
workers, or the CLI, and it does not claim a complete FP8 graph, physical
numerical correctness, performance, or readiness. Exact `sm89`, `sm90`, and
`sm120` are the only admitted targets; every other target fails before resource
construction.
