# Exact device execution profiles

`engine/device_profile` binds one immutable `StaticDevicePlan` and its matching
maximum-envelope `DeviceMemoryPlan` to one exact request shape. It derives all
token-input, weight, activation, workspace, and terminal-output byte views at
preparation time. Dispatch can then index immutable operation records and
region identifiers without allocating or repeating model-family decisions.
Every validated shape, region ID, view, operation profile, and complete profile
is opaque; callers can inspect checked accessors but cannot publish raw-field
substitutes.

This legacy bridge rejects catalog v4 before deriving any view. A later
numeric-aware profile slice must consume the complete authenticated numeric
layout and bind associated metadata operands such as weight scales. Until then,
catalog v4 and I8 static planning remain inert and cannot be presented as an
execution profile.

## Honest phase contract

The current semantic graph is stateless: causal attention has no KV-cache
input, page table, block mapping, or cache-position input. Consequently this
package supports only:

- `FullPrefill`, where every input token is a query and is present in context.
- `FullRecompute`, where generation reruns the complete sequence after a token
  is appended.

There is deliberately no decode constructor. A profile with one query token
and a longer claimed context would say that cached keys and values exist when
neither the static plan nor the memory plan provides them.

`ExactExecutionShape::kernel_shape` extracts the computational kernel-selection
key. Merely fitting inside the maximum activation arena does not make a kernel
compiled for another row count valid. A prepared executor must resolve an AOT
entry point or vendor plan for the exact batch rows, sequence tokens, and token
rows. Request phase stays separate: full prefill and full recompute with equal
dimensions execute the same stateless graph and may share one admitted kernel
profile.

## Buffer spaces and stable regions

Region identifiers are deterministic within a profile:

- `0` is the Int32 token-ID staging region.
- `1 + TensorRef` identifies a referenced weight-arena region.
- `1 + weight_tensor_count + ActivationRef` identifies an activation.
- the next identifier names the shared workspace.

Token staging is a separate reusable device allocation because
`DeviceMemoryPlan` owns only activation/workspace capacity. The profile exposes
both its exact logical bytes and its maximum-envelope capacity; the later
executor must allocate that capacity once and reuse it. Activation and
workspace views are checked against the already planned arena. Weight views
are checked against the static weight arena.

An operation may legitimately require zero workspace bytes. That operation's
workspace view remains a valid zero-length logical alias of the shared region;
AOT argument preparation must omit the workspace operand instead of attempting
to construct a zero-length device argument.

The terminal view retains the complete exact output plus one precomputed last
query-token row per batch item. Each row is an explicit bounded subview that
references its canonical backing region instead of reusing a region identity
with different geometry. Language-model-head outputs are marked as logits;
other legal terminal activations remain generic outputs.

## Remaining KV and paged-attention work

True incremental decode requires a separate design phase. At minimum, the
semantic and device plans must explicitly own checked KV geometry, per-layer K
and V regions, cache positions, page/block tables, sequence lengths, capacity
limits, and deterministic release. The scheduler must reserve and update those
resources, and attention launch contracts must name them in a fixed ABI. Only
after those inputs exist may this package add a constructible decode phase.

This package allocates bounded metadata only while building a profile. It does
not allocate device memory, load weights or artifacts, prepare an executor,
load kernels, launch CUDA, mutate KV state, or make support, readiness,
physical-GPU correctness, or performance claims.
