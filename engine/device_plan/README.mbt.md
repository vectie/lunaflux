# Static device planning

`engine/device_plan` is the immutable startup bridge between a validated
semantic model plan, a deterministic device-weight layout, and an exact
resolved kernel catalog. Kernel selection must name the exact execution plan.
Weights are reusable across full-context and paged plans only when their
verified content digest matches; every referenced tensor is then revalidated
against the operation's exact semantic BF16 byte size, so content equality
cannot rekey a foreign tensor schema. The plan preserves activation
dependencies, carries the validated shape envelope, and records one exact
kernel binding per operation.
The complete weight layout is also revalidated and its tensor count is pinned
in the resulting startup evidence, including regions not referenced by the
current operation graph.

Planning is bounded to 4,096 operations, 64 inputs per operation, and 8,192
inputs in total. Unsupported or inconsistent startup data fails with a typed
payload-safe error before an executor is created.

This package is planning only. It does not own a CUDA context or allocation,
load AOT artifacts, invoke cuBLASLt, launch kernels, allocate activation or KV
arenas, schedule requests, or prove physical-GPU correctness. In particular,
constructing `StaticDevicePlan` is not evidence that production CUDA execution
is available.
