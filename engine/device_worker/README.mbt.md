# Device worker aggregate

This package is the readiness owner for one device-worker process. `admit_plan`
validates and retains an independently expected full startup contract plus
immutable model, path, memory, kernel, artifact, and bootstrap evidence in an
opaque inert `DeviceWorkerPlan`. `prepare` first requires the received contract
to equal the admitted contract, then opens the exact
ordinal within the process-visible device set, streams weights from the
approved read-only model file, and prepares the complete paged executor.

`DeviceWorkerOwner::readiness_contract` succeeds only while the context,
weights, and executor are all live. No context, allocation, weights, executor,
module, function, stream, kernel argument, or native handle is exposed.

Cleanup is dependency ordered: executor, weights, context. A child cleanup
failure prevents its parent from closing and preserves authority for retry.
Preparation follows the same rule and returns `DeviceWorkerCleanupRequired`
when both startup and its first deterministic cleanup fail.

This package does not write the process `Ready` frame or own the worker
channel; the child entry point must call `readiness_contract` and only then
encode that exact value. CPU tests cover immutable admission and lifecycle
orchestration, while the nested device, weight, and graph owners provide their
own fault seams. Physical CUDA load/execute/close, sanitizer, leak, soak, and
benchmark evidence remains a promotion gate.
