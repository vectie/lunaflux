# Device worker aggregate

This package is the readiness owner for one device-worker process. `admit_plan`
validates and retains an independently expected full startup contract plus
independent model metadata, a precomputed immutable weight-file inspection, an
opaque `PagedExecutionAdmission`, and bootstrap limits in an inert
`DeviceWorkerPlan`. Admission requires the inspection's exact layout to equal
the layout retained by the aggregate execution admission and validates the
canonical paged model, bootstrap, and startup identities before any resource
opens. `prepare` first requires the received contract to equal the admitted
contract, then opens the exact ordinal within the process-visible device set,
completely re-admits the retained descendant beneath an independently approved
pinned root without re-inspecting it, streams weights through one reusable
fixed host buffer, and prepares the complete paged executor.

Weight readiness additionally requires successful terminal source-file close.
If both that close and cleanup of an otherwise-ready allocation fail, the
weight owner retains retryable `SourceClose` cleanup authority and worker
preparation cannot close the parent context or publish readiness.

`DeviceWorkerOwner::readiness_contract` succeeds only while the context,
weights, and executor are all live. No context, allocation, weights, executor,
module, function, stream, kernel argument, or native handle is exposed.

`execute_frame` authenticates a completion writer against one validated plan
frame, privately stages and executes the paged graph, appends the canonical
completion while the writer remains open, finishes the executor, and only then
submits and returns the validated frame. Any entered failure fail-stops the
owner as `WorkerFaulted`; it cannot become ready or
execute again, but remains explicitly closeable. Failure also best-effort
aborts the accepted completion writer; a double failure reports both bounded
causes. Writing the returned validated frame to transport remains outside this
owner.

Cleanup is dependency ordered: executor, weights, context. A child cleanup
failure prevents its parent from closing and preserves authority for retry.
Preparation follows the same rule and returns `DeviceWorkerCleanupRequired`
when both startup and its first deterministic cleanup fail.

This package does not write the process `Ready` frame or own the worker
channel; the child entry point must call `readiness_contract` and only then
encode that exact value. CPU tests cover immutable admission and lifecycle
orchestration. A positive-controlled native release harness additionally
prepares a genuine owner, warms it, and proves 128 exact final-prefill greedy
`execute_frame`/completion-authentication/plan-retirement cycles perform no
MoonBit managed, array, or string allocation and create no native resource.
Its fake device checks exact transfer, launch, synchronization, readback,
publication, and cleanup counts; injected launch/readback/non-finite faults
prove aggregate fail-stop and writer reuse. This is allocation and control-flow
evidence only. Ordinary prefill, decode, stochastic sampling, physical CUDA
numerical correctness, sanitizer/leak, soak, and benchmark evidence remain
promotion gates.
