# Device execution boundary

`device` owns LunaFlux's public accelerator abstractions. CUDA and cuBLASLt
handles remain opaque in `internal/cuda`; none appear in this package's public
interfaces or errors.

All GPU resources require deterministic release. Close dependants before their
owners:

1. close GEMM plans before their cuBLASLt handle;
2. close functions before their module;
3. close events, streams, allocations, modules, and cuBLASLt before the context;
   close allocation leases before their allocation;
4. retry a close that reports `Busy` or `DriverFailure`; a failed close retains
   ownership and does not invalidate the wrapper.

CUDA driver and cuBLASLt libraries are intentionally loaded once and retained
for process lifetime. Their dispatch pointers can therefore never outlive the
libraries that own them.

Each opened context pins its process-visible ordinal and immutable capability
snapshot. Startup planners compare that capability to their exact device target
before loading modules or allocating execution arenas; a plan selected for one
compute capability cannot be attached silently to another context.

`can_access_peer` is a directed, authority-free startup inventory query. It
validates two distinct visible ordinals and reports only the driver's bounded
boolean capability. It never creates a context or enables peer access; callers
must preserve directionality and fail closed when either ordered edge is absent.
On platforms without CUDA, including macOS, it remains truthfully unavailable.

Host transfers are synchronous. This is intentional: MoonBit buffers are
borrowed only for the duration of an FFI call and cannot safely outlive an
asynchronous transfer. `Allocation::copy_from_host` preserves the immutable
`Bytes` loading API. `Context::copy_from_fixed_host` accepts caller-owned
preallocated `FixedArray[Byte]` staging storage without materializing an
intermediate buffer; it also proves that the destination allocation belongs to
that exact context while both native resources are guarded against close.
`Context::copy_to_fixed_host` provides the symmetric allocation-free readback
into caller-owned staging storage, while `Allocation::copy_to_host` preserves
the allocating `Bytes` API for correctness-only callers. Phase 1 BF16 GEMM also
synchronizes its caller-owned stream before returning, so the plan, operands,
workspace, and stream cannot be closed while device work still references
them. Later asynchronous execution requires an explicit completion object that
owns those lifetimes.

Native resources reject close with `Busy` while an operation is active. Parent
creation and child retention use the same interlock, preventing concurrent
close from destroying a context, module, or cuBLASLt handle during child
construction.

`Context::lease_allocation` extends that interlock into an explicit lifetime
proof for prepared execution owners. A live lease keeps its exact allocation
retryably busy across calls while exposing no address, transfer, region, or
launch method. This lets an executor safely prebuild private argument lists
from caller-owned weights without stealing weight ownership. The executor must
close the lease only after it has made every prepared argument unreachable and
will never launch again.

Startup preparation may call `Context::validate_allocation_region` to prove
that a region belongs to the selected context and that its actual allocation
base plus offset satisfies a bounded power-of-two alignment, not merely that
the planned offset is aligned. The check is allocation-free and holds both
allocation and context lifecycle guards; it does not need to make the context
current because it reads only owned metadata. Launch and GEMM immediately
repeat the same native range, pointer-overflow, and resolved-address check so a
successful startup proof cannot weaken close-race or dispatch-time safety.

The current GEMM contract is deliberately narrow: a row-major BF16 activation,
a row-major `[output, input]` safetensors projection weight, a row-major BF16
output, FP32 accumulation, fixed alpha one and beta zero, and a reusable shape
descriptor. The cuBLASLt descriptor explicitly transposes the stored weight;
materialization never rewrites it. Zero-workspace algorithms are valid and use
a null workspace pointer. The ABI passes a null algorithm descriptor so
cuBLASLt performs its documented implicit heuristic query with default search
preferences; it does not claim deterministic or explicit algorithm selection.
An unsupported shape or missing ABI fails instead of switching to a reference
or CPU implementation. Physical-GPU correctness and performance are separate
Phase 1 evidence gates.

AOT functions retain a narrow synchronous correctness contract. Launch
geometry is bounded before the native call, and each of at most 32 parameters
is an immutable checked allocation region with an explicit power-of-two
alignment. Argument lists are prepared once and reused without launch-path
MoonBit allocation. Function, module, stream, and every referenced allocation
must share one context and remain active until the one mandatory stream
synchronization completes. Kernel scalar dimensions remain part of the exact
startup-selected AOT shape in this phase; arbitrary scalar and byte payloads
are intentionally absent from the public launch vocabulary.

Steady ordered execution uses `OrderedKernelExecutor`, not the general
synchronous function method. Startup flattens and revalidates every exact AOT
record, then the native owner holds long-lived active-operation leases and
references for its stream, functions, and argument allocations. Aliased close
therefore returns `Busy` until the executor is closed. Enqueue accepts only the
next scalar record index, never reconstructs launch metadata, and never
synchronizes. A nonblocking native operation gate rejects overlapping
enqueue, record, poll, abort, or reset calls with `Busy`, while a separate
active-operation interlock rejects concurrent close. A single reusable event
is recorded after all same-stream kernel
and collective submissions; polling is nonblocking. Fault cleanup first aborts
by synchronizing the retained stream, then destroys the event and releases
leases before functions, modules, the stream, or allocations can close. Event
destroy failure retains the exact owner and leases for retry.

The same owner has a startup-only CUDA graph mode. Its policy is fixed before
native construction: eager-only, captured-required, or captured with an
explicit eager fallback. Missing optional graph ABI symbols fall back only for
the last policy; a required capture fails closed. Capture consumes only the
already flattened function, geometry, and device-region records while every
dependency lease is live. The resulting graph exec can only be relaunched on
that exact retained stream: node updates and live-data graph construction are
not exposed. Warm launches allocate nothing, use the existing completion
event, and rearm only after completion. Graph-exec or event destruction failure
retains the owner and all dependencies for deterministic close retry.

Numeric capability admission is a separate, inert projection of one observed
`DeviceCapability`. The finite-E4M3 W8A8 v1 and symmetric-I8 weight-only v1
policies each have an independent closed allowlist that currently accepts only
the exact compute-capability pairs 8.9 and 9.0 with BF16 boundary support.
Neither policy treats architecture ordering as forward compatibility. Their
distinct feature tags produce distinct canonical identities even for the same
observed target.

The symmetric-I8 allowlist is software target admission only. It is not a
claim that any physical device, kernel artifact, module, materializer, loader,
or launch path supports I8 execution, and it publishes no readiness state. The
same is true of the finite-E4M3 device projection by itself: every later
artifact, startup-join, physical-correctness, and readiness gate remains
separate.
