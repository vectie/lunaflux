# Device execution boundary

`device` owns LunaFlux's public accelerator abstractions. CUDA and cuBLASLt
handles remain opaque in `internal/cuda`; none appear in this package's public
interfaces or errors.

All GPU resources require deterministic release. Close dependants before their
owners:

1. close GEMM plans before their cuBLASLt handle;
2. close functions before their module;
3. close events, streams, allocations, modules, and cuBLASLt before the context;
4. retry a close that reports `Busy` or `DriverFailure`; a failed close retains
   ownership and does not invalidate the wrapper.

CUDA driver and cuBLASLt libraries are intentionally loaded once and retained
for process lifetime. Their dispatch pointers can therefore never outlive the
libraries that own them.

Host transfers are synchronous. This is intentional: MoonBit buffers are
borrowed only for the duration of an FFI call and cannot safely outlive an
asynchronous transfer. Phase 1 BF16 GEMM also synchronizes its caller-owned
stream before returning, so the plan, operands, workspace, and stream cannot be
closed while device work still references them. Later asynchronous execution
requires an explicit completion object that owns those lifetimes.

Native resources reject close with `Busy` while an operation is active. Parent
creation and child retention use the same interlock, preventing concurrent
close from destroying a context, module, or cuBLASLt handle during child
construction.

The current GEMM contract is deliberately narrow: row-major BF16 inputs and
output, FP32 accumulation, fixed alpha one and beta zero, and a reusable shape
descriptor. The ABI passes a null algorithm descriptor so cuBLASLt performs its
documented implicit heuristic query with default search preferences; it does
not claim deterministic or explicit algorithm selection. An unsupported shape
or missing ABI fails instead of switching to a reference or CPU implementation.
Physical-GPU correctness and performance are separate Phase 1 evidence gates.

AOT functions use an equally narrow synchronous launch contract. Launch
geometry is bounded before the native call, and each of at most 32 parameters
is an immutable checked allocation region with an explicit power-of-two
alignment. Argument lists are prepared once and reused without launch-path
MoonBit allocation. Function, module, stream, and every referenced allocation
must share one context and remain active until the one mandatory stream
synchronization completes. Kernel scalar dimensions remain part of the exact
startup-selected AOT shape in this phase; arbitrary scalar and byte payloads
are intentionally absent from the public launch vocabulary.
