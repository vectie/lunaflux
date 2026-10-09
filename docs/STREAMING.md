# Bounded streaming and cold-prefix residency

Status: software integration implemented on branch `stream`, based on
`aad4e7e578b4e2d7f984ce3c9f448f4e90543121`. The branch has an isolated worktree;
uncommitted compiler and two-Spark work on `main` is not included.

## Motivation and evidence

NVIDIA's August 2026 [Polars streaming article](https://pola.rs/posts/gpu-streaming-backend/)
reports 1,118.3 seconds on a dual-socket Xeon versus 48.2 seconds on eight B200s
for the approximately 3 TB PDS-H suite. This is a dataframe benchmark, not an
LLM inference result. The useful mechanisms are bounded working sets,
backpressure, explicit residency and overlapping transfers with computation.
[RapidsMPF](https://docs.rapids.ai/api/rapidsmpf/stable/background/streaming-engine/)
is a design reference; LunaFlux retains its MoonBit control path and private
CUDA ABI.

Host spilling still consumes bandwidth and host capacity. DGX Spark's CPU and
GPU [share 128 GB of physical memory](https://docs.nvidia.com/dgx/dgx-spark/hardware.html),
so allocating a second host copy there does not extend physical capacity.
Capacity-tier admission must distinguish separate and shared physical memory.

## Functional design

Immutable startup plans describe capacity, page geometry and allowed transfer
regions. Pure functions map a state and an event to a next state or rejection.
The interpreter alone commits fixed-array cells and issues native effects.
Planning never opens a resource, hashes request data, performs I/O, or allocates
in steady state. Scalar value types represent transitions and completion IDs.
There is no global mutable runtime or environment-driven token policy.

## Ordered implementation

1. **Residency and admission.** A fixed-capacity, generation-safe owner reserves
   host slots and transfer credits before an effect. Pure transitions cover
   spill, restore, cancellation, transfer failure and terminal reclamation.
   Cold entries have no active readers. Host slots and device reservations
   remain leased while any transfer can access them. Generation exhaustion
   retires a slot; a stale completion can never publish a new occupant.
2. **Asynchronous transfer ABI.** Preallocate pinned host buffers and reusable
   transfer executors. A native owner retains the exact context and allocation
   leases, issues bounded copies on its own stream, polls one completion event,
   and retains cleanup authority after partial submission or close failure.
   Managed MoonBit byte arrays are never borrowed across an asynchronous call.
3. **Cold-prefix storage.** Compose residency with canonical per-layer K/V
   regions. Spill only complete, immutable, unreferenced prefix pages. Bind
   model, tokenizer, layout, scope and worker generation at the existing prefix
   boundary. Reserve restore destinations before transfer and publish pages
   only after the complete copy batch succeeds.
4. **Execution integration.** Restore into the existing stable device arena.
   The owner prevents graph execution with incomplete destinations; cancellation
   suppresses publication but cannot release DMA leases early. Scheduler policy
   consumes neutral readiness and budget values and imports no device package.
   The existing resident path remains the default until the new capability has
   its own admitted startup contract and positive qualification.
5. **Validation and selection.** Test transitions, capacity exhaustion, stale
   completions, identity isolation, every failure point and exact cleanup.
   Run native sanitizers and compare physical cached/recomputed logits and
   tokens. Measure restore latency, bytes moved, queue pressure, TTFT and decode
   interference. Choose restore only from measured costs under the admitted
   latency budget; never invent bandwidth or speedup evidence.

Startup weight loading can reuse the transfer machinery for bounded read/copy
overlap after these ownership gates. Workspace-aware prefill chunk planning is
a pure startup selection over already admitted shapes. Per-step dense weight
offloading is a separate capability: repeated transfers can dominate latency
and require segmented execution plans and their own numerical/performance gate.

## Ownership and failure rules

- A spill retains the source until all copies complete; a restore retains the
  host source and device destination until all copies complete.
- An enqueue failure may leave earlier copies live. The owner becomes poisoned
  and only a successful drain allows reclamation.
- A cancellation is a state transition, not an attempt to cancel CUDA DMA.
- A resource close failure retains a retryable owner, including construction
  that only partially acquired native resources.
- Host and device budgets include both copies while a transfer is in flight.
- Slot reuse increments a non-wrapping generation. Completion also binds an
  operation epoch, so a duplicate completion from an earlier transfer fails.
- Startup fixes all storage and transfer capacities. Saturation returns a
  scalar refusal without allocating a request-path error.

## Validation ledger

The unmodified base fails `moon check --target native --deny-warn` with 161
diagnostics from the current toolchain, including warning 79 trait-method
promotion changes. This is recorded separately from new-package validation.
No streaming physical-CUDA correctness or performance claim exists yet.

Foundation results before serving integration (Moon `0.1.20260920`, 2026-09-26):

- `moon info` and affected-package formatting/checking pass with the inherited
  warning exclusions below. Generated interfaces expose no internal CUDA types
  through public packages.
- The combined affected test run passes 246 tests. The pure residency/policy
  packages also pass 12 native release tests with all warnings denied.
- The existing allocation-instrumented executable, with positive controls,
  observes zero allocations across 10,000 warmed spill/restore/cancel/fail/drain
  and policy-selection cycles, then passes its existing scheduler/wire gate.
- ASan/UBSan passes 1,024 repetitions of 16 deferred-DMA scenarios (16,384
  total), including driver failure after accepting a copy, partial preparation,
  overlap rejection, delayed completion, drain/close failure and exact balance.
- Unfiltered repository-wide native check and test remain blocked by inherited
  toolchain diagnostics (the final check reports 144 errors, including warning
  79 promoted to an error). No repository-wide passing result is claimed.
- The physical page probe builds/checks locally but has not run on CUDA. It
  does not establish model-logit parity or representative transfer performance.

## Implementation ledger

| Step | Implemented on `stream` | Remaining gate |
| --- | --- | --- |
| 1. Residency | Pure transitions, fixed slots/credits, generation checks, quarantine/drain; scheduler pins spill sources and reserves every restore destination | Physical qualification |
| 2. Transfer ABI | `device.TransferPool` over private `internal/cuda`: pinned storage, reusable streams/events, asynchronous copies, range/overlap checks, retained allocation lease, partial-construction cleanup | Discrete-GPU correctness and transfer benchmark |
| 3. Cold payload | Canonical full-page K/V copies; the existing radix now retains host payload identities under the original model/tokenizer/layout/scope key, including shared ancestors | Physical cached/recomputed parity |
| 4. Execution | Value-type streaming IR, v5 startup capability, cooperative parent/child transport, stable destination arena, atomic restore publication, cancellation, timeout, post-reap invalidation and replacement epochs | Physical model-output parity |
| 5. Selection and gates | Measured restore/recompute policy and workspace-aware admitted prefill shapes wired into scheduling; bounded scan, sanitizer and allocation probes | Representative TTFT/decode-interference measurements and production qualification |

The feature is explicitly enabled by the digest-pinned descriptor versions
below. Legacy descriptors retain the resident path. No performance claim is
inferred from the Polars benchmark. Startup weight-copy overlap and per-step
dense-weight offloading remain separately scoped follow-on capabilities.

## Implemented worker transaction design

The parent owns logical prefix identity and allocator page generations; the
child owns the actual payload bytes, pinned allocation, streams and events.
Neither an `Entry` nor an `Operation` containing a MoonBit owner reference may
be serialized. A versioned scalar transaction protocol must carry the admitted
worker epoch, transaction sequence, host slot/generation and
device page index/generation. Every reply binds the same transaction. Startup
binds model/tokenizer/KV-layout/security-scope identities once; token execution
uses the resulting scalar authorities.

1. Select a complete immutable prefix with no active request references. Hold
   its radix identity and source page references while reserving host slots.
   Shared prefix ancestors cannot be reclaimed until all remaining references
   are accounted for.
2. Send bounded spill commands while retaining source reservations. Each page
   changes tier only after every layer's K/V copy completes. Mixed resident/cold
   prefixes remain cold misses to ordinary activation. Release the page's
   device cache reference only after its host payload is committed.
3. On a host hit, compare qualified restore/recompute costs and reserve every
   destination page plus transfer credit before submission. Keep the request
   out of active schedules while destinations are incomplete. A restore target
   is private until the whole prefix succeeds and its block table is published.
4. Cancellation marks the transaction, suppresses publication, and retains
   endpoints until completion/drain. Failed drain poisons the worker and keeps
   parent reservations until kill/reap proves that DMA ownership is gone.
5. Worker replacement invalidates all prior host identities. No restored page
   or late reply can cross a worker-generation boundary. Backpressure selects
   bounded waiting or recomputation, never an unbounded transfer queue.

Wire control operations must be serviced without blocking on CUDA completion;
normal graph plans are accepted only after their destinations are ready. The
first integration is conservative: graph work and transfers are fenced rather
than overlapped. Permitting independent-page compute/transfer overlap requires
a later hazard proof and a measured decode-interference budget.

The neutral `engine/streaming_ir` owns pure protocol decisions and scalar value
results; the interpreter owns effects. The 64-byte transfer frame rejects
noncanonical tags/padding and binds each response to the exact command. The
1024-byte startup v5 frame carries at most 32 shapes and a separate child epoch;
legacy v4 stays exactly 408 bytes. See [the architecture decision](STREAMING_ARCHITECTURE.md).

The service counts a live transfer as outstanding work, including after logical
cancellation. It refreshes the monotonic deadline before interpreting a reply.
Timeouts retain reservations until child cleanup; replacement binds a strictly
newer epoch. Spill candidate selection visits at most 64 radix anchors per turn.

## Activation contract

Use `schema_version: "lunaflux.runtime.v6"` on the existing BF16 descriptor,
including its required graph memory ceiling, and provide a `streaming` object:

| Required field | Meaning |
| --- | --- |
| `host_slots` | Fixed complete-page host capacity, 1–65,536 |
| `host_budget_bytes` | Pinned-memory ceiling; must cover every slot |
| `reserve_pages` | Target free device pages below which cold candidates spill |
| `restore_nanoseconds_per_page` | Qualified conservative end-to-end restore cost |
| `recompute_nanoseconds_per_page` | Qualified cost for the same complete logical page |
| `separate_host_memory` | Must be true; the worker independently rejects integrated GPUs |
| `transfer_timeout_millis` | Bounded transfer timeout, 1–600,000 |
| `workspace_budget_bytes` | Shared prefill workspace budget for one schedule |
| `prefill_shapes` | 1–32 unique `{ "tokens": ..., "workspace_bytes": ... }` profiles, including a one-token shape that fits |

Page bytes are derived from the authenticated KV layout across every layer and
both K/V components. Shapes must fit the scheduler/worker token envelope and
the policy budget cannot exceed the allocated activation arena. Selection
debits the workspace budget across selected rows and preserves the decode page
reserve. Costs and shapes must come from qualification of the pinned deployment;
the implementation does not fabricate calibration values.

### Additional worker routes

The same `streaming` object is required by these additive descriptor versions:

| Execution route | Streaming descriptor |
| --- | --- |
| Llama BF16 | `lunaflux.runtime.v6` |
| I8 weights with BF16 KV | `lunaflux.runtime.i8.v3` |
| Reusable FP8 weights with BF16 KV | `lunaflux.runtime.fp8-reusable.v4` |
| Qwen3 BF16 | `lunaflux.runtime.qwen3_bf16.v3` |
| Mistral BF16 | `lunaflux.runtime.mistral_bf16.v3` |
| Local tensor parallel | `lunaflux.runtime.tensor-parallel.v2` |

The existing loaders and launch routes select these versions; no new launch
schema or scheduler family branch is needed. Prior descriptor versions reject
the streaming object. Single-device versions require the graph-memory ceiling.
Tensor-parallel workspace budgets and prefill shapes apply per rank; host-budget
bytes cover the entire group and are divided before each rank's shard is admitted.
Replication cannot silently multiply the declared pinned-memory budget.

I8, FP8, BF16, and rank workers share the same `StreamingSession` interpreter.
Each session binds one cache for its lifetime. Tensor-parallel `GroupTransfer`
is immutable value-type IR: it rejects duplicate/stale/divergent rank results,
retains a completion mask, and publishes only when every shard is terminal.
Completed ranks are skipped during later polls. Restore cancellation may race
with completed shards and still suppresses publication for the entire group.
Maintenance spills finish normally; abandoning them uses whole-group recovery.
Any transfer timeout or protocol/device failure preserves page ownership until
all ranks have been reaped. Replacement uses a fresh group epoch.

## Integration validation (2026-09-26)

- 275 native scheduler, IR, radix, descriptor and worker-service tests pass with
  inherited dependency warnings excluded. New pure packages pass 14 release
  tests with **all** warnings denied.
- A separate device/worker/wire/cache run passes 369 tests with the same
  inherited dependency warning exclusions.
- The real scheduler/IR allocation gate passes 10,000 complete spill/restore/
  eviction cycles. It caught and removed optional-value boxing and temporary
  prefix-key allocation; both request keys and effect results are value types.
- Tests cover atomic multi-page restoration, shared ancestors, scope isolation,
  cancellation, deadline expiry, timeout ownership, stale replies, replacement
  epochs, host eviction, and workspace selection across multiple rows.
- ASan/UBSan still passes 16,384 deferred-DMA scenarios; every cycle also checks
  live discrete/shared-memory detection and balances native leases.
- `moon info` and `moon fmt` complete. The full warning-denied native check
  reports the same 144 inherited diagnostics. No clean full-repository result
  is claimed.
- Native Linux ARM64 passes the streaming process fixture (startup, cooperative
  copies, cancellation, replacement epochs), 19 worker-process tests, and the
  complete service fixture (cold reuse before activation, equal fixture output,
  cancellation and close). The broader legacy process E2E driver also passes;
  its fixture now consumes parent attestation, follows graph telemetry sampling,
  and checks the rooted supervisor's existing single-flight backpressure.
- Native Linux ASan/UBSan/LeakSanitizer passes all 16,384 deferred-DMA scenarios
  with leak detection enabled. An earlier x86-emulated container was unsuitable
  for process/sanitizer validation and supplies no passing evidence.

## Worker-family and rank-group extension (2026-09-26)

- 461 affected native tests pass across descriptors, shared execution, numeric
  workers, rank configuration, wire control, process ownership and service
  integration. The inherited warning exclusions below remain necessary.
  The pure streaming IR, residency, and policy packages pass 17 native release
  tests with all warnings denied and no exclusions.
- The scheduler/IR allocation gate passes 10,000 cycles including the immutable
  rank barrier. Both rank-wire and socket-backed rank-child-control allocation
  gates pass 1,000 streaming exchanges with zero measured allocations and
  working positive controls.
- Native Linux ARM64 passes all 19 rank-process tests, including skipped
  platform-dependent cases. A real two-process fixture passes skewed rank
  completion, cancellation after one shard completes, a subsequent restore,
  eviction, healthy close, and whole-group failure during a pending exchange
  followed by cleanup and reap. This fixture does not execute CUDA or NCCL.
- ASan/UBSan again passes all 16,384 deferred-DMA scenarios with exact native
  lease balance. The extension changes no native ABI implementation.
- The final unfiltered native check reports 161 errors in 21 files. Every
  diagnostic file is byte-for-byte unchanged from the original branch base;
  the earlier 144-error report above is historical. The unfiltered native test
  command also stops on inherited warnings. No full-repository pass is claimed.
  Checking the entire repository with the four warning exclusions reaches an
  additional compiler parser-consistency diagnostic (16) in the unchanged
  `benchmarks/qwen3_token_id_bridge/detokenize.mbt`; the affected-package native
  check passes with those exclusions.
- Physical cached/recomputed model-output parity, discrete-GPU transfer
  measurements, NCCL execution and production performance qualification remain
  outstanding.

## Reproducing the local gates

```sh
moon test engine/streaming_ir kv/residency scheduler/streaming_policy --target native --release --deny-warn
moon test scheduler/core prefix/radix runtime/descriptor_file engine/worker_service --target native --deny-warn --warn-list '-79-25-20-29'
moon test kv/host_cache device internal/cuda engine/device_step engine/device_worker engine/device_worker_bootstrap engine/device_worker_child engine/worker_wire --target native --deny-warn --warn-list '-79-25-20-29'
moon test engine/streaming_ir engine/rank_group_wire engine/rank_child_control engine/rank_group_process engine/tensor_parallel_rank_configure engine/tensor_parallel_group_transport engine/tensor_parallel_device_worker engine/tensor_parallel_worker_bootstrap engine/worker_service runtime/descriptor_file engine/device_step engine/i8_device_worker engine/fp8_device_worker_v3 engine/device_worker engine/device_worker_bootstrap kv/device_arena --target native --deny-warn --warn-list '-79-25-20-29'
moon run scripts/validate-streaming.mbtx
moon run tests/hot_path_alloc --target native --release --deny-warn --warn-list '-79-25-20-29'
moon run tests/rank_group_wire_alloc --target native --release --deny-warn --warn-list '-79-25-20-29'
moon run tests/rank_child_control_alloc --target native --release --deny-warn --warn-list '-79-25-20-29'
```

On native Linux, build `cmd/worker_echo` and run `tests/worker_process_e2e`
with `--streaming` followed by the absolute worker executable path. Run
`tests/streaming_service_e2e` with the same executable path.
The echo worker is a protocol fixture: it proves scheduler/wire/lifecycle
integration and deterministic token semantics, not CUDA payload correctness.
The production descriptor-based process spawning API intentionally rejects
unsupported hosts, including macOS.

For the rank-group extension on native Linux, build `tests/streaming_rank_echo`
and run `tests/streaming_rank_group_e2e` with its absolute executable path. Run
the latter again with `--failure` after the executable path to exercise failure
during an active transfer and post-reap cleanup. These packages contain only
synthetic metadata and protocol fixtures; they confer no deployment authority.

The warning exclusions above apply to inherited dependency diagnostics from
the base revision; repository warning configuration is unchanged. The new pure
packages run with warnings denied and no exclusions. The sanitizer runner uses
ASan/UBSan and explicit native host/stream/event/lease balance checks; macOS does
not supply LeakSanitizer; the separate native Linux run above supplies that gate.

## Completion criteria

The [9 October branch integration report](BRANCH_INTEGRATION_AND_MODEL_PREFLIGHT_2026-10-09.md)
records the fresh merged-main software checks, Linux fixture portability fixes,
completion-slot regression, and the separate model-execution blockers. Its
protocol/sanitizer passes do not close the physical criteria below.

The workstream is complete only when cold prefixes cross the worker boundary,
restore before request activation, reproduce the resident outputs, balance
all resources through cancellation/restart, and have measured physical
performance. A standalone residency model or fake-driver test does not fulfill
that end-to-end gate. Software completion and the remaining physical
qualification are recorded separately above.
