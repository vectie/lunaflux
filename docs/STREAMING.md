# Bounded streaming and cold-prefix residency

Status: implementation in progress on branch `stream`, based on
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

Local results (Moon `0.1.20260920`, 2026-09-26):

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
| 1. Residency | `kv/residency`: pure total transition function, fixed slot/free storage, bounded credits, owner/generation/attempt checks, quarantine/drain, cancellation and generation retirement; zero measured heap allocations over 10,000 warmed cycles | Downstream page ownership integration |
| 2. Transfer ABI | `device.TransferPool` over private `internal/cuda`: pinned storage, reusable streams/events, asynchronous copies, range/overlap checks, retained allocation lease, partial-construction cleanup | Discrete-GPU correctness, transfer benchmark and Linux leak gate |
| 3. Cold payload | `kv/host_cache`: canonical full-page K/V mapping for every layer, immutable budget plan, opaque cache/operation identities | Logical prefix key/index, parent page reservations and worker transaction protocol |
| 4. Execution | Opt-in paged-executor preparation, stable destination arena, stage fence, transfer-first teardown | Startup capability, child transport dispatch, restore-before-activation and worker-restart recovery |
| 5. Selection and gates | Pure prefill shape and restore cost policies, native failure probe, sanitizer runner, physical page probe | Policy hookup, cached/recomputed logits and token equality, TTFT/decode-interference measurements |

These are foundation APIs, not an enabled serving feature. Neither the request
scheduler nor worker startup silently opts into spilling. No performance claim
is inferred from the Polars benchmark. Startup weight-copy overlap and per-step
dense-weight offloading remain separately scoped follow-on capabilities.

## Worker transaction design (next integration boundary)

The parent owns logical prefix identity and allocator page generations; the
child owns the actual payload bytes, pinned allocation, streams and events.
Neither an `Entry` nor an `Operation` containing a MoonBit owner reference may
be serialized. A versioned scalar transaction protocol must carry the admitted
worker generation, transaction sequence, host slot/generation/attempt and
device page index/generation. Every reply binds the same transaction. Startup
binds model/tokenizer/KV-layout/security-scope identities once; token execution
uses the resulting scalar authorities.

1. Select a complete immutable prefix with no active request references. Hold
   its radix identity and source page references while reserving host slots.
   Shared prefix ancestors cannot be reclaimed until all remaining references
   are accounted for.
2. Send bounded spill commands while retaining the source reservations. Commit
   the host prefix only after every layer of every page completes. A prefix
   transaction is all-or-nothing; partial success cannot advertise a reusable
   prefix. Release the corresponding device cache references after commit.
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

The next implementation must extend the existing startup capability and
private worker transport together, including malformed-frame, duplicate-reply,
restart, shared-ancestor and cancellation fixtures. It must not add an
unvalidated environment switch or use integer IDs as unauthenticated pointers.

## Reproducing the local gates

```sh
moon test kv/residency scheduler/streaming_policy --target native --release --deny-warn
moon test kv/host_cache device internal/cuda engine/device_step --target native --deny-warn --warn-list '-79-25-20-29'
moon run scripts/validate-streaming.mbtx
moon run tests/hot_path_alloc --target native --release --deny-warn --warn-list '-79-25-20-29'
```

The warning exclusions above apply to inherited dependency diagnostics from
the base revision; repository warning configuration is unchanged. The new pure
packages run with warnings denied and no exclusions. The sanitizer runner uses
ASan/UBSan and explicit native host/stream/event/lease balance checks; macOS does
not supply LeakSanitizer, so the Linux leak gate remains outstanding.

## Completion criteria

The workstream is complete only when cold prefixes cross the worker boundary,
restore before request activation, reproduce the resident outputs, balance
all resources through cancellation/restart, and have measured physical
performance. A standalone residency model or fake-driver test does not fulfill
that end-to-end gate. Each stage's implementation and remaining work is recorded
below as it is completed.
