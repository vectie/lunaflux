# Operation-level diagnosis and split-decode dispatch fix

## Confirmed dispatch defect

Profiling the combined Qwen bundle (`07a3c33` runtime, head256 and MLP8/group2
artifacts) reveals that crossing the split-decode context threshold discards
the smaller execution-graph buckets. `prepare_paged_ordered_executors` previously
prepared only a maximum-envelope split-decode executor. Its selection bypassed
the ordinary decode bucket table.

The measured C1 short-vector trace contains 5,208 postprocess calls with grid
1024 × 32 and mean 42.62 µs, despite only one live query token. The same segment
uses QKV grid 2048 instead of the bounded grid 512. These are actual serving
launches, not isolated candidate measurements. The residual/norm kernel also
retains a 1024-block envelope; that independent issue is not fixed by this change.

The fix constructs split-decode bucket owners at startup. A pure transformation
matches `(operation ID, function index)` across the attention expansion and
reuses only unchanged kernels' bounded launch contracts. New partial/merge
attention functions retain their own exact dimensions. Dispatch is a fixed-array
lookup; numerical kernels, descriptor layout, and token-step allocations are
unchanged. The maximum-envelope executor remains the fallback.

Regressions cover identity remapping after inserted attention operations,
preservation of partial/merge geometry, C1/C8 owner selection, threshold/phase
exclusion, and the uncaptured-slot fallback. Physical A/B validation is pending
at this initial checkpoint; no speedup is claimed for the fix yet.

## Trace collection scope

All engines use Qwen3-0.6B BF16 on RTX 5060 Ti, identical token-ID vectors,
greedy fixed output lengths, no prefix cache, C1/C8, and excluded warmups.
Vectors are 59/256, 128/128, 512/64, and 1528/32 input/output tokens.

Nsight node-level tracing is diagnostic and its request throughput is not an
uninstrumented serving benchmark. A disposable supervisor forwards profiler
environment settings; the worker and kernel artifacts are unchanged.

The initial r1 traces lost final buffered events when process groups stopped.
Empty or partially covered long-vector windows must not be interpreted as zero
cost or compared. The r2 collection uses periodic CUDA buffer flushing and
explicit collection stop before process teardown. Raw files remain on the host
under `/dev/shm/lunaflux-op-profile-20260908-r1` and `-r2`; neither run overwrites
older traces. Per-operation attribution and complete-window validation follow
the r2 collection.
