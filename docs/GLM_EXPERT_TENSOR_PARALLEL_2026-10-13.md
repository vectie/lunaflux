# Grouped experts and tensor parallelism

**Latest completed full-model result:** non-expert TP plus the explicit native
FP4 expert path and a 64-row AOT envelope completes the six matched prompts in
**10.197 s**, versus a fresh vLLM **3.151630 s** (**3.235×** completion-time ratio).
The retained LunaFlux PP2 cohort was 16.509 s. The new run has one excluded
warmup; the old cohort included its first request. All six output counts match
the fresh reference, but only four rendered responses match. This is neither
full-model numerical parity nor production-quality qualification. Implementation
and numerical comparison remain separate. Details and limitations follow at
the end of this chronological report.

The measured starting point is 16.509 s for the six GLM prompts, versus
3.144176 s for the retained vLLM run. LunaFlux uses PP2/TP1; vLLM uses TP2/PP1.
This is not an equivalent distribution of work. Routed experts account for
431.8384 ms of the 1003.280928 ms two-stage diagnostic GPU trace. See
`GLM_SMALL_GROUP_SPARSE_2026-10-13.md` for measurement qualifications.

## Implementation decision

Keep three distinct concerns:

1. Numerical IR owns accumulation and publication rounds. Intermediate-axis
   tensor parallelism must reduce F32 down-projection partials **before** the
   expert output's BF16 rounding, routing weight, and deterministic combination.
   All-reducing the existing rounded expert outputs is not this operation.
2. Pure partition plans own row/column ranges and scale-block alignment.
   Gate/up are row-sharded and down is column-sharded. Packed payloads and their
   scale planes must be sliced together, without dequantization or whole-model
   host copies. A tensor-wide scale remains replicated.
3. Effect plans and device owners perform bounded startup transfers and ordered
   collectives. An expert-ID partition is EP, not intermediate-axis TP; a layer
   partition is PP, not TP. Runtime reports must retain those distinctions.

The same partition representation must support dense projections and attention
heads. Model adapters bind checkpoint names; CUDA/NCCL details stay in lowering.
External framework comparisons and quality evaluation remain outside compiler
selection. No payload hashing is added.

## Experiment boundary

Small expert groups need a separate useful-work mapping, not another blind
increase of the 16-row tile. The preceding three larger/gathered matrix trials
lost and are not repeated. New candidates must state their arithmetic contract,
test live and padded groups, and show selected source, correctness, memory
safety, and timing. A faster synthetic fixture is not full-model validation.

Sparks are exclusive per GPU; numerical probes use 2 GiB/no swap. Full-model
work remains within 96 GiB per node, with at least 16 GiB host headroom. No new
end-to-end speedup or complete TP implementation is claimed by this design.

## Implementation and bounded results

`ExpertTensorPartition` now describes packed gate/up row slices, down column
slices, and F32 partial publication. The loader reads strided planes through a
bounded scratch window (including scale planes), without expanding tensors or
building millions of per-row transfer descriptors. GLM's decoder inserts an
explicit collective boundary before each expert output publication. Attention,
dense layers and shared experts remain replicated: this is **expert TP**, not
vLLM-equivalent full attention/projection TP. The ordinary PP profile is unchanged.
The explicit CLI suffix is `-tiled-grouped-expert-tp`.

The multi-site queue regression executes 16 steps with two sums each: no
steady-state MoonBit allocation or blocking GPU synchronization, and no consumer
is enqueued before its reduction has been submitted. Packed column transfers
pass with scratch capacities 1, 3, 7, 16, 27, 64 and 128 bytes.

The two physical Sparks passed the generated whole/sharded expert fixture:
`/tmp/lunaflux-expert-tp-20261013-v4`, both ranks exact at rows 1 (64 outputs)
and 3 (192 outputs), explicit resource release. This does not prove full-model
quality or performance. v3 retained a diagnosed bootstrap failure from generating
the rendezvous ID in a process that had already exited. v4 fixes that lifetime.
v6's host CUDA-13.0 sanitizer cannot instrument the CUDA-13.4 NCCL image: it
fails inside NCCL initialization with `cudaErrorNoKernelImageForDevice`. That
gate remains open; it is not a clean sanitizer result. Earlier v1/v2/v5 setup
failures are retained alongside their output.

### Grouped kernels: paired experiments, not promoted regressions

`/tmp/lunaflux-glm-grouped-pair-20261013-v4` freezes the original v3 source and
the shared gate/up input-staging candidate in one executable. Signed, nonuniform
NVFP4 payloads/scales and nondyadic global/input scales replace the earlier
dyadic-only timing fixture. Activation, down output and final output are bitwise
equal. Five alternating trials include the whole four-kernel chain:

| Live rows | Original median | Shared staging median | Decision |
| --- | ---: | ---: | --- |
| 1 | 1467.974 us | 1482.067 us | 0.96% slower; do not enable for decode |
| 3 | 1889.625 us | 1912.970 us | noisy; no demonstrated win |
| 16 | 4576.426 us | 4441.494 us | 2.95% faster on this fixture |

Shared staging remains an explicit generic physical schedule option, default
unchanged. Its earlier all-16 memcheck/leak and one-row race/sync tests passed.
The scalar singleton attempt (`...grouped-performance-20261013-singleton-v1`)
lost, and was removed. A register-only BF16 MMA singleton experiment
(`...grouped-pair-20261013-register-v1`) also passes bitwise intermediate/output
checks but regresses row-one median from 1468.240 to 4189.978 us. It too is
removed from selectable compiler code; frozen generated source and binary stay
in the experiment root. Row16 was approximately unchanged. Fewer barriers do
not imply less execution time.

vLLM's selected reference uses FlashInfer CUTLASS W4A4, while this native path
still expands packed NVFP4 operands into BF16 for WMMA. This is a substantive
remaining implementation difference, not a claim that an extra IR layer alone
will close the gap. No new full-model speedup is established by these probes.

### Instruction-counter rejection

The paired register-only singleton root contains frozen `baseline.ncu-rep` and
`candidate.ncu-rep`, downloaded without hashing, and their raw counter CSVs.
`scripts/summarize-glm-grouped-pair-counters.mbtx` reproduces this selected
row-one gate/up comparison (profiled time is distinct from the unprofiled
four-kernel medians above):

| Counter | Original | Register-only singleton |
| --- | ---: | ---: |
| Gate/up profiled duration | 1.260672 ms | 3.137984 ms |
| Executed warp instructions | 124,659,152 | 152,652,732 |
| Average warp latency | 14.695807 cycles | 31.414019 cycles |
| Long-scoreboard per-issue ratio | 5.718480 | 24.592409 |
| Barrier per-issue ratio | 2.236370 | 0 |
| Tensor activity | 3.248920% | 1.301060% |
| Active-warps occupancy | 36.849723% | 38.020803% |

Both artifacts allocate 48 registers/thread and 19.456 KB shared/block. The
singleton branch bypasses the shared path but the combined kernel's launch
resource allocation still includes it. The loss is not explained by reduced
occupancy: dependent loads and 22.5% more instructions dominate despite removing
barrier stalls. Keep the rejected source frozen, not selected.

### Full-model memory boundary: no unsafe run

Clean ARM compilation of pinned `9d5c0393` completed at
`/tmp/lunaflux-checkpoint-build-glm-expert-tp-20261013-v1` (226 build tasks).
The CPU-only preparation `/tmp/lunaflux-glm-expert-tp-20261013-v1` stopped before
GPU allocation: compact routed-expert shards plus replicated non-expert
weights/state/workspace require **106,146,621,756 bytes per rank including the
2 GiB reserve**, or **98.86 GiB**. The configured limit remains **96 GiB**;
the shortfall is 3,067,406,652 bytes, approximately **2.86 GiB per rank**.

The separate metadata-only export
`/tmp/lunaflux-glm-expert-tp-metadata-only-20261013-v1` used a larger *planning*
ceiling solely to reveal the footprint. It was CPU-only under 4 GiB/no swap;
that number is not an authorized GPU budget or a runnable benchmark profile.
No full-model expert-TP benchmark was started. The last valid matched full-model
result is still the PP2 result at the beginning of this report.

Expert TP alone is consequently not a complete two-Spark solution for this
checkpoint under the agreed cap. Additional non-expert projection/attention
partitioning must reduce resident bytes before the full-model run. Reducing
precision, raising the cap, or quietly substituting CPU offload would not be
the tested implementation. Also still open: the compatible-runtime collective
sanitizer gate and the native low-bit expert operand path. Neither target is
reported as fully fixed.

## Follow-up: complete tensor partitioning and native low-bit execution

The user has explicitly raised this experiment's ceiling to **100 GiB per
Spark**, with zero task swap. The old 96 GiB rejection remains a historical
result, not the current limit. Preserve at least 16 GiB host-available memory;
the larger cap does not authorize overlapping GPU workloads.

Non-expert TP partitions independent attention heads and FFN intermediate
channels. Column-parallel producers retain full hidden inputs; row-parallel
consumers produce F32 partial sums, then an explicit collective precedes BF16
publication and residual updates. Small controls and normalization parameters
may remain replicated. Partition plans describe logical axes and rounding
boundaries; device instructions and collective ownership stay in their private
lowering/runtime layers. Test source selection and compact checkpoint slices,
not merely the presence of a plan in the compiler.

The low-bit grouped alternative must express activation quantization explicitly.
NVFP4-weight/BF16-activation WMMA and native W4A4 are different numerical
programs. A native block-scaled implementation needs independent quantized
arithmetic tests, matched output checks, sanitizer coverage, and measured
selection. Framework comparison remains an external benchmark, not a compiler
correctness oracle. No payload hashes or integrity scans are part of this work.

### Non-expert implementation stage (physical validation pending)

`TensorAxisPartition` and `ProjectionPublication` now carry head/channel slices
and the F32-before-BF16 collective boundary. KDA Q/K/V, expansions, convolution
history and recurrence state are head-sharded. Sparse Q-B/KV-B, KV history and
output projection are head-sharded; sparse indexing and low-rank inputs remain
replicated. Dense and shared FFNs shard intermediate channels. Compact uploads
reuse the packed rectangular-slice materializer, including strided output
weights. Rank zero no longer loads or computes a discarded vocabulary head;
token embedding and small controls are deliberately replicated.

Each full decoder has 132 explicit reduction sites: two per dense block and
three per MoE block. The decoder composes ordered compute/reduction/publication
boundaries at startup, without per-step array allocation. A local-only recurrent
executor rejects plans requiring a reduction. No TP output round occurs before
its collective. F32 reduction changes grouping and is not claimed to be bitwise
equivalent to the unpartitioned arithmetic on arbitrary inputs.

Warning-denied native check passes using the existing warning exclusions.
Two targeted batches pass (104/104 and 44/44; these overlap and are not summed).
The tests include all logical axis partitions, F32 scratch accounting, the
full 132-site contract, both attention source paths, dense/shared integration,
and text egress ownership. These are not yet GPU correctness, sanitizer, or
matched full-model timing results. The explicit TP benchmark now uses the
authorized 100 GiB planning/runtime ceiling; other benchmark modes retain their
previous 96 GiB limit.

### Native block-scaled expert implementation and first physical results

The explicit `-low-bit` variant carries `BlockScaledMatrixFloat32(16)` through
the independent expert numerical plan, compact rank plan, decoder owner and
kernel binding. Input/product global-amax and block-16 quantization are explicit
GPU stages, not implicit changes inside a GEMM. CUDA lowering uses native E2M1
block-scaled matrix instructions, with E4M3 block scales and separate global
scale publication. The eight-stage grouped path retains F32 partial output for
TP before collective/BF16 publication. Routed weights remain packed; there is
no persistent BF16 expansion. Defaults are unchanged. Per-rank product
quantization is an explicit numerical choice, not claimed to be bitwise equal
to unpartitioned BF16 activation execution.

Independent scalar packed-arithmetic fixtures passed on Spark .179:

- `/tmp/lunaflux-native-fp4-fragment-20261013-v4`: all 128 fragment outputs exact.
- `/tmp/lunaflux-native-fp4-grouped-20261013-v3`: actual generated grouping,
  activation quantization, gate/up, product quantization, down and combination
  match at live rows 1, 3, 16, 17, zero rows and reuse. Memory/leak/race/sync
  sanitizers report zero errors. This is not a full-model accuracy or speed claim.

CUDA 13.0 requires architecture-specific cubin compilation for these operations:
`--generate-code arch=compute_121a,code=sm_121a`. Plain sm121 and sm121a with a
generic fallback PTX target failed compilation; failed probe roots are retained.

The first complete non-expert TP run, pinned to `9c6d11fe`, compiled both rank
modules and loaded below the 100 GiB cap, but failed at the first NCCL launch
with `invalid resource handle`. Both ranks released their contexts and had zero
task swap. Root: `/tmp/lunaflux-model-bench-glm-fulltp-20261013-v1`. No prompt
timing from that attempt is valid. A runtime-primary-context alternative and a
small real-ABI two-rank context A/B probe are implemented; the context explanation
is a hypothesis until that identical-communicator experiment runs. Retain/release
ownership is explicit, with no primary-context reset or driver configuration change.

The subsequent real-ABI context A/B confirms this cause. Pinned `1f75a105` uses
the same NCCL library and four 32-element F32 all-reduces in both modes:
`/tmp/lunaflux-tensor-context-private-20261013-v1` reproduces CUDA error 400 at
the first collective on both ranks; `/tmp/lunaflux-tensor-context-runtime-20261013-v1`
exits successfully, all 32 outputs exact in all four repetitions on both ranks.
Runtime-context peak memory is 1,814,523,904 / 1,815,592,960 bytes, below the
2 GiB per-rank diagnostic limit, with task swap disabled. The full model is
still pending; the A/B is not a throughput result.

The independent grouped fixture also passes the F32-partial/deferred-publication
variant in `/tmp/lunaflux-native-fp4-partial-20261013-v1` at the same six row/reuse
states with all sanitizers clean. This tests the actual generated partial path,
not a complete two-rank matrix comparison. Latest affected native tests pass
162/162; warning-denied native check and both benchmark runner checks pass.

### Completed full-model comparisons (TP and native FP4)

Implementation is pinned to `1f75a105`; the clean ARM build is
`/tmp/lunaflux-checkpoint-build-glm-fp4tp-20261013-v1`. All successful native
cohorts log both selected arithmetic and all 132 collective sites, exit zero,
and report explicit context closure on both ranks. The old default arithmetic
is unchanged; the native FP4 numerical program is explicit, not an automatic
replacement based on agreement with another framework.

| Runtime and artifact | Six completion times, seconds | Sum |
| --- | --- | ---: |
| Retained PP2, BF16-activation grouped experts | 2.100, 2.099, 3.226, 3.293, 3.324, 2.467 | 16.509 |
| TP2, BF16-activation experts, rows16, one warmup | 1.493, 1.307, 2.467, 2.575, 2.593, 1.912 | 12.347 |
| TP2, native FP4 experts, rows16, one warmup | 2.170, 1.502, 2.386, 2.426, 2.418, 1.660 | 12.562 |
| TP2, native FP4 experts, rows64, one warmup | 1.240, 1.256, 2.124, 2.179, 2.191, 1.207 | **10.197** |
| Fresh vLLM TP2, one warmup | 0.385795, 0.393984, 0.651295, 0.669784, 0.672840, 0.377932 | **3.151630** |

The input-token vector is `[22,23,22,23,24,36]`, identical literal IDs across
engines. New FP4 and reference output-count vectors are `[4,4,8,8,8,3]`.
The BF16 TP control instead produces three tokens on the second prompt, so its
34-token total is **not equal work** to the 35-token FP4/reference totals.
For the three equal-length eight-token requests, FP4/rows16 is approximately
3.3%, 5.8%, and 6.7% faster than the BF16 TP control. The table does not establish
a large independent FP4 end-to-end gain. Numerical changes affect autoregressive
outputs; they must remain visible rather than being hidden by timing totals.

The 64-row profile uses the same compiler and numerical choices but fits each
prompt in one prefill pass, rather than two or three. Its six first-token times
are `[0.548,0.567,0.550,0.586,0.593,0.739]` seconds. It is 38.2% less completion
time than the retained PP2 cohort (1.619× speed), but still 3.235× vLLM's time.
It represents 3.43 output tokens/request-second including prefill, versus 11.11
for vLLM; these are **not decode-only throughput**. These are individual cohorts,
not confidence intervals. The BF16 KV / reference FP8-MLA KV distinction remains;
native recurrence is the explicit ordered-token profile in these runs, not a
claim that every possible optimization/profile has been enabled.

The native rows64 cold warmup itself takes 4.733 s (first token 4.044 s). Both
engines exclude one warmup, and the external runner retains it separately.
`compare-glm-matched-benchmark.mbtx` now records actual TP topology, numerical
profile, warmup count, and output-count matches. No comparison is imported by
compiler selection. The original cold TP/FP4 cohort remains at
`/tmp/lunaflux-model-bench-glm-lowbit-tp-20261013-v1`, not overwritten or relabeled.

Completed roots:

- `/tmp/lunaflux-model-bench-glm-bf16-tp-warm-20261013-v3`
- `/tmp/lunaflux-model-bench-glm-lowbit-tp-warm-20261013-v2`
- `/tmp/lunaflux-model-bench-glm-lowbit-tp-rows64-20261013-v1`
- `/tmp/lunaflux-reference-bench-vllm-glm-fp4tp-20261013-v1`

The BF16 control v2 failed before loading with `Address already in use`; its
logs and zero-swap release are retained. The v3 retry uses a fresh directory
after the socket clears. vLLM completes all six HTTP requests, then the requested
paired stop exits ingress normally and terminates headless rank1 with status137.
The watcher reports that nonzero teardown; this is not recorded as a clean
vLLM shutdown or an inference failure. Both GPUs are verified released before
the next native run. Minimum observed reference host-available memory is
17.405 GiB; no watchdog memory-floor breach occurs.

The rows64 native plan is 98,252,360,500 / 99,521,773,364 bytes (both below
100 GiB). Its systemd memory peaks are 59,306,721,280 / 59,637,116,928 bytes;
task swap peaks are zero and minimum host-available memory is 23.047 GiB.
Cgroup accounting and planned GPU residency are distinct measurements, not
additive or interchangeable. There is no overlapping GPU workload or payload
hash/integrity scan.

### Remaining measured costs, not a claim of vLLM parity

The fresh two-rank Nsight Systems trace is
`/tmp/lunaflux-model-bench-glm-lowbit-tp-trace-20261013-v1`. It is a **rows16**
diagnostic with two requests (one warmup), two model steps each; it is not the
rows64 end-to-end timing above. Actual kernel names show native grouped matrix,
quantization, and NCCL execution. Per-rank sums are:

| Kernel family | Ingress | Egress |
| --- | ---: | ---: |
| All kernels | 1614.321 ms | 1626.269 ms |
| 528 all-reduces (132 × 4 steps) | 517.732 ms / 32.1% | 416.539 ms / 25.6% |
| Routed expert matrix kernels | 352.832 ms / 21.9% | 357.126 ms / 22.0% |
| Expert activation quantization | 20.578 ms / 1.27% | 21.180 ms / 1.30% |
| Expert grouping and combination | 32.723 ms / 2.03% | 33.513 ms / 2.06% |
| Shared experts | 129.244 ms / 8.01% | 133.102 ms / 8.18% |

These are duration sums, not an additive explanation of the completion-time gap.
Collective kernel duration includes peer-readiness waiting; it is not pure CX7
wire time. The two ranks overlap and must not be summed as serial execution.
The next measured targets are collective submission/readiness and native FP4
matrix scheduling, not activation packing alone. A smaller kernel instruction
set or another IR layer does not establish equal performance.

The retained .nsys-rep files, generated SQLite/CSV and the external
`summarize-glm-tp-timeline.mbtx` reproduce these categories. No new Nsight Compute
instruction-counter claim is made from this timeline. The actual numerical
programs, representative kernel sanitizers, ABI lifecycle checks, full-model
completion, text comparisons and throughput remain separate validation levels.

### Final capacity and build-cache checks

The actual-source FP4 deferred-publication fixture was also exported with
capacity64, matching the final AOT envelope. In
`/tmp/lunaflux-native-fp4-partial64-20261013-v1`, live rows 1, 3, 16, 17, 64,
zero rows and reuse match the independent scalar packed-arithmetic oracle;
memcheck/full-leak, racecheck and synccheck report zero errors. The fixture has
hidden64/intermediate128/two experts, so this is capacity/ownership coverage,
not an independent full-model accuracy proof.

The first final full native suite stopped with a SIGSEGV in the NCCL collective
white-box fixture. LLDB located the null function call at
`collective_interop.c:364`. The fixture object was still dated Oct12, before the
Oct13 CUDA ABI table change, while the linked resources/interop objects were
new. A clean separate target directory passed all 27 CUDA/NCCL tests without
source changes. The mixed cached CUDA/NCCL package directories were moved to
`/tmp/lunaflux-glm-tp-clean-abi-20261013-v1/stale-debug-packages`, preserving the
failure for inspection, and the full suite was restarted after rebuilding them.
This is an incremental native-header dependency/cache failure, not a reason to
change collective arithmetic or suppress a test. Source-pinned Spark builds
and the physical runs above were clean builds.
