# Grouped experts and tensor parallelism

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
