# Prefill score deforestation

The long-prefill investigation separates fixed 256-token forward granularity
from the cost of each attention fold. The first lowering correction realizes
the existing backend-neutral `FuseScoreTransform` region: compose scale and
causal mask with the softmax consumer instead of materializing another shared
score tensor. This is map/fold deforestation, not a model-specific heuristic.

The CUDA lowering already requires that region. Its terminal implementation
may use CUDA primitives, but model builders, scheduling, and the semantic
optimizer remain device-neutral. Keep the original matrix dot, softmax
reduction order, probability type, and explicit float multiplication rounding.
Do not introduce runtime tuning or change numerical tolerances.

The second change factors the shared right operand of the query-map dot:
`map(query, fold(reduction, dot(query, key)))` becomes a reduction fold over
a tuple of query accumulators. Each accumulator keeps its original reduction
order. The semantic optimizer records operand sharing; CUDA chooses the
matrix-fragment granularity and retains one accumulator per query subtile.
This trades additional accumulator registers for fewer shared K loads, so
identical-shape paired measurements are required before claiming a benefit.

## Implemented changes

- `02dfd16`: deforest the scale/mask intermediate into its softmax consumer.
- `67f07be`: add the pure `FactorSharedKeyAcrossQueryMap` pass and its CUDA
  lowering. The pass is idempotent; tests include order-sensitive floating
  point inputs to distinguish interchange from reduction reassociation.

Neither change has model-name branches, runtime compilation, request-path
tuning, or extra token-step allocations. Both apply to the generic attention
compiler. This follows the MoonBit implementation/refactoring guides: focused
packages, immutable optimization values, targeted tests, and separate commits.

## Detailed physical experiment

RTX 5060 Ti, UUID `GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`,
PCI `00000000:17:00.0`, CUDA 13.1.115. Runs are sequential; no other GPU
workload was active. The old exporter is `f69e798`; the combined new exporter
is `67f07be`. The numeric fixture, tile dimensions, launch geometry and probe
are identical within each pair.

Each probe uses 10 warmup launches and nine samples of 40 launches, measured
with CUDA events. Three independent old/new trials alternate order. Values
below are medians of the three within-trial medians. These are isolated
synthetic attention timings, not Qwen token throughput.

The final probes use production flags: `-O3 --fmad=false --ftz=false
--prec-div=true --prec-sqrt=true --maxrregcount=128`. Earlier retained runs
used the probe's default FMA setting; do not mix their timings with this table.

Primary schedule: query tile 32, KV tile 32, head dimension 128, 256 threads,
45,056 bytes shared memory. Query heads 16, KV heads 4, page size 16.

| Query tokens | Context tokens | Old µs | New µs | Old/new |
|---:|---:|---:|---:|---:|
| 16 | 512 | 72.12 | 63.72 | 1.132x |
| 16 | 1024 | 140.58 | 124.06 | 1.133x |
| 16 | 2048 | 275.24 | 242.49 | 1.135x |
| 16 | 4096 | 547.28 | 481.66 | 1.136x |
| 64 | 512 | 89.73 | 80.76 | 1.111x |
| 64 | 1024 | 172.87 | 155.70 | 1.110x |
| 64 | 2048 | 339.30 | 305.16 | 1.112x |
| 64 | 4096 | 672.42 | 605.05 | 1.111x |
| 128 | 512 | 120.36 | 108.57 | 1.109x |
| 128 | 1024 | 233.50 | 210.22 | 1.111x |
| 128 | 2048 | 460.77 | 414.03 | 1.113x |
| 128 | 4096 | 915.60 | 821.81 | 1.114x |

The wider query-tile-64 schedule also improves under these flags: 1.173–1.183x
for 16-query long-context probes and 1.084–1.091x for 64/128-query probes. This
does not establish that it beats the query-tile-32 schedule; it does not here.

Disassembly of the default-flags paired probes shows the mechanism:

| Query tile | Static generic `LD.E` instructions | Static `BAR.SYNC` | Static `HMMA` |
|---:|---:|---:|---:|
| 32 | 160 → 128 | 13 → 12 | 40 → 40 |
| 64 | 320 → 224 | 13 → 12 | 80 → 80 |

These are static opcode counts, not measured dynamic instructions or memory
transactions. The 32-query probe's registers increase from 40 to 58, with
zero spill stores/loads. Retaining more independent accumulators has a real
resource cost; the measured improvement is not evidence of free register use.

## Full Qwen A/B

The final A/B rebuilds the four serving executables from `67f07be` for both
arms. Only the prefill modules differ: the old aligned-partial bundle versus
newly emitted prefill, wide-prefill, and partitioned-prefill modules. All
other modules, weights, release geometry, 256-token chunk bound, greedy
sampling, and input IDs are held fixed. No measured projection row variants
are newly installed. Production deployment was not touched.

Qwen3-0.6B BF16, one warmup and two measured trials per cell; means of trial
output throughput include prefill and client overhead but exclude startup.
Prefix reuse is disabled and EOS is ignored for fixed output counts.

| Input | Output | C | Old tok/s | New tok/s | Change |
|---:|---:|---:|---:|---:|---:|
| 59 | 256 | 1 | 232.62 | 232.31 | −0.14% |
| 59 | 256 | 8 | 968.78 | 970.85 | +0.21% |
| 128 | 128 | 1 | 222.80 | 223.00 | +0.09% |
| 128 | 128 | 8 | 893.93 | 895.10 | +0.13% |
| 512 | 64 | 1 | 179.27 | 180.79 | +0.85% |
| 512 | 64 | 8 | 502.45 | 504.44 | +0.39% |
| 1528 | 32 | 1 | 86.37 | 88.40 | +2.35% |
| 1528 | 32 | 8 | 126.39 | 129.98 | +2.84% |

For the longest prompt, mean TTFT is 214.5 → 204 ms at C1 (−4.90%) and
1027.875 → 996 ms at C8 (−3.10%). C8 post-first-token spacing is
31.077 → 30.286 ms; this includes overlap with other requests' prefills, not
just isolated decode execution. Short-cell sub-percent changes are noise-scale
in this small sample; this is not a confidence-interval or tail-latency study.

The separately retained score-only A/B (`02dfd16`) improved longest-prompt C8
throughput by only 0.97%; longest-prompt C1 regressed 1.07%. The combined
result above, not that first isolated change, is the completed implementation.

Against the earlier same-day measurements in
[the three-engine report](QWEN_PERFORMANCE_DISTANCE_2026-09-08.md), vLLM and
SGLang remain about **3.41x and 3.23x** faster in longest-prompt C8 throughput.
Those competitors were not rerun in this compiler A/B; this is an indicative
cross-run comparison, not a fresh three-engine campaign.

## Correctness and remaining bottleneck

- 106 affected native tests passed with warnings denied; formatting and
  interface generation passed. This is not a claim about the unrelated full
  working-tree suite.
- Paired probe output files are byte-identical at every tested element, for
  three trials across each tested tile schedule. Short/ragged cases use an
  exhaustive scalar reference; long contexts use deterministic reference
  sampling. Existing tolerances were not relaxed; nonfinite outputs now fail
  explicitly. Production-flags probes cover the two matrix schedules.
- Memcheck, racecheck, synccheck and initcheck pass for the primary schedule,
  including a repeat under production floating-point flags.
- Every measured Qwen request completed with the exact requested output count.
  The existing 59/256 C8 cell still has three distinct token sequences in both
  arms; other cells have one. This does not prove cross-engine equivalence or
  resolve the pre-existing concurrency-dependent sequence variation.

**The long-prefill problem is reduced, not fixed in full.** The kernel gain
does not remove repeated full-model forwards or the MLP/projection cost.
The unchanged 256-token bound still requires at least six prompt chunks for
1,528 tokens. The earlier detailed trace also showed mixed forwards sending
decode rows through the matrix-prefill route. Neither issue is changed here.
The next controlled experiments should vary legal AOT/workspace-backed prefill
granularity and separate mixed-batch phase work, then measure their effect on
TTFT and decode interference. Adding more equational passes alone cannot
remove those execution-plan costs.

## Reproduction and retained outputs

Remote roots: `/dev/shm/lunaflux-score-fusion-20260908-r1` (score-only) and
`/dev/shm/lunaflux-score-fusion-20260908-r2` (combined). `production-flags`
contains the final matched-flags probes; `serving` contains the Qwen A/B.
Raw request JSON/SSE, outputs, source archives, generated kernels, recipes,
build logs, sanitizer logs, disassembly and MoonBit runners are retained.

Archive: `/private/tmp/lunaflux-prefill-compiler-results-20260908.tar.gz`.
SHA-256: `bbd08d95e42219a3a4dcd2e919c666fca5f5cef2afd108398cf58b16ca815bcf`.
All owned benchmark server groups were stopped and the final GPU process
list was empty.
