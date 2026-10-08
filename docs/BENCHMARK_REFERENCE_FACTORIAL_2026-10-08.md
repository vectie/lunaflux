# Reference attention + projection substitution: 4.5% less completion time

The controlled serving experiment has a positive, partial result. Keeping the
frozen LunaFlux model, scheduler, KV storage and serving graph, replacing
attention with an adapted pinned vLLM FlashAttention-2 implementation and the
selected output/down GEMMs with cuBLAS reduces median completion time from
**4898 to 4678.5 ms (4.48%)**, or **104.53 to 109.44 output tok/s**.

This is an **offline causal diagnostic**, not a production optimization or
model-quality admission. No production compiler/runtime source was changed.
vLLM and SGLang were not rerun in this experiment. The earlier incorrect
all-row attention rewrite and its withdrawn 2.63% gain are not used here; see
[the missing-write diagnosis](MIXED_ATTENTION_ROW_COVERAGE_DIAGNOSIS_2026-10-08.md).

## Question and controls

The question is whether better attention/projection implementations actually
reduce LunaFlux serving time, rather than whether isolated counters look better.
This follows the [equal-work attribution](BENCHMARK_EXACT_MIXED_WORK_2026-10-07.md)
and [projection-only substitution](BENCHMARK_REFERENCE_GEMM_SUBSTITUTION_2026-10-07.md).

- Spark .179, GB10 sm121, 48 SMs; Qwen3-0.6B BF16.
- Input vector `[8192,8192,8192,8192,8192,8192,8192,8192]`, output vector
  `[64,64,64,64,64,64,64,64]`, varied prompts, prefix reuse disabled.
- Same frozen serving artifacts and parent, copied diagnostic worker with only
  a private driver-loader substitution seam. No new scheduler or IR policy.
- One compiled shim shared by all timing arms. Mode is selected once at
  startup; no diagnostic comparisons or outlier capture in timed execution.
- One discarded warmup and three measured waves per fresh start. Two starts
  per arm, in order: control, attention, projections, both, both, projections,
  attention, control. Every request body is checked equal across arms.
- One GPU workload at a time. Worker 64 GiB/no swap; controller/compiler
  8 GiB/no swap; admission floor 32 GiB MemAvailable. Minimum sampled
  MemAvailable in timing was 103,732,576 KiB (about 98.93 GiB).

Raw filenames reuse the earlier runner's arm labels: **`output` means attention
only; `down` means output and down together; `both` means all replacements**.
The legacy `reference-swap.v1` JSON scope text mentions GEMM only. This table,
the shim mask, loaded source and trace define the factorial experiment; raw
receipts are preserved, not rewritten to conceal the inherited names.

## Unprofiled timing

| Substitution | Completion ms | Output tok/s | Median TTFT ms | Median TPOT ms | Less completion time |
| --- | ---: | ---: | ---: | ---: | ---: |
| None | 4898.0 | 104.53 | 1492.0 | 50.29 | — |
| Attention only | 4789.5 | 106.90 | 1459.0 | 48.83 | 2.22% |
| Output + down only | 4800.5 | 106.66 | 1418.5 | 49.58 | 1.99% |
| Both | **4678.5** | **109.44** | **1383.5** | **48.01** | **4.48%** |

These are medians of six waves per arm. TTFT/TPOT are medians over measured
requests; they cannot be added to reconstruct a wave's completion time.

| Arm | First start: three wall times, ms | Second start: three wall times, ms |
| --- | --- | --- |
| Control | 4866, 4882, 4888 | 4908, 4934, 4950 |
| Attention | 4759, 4780, 4784 | 4795, 4807, 4809 |
| Projections | 4779, 4795, 4804 | 4800, 4801, 4822 |
| Both | 4658, 4666, 4680 | 4677, 4680, 4685 |

Combined-vs-control fresh-start median reductions are 4.42% and 5.15% on
the two sides of the order. This is consistent direction across two starts,
not six independent starts or a broad confidence interval. The two isolated
effects sum to 208 ms; combined saves 219.5 ms. The small difference is not
enough to claim a separate interaction optimization.

For context only, the previous framework measurements were 4494 ms vLLM and
4545.5 ms SGLang. The new diagnostic would be 4.11% / 2.93% above those old
numbers, but **that is not a fresh paired framework comparison** and does not
establish a new production gap or parity.

## Actual dispatch and measured-wave GPU accounting

Two separate Nsight captures use the existing offline profiling parent and
the same diagnostic workers. These are not unprofiled throughput samples.
Each contains exactly 192 `cuGraphLaunch` calls: 96 warmup steps, then 96
measured steps. The analysis excludes the first 96 and startup vendor calls.
Both measured sequences have exactly matching names, order, grids, blocks and
shared-memory sizes for every **unaffected** kernel, step by step: 14,726 calls.
This checks unaffected launch work, not bitwise equality of intermediate data.

| Measured-wave activity | Control ms | Both ms | Savings ms |
| --- | ---: | ---: | ---: |
| Attention, including replacement adapter/no-ops | 3024.127 | 2917.275 | **106.851** |
| Selected output + down | 378.118 | 257.726 | **120.392** |
| All unaffected kernels | 1384.965 | 1408.280 | **-23.315** |
| Total summed kernels | 4787.210 | 4583.281 | **203.928** |
| GPU activity union, including copies/sets | 4785.873 | 4582.165 | 203.708 |
| Profiled client completion | 4898 | 4708 | **190** |

Original attention launches disappear in the combined trace, as do the two
eligible original projection geometries. The replacements are positively
observed: 2685 calls each to page adapter, mixed attention, decode partial and
decode combine; 1954 graph calls to the vendor projection kernel. Other
projection geometries remain untouched.

The gap between net GPU activity savings and profiled completion savings is
13.708 ms. Inactive time inside the first-to-last GPU envelope increases from
92.786 to 104.406 ms; outside that envelope contributes the remaining roughly
2.1 ms. This does **not** indicate a newly discovered large host bubble.
The separate unprofiled median savings are 219.5 ms, not the single profile's
190 ms. Do not combine numbers from these different timing populations.

Unchanged kernel activity increases 23.315 ms, mostly in the prefill/mixed
segment. Identical code/geometry does not imply identical timings after a
different preceding kernel. This capture does not distinguish cache, clock,
data-dependent behavior or run variability as the cause; it is not another
proven compiler defect.

### Why reference decode fails to improve the whole decode segment

Steps 0–32 contain prefill/mixed work; steps 33–95 are pure decode. The rotary
and QKV geometry transition and all unchanged launch signatures are retained
in `profile/comparison.json` for audit.

| Pure-decode attention component | GPU ms |
| --- | ---: |
| Original complete attention chains | 1593.095 |
| Reference partial + combine calculation | 1551.099 |
| Page-table adapter | 8.161 |
| Inactive mixed/prefill launches | 35.398 |
| Total adapted reference attention | **1594.658** |

The reference calculation saves **41.996 ms**, but the adapter plus inactive
prefill launches cost **43.559 ms**. The whole decode attention path is
therefore 1.563 ms slower, not faster. These costs are included in all results.
Prefill/mixed attention accounts for the attention gain: 1431.032 to
1322.617 ms, saving 108.414 ms.

There is a second bounded-adapter penalty: selected base down launches that
occur in pure-decode steps cost 3.584 ms originally and 16.572 ms as vendor
full-envelope GEMMs. The diagnostic geometry guard cannot use geometry alone
to distinguish those live-row domains. Overall projections still win because
their prefill/mixed activity falls from 374.534 to 241.154 ms.

Accordingly the first 33 steps' GPU envelope shrinks by 212.716 ms, while the
last 63 steps' envelope grows by 20.952 ms. **This experiment's gain is mainly
prefill/mixed, not a universal decode acceleration.** This is a measured reason
not to turn the diagnostic shim into a production fallback.

## Implementation boundary

Reference attention uses the pinned vLLM FlashAttention-2 inference source,
with traits `<128,64,128,4,false,false,bf16>`, 128 threads, 81,920 dynamic shared
bytes. Pure decode reinterprets the two GQA query heads as query rows and uses
three KV partitions plus the upstream four-row combine. Prefill/mixed is
causal and covers all live rows, including single-query companions.

This is **not** the byte-identical stock vLLM binary or host ABI. LunaFlux keeps
its 8-token paged KV layout. A GPU adapter converts the compact CSR page table
to a padded rectangular table; reference masked lanes can read padded slots.
The diagnostic launches both phase branches and lets the irrelevant branch
return. Its mixed grid remains `32×8×16`, including pure-decode no-op calls.
No adapter or branch cost is subtracted from reported speed.

cuBLAS replaces only base output `grid=512, block=256, dynamic_shared=8192`
and down `grid=256, block=128, dynamic_shared=0`. Both use M=2048,N=1024;
K=2048/output or 3072/down, BF16 IO and F32 accumulation, with reduced-precision
reduction disallowed. All other shapes pass through. The observed vendor
kernel is `nvjet_sm121_tst_mma_128x192x64_2_32x96x64_tmaAB_bz_TNNN`.
This is a vendor substitution, not a claim to reproduce vLLM's exact GEMM.

## Correctness: independent oracle, preserved failed cross-check

1. The full-output reference oracle uses zero Q/K and nonconstant V, giving an
   independent causal-mean answer. It covers 2048-token prefill, mixed vectors
   `[2030,1,1,1,1,1,1,1]` and `[91,1,1,1,1,1,1,1]`, C8 and C1 decode,
   shuffled physical pages, unequal histories and alternating poisoned outputs.
   **17,170,432 output checks pass**, with untouched padding and unchanged KV.
2. Memcheck, racecheck, initcheck and synccheck pass on the small mixed/decode
   oracle. This does not mean the complete serving application or later
   diagnostic outlier-capture hook received a full sanitizer campaign.
3. Separate projection shadow comparison checks **7,504,216,064 live values**:
   zero violations of `abs(delta) <= .01 + .02*abs(original)`, 444 nonbitwise
   values, maximum absolute difference .015625, clean explicit release.
4. Attention shadow comparison checks **7,548,829,696 live values**. It has
   **92 violations** of that cross-original tolerance. This gate remains
   **failed**, its output is preserved, and it is not reclassified as passing.
5. All 92 outliers' exact Q/K/V inputs were captured, with no truncation, and
   independently evaluated in FP64 on CPU at the same numeric bound, now against
   FP64 truth. Reference passes **92/92**, original **68/92**; reference is closer
   for 76/92. Maximum reference error is .00963572165308; original .0202647246809.
   The independent check supports exploratory timing despite the failed
   cross-original test. It is an outlier-selected audit, not a complete
   FP64 validation of every activation or whole-model quality.

All unprofiled timing runs return 64 tokens/request, empty runtime stderr,
successful drain/child closure and explicit release. Output sequences are
not uniformly repeatable even in control: within-arm mismatches against the
first same-row result are 11/40 control, 7/40 attention, 8/40 projections,
4/40 both. Against the first control vectors, mismatches are 22/48 attention,
12/48 projections, 24/48 both. These are whole-vector mismatch counts, not
error rates or quality scores. **No model-quality/production parity claim.**

Setup failures are preserved and excluded: MoonBit seam/parse and compile
errors; missing license filename; an unused dropout wrapper; stopped redundant
shim builds; an initial compact-page-table out-of-bounds read; missing captured
descriptor state; and a partial-vs-final attention ABI mismatch. The latter
two were diagnostic adapter bugs, not demonstrated production engine defects.
The corrected partial ABI is Q,K,V,workspace; the final-writer ABI is Q,O,K,V.

The projection-only runner initially hashed factorial `verify.so` rather than
the separately built and loaded `projection-verify.so`. Raw receipt is retained;
`PROJECTION_SHADOW_IDENTITY_CORRECTION.txt` binds the actual library separately.

## Architectural conclusion

There is now intervention-based evidence of a real kernel-chain contribution:
**attention and selected projections recover about 4.5% of completion time**.
It is not proof that all of the old approximately 10% gap is understood or fixed.
Neither more IR layers nor a speculative host-overlap rewrite follows from it.

The next production design should express live-row work domain and phase in
the existing pure execution/physical plan, then lower a phase-appropriate
complete chain. It must avoid launching a large inactive prefill grid during
decode, avoid paying a page-format conversion per layer where a compatible
layout can be planned once, and not select a full-envelope vendor GEMM merely
because the base launch geometry matches. Those are concrete, now-measured
adapter costs, not a warrant for a Qwen-specific runtime branch. Any production
implementation still needs full-domain correctness and a fresh paired serving
test; eliminating those costs is not yet an observed additional speedup.

The AKO measured-loop discipline shaped this work: one bounded four-arm
experiment, actual-dispatch checks, warmup separated from timing, explicit
negative results and no automatic promotion based on a kernel-only win.

## Evidence identities

- Remote root: `/home/wlc004s/lunaflux-reference-factorial-20261008.aj3VkY62`.
- GPU UUID: `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`; PCI `0000000F:01:00.0`.
- nvcc 13.0 SHA-256:
  `fbb111f057786ddd10ba723d993cc7dd43abf978b6baa32fedd3c9d806dc79e1`.
- Frozen serving bundle SHA-256:
  `f7afc9b5a0a0a6fa148ca027c26c3c4c9f14b6580550d743c443d3ade94fb58a`.
- Pinned reference image:
  `sha256:73e1b0f3230a9377a3109d20be7f8607963a41a715eb3000bc647f5f73c198ff`.
- Original `flash/flash_fwd_kernel.h` SHA-256:
  `242cbc331bd09d6ccfd023aae2d31461484b2a4801dcf66733b4151f9018622a`.
- One timing shim, `factorial-v4.so` SHA-256:
  `d6a0fd5b7c51df8fef3571ba1e352764763157011c0a8c552fe65f4a30ad3f72`.
- Copied diagnostic worker SHA-256:
  `3f2dd2d091a366ec3166dda87da3276ea3064f1a536d313c3a0fc92fc5854bf0`.
- Projection-only shadow SHA-256:
  `4577d02507b9db3a7ef0b254c6c0e17f94f6caabb949cf228883690f106d5c98`.

Helpers are under `benchmarks/gpu_pipeline/reference_factorial*`, with
`reference_attention_swap.cuh` and the conditional extension to
`reference_projection_swap.cu`. Production code and unrelated working-tree
changes are untouched.

The sealed archive contains **2978 checksummed files**, including raw requests,
the copied worker and shims, reference source/licenses, numerical inputs and
outputs, traces, and failed attempts. Build caches and model weights are omitted.
Archive SHA-256:
`988c7a8e4bbcca1dee81b46b2cf8f109f1cc74b44bd6926db519f28a605273fe`.

Downloaded without overwrite to
`/tmp/lunaflux-reference-factorial-verified-20261008.RnPCu4hz/measurement.tar.gz`.
The local archive hash matches; all 2978 extracted `FILES.sha256` entries
verified in the sibling `evidence/` directory. The GPU was idle after the
campaign. All seven new MoonBit automation helpers pass warning-denied native
checks; the three helpers with tests (finish, profile analysis, seal) pass 3/3.
Formatting is checked on those scripts. This is not a fresh full-project test
of the unrelated dirty working tree or a production release campaign.
