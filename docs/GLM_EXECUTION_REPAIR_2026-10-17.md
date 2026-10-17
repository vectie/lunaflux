# GLM whole-path repair

This work follows the full inventory in
[packed expert transport](GLM_PACKED_EXPERT_TRANSPORT_2026-10-17.md), not just
the failed producer/consumer experiment. Completion requires executable
selection and whole-model evidence, not new IR types alone.

The implementation boundaries remain numerical program → legal regions →
physical domains/layout/lifetimes → execution effects → private device lowering.
Comparison and performance observations remain external. No payload hashing.

## Work ledger and bounded experiments

| Region | Required repair | Acceptance boundary |
|---|---|---|
| Expert operands | Preserve valid-row masks through packing; combine bulk transfer, consumer geometry and fragment lookahead | Exact full chain, padding/tails, sanitizer, five alternating paired trials |
| Quantization/epilogues | Parallel global reduction and avoid redundant materialization without moving BF16 publication | Exact payload/scales and complete chain timing |
| Non-expert projections | Buffered operand transport through physical plans; shared-input region reuse | Ordered MMA fold preserved, odd tails, full selected matrix chain |
| KDA | Shared-input projections, joint convolution, efficient chunk implementation | State/chunk equivalence and short/long timing separately |
| Sparse attention | Propagate producer's written prefix; share selection work; compressed-history alternative | Malformed-list behavior retained for unconstrained inputs; causal/history tests |
| Hyper-connections | Fuse post/pre/norm across semantic region boundaries | Preserve each declared publication/rounding and state lifetime |
| Shared experts | Independent branch execution with explicit join | No alias hazards, cancellation and complete critical-path timing |
| Tensor parallel | Smaller reduction domain under explicit numerical law; collective capture | Numerical program change disclosed; rank agreement, cancellation and bytes/timeline |
| Head | Shard vocabulary with global deterministic argmax | Tie semantics and rank-tail timing including collective |
| Evaluation | Fixed-work controls, broader rows/history and full-model ablations | Timing/token vectors, actual selected source and memory observations |

First campaign: exact physical-domain repairs and buffered matrix transport;
at most three schedule variants, five alternating component trials per cell.
Only successful candidates proceed to combined full-model testing. Changes to
scale/reduction order are separate numerical alternatives, never justified by
agreement with vLLM alone. Failures and non-wins remain recorded.

The original head and worker containers stay running during source work. Stop
only those two for GPU measurements, retain the NFS container, and restart the
same containers afterward. Task limits: 100 GiB per Spark, independently
observed 16 GiB MemAvailable reserve, no task swap and no overlapping GPU jobs.
The original deployment is restored unchanged even though its original memory
configuration does not meet this benchmark reserve.

No implementation or performance completion is claimed by this ledger.

## First physical repairs and measured alternatives

The sparse index producer now exposes its maximum written prefix through its
immutable physical plan. Only consumers bound to that producer shorten their
traversal; generic arbitrary-list validation still scans the complete extent.
The selected-list storage stride, stable ordering and arithmetic are unchanged.
At history 64 the old consumer unnecessarily traversed 2051 slots instead of
the producer's maximum 67. Actual generated-source paired trials at 22 queries
improve median .227632 to .062645 ms; single-query .073776 to .022582 ms.
History 4096 is unchanged (.77828/.77849 ms). This is a sparse kernel result,
not whole-model throughput. Root: `lunaflux-glm-sparse-retained-prefix-20261017-v1`.
Malformed lists, odd dimensions, finite/tail checks and all three sanitizers
pass with full leak checking, bounded 2 GiB/no task swap.

`MatrixStaging` separates physical two-slot publication/retirement from the
ascending K16 numerical fold. Explicit GLM `-buffered-matrices` propagates the
same choice through export, runtime reconstruction and recurrent projections.
Standalone generated-source probes include unaligned and odd-stride operands;
all outputs match the retained implementation bitwise and all sanitizers pass.
Three K schedules were tested, not an unbounded search:

| Shared K | R1 median ms old/new | R22 median ms old/new | R64 median ms old/new |
|---|---:|---:|---:|
| 128 | .11458/.12089 | .18219/.15521 | .20131/.33128 |
| 64 | .110323/.112876 | .178185/.111222 | .195513/.213170 |
| 32 | .105600/.107102 | .194826/.146491 | .195793/.148196 |

The K128 and K64 schedules are not blanket winners. K32 goes to explicit
whole-model testing, not default promotion. Roots:
`lunaflux-glm-buffered-matrix-20261017-v{1,2,3}`. Actual K32 Nsight replay at
R22 shows instructions **5.973M to 8.418M**, static shared memory 9.216 to
14.336 KB, long-scoreboard 63.11% to 58.61%, barriers 20.26% to 20.86%, and
profiled duration 329.344 to 305.024 us. A second pair is 326.272/296.448 us.
Thus copy overlap helps, but does not remove barrier or address-instruction
cost; the profiled gain is smaller than the unprofiled component result.

The indexed expert alternative uses cooperative double buffering without
packed-row materialization or producer polling. Its source has the same eight
launches and activation allocation as indexed transport. At full 288-bank
R22/capacity the independent-input variant changes median 6.786 to 6.532 ms;
R64 13.152 to 12.734 ms. R3 remains noisy and is not a demonstrated win.
R1/R3/R22/R64 capacity and trimmed chains are bitwise, with poisoned padding;
R3 full-bank capacity memcheck/leak/race/sync pass under 8 GiB/no task swap.
Root: `lunaflux-glm-fp4-performance-indexed-buffered-20261017-v2`.
This is still individual asynchronous copies, not CUTLASS TMA or its wider
consumer geometry. The earlier separate-role experiment remains a non-win.

The finite nonnegative max tree removes seven CTA-wide barriers from the
global quantizer without changing payload/scales. Its isolated R22 expert
chain median 6.781 to 6.748 ms is within observed variation, not a claimed
throughput improvement. Root: `lunaflux-glm-fp4-performance-max-tree-20261017-v2`.
The earlier v1 also changed gate/up input sharing and is explicitly confounded;
do not attribute its result to the max tree. v2 uses the actual selected
independent gate/up staging. No numerical reduction/scale law changes.

These repairs do **not** complete the remaining cross-boundary HC fusion,
shared-expert overlap, collective capture/reduction-domain, head sharding,
shared-input KDA transport, compressed sparse-history and expert TMA/fragment
lookahead items above. Full-model results must be recorded separately below.

## Shared storage and the second pipeline level

The first buffered expert source allocated a separate shared ring for gate
and up despite their sequential lifetimes. The executable `SequentialScratch`
binding now reuses that storage. In the v3 indexed-buffered component, gate/up
shared storage returns from 27,648 to 13,824 bytes and active-warps occupancy
returns to approximately 46%. Nevertheless, the paired NCU replay is
4.2110/4.2182 ms (retained/new): **recovering occupancy did not recover a gate
speedup**. Instructions are 96.53M/111.88M; barrier stalls 18.68%/19.72%;
long-scoreboard 76.36%/75.80%. Down improves 2.2106/2.0700 ms while using more
instructions and shared memory. Unprofiled complete-chain medians improve only
6.763832/6.656288 ms at R22 and 13.14326/12.92713 ms at R64. All R1/3/22/64
capacity/trimmed comparisons remain bitwise; R3 full-bank capacity passes
memcheck/full leak/race/sync. Root:
`lunaflux-glm-fp4-performance-indexed-buffered-20261017-v3`.

Commit `8ae1c7b9` introduces independent register read-ahead ownership through
`ExpertGroupingPlan.fragment_slots` and the existing pure `OrderedOperandReads`
action plan. Two register slots load upcoming operands before consuming the
current ones, without changing the ascending K64 MMA fold. Global-to-shared
transport and shared-to-register reads are separate choices. The selection
reaches both GLM recurrent/sparse expert owners, startup budgets, generated
source and the explicit `-lookahead-experts` mode. Ordinary selection remains
one slot. Unsupported BF16/transport combinations reject rather than silently
ignoring a requested schedule. The generic expression-only BF16 renderer also
rejects buffered schedules without concrete operand-address contracts
(`b7d13e38`); address-aware dense/segmented lowering supports them.

Two read-ahead candidates were measured, not an unlimited search:

| Component candidate | R22 complete-chain median old/new ms | R64 old/new ms | Losing small shape |
|---|---:|---:|---|
| Indexed, two register slots | 6.750/6.663 | 13.143/12.963 | R3 .9183/.9468 |
| Buffered, two register slots | 6.767931/6.654664 | 13.144984/12.937799 | R3 .960227/.967754 |

Both compare with the frozen pre-max-tree baseline, so these are composed
physical changes, not isolated read-ahead gains. The separately measured max
tree was within variation. The read-ahead component roots are
`lunaflux-glm-fp4-performance-fragment-lookahead-20261017-v{1,2}`. All full-bank
R1/3/22/64 capacity/trimmed chains match retained payload/scales/output bitwise;
R3 capacity memcheck/full leak/race/sync are zero. Wrapped 704/576 dimensions
exercise three shared epochs and a partial final epoch; the buffered two-slot
source also passes all gates in `lunaflux-glm-fp4-performance-lookahead-wrap-20261017-v2`
under 2 GiB/no swap. Wrapped v1 was prepared with a mismatched full-width
baseline but **never executed**. It is retained as invalid preparation; the
probe now checks paired geometry/storage contracts at compile time.

The composed two-level NCU component capture reports:

| Metric | Gate/up retained/new | Down retained/new |
|---|---:|---:|
| Replay ms | 4.255168/4.201248 | 2.150784/2.068256 |
| Warp instructions, millions | 96.530/109.476 | 49.418/57.488 |
| Static shared KB | 13.824/13.824 | 6.912/13.824 |
| Active warps % | 46.68/46.98 | 81.32/45.68 |
| Barrier stalls % | 20.06/19.94 | 13.18/16.12 |
| Long-scoreboard stalls % | 75.22/75.46 | 72.69/78.09 |
| L2 read sectors, millions | 36.489/36.486 | 18.231/18.237 |

This does not demonstrate elimination of load waits. The supporting
instructions remain higher with essentially identical read traffic. Compared
with buffered v3, instruction counts fall slightly, but different captures
cannot establish the isolated timing contribution. CUTLASS's TMA granularity,
persistent work distribution and wider two-consumer geometry are still absent.
The measured second pipeline level is implemented; the larger expert gap is
not solved by merely calling this implementation a pipeline.

The retained installed CUTLASS component provides a useful counterexample to
single-metric explanations. Its gate/up is 3.247 ms with **lower** active-warps
occupancy (22.9%), **higher** L2 reads (45.808M sectors) and 79.031M instructions.
Its tensor activity is 21.1% versus this native candidate's 2.12%, but the
128-row versus 16-row physical tiles execute different padded mathematical
work. That activity ratio is not a tenfold useful-throughput claim. For down,
CUTLASS has **more** instructions (64.373M versus native 57.488M) and still
finishes sooner (1.745 versus 2.068 ms). Instruction count alone is therefore
also insufficient. Transfer issue granularity, producer/consumer dependency
edges, fragment schedule, work distribution and publication all matter.
The bulk-transfer producer changes where waiting occurs; warp-averaged stall
percentages across different warp roles are not fractions of wall time.
These are component observations from the installed implementation, not a
decomposition of the exact vLLM serving gap. The wider tile, TMA and persistent
scheduler form a coupled alternative; borrowing only one is not equivalent.

## Full-model ablation protocol

All native cohorts use the identical six literal prompt ID arrays, one excluded
warmup, TP2/PP1, row capacity 64, history 64, shape-aware projections, queued TP
and compute-interval graph replay. Input token counts are 22/23/22/23/24/36;
retained native output counts are 4/3/8/8/8/3. Startup is excluded. The native
CLI completion/retirement boundary and reference buffered HTTP boundary remain
different; BF16 expanded native KV and FP8 reference MLA KV also remain
different. Comparison remains external to the compiler.

The first fresh control totals 4.895 s versus the retained 4.507 s, with the
first three requests unusually slower. Its raw run is preserved, not selected
as a favorable denominator. A preliminary explanation blaming simultaneous
preparation was withdrawn: preparation ended during loading, before measured
requests. A second control is required. The prefix/max-only and buffered
matrix cohorts total 4.466 and 4.463 s. Their 3 ms difference is not a reliable
whole-model buffered-projection win, despite the isolated R22 projection gain.
All six native output ID arrays are unchanged. Later combined, read-ahead,
repeat-control and fresh-reference results belong below, not inferred here.

Raw roots share `/tmp/` and suffix `-20261017-v1`:
`lunaflux-model-bench-glm-repair-{control,prefix,matrices,combined}`;
the repeated control uses `v2`. Every result is a cohort, not a confidence
interval or a correctness certificate against a different framework.

## Coverage is not complete

| Family | What actually changed in this campaign | Still not implemented here |
|---|---|---|
| Sparse execution | Producer-bounded selected-index traversal | Once-per-query shared compaction; compressed MLA history representation |
| Global quantization | Exact two-level intra-workgroup maximum; seven fewer block barriers | Multiple-workgroup global-amax; producer epilogue/quantization fusion |
| Non-expert matrix | Two shared operand slots, three bounded K alternatives, selected recurrent projection integration | Full coverage of sparse/shared/dense projection families; physically merged shared-input regions |
| Routed experts | Correct packed valid rows; sequential ring reuse; indexed buffering; independent register read-ahead | Bulk TMA lowering and wider persistent consumer schedule; fewer publication/materialization boundaries |
| KDA | Buffered recurrent projection alternative only | Shared-input reuse across projection regions, joint convolution, improved chunk-prefill selection |
| Hyper-connections | Unchanged in this campaign | Cross-sublayer post/pre/norm composition |
| Shared experts | Unchanged in this campaign | Explicit fork/join execution and measured overlap |
| Tensor parallel | Existing queued collectives and interval graphs retained | Collective-inclusive capture; justified numerical alternative for reduced message domain |
| Vocabulary head | Unchanged in this campaign | Distributed vocabulary rows and deterministic global selection |

The single-block quantizer's reduction tree is not a multi-SM global reduction.
Likewise, fragment read-ahead does not create TMA or wider work ownership.
Neither feature presence nor a green test suite closes the other rows above.
No default promotion is justified solely by an isolated component win.

## Completed native full-model ablations

The repeated control is 4.520 s, close to the retained 4.507 s. The first
4.895 s control remains in the evidence; its slower requests have no established
cause. All six output ID arrays are unchanged in every candidate below.

| Input tokens / native output tokens | Repeated control ms | Prefix/max ms | + buffered matrices ms | + buffered experts ms | + register read-ahead ms | Final TTFT ms |
|---|---:|---:|---:|---:|---:|---:|
| 22 / 4 | 586 | 576 | 585 | 582 | 563 | 294 |
| 23 / 3 | 506 | 503 | 503 | 496 | 496 | 310 |
| 22 / 8 | 934 | 921 | 919 | 913 | 906 | 304 |
| 23 / 8 | 943 | 941 | 937 | 923 | 933 | 323 |
| 24 / 8 | 959 | 945 | 935 | 944 | 936 | 330 |
| 36 / 3 | 592 | 580 | 584 | 584 | 585 | 398 |
| Total | **4520** | **4466** | **4463** | **4442** | **4419** | **1959** |

The complete candidate takes 2.23% less time than the repeated control, or
1.95% less than the retained control. TTFT totals change 1.983 to 1.959 s
(1.21% less). Completion-after-first-token totals are approximately 2.537 to
2.460 s; millisecond-rounded telemetry makes sub-millisecond sums inexact.
This is one full-model cohort per candidate, not a statistical confidence
interval. In particular, the 3 ms matrix increment and 23 ms read-ahead
increment cannot independently establish reliable whole-model wins.

Final root: `/tmp/lunaflux-model-bench-glm-repair-lookahead-20261017-v1`.
The source-pinned runtime is
`/tmp/lunaflux-glm-expert-tp-repair-lookahead-20261017-v1`, built from
`8ae1c7b9`. Both export and execution record `recurrent_matrix_buffered=true`,
`operand_slots=2`, `expert_fragment_slots=2`, and
`expert_operand_transport=buffered`; both ranks prepare 91 interval graphs.
The alternatives therefore reach the running model, not just test sources.
The three preceding runtime/run roots use `repair-prefix`, `repair-matrices`
and `repair-combined`. No request-path profiling or comparison is added.

## Selected full-model timeline: what actually improved

New trace:
`/tmp/lunaflux-model-bench-glm-repair-lookahead-trace-20261017-v1`.
Retained trace:
`/tmp/lunaflux-model-bench-glm-stable-ranks-trace-20261017-v1`.
Both use the same literal short prompt, two output tokens and one excluded
warmup. The offline `analyze-glm-selected-trace.mbtx` reports actual symbol
families, matrix roles, enqueue intervals and step spans for both ranks.
The table uses **one rank**, egress warmed prefill step 3; never add TP ranks.

| Selected region | Retained ms | Candidate ms | Interpretation |
|---|---:|---:|---|
| Routed expert gate/up | 74.271552 | 74.188384 | Largest component remains essentially unchanged |
| Routed expert down | 36.279680 | 34.279296 | 5.51% less, about 2 ms recovered |
| Product global quantizer | 8.150592 | 8.176160 | No demonstrated gain from the intra-CTA max tree |
| Recurrent merged-input projections | 22.254656 | 22.527168 | Slightly worse; isolated projection gain does not transfer universally |
| Recurrent output projection | 9.271264 | 8.149696 | 12.10% less, about 1.12 ms recovered |
| Sparse attention leaf | 1.549728 | .308256 | 80.11% less, about 1.24 ms recovered; prefix repair is selected |
| Shared expert family | 29.179168 | 29.110880 | Unchanged implementation |
| Hyper-connection family | 17.305792 | 17.314336 | Unchanged implementation |
| Collective family | 23.376736 | 24.074464 | Includes peer waiting; not pure wire-transfer cost |

Ingress independently reports gate/up 74.128224 ms, down 34.236 ms and sparse
attention .309824 ms in its warmed prefill. The winning effects therefore do
not depend on choosing one unusually favorable rank. Egress decode also
reduces down 4.672000 to 4.333472 ms and sparse attention .827712 to .225536 ms.
Gate/up decode changes 9.756416 to 9.536704 ms, a much smaller effect.

This explains why the whole-model gain is small without inventing an
unmeasured cause: an 80% leaf improvement saves only about 1.24 ms in this
short prefill, while gate/up still costs about 74 ms, shared experts 29 ms,
and hyper-connections 17 ms. The changes do not address those latter critical
regions. The three measured prefill savings above total roughly 4.36 ms,
not 80% of the model. They are profiled region deltas, **not** an additive
decomposition of the unprofiled vLLM completion gap.

The candidate egress prefill span is 274.909 ms with 13.320 ms without a kernel;
the retained span is about 274.785 ms with 9.112 ms without a kernel. Both
execute 91 graphs and 90 collectives, with no `StreamQuery` before the final
collective. The candidate decode collective sum increases 22.093 to 42.054 ms
while its total profiled step becomes shorter. These are rank/host-wait and
instrumentation contexts, not evidence that the unprofiled TP path regressed
or that its waiting problem is solved. Do not use profiled step span as the
unprofiled end-to-end speedup.

### Source comparison changes the explanation

The installed reference `reference-glm-kda.py` explicitly clears `quant_config`
for KDA projections. These projections are BF16 in both frameworks; the
reference's speed must not be attributed to four-bit KDA weights. Its
physically merged input projection differs from our disjoint virtual matrix
regions even though each is exposed as one launch. The measured candidate
does not fix that shared-input reuse gap.

For experts, the counter table above shows unchanged read volume and persistent
load dependencies despite the extra pipeline level. CUTLASS's wider persistent
schedule, bulk transfers and distinct producer/consumer roles remain a coupled
alternative, not something achieved by adding a second buffer. Separate
quantization/publication, shared-expert overlap, cross-boundary hyper fusion,
collective capture and head sharding are independent remaining differences.
Some reference capabilities are present in source but still lack proof that
this serving invocation executes them; keep that distinction rather than
assigning every missing percentage to a convenient hypothesis.

## Validation and memory for native work

Full native suite: **5175/5175 pass**. Warning-denied native check passes with
the unchanged migration exclusions `-20-25-29-35-79-92`; `moon info` reports
1057 existing migration warnings and zero errors. Affected compiler/expert/GLM
tests pass 55/55, matrix-guard packages 56/56, benchmark runner 3/3 and probe
runner 1/1. The clean `8ae1c7b9` ARM build reports 747.0 MiB peak and zero swap.
The changed CUDA sources have the deterministic, sanitizer and leak coverage
recorded above; this does not imply every long-context shape was tested.

Native owners remain capped at 100 GiB each with zero task swap. Minimum
observed host available memory is 22.649/22.514/22.565/22.636 GiB for
prefix/matrices/combined/read-ahead respectively, above the 16 GiB reserve.
The final startup plans are 98,648,722,228 and 99,918,135,092 bytes, each
below 107,374,182,400 bytes. Cgroup process peaks are not substituted for
total unified-memory accounting; the independently sampled host reserve is
required as well.
The selected trace also passes the reserve guard. No benchmark overlaps
another GPU owner, and no payload hashes or integrity scans are performed.

Fresh reference timing and original-service restoration are recorded below
after their actual completion, not inferred from earlier runs.

## Fresh vLLM comparison: the major gap remains

Reference root:
`/tmp/lunaflux-reference-bench-vllm-glm-repair-20261017-v1`.
The installed reference is unchanged: vLLM
`0.1.dev20051+g487ecf187`, `FLASHINFER_CUTLASS` NVFP4 experts,
`FLASHINFER_MLA_SPARSE_SM120` with FP8 MLA KV, and PYNCCL TP2/PP1.
Context limit 256, C1, maximum output 8, prefix cache and speculation off,
one excluded warmup. The same six literal input-ID arrays are checked by the
offline comparison. All six measured requests return HTTP 200.

| Category | Input tokens | Native / reference output tokens | Native / reference completion ms |
|---|---:|---:|---:|
| Fact | 22 | 4 / 4 | 563 / 385.359 |
| Arithmetic | 23 | 3 / 4 | 496 / 390.374 |
| Chinese | 22 | 8 / 8 | 906 / 656.062 |
| JSON | 23 | 8 / 8 | 933 / 672.195 |
| Code | 24 | 8 / 8 | 936 / 670.881 |
| Retrieval | 36 | 3 / 3 | 585 / 375.941 |

The **five equal-reported-output-count cases** total **3.923 s native versus
2.760438 s reference: native is 42.12% longer**. The repeated native control
on those same five cases is 4.014 s (45.41% longer than this fresh reference).
All six sum 4.419 versus 3.150812 s, but the arithmetic case has unequal work;
do not present its 40.25% ratio as an equal-work result. The previous reference
fact answer emitted five tokens; this fresh one emits four, so the earlier
four-case subset cannot be silently reused. The report retains both cohorts.

The machine-readable external report is
`/tmp/lunaflux-model-bench-glm-repair-lookahead-20261017-v1/comparison-fresh-reference.json`.
The five-case aggregate produces 31 tokens in each framework, or 7.90 versus
11.23 output tokens per completion second **including prefill**; this is not
decode-only throughput. JSON output text still differs at equal count, and
eight-token truncation is not answer-quality validation. CLI completion versus
buffered HTTP, different KV representation/precision, generated token paths
and single-cohort variation remain disclosed limitations. No matched vLLM
TTFT is available from these buffered requests. The original reference loader
also warns about gate/up global scales; comparison is not a correctness oracle.

Minimum observed reference host available memory is **17.247807 GiB**, above
16 GiB. Reference owners are capped at 96 GiB with no task swap; native owners
remain at 100 GiB. The query controller stops the pair only after collecting
all six responses. Head exits zero; worker exits 137 during that deliberate
teardown. The memory watcher then reports its generic failed-rank/floor error
with **both hosts already above 117 GiB available**. The retained observations
show no reserve breach; that post-stop watcher error is not evidence of OOM.
Both GPU owner lists were empty before original-service restoration.

This completes the first bounded physical-repair campaign, **not the entire
work ledger**. Source presence, 5175 passing tests, and small measured gains
do not close the remaining architecture rows. In particular, the largest
gate/up component is still flat in the selected full model. Repeating the
same buffering variants would not justify a new conclusion; the remaining
coupled transport/consumer and cross-region changes need their own executable
implementations and matched gates. They are not being labeled fixed here.
