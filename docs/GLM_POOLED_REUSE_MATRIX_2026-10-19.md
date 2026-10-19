# Pooled-key reuse and matrix sparse attention

Status: compiler implementation and measured full-model experiments completed;
online promotion is blocked by service qualification. Restoration of the prior
build also fails the host-reserve gate. **The GLM endpoint is offline.** No
candidate described below is claimed to be an online runtime.

## Measured scope

The frozen baseline is the actual source retained at
`/tmp/lunaflux-long-paged-sparse-20261019-v5/next.cuh`. Only its symbol namespace
is changed for side-by-side replay. The probe uses 64 query rows, 32 heads,
256-dimensional Q/K/V, 8,192-token logical history, 513 physical 16-row pages
and 2,048 selected keys plus the incomplete pooling tail.

The existing 7,973-input/one-output HTTP result is 55.779 s versus vLLM
17.656 s (3.16x). The earlier roughly 10% gap described short CLI workloads,
not this long HTTP prefill. See GLM_LONG_PREFILL_SELECTION_2026-10-19.md.
No new full-resident software timeline is attempted: previous attempts
violated the 16 GiB host reserve and were stopped.

## Architecture and experiment sequence

1. PoolUpdatePlan gives one writer to each physical pool touched by the
   current write footprint. Unchanged pooled keys remain resident. This is
   independent of request ordering and still recomputes same-generation
   overwrites. Resident validity is initialized once, outside the token path.
   Page-table inputs and persistent read/write state are explicit effects.
2. AttentionAccumulationLaw::TiledFloat32 explicitly permits parallel F32
   reductions and matrix products. Ordered arithmetic stays the default.
   BF16 probability publication remains in the numerical program.
3. SelectedQueryReuse owns bounded score storage, query cohorts and
   per-sequence ownership. Every row retains its own selected-key mask and
   causal boundary. A mixed-sequence cohort is partitioned, not treated as
   one request. Device lowering implements matrix layout; no model,
   reference output or benchmark dependency enters IR.
4. The option propagates through the sparse projection owner, decoder
   slice/stage and AOT/runtime service configuration. The explicit service
   value is `sparse_attention_accumulation: tiled-f32`. Code presence is not
   serving-selection evidence: export and runtime must be tested.

The query-reuse schedule traverses a dense physical key domain with each
row's sparse mask. This trades extra dot arithmetic for shared K/V reads.
It stores F32 scores, publishes BF16 probabilities, then performs a shared
value matrix product. Its larger scratch allocation must pass the full
100 GiB/Spark and 16 GiB host-reserve gate before deployment.

## Preserved experiments

All roots below are /tmp/lunaflux-long-paged-sparse-20261019-vN.

| Version | Change | Observation / decision |
|---|---|---|
| v6 | Incremental pool ownership, serial duplicate-writer scan | Pool 0.00655 to 0.0127 ms: rejected schedule |
| v8 | Parallel pool ownership; one live matrix query row | Pool about 0.00406 ms; attention about 14.7 to 13.7 ms, modest benefit |
| v9 | 16-query history reuse, serial softmax | Normal timing about 14.3 to 8.5 ms; **not qualified**: sanitizer exposed an empty-tile flag-reader race and output instability |
| v10 | Reader retirement before empty-tile reuse; parallel softmax | Normal late probe about 14.5 to 6.5 ms; full 64-row memcheck/leak/race/sync/init gates pass |
| v11 | Cohort prefill plus row-wise dispatch below 16 active queries | Late attention 6.394–6.700 ms versus 14.254–14.367 ms; full 64-row sanitizer gates pass |

v8 Nsight Compute capture (counters-next.ncu-rep): old/new attention
executed 1.737/1.072 billion warp instructions but read 145.35/141.97 million
L2 sectors (4.65/4.54 GB). Profiled time 14.294/13.326 ms. Long-scoreboard
contributions were 61.5%/57.5%; barriers 21.3%/23.3%. These are warp-stall
metrics, not additive fractions of whole-service time. Index selection
remained about 7.55 ms with 212 million instructions and 12.1% active-warps
occupancy. Pooling microseconds cannot explain seconds of service latency.

v9's failure is retained rather than relabeled as an arithmetic difference:
empty tiles could advance and overwrite the shared flag before all readers
retired. v10 adds reader retirement before the uniform skip. Normal v10
output differs from ordered accumulation, as permitted by a different
numerical program; this is not a bitwise-equivalence claim or a waiver.

## Qualification boundary

Comparison remains external. Diagnostics report exact selected indices,
all-output error statistics, an independent F64 attention calculation with
the declared BF16 probability boundary, and sanitizer results. A bounded
analytic fixture is not full-model correctness. Mixed sequences, invalid
selection, page-generation reuse, tails and deterministic replay require
separate coverage. Full-model complete answers and matched unprofiled timing
remain necessary before changing the online service.

No payload hashes or integrity scans are performed. Only task-owned paths
may be committed; unrelated toolchain edits must remain untouched.

## Executable qualification

Implementation 2fa80cd2 is explicit, not an unconditional default numerical
change. Clean pinned ARM build at
`/tmp/lunaflux-checkpoint-build-glm-pooled-matrix-20261019-v1` passes
(1.2 GiB reported peak, zero swap). The full-model AOT/configuration at
`/tmp/lunaflux-glm-http-pooled-matrix-20261019-v1` selects tiled-f32 on both
ranks; all 11 sparse-layer sources contain the cohort dispatch. Context8192,
8 slots,16 lanes,513 physical pages and64 prefill rows are unchanged.

`/tmp/lunaflux-query-reuse-20261019-v2` runs all rows/heads against an
independent F64 calculation on 12 fixtures: mixed sequences, GQA, noncontiguous
pages, generation mismatch, malformed selections, empty selections/counts,
1/15/16/17/31 active rows and inactive tails. Three repeated executions are
bitwise deterministic within each selected program. Full fixtures pass
memcheck/leak/race/sync/init gates. A BF16 rounding envelope on an analytical
fixture is not a relaxed model tolerance or full-model quality evidence.

`/tmp/lunaflux-paged-sparse-pool-reuse-20261019-v2` passes five stateful
mixed-request/reorder/generation-reuse steps, plus a same-generation overwrite
of a completed pool. The updated pool matches recomputation and untouched
pools remain bitwise. All sanitizer/leak gates pass.

v10's counters show the substantive traffic reduction: attention L2 read
sectors145.51M to37.38M (4.66GB to1.20GB), warp instructions1.737B to260.33M.
NCU durations14.524/7.020ms are separate from unprofiled event times.
Long-scoreboard remains59.3%, barrier12.7%, active warps42.6%. Higher occupancy
is not the objective: the smaller data demand is what this schedule changes.
The current index leaf still takes about7.6ms and remains a distinct frontier.
Do not attribute the complete HTTP gap solely to these synthetic-input leaves.

## Full-service memory finding and correction

Initial explicit 2fa80cd2 warm one-output vector (165/2085/6181/7973 input
tokens) completed in 1.221/11.247/32.939/44.459 seconds. The separate first
vector was 1.270/11.205/32.857/43.286 seconds. These are not confidence
intervals. The subsequent complete-answer C4 test did **not** qualify:
rank0 MemAvailable reached 16,428,828 KiB (15.67 GiB), and the external
watchdog stopped both ranks. GPU memory stayed below 100 GiB and both owned
units reported zero swap. The incomplete responses are retained at
`/tmp/lunaflux-glm-service-benchmark-pooled-matrix-answers-20261019-v1`.

The new dense score scratch was independently allocated by all 11 sparse
layers even though their lifetimes do not overlap. A pure `SerialScratch`
storage plan now budgets the maximum, not the sum, for explicitly serialized
phases. The decoder lends one allocation through the projection/attention
frames, closes every borrower before the allocation, and never aliases
persistent KV, pooled keys or selected indices. Expert producer forks do not
access it and join within their layer. This saves 640 MiB per rank at this
geometry without shrinking context/admission capacity or changing arithmetic.
The existing generated CUDA/AOT source is unchanged. This ownership correction
requires a fresh full-service memory/concurrency retest before deployment.

The correction is committed as0ee75887. Clean pinned ARM build succeeds
(1.2 GiB peak, zero owned-unit swap). Both exported rank sources compare
directly equal to2fa80cd2 and the existing AOT files are reused without GPU
compilation or payload hashing. The local phase suite passes5310/5310 with
the existing migration-warning exclusions; targeted `moon info` reports1065
existing migration warnings and zero errors. Bare warning-denied validation
remains blocked by the already recorded unrelated migration diagnostics.

## Remaining source differences (not an additive timing attribution)

The actual installed vLLM variant is retained at
`/tmp/lunaflux-glm-long-indexer-vllm-20261019.py`; it is not identical to
the model coverage in the local `../vlm` checkout. Its pooled index path calls
DeepGEMM matrix logits and has a no-scoring path when all candidates are
selected. LunaFlux's current paged index still assigns each query to one
workgroup, and each candidate lane loops through 32 heads and 128 components
in ordered scalar F32 arithmetic before stable selection. The late probe's
roughly 7.6 ms index duration is therefore a separate executable gap; caching
the pooled keys does not remove scoring work.

The new matrix attention retains the existing expanded BF16 K/V representation.
vLLM's selected implementation uses compressed MLA/FP8 state. Query reuse
reduces repeated native reads, but does not make these representations or
memory demand equivalent. Converting the representation would require a
separate semantic/numerical and ownership change, not a hidden lowering cast.
The synthetic 64-query probes identify these mechanisms; they do not establish
their individual fractions of the remaining end-to-end service gap.

## Final-kernel counters and fresh reference

After the reference pair released both devices, v11 was captured serially in
the bounded2 GiB/no-swap probe. The final exported kernel includes the
small-row dispatch, not just the earlier cohort prototype. Raw report:
`/tmp/lunaflux-long-paged-sparse-20261019-v11/counters-next.ncu-rep`.

| Late-history attention counter | Frozen old | Final v11 |
|---|---:|---:|
| Profiled duration |14.216 ms|6.941 ms|
| Warp instructions |1,737,421,824|263,902,848|
| L2 read sectors (32 B) |145,337,501|37,241,952|
| Long-scoreboard contribution |61.21%|57.71%|
| Barrier contribution |21.20%|12.44%|
| Active warps |80.61%|43.06%|
| Issue-active metric |23.91%|7.88%|

Lower instruction demand and traffic reduce duration despite lower occupancy
and issue-active percentages. Neither percentage is an objective by itself.
The final index probe remains7.681 ms,212,210,635 warp instructions,12.08%
active warps,51.47% long-scoreboard contribution. It is now slower than the
attention leaf in this late-history probe, not proven the largest whole-model
component. The common scratch ownership change does not alter these kernels.

Fresh vLLM uses the same NVFP4 checkpoint,TP2,8192 context,8 maximum sequences,
64-token prefill chunks,256 MiB KV and capture maximum8 as the previous bounded
baseline. Prefix caching and speculation are disabled. It completes both
one-output vectors and all64 complete-answer requests. Warm second-vector
times for165/2085/6181/7973 inputs are0.808/6.057/15.018/17.642 seconds.
Complete8K answer takes18.457 seconds and10 generated tokens, exact label.

The broader reference answer vector has57/64 exact answers and64/64 completed
requests. Seven responses contain the label but also repeated reasoning after
the first `</think>`, so they fail the explicit exact-label task. This is an
observed output-format failure, not a proven vLLM implementation bug. The
reference is not a correctness oracle; no parser, stop list or reference text
is changed to hide these failures. Output lengths differ and answer-cohort
times must not be presented as equal-generated-work throughput. The four
one-output comparisons remain the fixed-output latency control.

## Small-row arithmetic control (not yet qualified)

The shared-scratch service's warm fixed-output vector is
1.231/11.902/32.636/43.177 seconds. The 7,973-token latency is 22.6% below
the previous55.779 seconds, but still2.45x the fresh17.642-second reference.
Its complete-answer short C16 group contains one exact-format failure:
lane12 returns the correct orchid-82 label followed by repeated reasoning.
The previous ordered service returned only the label for that prompt.
This is not yet proof that changed decode arithmetic is the cause: batching
and upstream prefill arithmetic must not be conflated.

A bounded follow-up keeps matrix query reuse for a full physical cohort,
but uses the existing ordered row program for fewer live queries than the
cohort size. This avoids one-live-row MMA work without changing the matrix
prefill algorithm, page ownership, or persistent state. Static shapes that
cannot support a reuse cohort still retain their explicit matrix alternative.
No output parser, reference or acceptance condition is changed. The candidate
must pass generated-source checks, the mixed-query analytic/sanitizer fixture,
and a fresh full-service answer/memory check before any promotion.

The initial shared-scratch campaign has now completed: all64 requests finish,
63/64 meet the exact-label requirement. The only failure is the short C16
case above; the remaining2K C1/4/8/16,6K C1/4 and8K C1 groups are exact.
The8K complete answer takes44.375 seconds with10 output tokens, versus the
fresh reference18.457 seconds with10 output tokens. The failed qualification
stops its private ranks without changing the public service installation.
The watcher's one-rank-exit message occurs during this intentional paired
shutdown, not from a memory-limit breach.

| Actual input tokens, one output | Previous LunaFlux | Shared-scratch matrix | Fresh vLLM | Current ratio |
|---|---:|---:|---:|---:|
|165|1.281 s|1.231 s|0.808 s|1.52x|
|2,085|13.452 s|11.902 s|6.057 s|1.97x|
|6,181|42.323 s|32.636 s|15.018 s|2.17x|
|7,973|55.779 s|43.177 s|17.642 s|2.45x|

These are matched warm observations, not statistical confidence intervals.
All serialized prompts and actual prompt/output counts are checked by the
external comparison report. Service startup/model loading is excluded.

Across354 samples per native rank, minimum available host memory was
16,821,888/18,885,480 KiB and peak GPU allocation97,663/97,690 MiB.
Both ranks remain below100 GiB and above the16 GiB reserve; the rank0
margin is narrow. The fresh reference's208 samples per rank record minima
17,558,236/18,567,744 KiB and peaks94,546/94,544 MiB. Neither used owned-unit
swap. Do not confuse system-wide `MemAvailable` with cgroup resident bytes.

Small-row control5cc72d4c passes all5,310 native tests with existing migration
warning exclusions, and the12-case actual-source GPU fixture at
`/tmp/lunaflux-query-reuse-20261019-v3` passes the independent analytical
calculation, three deterministic replays, memcheck/full leak/race/sync/init.
This establishes local execution safety, not yet complete-answer quality.

## Ordered-small-row retest and rejected promotion

Pinned5cc72d4c builds on ARM (336 tasks,1.5 GiB reported peak, zero unit swap).
Both AOT exports select matrix cohorts in all11 sparse layers and the ordered
row program below16 live queries. The29 short C1/4/8/16 answers are exact,
including the previously failed lane12. This supports the small-row control,
but does not prove a sole arithmetic cause: physical batching also varies.
No stop rule, parser, prompt or reference is changed.

The first/second fixed-output vectors are
1.230/11.516/34.027/45.472 and1.230/12.228/35.285/45.909 seconds.
The second8K result is17.7% below the frozen55.779-second native baseline,
but remains2.60x the fresh17.642-second vLLM result. Do not advertise the
earlier43.177-second experimental result as this candidate's speed.
Actual full-model cubin resource metadata also exposes a tradeoff:
48 registers/27,608 shared bytes became56 registers/44,056 shared bytes per
sparse-attention block. The combined entry point reserves both ordered-row
and matrix-cohort shared arrays. This is a concrete resource cost, not yet
an exact attribution of every millisecond of the timing difference.

The long answer campaign then fails the unchanged memory gate during2K C8.
All29 short answers and the complete2K C1/C4 groups pass before that failure;
the full64-answer campaign does NOT complete. Rank0 available host memory
falls to16,748,440 KiB (28,776 KiB below the16 GiB threshold). Peak GPU
allocation remains97,663/97,690 MiB and owned-unit swap stays zero. The
watchdog stops both private ranks and incomplete responses are retained at
`/tmp/lunaflux-glm-service-benchmark-cohort-ordered-long-20261019-v1`.
No tolerance waiver, cache flush, capacity reduction or unrelated workload
termination is used to turn this into a pass.

The previous shared-scratch run passed with only44,672 KiB rank0 headroom;
that margin is not robust enough to qualify deployment. GPU allocation is
288 MiB lower than the previous online build, so the host-availability
failure must not be casually described as a demonstrated new GPU allocation
leak. Further host/device residency attribution and extra headroom are needed.
Pending deployment/route files were prepared locally only; they were never
installed or applied. Restore the prior13a9ff61 service and verify its
customer-facing route before closing this iteration.

The final boundedv12 diagnostic accidentally omitted its legacy
`LUNA_LONG_MATRIX=1` environment setting. Its exported header and run report
explicitly say `matrix=false`/`matrix=0`: it is an ordered-attention,
pooled-reuse control, NOT a hardware capture of5cc72d4c's selected service
matrix branch. Its all64-row sanitizer gates pass, but its timing/counters
must not be substituted for the matrix measurements. The last measured
matrix counters remainv11 (before the ordered-small-row resource change).
The service AOT/configuration independently confirms matrix selection in
all11 layers; the diagnostic misconfiguration did not affect HTTP results.

The external probe runner now offers explicit `export-matrix`, passes the
numerical selection directly to the exporter, and checks the returned
selection marker. Legacy `export-next` and its environment override remain
supported. The regression passes;v13 is an explicit matrix source export
only, NOT another executed GPU experiment. No compiler/reference dependency
or production startup validation is added by this diagnostic correction.

## Restoration attempt: API pass, memory failure, endpoint offline

The retained13a9ff61 production units were restarted through the existing
paired controller without installing any candidate or applying route patches.
The existing company route passes model listing,323 arithmetic,42 multi-turn,
a coherent complete Chinese SSE explanation, and the literal7,973-token
orchid-70 answer. Backend metrics at that instant show4 admissions/completions,
zero failures/restarts, zero active/queued requests and all513 KV pages free.
Those successful replies do not qualify ongoing service residency.

The concurrent production observer then records rank0 `MemAvailable` of
16,636,728 KiB (140,488 KiB below16 GiB) and stops both exact owned units.
GPU peaks remain97,951/97,979 MiB and owned-unit swap remains zero.
Rank1 minimum is19,052,732 KiB. Root:
`/tmp/lunanexa-glm-restored-baseline-observation-20261019-v1`.
The controller is explicitly stopped afterward to clear desired-running state.
Thus neither the new candidate nor the previous build may be described as
successfully online after this iteration. No unrelated workload is killed and
the reserve is not relaxed. Additional measured residency headroom is required
before restarting customer service; response correctness alone is insufficient.

## Follow-up: bounded serial expert storage

The next memory experiment keeps context8192,8 slots,16 admission lanes and513
pages. Packed routed-expert activation/row-transport buffers are currently
owned separately by every MoE layer, although each layer's producers join and
its down/collective/publication consumers retire before the next layer begins.
Use the existing pure `SerialScratch` plan to lend one routed-activation buffer
across these serialized layers. Keep shared-expert activation separate because
the two producers may overlap; do not alias weights, routing state, quantizer
error flags, recurrent state, KV or HC values. This changes storage ownership,
not arithmetic or CUDA source. Measure planned bytes and actual GPU/host
residency and rerun the complete-answer concurrency gate before promotion.
The host-reserve failure itself is not yet attributed to a leak.

The pure full-service plan reports42 routed frames of19,957,776 bytes each.
One shared routed arena instead of42 saves818,268,816 bytes (780.36 MiB) per
rank. This is a planned allocation reduction, not yet measured host headroom.

## Follow-up: routed arena and remaining physical gaps

Memory-only commit `977ebda6` passes all 5,311 native tests with the existing
migration exclusions and a clean pinned ARM build. The private service root is
`/tmp/lunaflux-glm-http-expert-arena-20261019-v1`; direct source comparison
allowed reuse of the preceding cohort/ordered AOT. No payload hashes were used.
It retains 8,192 context, eight resident slots, 16 admission lanes and 513
physical pages. Initial measured GPU peaks are96,881/96,909 MiB, versus
97,663/97,690 before the routed arena: about782 MiB less on each rank.

The isolated one-output-token control is
`/tmp/lunaflux-glm-service-benchmark-expert-arena-latency-20261019-v2`:

| Actual prompt tokens | Memory-only completion, seconds |
| --- | ---: |
|165|1.254|
|2,085|11.415|
|6,181|34.068|
|7,973|45.503|

This is essentially unchanged latency, not a kernel speedup. The failed v1
request used the public alias against the private service, which advertises
`glm-5.3`; its validation-error response is retained, not a GPU failure.
The complete-answer/memory campaign is separate and must finish before a
residency qualification claim.

Two additional source candidates pass the full5,313-test native suite but
are not yet GPU-qualified or installed in the public service:

- `2cc4bd3a`: the existing physical `SequentialScratch` plan overlays the
  mutually exclusive ordered-row copy tile and matrix-cohort membership mask.
  The hypothesis is lower shared-memory residency cost, not less arithmetic.
- `acb96a7b`: `CandidateScorePlan` partitions independent candidate ownership,
  with a pure score-publication/selection dependency in `parallel_ir`.
  At64 queries and2,048 candidates, the score launch has1,024 blocks instead
  of keeping all work inside64 query blocks. Stable selection remains a
  separate64-block consumer with exactly the original head/component folds
  and tie order. The frame binds both stages into its existing ordered queue;
  its fake-device32-step regression has zero hot allocations or added waits.

This follows the reference's separation of score production from selection,
not its FP8/FP4 DeepGEMM numerical program or its short-context score bypass.
The new actual-source probe times both score and selection launches together.
The frozen baseline is the actual5cc72d4c matrix/ordered source in v13, not
the mistakenly exported ordered-only v12. Hardware and service results must
be recorded separately from these hypotheses.

### Completed memory gate and physical trials

The memory-only complete-answer root
`/tmp/lunaflux-glm-service-benchmark-expert-arena-answers-20261019-v1`
finished64/64 requests:63 exact, one short C8 format failure (lane5 returns
the correct label, then extra reasoning and the label again). The failure is
retained; containing the expected string is not an exact-answer pass.
All2K C1/C4/C8/C16,6K C1/C4 and near8K C1 answers were exact. The monitored
minimum host availability was17,476,104/19,812,148 KiB, above16,777,216 KiB;
GPU peaks96,881/96,909 MiB, owned swap zero. Both ranks stopped and released
their GPU allocations before the next GPU probe. This resolves the observed
reserve failure on this campaign, not arbitrary-load capacity.

Actual-source physical comparisons:

- v14 failed local export because the test used a nonexistent ReadOnlyArray
  conversion; no GPU result. v15 compiled the baseline but did not execute:
  review found its frozen incremental-pool ABI required more arguments and
  history-write replay. The external harness was corrected, preserving both
  old full-pool and incremental baselines; no reference arrays were edited.
- `/tmp/lunaflux-long-paged-sparse-20261019-v16`: the first partitioned index
  improved8K but regressed at256/2K. Keep this negative result. Query-major
  fixed-capacity tiles scatter live work among inactive blocks.
- `98c629fd` consumes a pure live-domain ownership mapping: live queries are
  fastest varying, candidate tiles follow. This compacts active work without
  host readback, numerical changes or an empirical short-context threshold.
- `/tmp/lunaflux-long-paged-sparse-20261019-v17`: matched baseline5cc72d4c,
  latest source, three alternating-order trials/five launches per timing.
  The index timing includes BOTH score publication and selection. Median ms:

| History/live rows | Old index | New index chain | Old attention | New attention | Combined reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
|64/64|0.09256|0.09413|0.18061|0.11750|22.5%|
|256/64|0.23673|0.23709|0.34117|0.22769|19.6%|
|2048/64|1.82657|1.37822|3.26632|1.82491|37.1%|
|4096/64|3.73613|2.62871|6.56831|3.75660|38.0%|
|8000/64|7.59068|5.37571|10.18057|6.62396|32.5%|
|7997/29|4.27743|2.76221|4.94118|4.94295|16.4%|
|7995/15|4.27845|1.66429|6.07270|5.14982|34.2%|
|7996/1|4.26799|0.78939|1.46777|1.45359|60.9%|
|7994/16|4.27578|1.81731|4.60163|4.74454|26.1%|

Every selected index and BF16 output is bitwise equal on all nine fixtures;
independent mathematical bounds also pass. v16 and v17 full leak/memcheck,
racecheck, synccheck and initcheck all report zero over64/15/1/16 live rows.
This is synthetic exact-geometry evidence, not literal service prompts or a
full-model quality claim. Attention alone has no gain at29 rows and a3.1%
regression at16; the complete measured chain still improves in both cases.
The short index chain has about1.6 microseconds extra launch overhead.

The v17 Nsight report explains a real storage/latency-hiding improvement:

| Counter | Old attention | New attention |
| --- | ---: | ---: |
|Static shared bytes/block|43,032|26,648|
|Shared-memory resident-block limit|2|3|
|Registers/thread|56|56|
|Executed warp instructions|259,288,704|259,288,704|
|Active-warps percentage|27.50%|42.78%|
|L2 read sectors|37,338,290|37,348,222|
|Profiled time|10.5495ms|7.0071ms|

Thus this attention gain does not come from fewer instructions or fewer
bytes. The independent shared regions had limited residency; overlaying them
allows more work to hide the same loads. Load-dependency stalls remain57.45%
of average active-warp latency, not zero and not a fraction of total wall time.

Old combined index:7.6975ms,212.21M warp instructions,12.06% active warps.
New score producer:4.8415ms,108.84M instructions,77.88% active warps;
selection:0.8292ms,99.51M instructions. Instructions fall only about1.8% in
total. Score-producer L2 reads increase substantially (34.40M sectors versus
1.24M for the old combined index); improved latency hiding has a traffic
trade-off, so this is not an optimality claim. Profiled and unprofiled timings
are kept separate.

Pinned ARM98c629fd builds successfully,1.3GiB reported peak and zero swap.
Fresh full-service AOT preparation is
`/tmp/lunaflux-glm-http-live-index-20261019-v1`; its actual generated rank
source contains all11 score-producer symbols and the live-domain mapping.
Public GLM remains offline pending completed service qualification; source
presence and isolated kernel gains alone do not authorize a speed claim there.
