# Mixed attention chain repair

> **Correction after row-coverage investigation:** the unified serving
> diagnostic omitted decode writes in its selected partitioned-prefill tail.
> The reported 2.63% throughput increase is therefore **not a valid
> correctness-preserving speedup**. The timing is retained as failed-experiment
> evidence, not a result to promote. See
> [the reproduced cause](MIXED_ATTENTION_ROW_COVERAGE_DIAGNOSIS_2026-10-08.md).

The matched 8,192/64 C8 trace attributes 279 ms of the LunaFlux/vLLM GPU-time
gap to mixed steps. This is not proof that serialization accounts for all of
it: each component may compete for the same execution resources.

## Bounded first experiment

Keep the selected prefill, decode-partial and merge artifacts, grids, row
vectors and numerical laws fixed. Compare captured ordered execution with a
captured fork/join: prefill writes only prefill rows; decode partial writes its
private workspace, then merge writes only decode rows. Both must complete
before the downstream output projection or any workspace reuse.

Replay all 29 mixed vectors from the matched serving trace using five
alternating timing pairs per vector. Require bitwise whole-output equality,
unchanged KV, the existing sampled FP64 oracle, and sanitizer success.
Only a useful complete-chain result justifies runtime integration and paired
serving verification. The budget is one dependency change, not a search over
new kernel schedules. A no-win remains a diagnostic result.

## Architecture constraint

If retained, independence belongs to a pure startup execution plan. CUDA
streams/events are terminal lowering, not model or scheduler policy. The
executor retains all allocations and functions through the joined completion;
cancellation never releases an in-flight branch. No analysis, allocation,
capture or extra host readback enters the token-step path. Ordered execution
remains a numerically identical legal schedule, not a second model engine.

## Results and bounded follow-up

The dependency-only experiment completed. Its aggregate benefit was too small
to justify adding production stream/event ownership as the main repair. One
additional, bounded hypothesis was then tested: use the existing matrix
attention artifact for a single ragged worklist containing **all** query rows,
instead of a prefill worklist followed by a separate split-decode chain. This
changes decode's numerical execution; it is not labeled bitwise-equivalent.

Both comparisons replay the same 29 exact mixed vectors, fragmented page
mapping, runtime launch bounds, and frozen cubins. Both arms are graph-captured;
removing host launch overhead cannot be counted as the candidate's gain.

| Complete-chain replay | Ordered sum of medians, µs | Candidate sum, µs | Reduction | Per-vector decisions |
| --- | ---: | ---: | ---: | --- |
| Same kernels, fork/join | 45,501.95 | 44,483.50 | 2.24% | 3 improved, 26 inconclusive |
| Unified matrix row worklist | 45,567.58 | 39,289.64 | 13.78% | 26 improved, 2 inconclusive, 1 regression |

These are sums of one-layer replay medians, not serving time or a confidence
interval. Each vector has five alternating timing pairs. The predeclared
per-vector improvement rule requires every pair to improve by at least 2%.

Representative unified-worklist results:

| Exact row shape | Ordered µs | Unified µs | Time reduction |
| --- | ---: | ---: | ---: |
| `2047:0, 1:8192` | 459.96 | 489.01 | **−6.31%** |
| `2043:6093` plus five unequal-history decode rows | 2,405.77 | 1,960.81 | 18.50% |
| `2041:4004` plus seven unequal-history decode rows | 2,239.36 | 1,658.91 | 25.92% |
| `106:8086` plus seven unequal-history decode rows | 1,360.69 | 1,271.69 | 6.54% |

All 29 vectors passed unchanged-KV and sampled FP64-oracle checks. Fork/join
also passed bitwise whole-output comparison. Unified output is **not** bitwise
equal: the representative late vector has maximum differential error
`0.000488281` and sampled oracle error `0.000389393`. Memcheck, racecheck,
initcheck and synccheck passed for that vector in both experiments. This does
not establish model-quality equivalence or sanitizer coverage of every shape.

## Actual serving substitution

A new isolated copy of the frozen worker source makes two diagnostic changes:

1. Publish tile metadata for the full row domain, including decode rows.
2. Prepare mixed owners with the original complete attention launch list,
   without the narrowed prefill arguments and appended decode companions.

The model, weights, projection kernels, sampling, scheduler, and original
serving installation are unchanged. This is a **diagnostic worker**, not a
production-selected compiler policy. No copied working-tree changes were used.

Qwen3-0.6B BF16, Spark .179 / GB10, input vector eight × 8,192, output vector
eight × 64. Order is control → unified → unified → control. Each fresh start
has one excluded warmup and three measured waves, giving six waves per arm.

| Arm | Median wave ms | Output tok/s | Median TTFT ms | Median request mean TPOT ms |
| --- | ---: | ---: | ---: | ---: |
| Frozen control | 4,917 | 104.13 | 1,501 | 50.33 |
| Unified diagnostic | 4,791 | 106.87 | 1,456 | 49.14 |

**Observed completion time fell 2.56%; output throughput increased 2.63%,
but this serving result is invalidated by missing decode-row writes.** Control
wave range was 4,886–4,950 ms; unified was 4,780–4,808 ms. This is a real
timing change from an incorrect implementation, not a valid serving gain or
closure of the remaining performance gap.

The immediately preceding matched vLLM/SGLang measurements were 4,494/4,545.5
ms (113.93/112.64 tok/s). The incorrect diagnostic's timing corresponds to
6.61%/5.40% longer completion time, but these are **not valid competitive
performance gaps** because the diagnostic does not produce complete outputs.
References were not restarted in this experiment; this is not a new matched
three-engine campaign or a result for every context length.

### Dispatch really changed

A separate Nsight Systems capture includes warmup and one measured wave:
192 graph launches, 66 prefill/mixed graphs and 126 single-query graphs.
**No prefill graph contains a separate decode-attention companion.** The
existing partitioned prefill tail remains selected where its graph requires
it; the experiment does not falsely report every prefill as one identical
kernel. Single-query graphs retain their original decode routes. The trace
contains 39,688 kernel invocations over both waves.

This is selected-dispatch proof, not a new instruction-counter capture. It
does not establish the exact load/barrier contribution of every saved µs.

### Numerical issue prevents global enablement

The workload generator uses the same deterministic token input for each row
across trials. Compared with each arm's first measured wave, full output
sequences repeated in **35/40 control comparisons versus 4/40 unified**.
Only 4/48 cross-arm sequences were identical. Admission order and phase/shape
selection can change between concurrent waves; this is **not proof of a data
race**, and random-token performance prompts are not a quality corpus.
The follow-up now identifies a concrete diagnostic rewrite error: the
partitioned merge retained its prefill-only write domain. Every one of the
36 differing repeated sequences first diverges at the predicted bad-tail
token. The local attention tolerance checks missed this because the 29-vector
replay used the ordinary matrix artifact even for the final tail, while
serving actually selected a partitioned pair.

Decision: **do not enable this globally or call the problem fixed.** Preserve
the incorrect implementation and measurements as a failed experiment. Production
integration needs an explicit whole-row numerical contract, phase/shape
consistency and teacher/logit validation. The early mixed regression also
requires measured whole-chain selection, not an unconditional all-shape switch.

The overlap probe yields little benefit on its measured chains. The serving
substitution cannot establish that changing the mixed work domain yields
more: missing tail writes invalidate that comparison. Correct final-writer
coverage must precede another serving performance conclusion.

## Safety, reproducibility and source boundaries

- GPU: `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`, PCI `0000000F:01:00.0`.
- nvcc SHA-256:
  `fbb111f057786ddd10ba723d993cc7dd43abf978b6baa32fedd3c9d806dc79e1`.
- Diagnostic worker SHA-256:
  `e83d46a28bb1239d9e1398df5f63fb9429a0684745e4fb1e5eb7f44e9ad1babb`.
- Frozen serving identities are recorded in
  [the matched comparison](BENCHMARK_EXECUTION_GRAPH_COMPARISON_2026-10-07.md).
- Fork/join: `/home/wlc004s/lunaflux-mixed-fork-join-20261008.ZOVMDIUS`.
- Unified replay: `/home/wlc004s/lunaflux-mixed-unified-20261008.kASJ9zTj`.
- Serving: `/home/wlc004s/lunaflux-mixed-serving-20261008.fZfbeldp`;
  completed ABBA is `timing-v2`, selected trace is `profile`.
- Failed script setup attempts are preserved and excluded: an initial
  `.mbtx` array conversion, a receipt written to the wrong diagnostic path,
  and a retry-script parse error. They are not GPU timing results.

GPU workloads were serialized. Serving was limited to 64 GiB/no swap;
controllers/probes to 8 GiB and bridges to 2 GiB. Both serving arms retained
more than 99 GiB MemAvailable (32 GiB floor). All measured requests completed.

The local changes are offline probes, MoonBit experiment/report helpers, and
this report. The production graph/compiler/ABI is not changed. The diagnostic
rewrite is model-independent but fails to preserve complete row writes. Any
retained implementation must expose the row-domain/numerical choice in the existing pure plan and
measured route selection, not hide a model-name condition in CUDA lowering.

## Downloaded evidence and checks

All three archives were downloaded into a fresh, non-overwriting directory:
`/tmp/lunaflux-mixed-chain-verified-20261008.JV0wzFfT`.
Local archive hashes match the remote seals; every extracted `FILES.sha256`
entry verifies (172 fork/join, 335 unified replay, 770 serving files).

| Archive basename | SHA-256 |
| --- | --- |
| `lunaflux-mixed-fork-join-20261008.ZOVMDIUS.tar.gz` | `793e69bcc706a9b20a9127a7854813b476622b60f41164b0fd7e024cf4f74167` |
| `lunaflux-mixed-unified-20261008.kASJ9zTj.tar.gz` | `810bdfc46c786c491bf43071781c686c0c8a06ff48ed6fe6868aa9ecc867d327` |
| `lunaflux-mixed-serving-20261008.fZfbeldp.tar.gz` | `1184c4362b67ef5a1dfa436c86e33cfd6fc1059f02b8485d432bae64ef3b6d30` |

The archives retain diagnostic sources, commands, outputs, failed setup
attempts, the diagnostic worker, and the serving trace. They exclude copied
model/deployment trees and build caches; frozen dependencies remain identified
by their recorded paths and hashes. The sealed report predates this download
index, avoiding a circular archive hash.

The final self-contained probe source was rebuilt and reran all 29 unified
vectors plus the four sanitizer tools successfully. MoonBit helper checks and
focused tests passed, including the four AKO harness tests; the C++ launch
geometry test passed with warnings treated as errors. This is not a full
repository suite result: no production package was changed, and unrelated
working-tree changes were preserved.

Before another speed claim, the next repair must isolate the numerical issue:
replay identical captured Q/K/V and row domains across phase schedules, check
every output row against a numerical reference, and compare teacher-forced
logits on meaningful fixed prompts. Only then can whole-chain measurements
select unified versus partitioned row execution through a pure startup plan.
Changing admission timing to make random-token sequences agree would not be a
numerical fix.
