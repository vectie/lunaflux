# Workload diversity: measurements and limits

The original uniform regression matrix is useful but insufficient: requests
within a case share length, token sequence and arrival time. Long prompts cycle
one 1,528-token synthetic sequence. This is not a natural-language quality set.

## Controlled first pass

RTX 5060 Ti, Qwen3-0.6B, uninstrumented row-tail-capacity-fixed worker. Evidence:
`/tmp/lfdiversity.bxHEX5/lunaflux` on the benchmark host. One warmup and three
measured repetitions, rotating case order. All requests completed with the
requested output count. Client launch lag p95 was at most 1 ms. These are not
new vLLM/SGLang comparisons or a before/after optimization experiment.

Every case has average input/output counts 2,048/64. Ragged inputs per eight
requests: 63, 65, 511, 513, 3583, 3585, 4031, 4033. Ragged outputs: 16, 112, 32,
96, 48, 80, 64, 64. Staggered requests arrive 25 ms apart.

| Case | C8 median wall ms | C16 median wall ms | C8 p95 TTFT ms | C16 p95 TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| Uniform shared sequence | 1215 | 2182 | 624 | 1277 |
| Uniform distinct sequences | 1219 | 2184 | 628 | 1277 |
| Ragged input | 1558 | 2427 | 723 | 1438 |
| Ragged input/output | 1774 | 2727 | 724 | 1432 |
| Uniform staggered | 1221 | 2186 | 465 | 909 |
| Ragged staggered | 1788 | 2760 | 612 | 1123 |

**Equal token counts are not equal attention work.** C8 uniform causal prefill
pairs sum to 16,785,408; ragged pairs sum to 29,376,516 (75.0% more). Therefore
the ragged-input slowdown cannot be assigned entirely to batching or padding.
Varying output length also changes active decode batch sizes and tail work.
Arrival-relative TTFT can improve even when whole-campaign completion does not.

The shared/distinct control is nearly unchanged. Thus repeated token contents
alone do not explain this measured runtime's performance. Uniform versus
staggered completion also barely changes under this particular 25 ms pattern.
Neither observation proves general absence of prefix or scheduling effects.

## Why previous compiler work did not guarantee speedup

1. The measured stage/window search retained the baseline artifacts; improving
   selection infrastructure without changing selected instructions cannot by
   itself speed up the GPU work.
2. Resource-only selection was slower than timing-based selection. Occupancy
   is a constraint, not a performance objective.
3. Complete ingress fusion lost to partial fusion in the measured chain; the
   selected implementation consequently stayed partial.
4. Uniform per-row attention microbenchmarks do not reproduce real steps that
   combine one long prefill row with short decode rows. A microbenchmark winner
   must be checked on that actual row distribution before production selection.
5. A faster individual kernel has only its fraction of end-to-end time available
   to improve. Report changed code, selected route, invocation count and total
   time together rather than counting implemented passes as performance gains.

## Remaining controls, not yet completed

- Same exact request multiset with reversed staggered arrival order, preserving
  causal work and token contents while changing scheduling exposure.
- Identical workload vectors on baseline and new runtime, including per-request
  TTFT, completion, token intervals, exact selected route and mixed row shapes.
- Natural prompts, non-cycling token sequences, longer output tails, bucket-edge
  shapes and held-out shapes; keep tuning and evaluation observations separate.
- Current measured-route bundle physical qualification and independent numerical
  diagnosis of the previously observed final-token disagreement.

Do not label all seven integration items complete based on this table. The
first invalid >4096-input attempt in `/tmp/lfdiversity.j30cOV` is preserved as
an out-of-capacity benchmark setup failure, not an inference performance result.

Downloaded archive `/tmp/lunaflux-workload-diversity-20260914.tar.gz` matches the
remote SHA-256 `6e5a14cd6c9f1d27c3a1f4369f5f72cdd3b308ffcc5367dc290dd25a2a7fe4c0`.
Measured-route production integration was committed as `01ba1c46`; native check
and all 3,757 tests passed. This is implementation validation, not a claim that
the new measured route bundle has passed physical qualification.

## Same-multiset order control and alternate attention

`/tmp/lfdiversity-order.3PpwqG` uses the rebuilt integration worker with the
byte-identical baseline bundle. Reversing the same eight request tuples in each
staggered group changes C8 median wall time from 1785 to 1764 ms and C16 from
2762 to 2730 ms. This preserves tokens, lengths and causal work, unlike the
uniform/ragged comparison. Three measured trials do not establish a universal
scheduling benefit. C8 request p95 mean inter-token time changes from 30.935 to
20.666 ms, demonstrating why completion throughput alone is insufficient.

`/tmp/lfdiversity-wide.EU0jMm` uses the same request bodies and worker with the
c324 wide-prefill artifact instead of the c322 alias. No synthetic measurement
table was used to force selection. Median wall times:

| Case | Baseline C8 | c324 bundle C8 | Baseline C16 | c324 bundle C16 |
| --- | ---: | ---: | ---: | ---: |
| Uniform distinct | 1218 | 1217 | 2187 | 2182 |
| Ragged input | 1557 | 1568 | 2426 | 2420 |
| Ragged input/output | 1771 | 1770 | 2678 | 2698 |
| Ragged staggered | 1785 | 1785 | 2762 | 2755 |
| Ragged reverse staggered | 1764 | 1761 | 2730 | 2732 |

There is no stable end-to-end improvement in these measurements. The expanded
workloads did not reveal a hidden large c324 speedup. Actual selected-kernel
profiling is being checked separately; bundle presence alone is not invocation
proof.

Across 504 measured requests (32,256 tokens), 62 request sequences differ between
the two bundles; each bundle has five differing repeated requests among its
336 within-bundle repeat comparisons. Cross-bundle differences are in ragged
cases, including repeatable first differences at positions 4, 10, 22 and 51;
they are not confined to the final token. This exposes coverage missing from
uniform tests, but does not establish which output is numerically correct.
Greedy divergence must be investigated against reference logits/tolerances;
do not relabel token disagreement as proven accuracy loss or harmless rounding.
The alternate bundle remains experimental, not the new default.

## Actual selected-kernel trace

The separate 4096/64 C8/C16 trace at
`/tmp/lfwide-selection-trace.ygncsI/lunaflux/trace-export.sqlite` confirms 2,688
calls to `lunaflux_attention_prefill_tile_compiler_v1_c324`. The prior row-tail
baseline trace has the same 2,688 prefill calls, 8,288 decode-attention calls and
82,560 total kernel calls over its four warmup/measured cases.

| Aggregate over the four profiled cases | c322 baseline ms | c324 ms |
| --- | ---: | ---: |
| Prefill attention | 2770.429 | 2741.994 |
| Decode attention | 4098.247 | 4098.238 |
| All kernels | 13078.436 | 13052.361 |

Thus the selected prefill kernel improves only 1.03% here, not the roughly 14%
seen on the earlier uniform-row microcase. It represents about 21.2% of baseline
GPU kernel time; saving 28.435 ms there predicts only about 0.22% of total kernel
time. Observed aggregate savings are 26.075 ms (0.20%). This directly explains
why this substitution does not materially move end-to-end throughput. It is not
an absent-kernel-selection problem. These are single profiling comparisons,
not statistical proof of a 0.20% production speedup.

The remaining investigation must measure matched **actual row distributions**,
history, cache conditions and instruction/memory behavior before proposing
another compiler pass. Uniform-row microbenchmarks and generic bucket upper
bounds are not substitutes for those inputs. This trace does not diagnose the
instruction-level reason the microcase benefit disappears.
