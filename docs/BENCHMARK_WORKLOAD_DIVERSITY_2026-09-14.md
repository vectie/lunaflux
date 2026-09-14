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
