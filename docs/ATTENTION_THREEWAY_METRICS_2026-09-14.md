# Matched logical attention workload — 2026-09-14

This is a **kernel-library comparison**, not an end-to-end serving benchmark.
No production kernel, scheduler, deployment, or baseline installation was changed.

## Test contract

- RTX 5060 Ti, UUID GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6, PCI 17:00.0.
- Qwen geometry: BF16 Q/K/V/output, 16 query heads, 8 KV heads, head dimension
  128, causal attention, scale `1/sqrt(128)`, no window or soft cap.
- Same deterministic logical Q/K/V values, per-row query lengths and history
  lengths as the LunaFlux fixture. Physical pages are permuted.
- `q` is total query tokens, not tokens per row; rows are not HTTP concurrency.
  Multi-row history is `base + (row % 3) * 8`, including base-zero cases.
- Six process blocks, five measurements/block, 30 launches/measurement. Baseline
  engine order alternates across blocks. LunaFlux was rerun after the baseline
  sweep on the same day; it was not interleaved with the two baseline engines.
- Independent sampled FP64 reference checks cover first/middle/last query and
  heads 0/7/15 in every row for each baseline case. All 288 baseline case-runs
  pass, maximum sampled absolute error below 0.000988 (threshold 0.003).
  LunaFlux's 480 paired observations pass its scalar referee, bitwise A/B and
  unchanged-KV checks. This does not prove arbitrary model/batch numerics.

### Baseline identities and unavoidable layout differences

vLLM 0.24.0 / torch 2.11.0: installed `flash_attn_varlen_func`, `fa_version=2`,
paged KV, preallocated output. Its FlashAttention split-KV family is present in
the saved service trace. It rejects page size 8, so identical logical KV is
repacked to permuted 16-token pages **outside timing**.

SGLang 0.5.2 / torch 2.8.0 / flashinfer-python 0.3.1: installed FlashInfer
`BatchPrefillWithPagedKVCacheWrapper`, FA2, NHD, page size 8, causal, preallocated
output. This is a controlled paged-library call, **not** an assertion that
SGLang selects that exact call for every shape. Saved serving traces also use
ragged current-token attention and noncausal paged history attention; replaying
that complete split/merge composition is also measured below. The cascade test
uses causal ragged current KV, noncausal paged historical KV and `merge_state`;
with no history it uses ragged alone. KV writes and planning are excluded from
all variants. This follows the installed attention backend, but is not a replay
of the full serving scheduler.

LunaFlux uses the existing c318 synchronous and c322 single-storage async
artifacts, metadata ABI v1, page size 8, current-input stride 4096 BF16 values.
Baseline Q is contiguous. Thus logical work is matched, but physical layout and
launch geometry are deliberately backend-specific, not identical.

Baseline CUDA-event timings cover graph replay of the whole prepared call to
avoid Python submission gaps. LunaFlux uses the existing native direct-launch
CUDA-event loop. Small-query comparisons can include different launch overhead;
do not interpret those ratios as pure arithmetic efficiency. Planning, layout
conversion, JIT compilation and reference checking are outside timing. Reused
buffers make these warm-cache tests; cold-cache NCU runs are separate.

## Fresh timing matrix

Median microseconds, 30 observations per cell. Neither LF column is selected
per shape after the fact; both variants are shown.

| Total q | Rows | History base | LF sync | LF async | vLLM FA2 | SGLang's FlashInfer paged |
|---:|---:|---:|---:|---:|---:|---:|
|63|1|0|19.52|19.03|8.33|8.35|
|64|1|0|19.43|18.89|8.31|8.35|
|65|1|0|22.64|24.19|8.29|8.38|
|512|8|0|38.33|39.76|22.69|14.52|
|520|8|0|55.60|59.54|41.13|24.75|
|1528|1|0|460.23|499.28|295.62|283.52|
|1528|8|0|112.69|122.83|84.16|54.93|
|2048|1|0|762.94|829.23|494.86|459.31|
|2048|8|0|164.07|177.51|127.20|82.57|
|2048|16|0|119.30|128.19|98.46|54.90|
|2048|1|2048|2184.01|2243.03|1343.61|1263.19|
|2048|8|2048|1639.51|1632.34|982.75|879.86|
|1528|8|4096|2583.60|2366.24|1349.84|1271.65|
|2048|1|4096|3624.69|3667.29|2199.37|2062.84|
|2048|8|4096|3147.21|3104.31|1842.48|1698.10|
|2048|16|2048|1609.06|1592.12|957.57|859.62|

The 1528/8/history4096 async improvement reproduces, but async still takes
1.75 times vLLM's and 1.86 times FlashInfer's time in this controlled test.
For 1528/1/history0, synchronous LF takes 1.56/1.62 times baseline time, and
async regresses. Therefore the remaining gap cannot be explained solely by
the earlier end-to-end measurement failing to activate async.

### SGLang ragged/history/merge composition

Same 30-observation timing protocol, median microseconds. This is a separate
sweep, not simultaneous or interleaved with the earlier runs.

| Total q | Rows | History base | SGLang composition |
|---:|---:|---:|---:|
|63|1|0|8.34|
|64|1|0|8.33|
|65|1|0|8.37|
|512|8|0|20.80|
|520|8|0|34.75|
|1528|1|0|279.02|
|1528|8|0|71.32|
|2048|1|0|452.07|
|2048|8|0|112.33|
|2048|16|0|101.16|
|2048|1|2048|1292.59|
|2048|8|2048|936.98|
|1528|8|4096|1312.38|
|2048|1|4096|2123.98|
|2048|8|4096|1757.74|
|2048|16|2048|918.04|

Against this composition, LF sync is 1.65x slower at 1528/1/0 and LF async
is 1.80x slower at 1528/8/4096. Including merge changes the comparison,
but does not eliminate the gap.

## Hardware counters

Five representative geometries were profiled: 1528/1/0, 1528/8/0,
2048/1/2048, 2048/8/2048 and 1528/8/4096. Baseline paged calls have
20 reports (two engines, five geometries, cache-control all/none), each with
one kernel record. LF supplemental traffic collection has ten reports
(two variants, five geometries, cache-control none). NCU timings are diagnostic;
the ordinary timing matrices above are the performance results.

The following are cache-control-none measurements, not cold DRAM traffic.
Instruction counts are executed warp instructions, not scalar-thread counts.

| q/rows/history | Metric | LF sync | LF async | vLLM paged | SGLang paged |
|---|---|---:|---:|---:|---:|
|1528/1/0|Warp instructions, million|32.43|43.81|11.15|13.06|
|1528/1/0|Tensor active, % elapsed|42.74|39.43|68.04|70.82|
|1528/1/0|Achieved occupancy, %|15.84|15.85|8.32|15.49|
|1528/8/4096|Warp instructions, million|166.93|222.42|50.40|64.38|
|1528/8/4096|Tensor active, % elapsed|40.84|44.79|77.28|85.24|
|1528/8/4096|DRAM read, decimal MB|171.66|171.27|148.41|147.05|
|2048/8/2048|DRAM read, decimal MB|84.17|84.23|84.39|84.32|

Key conclusions:

1. **The deficit is not simply lower occupancy.** vLLM uses 254 registers/thread
   and 82,944 bytes shared/block versus LF's 168/173 registers and 50,192 bytes.
   Despite lower occupancy, its Tensor pipeline is substantially more active.
2. **Async exchanges one class of waits for others and adds work.** At
   1528/8/4096, LF long-scoreboard PC samples fall from 84,427 to 17,193,
   but short-scoreboard samples rise from 17,382 to 35,920 and wait samples
   from 50,202 to 72,159. Executed instructions rise 33.2%; measured time only
   improves 8.6%. These sample counts are not percentages of wall time and
   cannot be added up into a predicted speedup.
3. **Reading fewer DRAM bytes cannot by itself close the gap.** At
   2048/8/2048 all paged implementations read approximately 84 MB, yet LF takes
   roughly 1.6–1.9x the baseline time. At 1528/8/4096 LF reads about 16% more,
   much less than its timing deficit. Instruction scheduling, data movement
   between memory levels, and useful Tensor work remain the next isolation
   targets; counters alone do not identify a unique source-level fix.
4. Small warm-cache DRAM counts, occasional derived L2 hit rates above 100%,
   and an impossible 160.99% merge occupancy measurement
   are replay/counter limitations. Do not interpret them as literal cache
   efficiency or derive peak-bandwidth claims from them. Clocks were not locked.

The SGLang composition collection completed ten additional NCU reports with
26 kernel records: one ragged kernel for the no-history single-row case, and
ragged + paged + merge for each remaining case, in both cache modes. At
1528/8/4096 these three kernels execute 2.881M + 63.034M + 1.076M = 66.991M
warp instructions, versus LF async's 222.424M. Their diagnostic durations are
48.90 + 1233.15 + 23.90 microseconds; these are not a replacement for the
ordinary whole-composition median of 1312.38 microseconds. Including merge
does not account for LF's excess instruction count.

Explicit LF local-load/store sector counts are zero for all ten supplemental
profiles. Hardware shared-bank counters are **not zero**: at 1528/8/4096,
load/store counts are 29,446/603,192 (sync) and 29,056/4,245 (async).
At 1528/1/0 they fall from 50,414/131,051 to 30,779/7,691 even though async
slows down. These aggregate counters neither establish software layout conflicts
at a particular instruction nor substitute for measured elapsed time.

Still outside this measurement: full-serving CPU/GPU idle intervals, queue and
batch distributions, graph fallback frequency, KV-write cost, all model
operators, and end-to-end TTFT. This completes matched attention measurements,
not the entire serving-performance investigation.

## Raw result locations

Remote:

- `/tmp/lunaflux-attention-baseline-timing-20260914-r1`: frozen numerical fixture,
  commands, per-process logs, device snapshots, 32-cell SUMMARY.json.
- `/tmp/lunaflux-attention-lf-retest-20260914-r1`: fresh LF A/B, correctness,
  artifact/tool hashes, per-process logs, 16-cell SUMMARY.json.
- `/tmp/lunaflux-attention-baseline-counters-20260914-r1`: 20 NCU reports;
  use `FULL_SUMMARY.json`, not the earlier partial `SUMMARY.json`.
- `/tmp/lunaflux-attention-traffic-20260914-r1`: ten LF NCU reports and SASS,
  `FULL_SUMMARY.json` with explicit DRAM and memory-sector counters.
- `/tmp/lunaflux-attention-cascade-timing-20260914-r1`: 96 case-runs,
  frozen composition fixture and 16-cell `SUMMARY.json`.
- `/tmp/lunaflux-attention-cascade-counters-20260914-r1`: ten reports,
  26 kernel records in `FULL_SUMMARY.json`, preserving the full composition.

The temporary pytest dependency was installed into an isolated `/tmp` directory,
not either baseline environment. Numerical tests use baseline-native tensor
APIs; orchestration and summaries use `.mbtx` per the MoonBit agent guide.

All six result directories were archived and downloaded without overwrite to
`/tmp/lunaflux-threeway-attention-20260914.2FdhAG/results.tar.gz`.
Remote and local SHA-256 both:
`5f62780eec5f7e820ac38cc6c5de4792cceab609eba909e51594814a8886f829`.
