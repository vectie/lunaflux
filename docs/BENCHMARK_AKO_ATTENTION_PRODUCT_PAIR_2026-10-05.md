# Dual-Spark attention product-pair experiment

## Result

Both Sparks completed the same nine-cell experiment. **No variant has a robust
win; none is enabled in serving.** QK pairing regresses consistently, while PV
pairing is within noise. This does not replace the preceding measured 5.47%
C2 serving improvement or solve its output-trajectory qualification issue.

The AKO loop prevented a source-level scheduling change from becoming an
unmeasured production regression. No production compiler, model, scheduler,
numeric contract, route table, or release binary changed in this experiment.

## One hypothesis, bounded experiment

The [selected c322 counters](BENCHMARK_AKO_LONG_STAGE_AND_C2_DECODE_2026-10-05.md)
identified fixed instruction dependencies as the leading warp-latency
contribution. Hypothesis: loading two independent RHS fragments and issuing
their MMA halves in half-major order improves latency hiding without changing
each accumulator's ordered reduction.

Three ablations: QK only, PV only, and both. Three exact workload cells per
variant, five alternating-order pairs per cell, thirty event-timed repetitions
per side. Eighteen cells across two devices: ninety paired timing observations.
The fixed budget ended after these cells and one baseline/candidate counter pair.
There was no search-until-success loop or selectively repeated winning cell.

The diagnostics generate CUDA-only instruction ownership alternatives from the
immutable selected source. They do not claim to be admitted compiler candidates:
the original recipe's semantic/schedule digests remain frozen and an explicit
`experimental_lowering=half-major-product-pair-v1-*` tag identifies the change.
The source hash is updated; baseline identity is checked before transformation.
These recipe files are probe inputs, **not production admission receipts**.

Per-output arithmetic, softmax, probability packing, causal masks, shared layout,
copy/effect order, query geometry and strict-natural-exponential law are unchanged.
Only independent matrix outputs interleave. No model-family condition or new
runtime work is introduced. A positive result would require implementation in
typed physical scheduling and CUDA lowering, followed by serving selection and
end-to-end validation. This negative experiment is retained only in offline
benchmark automation; no alternative production engine remains.

## Devices and frozen identities

- .178: GB10/sm121, UUID `GPU-9c3d3cf0-439a-5da2-67e9-20414255879f`.
- .179: GB10/sm121, UUID `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`.
- Baseline source SHA-256:
  `a6c232ab5048ba8df9b1e61ce687b756326808f1ffedd26e22985865f8bb704b`.
- Baseline cubin SHA-256:
  `9aedf7709b833ad4ba05bf017d54199b33908e89d63c7fae51e73f94cf0a655d`.
- CUDA 13.0.88 nvcc SHA-256:
  `fbb111f057786ddd10ba723d993cc7dd43abf978b6baa32fedd3c9d806dc79e1`.

Both devices measure their own paired samples. Do not compare raw cross-machine
times as though they were a single controlled pair. GPU work is serialized on
each device; the final sanitizer jobs ran simultaneously across the two devices.
User-systemd bounds: preparation 8 GiB/300 s; trials 16 GiB/600 s; counter capture
16 GiB/240 s; sanitizers 16 GiB/180 s. The trial runner checks the existing
32-GiB MemAvailable reserve before and after each workload. All samples pass
that reserve. These are host-available snapshots, not continuous GPU peak-memory
measurements. No large-model or million-token allocation was attempted.

## All workload results

Q=2048 in every cell. H is prior KV history per request. R is active request
rows; the runtime launch still uses its actual 32-row profile bucket. Negative
gain means slower. Values are medians of the five **paired** reductions, not
ratios of independently sorted medians.

| Changed products | R | H | .178 paired gain | .179 paired gain |
| --- | ---: | ---: | ---: | ---: |
| QK | 1 | 28672 | −7.43% | −5.41% |
| QK | 2 | 28672 | −4.38% | −2.91% |
| QK | 2 | 8192 | −2.26% | −2.32% |
| PV | 1 | 28672 | +0.08% | −0.22% |
| PV | 2 | 28672 | +0.57% | +0.42% |
| PV | 2 | 8192 | −0.08% | −0.26% |
| QK + PV | 1 | 28672 | −6.65% | −6.20% |
| QK + PV | 2 | 28672 | −3.58% | −4.05% |
| QK + PV | 2 | 8192 | −2.17% | −2.15% |

No cell meets the 3% minimum gain in every pair. Even the positive PV medians
contain losing pairs. Raw samples and worst-pair reductions remain in the
downloaded manifests and the generated report.

## Matched hardware counters explain the non-win

On .179, NCU captures exactly the baseline and QK+PV candidate at Q2048/R2/H28672:
same symbol `lunaflux_attention_prefill_tile_compiler_v1`, grid `63×16×1`,
128 threads. These are instrumented single-invocation durations, not serving
timings or a vLLM/SGLang comparison.

| Metric | Baseline | QK + PV pair |
| --- | ---: | ---: |
| Profiled duration | 8.441536 ms | 8.788928 ms |
| Registers/thread | 235 | 244 |
| Register-limited resident blocks | 2 | 2 |
| Executed warp instructions | 1,125,435,264 | 1,117,057,920 |
| HMMA instructions | 119,668,736 | 119,668,736 |
| Non-transposed LDSM instructions | 37,396,480 | 37,396,480 |
| Transposed LDSM instructions | 29,917,184 | 29,917,184 |
| 128-bit async LDGSTS instructions | 14,958,592 | 14,958,592 |
| MOV instructions | 102,763,520 | 103,704,576 |
| NOP instructions | 21,502,976 | 32,721,920 |
| Eligible warps/scheduler | 0.413235 | 0.405906 |
| Issue-active fraction | 33.885% | 32.825% |
| Mean warp latency/issued instruction | 5.568427 | 5.856061 |
| Fixed wait contribution to mean latency | 36.507% | 35.575% |
| Long-scoreboard contribution | 14.296% | 12.505% |
| Barrier contribution | 2.114% | 2.626% |
| Source-derived excessive shared wavefronts | 0 | 0 |

Total instructions decline only 0.74%; NOPs increase 52.17%, MOVs increase
0.92%, eligible warps/issue activity decline, and the matched capture takes
4.12% longer. The compiler did not turn the proposed independent-product issue
order into a faster hardware schedule. Occupancy's register limit stays at two
blocks; the extra registers are **not proof of an occupancy drop**. Likewise,
the smaller fixed-wait percentage is not an improvement in completion time:
the mean latency denominator itself grows about 5.17%.

SASS/counters establish the changed instruction mix and poorer latency hiding.
They do not prove that all regression is caused by NOPs or register allocation.
Neither shared-memory conflict chasing nor further source-only MMA grouping is
supported as the next fix. The next candidate must reduce actual supporting
work/dependency latency or widen useful scheduling freedom within the resource
budget, and must be measured with the same complete query/history/row vector.

## Validation and saved records

- All eighteen cells: complete-output bitwise equality and maxabs=0; unchanged
  KV; independent sampled FP64 oracle within the probe's 0.003 contract.
- .178 and .179 both pass memcheck, racecheck and synccheck at Q129/R2/H128,
  with bitwise equality and zero errors/hazards. This irregular-tail boundary
  is a diagnostic check, **not long-context sanitizer admission**.
- Both new `.mbtx` scripts pass focused warning-denied native tests and formatting.
- No serving campaign ran for these losing variants. There is no new token/s,
  TTFT, output-quality or baseline-framework speedup claim.
- Both GPUs returned idle.

Preparation failures are preserved: the first generated version left an unused
single-product helper and failed the CUDA warning-denied build; .178 initially
lacked the baseline source file. The source was copied into a new anchor rather
than mutating its frozen parent. Final successful directories are:

- .179: `/home/wlc004s/lunaflux-ako-attention-pair-20261005.2NTlePfR/experiment-v2`.
- .178: `/home/wlc003s/lunaflux-ako-attention-pair-20261005.iplwk9Nh/experiment-v3`.

Archives downloaded without overwrite under
`/tmp/lunaflux-ako-attention-pair-20261005.8e4v1Vy0`:

- `measurement179.tar.gz` SHA-256:
  `fbd8fb0d805c13fb1b27bd6a3cc5b8b73db32d0e1f7de18c9c8c8b8d22f27b53`.
- `measurement178.tar.gz` SHA-256:
  `0e551d55ca7a29c714d43b550ea865965bef427d1f3f5ccf31ad90d3b93caea6`.

Local hashes match remote hashes. All **233** archived manifest members verify
locally. `final-report/report.json` records every paired cell and both opcode
tables. Reproduction automation: `ako_attention_product_pair.mbtx` and
`ako_attention_product_pair_report.mbtx` in `benchmarks/gpu_pipeline`.
