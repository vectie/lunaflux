# Fresh Qwen performance distance — 2026-09-08

## Scope and method

All three engines were freshly measured, sequentially, on the same RTX 5060 Ti
(`GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`, PCI `00000000:17:00.0`).
No simultaneous GPU benchmark or profiler was running.

- Model: the existing admitted Qwen3-0.6B BF16 weights, identical input token IDs.
- Greedy generation, fixed output lengths, EOS ignored, prefix/radix reuse off.
- Workload vector: input/output pairs `(59,256)`, `(128,128)`, `(512,64)`,
  `(1528,32)`, each at concurrency 1 and 8.
- One warmup and two measured trials per cell; arithmetic mean of trial output
  tokens/sec. Throughput includes prefill and client/server overhead, excluding
  model startup. TTFT and decode spacing are client-observed means across the
  measured requests. Decode spacing excludes the first token.
- This is a short closed-loop comparison, not a saturation curve, tail-latency
  study, or statistically strong estimate of small percentage differences.

LunaFlux executable code is commit `f69e798`; HEAD `3160a72` differs only in
documentation. The four release executables were rebuilt from that exact source.
The Qwen AOT bundle is the validated aligned-partial bundle from `8b1f8f5`.
The new row-variant infrastructure is present, but new measured row variants
are **not installed in this bundle**. Thus this run measures the current usable
Qwen serving configuration, not the synthetic dense fixture's 21% improvement.

Baselines use their existing pinned environments and launchers: vLLM 0.24.0
and SGLang 0.5.2, BF16, FCFS, maximum 32 running requests, stream interval 1,
single GPU. Full launch arguments and server logs are archived. This is not a
comparison against arbitrary versions or differently tuned deployments.

## Throughput

Output tokens/sec; higher is better. Ratios are competitor / LunaFlux, not
percentage latency reductions.

| Input | Output | C | LunaFlux | vLLM | SGLang | vLLM / LF | SGLang / LF |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 59 | 256 | 1 | 233.7 | 274.9 | 264.6 | 1.18x | 1.13x |
| 59 | 256 | 8 | 972.0 | 1848.4 | 1760.2 | 1.90x | 1.81x |
| 128 | 128 | 1 | 223.4 | 270.6 | 258.9 | 1.21x | 1.16x |
| 128 | 128 | 8 | 895.1 | 1750.4 | 1646.3 | 1.96x | 1.84x |
| 512 | 64 | 1 | 180.3 | 247.1 | 240.2 | 1.37x | 1.33x |
| 512 | 64 | 8 | 500.7 | 1187.9 | 1131.5 | 2.37x | 2.26x |
| 1528 | 32 | 1 | 86.6 | 178.3 | 171.6 | 2.06x | 1.98x |
| 1528 | 32 | 8 | 126.3 | 443.7 | 419.7 | 3.51x | 3.32x |

LunaFlux is essentially unchanged from the previous full-Qwen run. This is
consistent with using the same Qwen kernel bundle; artifact selection support
alone does not turn a microbenchmark winner into a serving optimization.

## Latency breakdown

Each triple is LunaFlux / vLLM / SGLang. All values are milliseconds.

| Input/output | C | Mean TTFT | Mean post-first-token spacing |
|---|---:|---:|---:|
| 59/256 | 1 | 20.5 / 17.5 / 27.0 | 4.20 / 3.57 / 3.67 |
| 59/256 | 8 | 49.9 / 35.2 / 56.4 | 8.03 / 4.17 / 4.29 |
| 128/128 | 1 | 24.0 / 23.5 / 28.0 | 4.31 / 3.52 / 3.65 |
| 128/128 | 8 | 94.6 / 49.2 / 62.5 | 8.16 / 4.19 / 4.35 |
| 512/64 | 1 | 64.5 / 26.0 / 29.0 | 4.56 / 3.67 / 3.73 |
| 512/64 | 8 | 345.7 / 92.5 / 98.8 | 10.69 / 5.30 / 5.52 |
| 1528/32 | 1 | 212.5 / 56.5 / 60.5 | 5.00 / 3.87 / 3.98 |
| 1528/32 | 8 | 1028.4 / 223.9 / 245.9 | 31.13 / 10.83 / 11.60 |

The highest priority remains long-prefill concurrent execution: its C8 TTFT is
4.59x vLLM and 4.18x SGLang; post-first-token spacing is 2.88x and 2.68x.
These are end-to-end symptoms, not isolated kernel attribution. This run does
not distinguish attention, projection, batching, or queueing costs by itself.

## Correctness scope and retention

Every measured request reached its expected terminal event and output count.
Token sequences were retained, not assumed identical. The 59/256 C8 cell had
3 unique sequences for LunaFlux, 2 for vLLM, and 2 for SGLang across measured
requests. vLLM also had 2 sequences in 128/128 C8 and 512/64 C8. Other cells
had 1 per engine. These observations neither prove cross-engine numerical
equivalence nor establish that LunaFlux's existing C8 variation is harmless.

Run root: `/dev/shm/lunaflux-distance-f69e798-20260908-r1`.
Per-engine results: `/dev/shm/lunaflux-baselines-20260906-r1-ENGINE-distance-20260908-r1`.
All raw SSE, request/trial JSON, launcher arguments, build/server logs, source
archive, release executables, summary, and file hashes are retained.

Local archive: `/private/tmp/lunaflux-distance-results-20260908-r1.tar.gz`.
SHA-256: `6486b139a4edb03d0ecd0f232a093fabe3102a87bbf7753fb5386eb79790faf3`.
All owned benchmark server groups were stopped; the final GPU process list is
empty. Production deployment was not changed.
