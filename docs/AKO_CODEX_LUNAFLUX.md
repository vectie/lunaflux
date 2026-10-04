# AKO-style optimization for Codex and LunaFlux

## Adaptation

Reviewed `../ako` (AKO4ALL), revision
`bbd0e1cf1ce2fb19d4322a932680f4c3d175d80e`, MIT, TongmingLAIC 2026.
The useful component is its profile → hypothesis → change → paired measurement
→ record loop, not its Python KernelBench evaluator or Claude-specific setup.
Upstream speedup claims have not been reproduced here.

The Codex skill is `.agents/skills/lunaflux-ako/SKILL.md`; the upstream license
is retained alongside it. A local symlink at
`/Users/kq/.codex/skills/lunaflux-ako` makes it discoverable outside this checkout.
Skill metadata follows the [Codex skill format](https://learn.chatgpt.com/docs/build-skills).
No global agent settings or permissions were changed.

| Upstream behavior | LunaFlux adaptation |
| --- | --- |
| Claude skill plus Python/PyTorch evaluator | Codex skill plus native MoonBit `.mbtx` automation |
| Standalone kernel speedup | Runtime launch geometry, full kernel chains, then serving A/B |
| Broad language/kernel rewriting | Pure compiler plans and explicit terminal backend lowering |
| Whole-tree staging and best-commit restoration | Selective owned-file commits; preserve unrelated edits and failures |
| Open-ended optimization loop | One route-selection hypothesis in this pilot; finite paired trials |
| GPU benchmark wrapper | Serialized GPU work, user-systemd limits, no swap, 32 GiB memory reserve |

This is an offline development workflow. It adds no production Python, JIT,
token-step allocation, filesystem checks, cryptography, or profiler dependency.
Existing functional multi-layer compiler boundaries remain unchanged.

## Components

- `benchmarks/gpu_pipeline/ako_trial.mbtx`: JSON-driven paired BF16 probe runner.
  Records commands, raw output, all five paired samples, memory observations and
  artifact identities. Rejects partial captures, invalid/errorful measurements
  and a supposed robust win containing a paired regression. Correctness is
  owned by the probe and additionally checked against its 0.003 error contract.
  A successful result means a completed experiment, not release promotion.
- `ako_long_prefill.mbtx`: stages frozen artifacts, measures exact query/row/
  context shapes, binds missing measured-route slots, and prepares a new
  benchmark-only deployment. Phases: `prepare`, `bind`, `serve-prepare`.
  Existing artifacts and directories are never overwritten by the adapter.
- `ako_serving_trial.mbtx`: reuses the frozen bounded serving launcher and token
  generator. Seven cells, two fresh starts, one excluded warmup and three
  measured trials per cell/start. Reference engines are not remeasured by this
  route-only A/B adapter.
- `ako_report.mbtx` retains completion/TTFT samples and full per-request token
  vectors; `ako_finish.mbtx` runs scoped sanitizer/counter gates and seals the
  completed experiment without overwriting an archive.
- `selected_policy_probe.cu`: the diagnostic history bound now follows declared
  page capacity instead of the old hardcoded 8K experiment ceiling. Checks use
  64-bit arithmetic before allocation; aggregate pages remain checked as well.

The trial JSON has schema `lunaflux-ako-trial-v1`, explicit absolute executable
and artifact paths, GPU UUID, `reserve_kib`, `minimum_gain`, and an array of
`{label,args}` workload entries. It is not a generic correctness evaluator:
the supplied probe must enforce deterministic correctness and print the
documented paired timing/error fields. External process limits remain required.

## Pilot hypothesis and controls

The [long-context baseline](BENCHMARK_LONG_CONTEXT_2026-10-04.md) found that the
measured attention route table stopped at 8K while serving allowed 32K inputs.
Hypothesis: above that boundary, an existing fast compiler kernel is not
selected because its measured slots are absent. This is a selection experiment,
not a new kernel algorithm or proof that all long-context cost comes from it.

Machine: Spark GB10, sm121, GPU
`GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`; Qwen3-0.6B BF16 for serving.
Frozen source/runtime:
`/home/wlc004s/lunaflux-domain-serving-v4-20261004.QmE9WZ7s`.
Pilot output:
`/home/wlc004s/lunaflux-ako-pilot-v2-20261004.bfWwShhe`.

Probe comparison: frozen wide kernel versus frozen compiler c322, using actual
runtime launch geometry. Query count 2,048, rows 1/2/4, final per-row contexts
8,192/16,384/32,768; plus the 1,792-query final partial chunk at context 32,512.
Five alternating-order paired trials with 30 CUDA-event-timed repeats each.
Timing uses warmed/reused input buffers; this is **not** Ako's fresh-random-input
timing regime. Deterministic varied BF16 inputs, shuffled physical page mapping,
sampled independent FP64 softmax oracle, output comparison and unchanged KV
checks are performed before timing.

The microprobe ran under 8 GiB MemoryMax, zero swap, 900-second timeout. Serving
uses the existing 64 GiB engine / 2 GiB bridge envelopes, zero swap, bounded
runtime, and the 32 GiB reserve. No 1M-context allocation is attempted.

The first attempt at
`/home/wlc004s/lunaflux-ako-pilot-20261004.2gC2yByx` stopped before GPU work:
the wide recipe directory did not contain its cubin. The adapter now stages
the exact frozen recipe and binary together. Additional materialization attempts
preserve their logs; canonical release/executable paths fixed symlink rejection
without weakening deployment validation.

## Results

All ten probe shapes passed the numerical/KV checks. c322 reduced GPU time by
approximately 76–77% relative to the wide kernel, including the partial tail.
Representative median timings:

| Rows | Context | Wide ms | c322 ms | Reduction |
| --- | ---: | ---: | ---: | ---: |
| 1 | 8,192 | 8.221 | 1.859 | 77.4% |
| 1 | 16,384 | 17.140 | 3.909 | 77.2% |
| 1 | 32,768 | 35.532 | 8.320 | 76.6% |
| 2 | 32,768 | 37.734 | 8.633 | 77.1% |
| 4 | 32,768 | 38.248 | 8.887 | 76.8% |
| 1, partial query | 32,512 | 30.770 | 7.331 | 76.2% |

No new arithmetic or kernel source was introduced. Only the missing measured
16K/32K route slots are added; measured controls below that remain intact and
there is no cross-bucket extrapolation. The frozen exporter successfully bound
the table to the runtime's exact route scope. Table SHA-256:
`a2a4e87c05bfd70636fc4f50cc34563d339a6b327bb4ede98e5dc6628e6a77b0`.
The new launch is
`1dfcc2e09f96ee603d2bc9d9ac381c2a08d2b43d328bbe8e654454cd7725fbc5`;
its worker remains
`d3b4d60d1eacb6698209fc14f88061b3a47c1fa9ac0db3c306eee454f9cc77b2`.

Serving A/B completed: two fresh starts per side, three measured trials after
one warmup per cell/start, 64 requested output tokens. All control runs preceded
the tuned runs; this is **not** a counterbalanced serving campaign. The unchanged
4K/8K controls show less than 1% timing movement. Every request produced all
64 tokens; prefix reuse was disabled, as in the frozen baseline.

| Input / concurrency | Baseline completion s | Tuned completion s | Reduction | Matched output vectors |
| --- | ---: | ---: | ---: | ---: |
| 4,096 / C1 | 0.684 | 0.687 | -0.4% | 6/6 |
| 8,192 / C1 | 0.999 | 1.006 | -0.8% | 6/6 |
| 16,384 / C1 | 2.473 | 1.829 | 26.0% | 6/6 |
| 32,512 / C1 | 7.394 | 4.210 | 43.1% | 6/6 |
| 32,512 / C2 | 15.122 | 9.011 | 40.4% | 0/6 |
| Mixed lengths / C4 | 11.338 | 7.958 | 29.8% | 2/6 |
| 32,512, broad vocabulary / C1 | 7.418 | 4.243 | 42.8% | 6/6 |

For matched 32K C1, first-token time fell **5.914 → 2.731 s** (53.8% less),
and output throughput rose **8.66 → 15.20 tok/s** including prefill time.
Post-first-token time was essentially unchanged. At 16K C1, TTFT fell
1.508 → 0.860 s. The broad-vocabulary 32K control also matched all output
vectors and improved, so the effect is not confined to the 12-token input pool.

C2 and mixed-length measurements are **fixed-work timing, not strict numerical
equivalence**. Both baseline and tuned runs have within-engine output variation:
C2 has two distinct vectors per side, and mixed C4 has four per side. The route
change adds further paired differences. The BF16 probe tolerance does not prove
greedy output stability near ties or full-model quality. These data must not be
used to promote a numerically equivalent concurrent deployment. No vLLM/SGLang
reference was remeasured in this A/B, and no cross-framework victory is claimed.

The experiment demonstrates a real selection-coverage gain using existing
compiler kernels—not that AKO itself creates a 4× algorithmic speedup, that new
IR layers caused this gain, or that the compiler is now optimal.

Local warning-denied native checks passed for the adapter scripts. Trial parser
tests passed, including partial/duplicate/nonfinite captures, correctness error
and paired-regression rejection. Eleven existing route/strategy tests passed
with known unrelated migration warnings 79/29/25 disabled; unmodified dependent
packages prevent the warning-denied default command from passing. No broad
toolchain cleanup is included here. Skill frontmatter passed Ruby safe-YAML
validation. The official validator's system Python lacks PyYAML; no dependency
was installed solely for that check.

The full 32K partial-chunk pair passed Compute Sanitizer memcheck, racecheck and
synccheck, with zero errors/hazards. Non-admin Nsight initially returned
`ERR_NVGPUCTRPERM`; that failed capture is preserved. A separately bounded
administrator replay completed successfully without changing driver settings.
Profiled durations are diagnostic, not substituted for unprofiled timing.

The 32K partial-chunk counter capture confirms two distinct AOT symbols and
their actual launch/resource differences:

| Counter / launch property | Wide c2001 | Compiler c322 |
| --- | ---: | ---: |
| CTAs launched | 1,520 | 1,008 |
| Threads / CTA | 64 | 128 |
| Registers / thread | 255 | 235 |
| Allocated shared memory / CTA, kB | 74.880 | 50.304 |
| Active warps, peak fraction | 4.06% | 15.99% |
| Tensor active cycles, peak fraction | 12.29% | 54.23% |
| Profiled duration | 31.015 ms | 7.137 ms |

Shared-memory values are Nsight's decimal kB. The wide schedule's larger
shared-memory reservation limits resident CTAs; c322 reports two resident blocks
and no local allocation in the probe. This corroborates the replay difference
but is not a universal claim that higher occupancy always wins. A server-side
per-step trace was not collected in this pilot; propagation is established by
the exact source-bound route table/deployment, unchanged worker/AOT hashes and
serving A/B—not a new full serving hardware timeline.

## Saved experiment and disposition

Remote archive:
`/home/wlc004s/lunaflux-ako-pilot-v2-20261004.bfWwShhe/measurement.tar.gz`.
Downloaded without overwrite to
`/tmp/lunaflux-ako-results-20261004.if3Il4RS/measurement.tar.gz` and extracted
into its new `evidence/` directory. Remote and local SHA-256 agree:
`bf56ccdb34c16216e9cd149435e907c1da857362e92b0c3b0abfda3abff3da7e`.
The archive contains raw paired trials, serving inputs/outputs/token vectors,
memory samples, counters, sanitizer output, recipes/cubins and bound route data.

Minimum serving MemAvailable was **99.52 GiB**, comfortably above the 32 GiB
reserve. All four serving instances acknowledged drain, exited zero and closed
their children; GPU returned idle. No production release or driver setting was
changed. This is a completed offline pilot with a reproducible C1 performance
win; concurrent numerical stability remains an explicit follow-up, not a pass.
