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
the table to the runtime's exact route scope. Serving A/B is in progress; the
microprobe numbers alone do not establish an end-to-end gain or framework win.

Local warning-denied native checks passed for the adapter scripts. Trial parser
tests passed, including partial/duplicate/nonfinite captures, correctness error
and paired-regression rejection. The official skill validator could not run in
the system Python because PyYAML is absent; no dependency was installed solely
for that check.
