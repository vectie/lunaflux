# Qwen compiler runtime benchmark — 2026-09-05

The latest measured compiler runtime is `aa61315`, including compiler changes
through `68a8690` and the subsequent v4 bundle compatibility fixes. On the
RTX 5060 Ti, the 12-point token vector shows essentially unchanged short
decode and small observed concurrent-prefill improvements. Median changes
against the historical `ef1da84` quick run range from -0.12% to +3.45%.
This is not a statistically established or output-equivalent speedup.

## Measurement scope

- Qwen3-0.6B BF16, the same existing token-ID request fixtures, greedy output,
  loopback native-framed runtime plus token-ID SSE bridge.
- RTX 5060 Ti, CUDA 13.1.115, driver 590.48.01. GPU UUID
  `GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`, PCI `00000000:17:00.0`.
  The RTX 2080 was not used; there were no concurrent GPU jobs.
- Token vectors: (59,256), (128,128), (512,64), (1528,32); concurrency 1/8/32.
  Each point ran three sequential trials, with 8/8/32 measured requests
  respectively. Each trial excluded one expected-output learning request and
  one additional warmup.
- Reused the committed `quick_token_id_benchmark.py`, with
  `--learn-expected --warmups 1 --timeout 300`. This is a repeated
  **single-prompt quick matrix**, not the earlier diverse-prompt four-engine
  campaign. vLLM, SGLang, and llama.cpp were not rerun.
- Throughput is total completed output tokens divided by measured wall time,
  including HTTP/bridge/runtime work but excluding startup and warmup.
  The runner records whole-response latency, **not TTFT or ITL**.
- Baseline is the earlier `ef1da84` measurement of the same request files,
  request counts and concurrency. It was not rebuilt or interleaved with the
  new trials. The range below is the new three-trial minimum/maximum, not a
  confidence interval.

## Throughput vector

Units are output tokens/s. `Different` counts measured outputs differing
from the current trial's excluded single-request greedy sequence, across all
three trials; it is not an independent numerical-reference verdict.

| Input | Output | Concurrency | Historical ef1da84 | New median | New range | Observed change | Different |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 59 | 256 | 1 | 85.476 | 85.403 | 85.389–85.429 | -0.09% | 0/24 |
| 59 | 256 | 8 | 493.020 | 499.419 | 491.769–499.867 | +1.30% | 24/24 |
| 59 | 256 | 32 | 1061.318 | 1064.559 | 1063.553–1067.482 | +0.31% | 85/96 |
| 128 | 128 | 1 | 85.069 | 84.968 | 84.729–85.137 | -0.12% | 0/24 |
| 128 | 128 | 8 | 467.605 | 469.360 | 463.207–471.507 | +0.38% | 24/24 |
| 128 | 128 | 32 | 953.978 | 964.707 | 954.409–966.256 | +1.12% | 96/96 |
| 512 | 64 | 1 | 65.420 | 65.374 | 65.301–65.403 | -0.07% | 0/24 |
| 512 | 64 | 8 | 288.574 | 292.600 | 285.951–295.657 | +1.39% | 22/24 |
| 512 | 64 | 32 | 416.892 | 431.279 | 428.000–432.232 | +3.45% | 78/96 |
| 1528 | 32 | 1 | 31.179 | 31.280 | 31.266–31.290 | +0.32% | 0/24 |
| 1528 | 32 | 8 | 78.726 | 80.393 | 80.323–80.406 | +2.12% | 0/24 |
| 1528 | 32 | 32 | 97.831 | 99.609 | 99.398–99.684 | +1.82% | 0/96 |

All 576 measured requests completed with the requested output length:
69,120 tokens total. All 37 benchmark/smoke stderr files and the final server
stderr were empty. The preliminary 32-concurrent-request smoke produced the
expected two-token sequence `[92648,4532]` in all 32 responses. The isolated
benchmark service then drained and exited, and the GPU process list was empty.
No production deployment was changed.

## Output consistency limitation

329/576 measured requests differ from their trial's single-request sequence.
All these differences are in concurrent points; all 96 measured c1 requests
match their own learned sequence. The historical baseline already had
116/192 such differences, so this is not a newly observed phenomenon, but
neither run establishes numerical correctness of the divergent continuations.

The new c1 outputs are stable across all three trials. Compared with the
historical c1 hashes, the 59→256 and 128→128 continuations changed; the 512→64
and 1528→32 continuations did not. Thus `--learn-expected` must not be read as
an external correctness check or proof of semantic equivalence. No numerical
reference/tolerance comparison was performed here, and no causality for the
differences is established. The throughput table is descriptive, not a
correctness-qualified performance win.

| Input/output | New learned token-sequence SHA-256 |
| --- | --- |
| 59/256 | `5324c13248cac67819d2c93fb0b793cdcaeb2e5cb63b0e5362ce0b053ce57095` |
| 128/128 | `404bd2fafda0e53d72ec3b9bcd7bf7c045cb10e62e8e6f9002a054ab7789f55b` |
| 512/64 | `0ccf08b647e2edc778f1391697eb36884032ebf014fbc38ececb7c7b938c0375` |
| 1528/32 | `61f0b435fdcd8081c6ae0b129f19f32a0b9ee082b210947263e2dd0816433ddf` |

## What the compiler build contains

The loaded v4 bundle contains nine modules, including residual/RMSNorm,
QKV/QKNorm/RoPE/KV-write, base/wide prefill, two-way and eight-way partitioned
prefill partial/merge pairs, and split decode attention.

| Attention role | Query tile rows | KV partitions | Threads | Shared bytes/block |
| --- | ---: | ---: | ---: | ---: |
| Base prefill | 32 | 1 | 256 | 45,056 |
| Wide-query prefill | 32 | 1 | 256 | 45,056 |
| Partitioned prefill partial | 32 | 2 | 256 | 65,536 |
| Deep partitioned prefill partial | 32 | 8 | 256 | 65,536 |
| Partition merges | 32 | — | 256 | 0 |
| Split decode | phase-specific | — | 256 | 34,076 |

In this build, **base and wide-query prefill are the same final cubin and
symbol**, SHA-256
`f2dc5a565c4653cf1f9ae1f9d0dc5112a1b340379560b766fd883ca802f9e6e0`.
They do not provide two different tile implementations for this workload.
The presence of partitioned modules does not prove a particular measured
request selected them; dispatch/individual-kernel activity was not profiled.

The projection CSE/DCE increment also leaves the current elaborated Qwen
projection graphs at a fixed point, as documented in
[compiler optimization](COMPILER_OPTIMIZATION.md). It adds general compiler
capability without changing those graphs' kernel computation. Consequently,
these structural compiler changes alone are not evidence of runtime speedup.
This benchmark cannot isolate the contribution of any individual compiler
pass. It also does not cover contexts longer than the listed vectors.

## Preparation fixes and reproducibility

Preparation exposed v3-only consumers of the already-v4 exporter output.
The following small fixes were committed and pushed before measurement:

- `3ee5f98`: materializer and root-plan augmentation accept v3/v4.
- `e602ad5`: root assembly accepts v3/v4, with both-version regression fixtures.
- `aa61315`: worker bootstrap routes v4 through the reusable-bundle parser,
  including a regression using actual exporter output.

The related root-assembly checks, six serving-validator tests, warning-denied
native check, and 41 bootstrap/admission tests passed. The rebuilt remote
release worker passed compilation and the live benchmark smoke. This does
not replace the separate sanitizer or full release campaigns.

Only committed source was uploaded; unrelated dirty/untracked model-family
work was excluded. Kernel artifacts were compiled from `68a8690`; the later
three commits fix consumption/bootstrap and do not change those kernels.
The remote source directory retains its original `68a8690` name.

- Source archive: `/tmp/lunaflux-source-aa61315-benchmark-20260905.tar.gz`,
  SHA-256 `ad1e1c4a2fff20a50ab1209a610e271e4c3c3bc2da96974c61ee426d5b1b8b03`.
- Worker binary SHA-256:
  `44a658987ded853c2ea3bb52ec88f0cce85e65479eb0de45aa6c65ffbc0bf1ce`.
- Model content SHA-256:
  `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
- Reusable bundle SHA-256:
  `f998a04282d5140e9760cf6dc343ea9a29044ff9e5183efda380629334c16812`.
- Results, compact module metadata, build/preparation/server logs and historical
  quick-run JSONs: `/tmp/lunaflux-benchmark-aa61315-results-20260905.tar.gz`.
  Downloaded to
  `/private/tmp/lunaflux-qwen-benchmark-68a8690.lArb77/lunaflux-benchmark-aa61315-results-20260905.tar.gz`.
  Archive SHA-256:
  `7d3ff0fd0fea618d7d35e1bdb40dc2c1a501acd9a6f58ec0e738f0e3855fcc1d`.

With the user's cleanup approval, four obsolete remote `_build` caches were
removed, freeing approximately 1 GB. Source, model, production and historical
result directories were preserved; those build caches are regenerable.
