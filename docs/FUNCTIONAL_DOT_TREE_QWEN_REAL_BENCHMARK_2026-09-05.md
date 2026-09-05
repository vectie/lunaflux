# Deterministic dot tree: real Qwen serving A/B — 2026-09-05

The new compiler-generated single-token ingress improves measured Qwen3-0.6B
single-request throughput by **13.7–38.1%**, not the 6.17× observed in the hot
single-layer microbenchmark. Concurrency 8/32 is essentially unchanged. Long
generation is **not token-identical** to the previous numerical schedule; this
is a performance result, not completed model-quality qualification.

## What actually ran

- Real admitted Qwen3-0.6B BF16 weights, RTX 5060 Ti, CUDA 13.1.115.
  GPU UUID `GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`, PCI `00000000:17:00.0`.
- Committed compiler implementation `2564c31`, recorded by source snapshot
  `a1cc6a51376db1d364ce28c79a05464e53008474`. No unrelated dirty-tree changes
  were packaged. The runtime/worker/bridge sources have no changes between
  `aa61315` and this commit, so both variants reuse the same existing binaries.
- Same model, model plan, weights, runtime, worker, bridge, greedy policy,
  launch dimensions and other eight kernel modules. A line-by-line comparison
  of the canonical runtime bundles finds only `module_1_module_sha256` and
  `module_1_module_hex` different. New deployment uses the existing exporter
  and materializer, not manual mutation of hashed runtime configuration.
- Loopback token-ID SSE endpoint, native-framed c32 benchmark profile,
  concurrency 1/8/32. Both instances started separately, published readiness,
  served requests, acknowledged drain, and exited zero with `child_closed=1`.
  Both server stderr logs are empty; the GPU has no test processes afterward.
  Production deployment was not changed.

## Token-vector measurements

Each cell is the median of three trials. Per trial there are 4 measured requests
at C1, 8 at C8, and 32 at C32; one excluded reference-learning request and one
excluded warmup precede each trial. Fixed output lengths use `ignore_eos=true`.
There are **1,056 measured requests and 126,720 generated output tokens** across
both variants, excluding warmups, raw sequence samples and the independent
reference checks below. Every cell returns its full requested token total.

| Input/output tokens | Concurrency | Old output tok/s | New output tok/s | Change | Old response p50 ms | New response p50 ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 59/256 | 1 | 85.57 | 118.21 | +38.1% | 2992.5 | 2161.1 |
| 59/256 | 8 | 498.37 | 500.84 | +0.5% | 4082.3 | 4071.2 |
| 59/256 | 32 | 1064.15 | 1065.77 | +0.2% | 7607.3 | 7591.9 |
| 128/128 | 1 | 84.84 | 116.78 | +37.7% | 1507.6 | 1095.3 |
| 128/128 | 8 | 464.96 | 471.76 | +1.5% | 2184.5 | 2145.8 |
| 128/128 | 32 | 956.83 | 955.37 | −0.2% | 4167.7 | 4180.6 |
| 512/64 | 1 | 65.33 | 83.06 | +27.1% | 978.9 | 769.6 |
| 512/64 | 8 | 292.56 | 286.28 | −2.1% | 1685.2 | 1730.0 |
| 512/64 | 32 | 432.77 | 432.30 | −0.1% | 4399.6 | 4407.5 |
| 1528/32 | 1 | 31.25 | 35.54 | +13.7% | 1023.7 | 899.6 |
| 1528/32 | 8 | 80.11 | 80.47 | +0.4% | 3095.4 | 3082.4 |
| 1528/32 | 32 | 99.40 | 99.43 | +0.0% | 10060.2 | 10058.5 |

These are complete-response latencies, **not TTFT or inter-token latency**.
The existing quick client reads the entire SSE response. Throughput includes
prefill, decode and client/bridge overhead; it is not a standalone decode kernel
rate. Old and new run in separate blocks, old first, rather than randomized
alternation. No other GPU workload overlaps either block. CPU-only preparation
of the new test deployment overlaps part of the old block. No confidence
interval or broader sustained-load claim is inferred from three small trials.

The shape dependence agrees with the implementation: the new branch runs only
when the packed token count is one. Multi-token WMMA is unchanged. Larger
concurrent batches therefore see little benefit. Long prefill still spends time
in unchanged computation. This is an explanation from code and observed shape
trends, not a new component-level profiler breakdown.

## Real-weight numerical observations

All C1 measured requests match their own learned reference; the learned C1
hash is also stable across all nine trials/concurrency settings for each
variant and token vector. That is repeatability, not independent correctness.

Full raw old/new C1 sequences are retained. Zero-based first divergence:

| Input/output tokens | First differing output index | Exact old/new sequence |
| ---: | ---: | --- |
| 59/256 | 24 | No |
| 128/128 | 95 | No |
| 512/64 | 42 | No |
| 1528/32 | None | Yes, all 32 tokens |

Changing the explicit FP32 reduction tree changes rounding and can change greedy
decisions; autoregressive continuation can then diverge. These results do not
establish whether the changed long outputs are equally good or degraded.
Teacher-forced/logit comparison against an independent implementation and a
broader quality corpus remain necessary to resolve that question. In
particular, the earlier exactly representable dyadic kernel fixture does not
answer it.

Concurrent outputs already differed from C1 before this change. Summing the
quick client's `incorrect_requests` field (which means token mismatch against
its self-learned C1 output, not a task-quality score):

| Input/output tokens | C8 old/new mismatches, out of 24 | C32 old/new mismatches, out of 96 |
| ---: | ---: | ---: |
| 59/256 | 24/24 | 85/96 |
| 128/128 | 24/24 | 96/96 |
| 512/64 | 21/21 | 78/81 |
| 1528/32 | 0/0 | 0/0 |

Those mismatch cells retain the client's nonzero exit code rather than being
relabeled as correctness passes. Detailed mismatch records are capped at eight
by the existing client; full raw SSE is retained for the separate C1 samples.

An additional existing independent Transformers reference uses a 23-token
prompt and expected greedy IDs `[92648, 4532]`. The new runtime matches **44/44
measured requests**: 4 at C1, 8 at C8, 32 at C32, with empty stderr. This narrow
two-token smoke test is not a substitute for long-output numerical/quality
validation. No vLLM, SGLang or llama.cpp performance rerun was performed here.

## Reproduction and saved artifacts

Remote results: `/tmp/lunaflux-dot-tree-real-20260905-r1`.
New measured deployment:
`/dev/shm/lunaflux-dot-tree-real-deployment-20260905-r2`.
The first unused bundle draft had different combined-attention launch metadata;
it was caught before new-kernel serving, preserved, and replaced by the
metadata-identical r2 bundle. Initial setup also required correcting the test
driver's PATH/cwd and moving the final deployment to memory-backed storage when
the host's disk lacked space. Neither draft produced new-kernel timing rows.

The result archive contains requests, raw samples, per-trial JSON/stderr/exit
codes, server lifecycle logs, the exact client, MoonBit orchestration and
summarizer, source snapshot and old/new bundles. It excludes deployment weight
copies. Hash verification runs outside the measured request path.

Downloaded result archive:
`/private/tmp/lunaflux-dot-tree-real-results-20260905-r1.tar.gz`, SHA-256
`18d58704923c97f2684565b8622c793d616d8619ec5672b96e122a842267a7c8`.
Its checksum matches the remote archive; `FILES.sha256` records the individual
result files.

Key identities:

- Source archive SHA-256:
  `86e600c4df0d2fc741ecdfdced068f42e3484612ed7e0f76a039a78bf734e5c3`.
- Shared runtime executable SHA-256:
  `f5b838d3eba5feff48c5e854e05c80261fd53f736bc7bda205d1b035b9ea3f03`.
- Old ingress CUBIN SHA-256:
  `cbef54174f67c11ae99ea77464e6b1050fd8551482b21cb80343db25edea9fa0`.
- New ingress CUBIN SHA-256:
  `4dcd31411241ed29c3f6e56cceee819cbbff16f9dedd7e333889de14255b5c2b`.
- New measured launch SHA-256:
  `31204bf3c34af78ce811b8bbfbad53324279b629a77572004d21c1c8f7cb8c2b`.
- New measured runtime bundle SHA-256:
  `b8f4b9654a51f926cea412ef8f797bdd4d73394c5838783d1c20095ad01c8bb7`.

The compiler architecture remains unchanged by this testing work: a pure,
explicit numerical policy and deterministic reduction schedule, with hardware
operations confined to CUDA lowering. The strategy is general; the measured
benefit here is specifically its single-token fused-Qwen ingress consumer.
