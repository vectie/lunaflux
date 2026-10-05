# AKO mixed-route serving and parallel decode frontier

## Bounded work on both Sparks

After the complete mixed-chain measurements, .179 independently bound its own
observations through the existing startup attention table and ran four fresh
uninstrumented serving starts in ABBA order. Concurrently, .178 checked whether
the partitioned-decode advantage extends to larger batches. One GPU workload
per machine; no production service or kernel bytes were changed.

The serving budget was two cells, four starts and four trials per cell/start:
trial zero is warmup, leaving six timing samples per side/cell. Every request
body is checked byte-for-byte across trials and starts. Every 64-token output
vector, raw timing sample and memory sample is retained. Prefix reuse is off.
The outer serving jobs have finite runtime, MemoryMax=48 GiB and zero swap;
the launcher retains its existing per-child limits and 32 GiB available-memory
reserve. Decode probes use an 8 GiB limit and the same reserve.

## Fresh serving results

Qwen3-0.6B BF16, input 32,512 tokens per request, output exactly 64 per request,
prefill chunk budget 2,048. Baseline already contains measured pure-prefill
routes; the candidate adds measured **mixed** routes without removing those
records. The baseline/candidate worker, model, AOT kernels and request bodies
are unchanged. This is not a new vLLM/SGLang comparison.

| Concurrency | Baseline wall, ms | Mixed-tuned wall, ms | Completion reduction | Baseline / tuned output tok/s | Baseline / tuned TTFT, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 control | 4572.0 | 4594.5 | -0.49% | 14.00 / 13.93 | 3107.0 / 3123.0 |
| C2 | 9855.5 | 9476.0 | 3.85% | 12.99 / 13.51 | 4926.5 / 4739.0 |

These are medians; the small C1 difference is not evidence of a systematic
regression. The whole attention-chain improvement (~3× at the mixed step)
does not imply a 3× whole-model improvement: it affects a small fraction of
the total serving work.

**Numerical caveat:** baseline C2 output vectors vary across identical-request
trials, changing up to 41/28 positions for the two rows relative to the first
trial. Candidate C2 vectors are repeatable in all eight waves across its two
starts. C1 is repeatable on both sides. This does not establish quality
equivalence or resolve the earlier batch-dependent BF16 near-tie investigation.
Report this as fixed input/output-length timing, not identical greedy output.

The table uses .179's whole-chain observations for the mixed query buckets
2048/4096, two rows, context bucket 32768; candidate 1 beats candidate 5.
Existing pure-prefill table bytes remain an exact prefix. The immutable source
scope is `854fff066aebc53d7a7e6de80744a0bf0142fc9f6792449861018d678bf83ae8`;
the newly materialized runtime bundle is
`f2e7d310a8c3203d0f92446187b19aba6058664e85b088e7091423b004c3c9d5`.

## Larger decode batches on .178

Five alternating paired probes compare ordinary c468 against the partitioned
partial + merge chain from the **same** c468 cubin, preserving full output/KV
checks and the sampled FP64 oracle.

| Rows / history | Ordinary median, µs | Partitioned median, µs | Median paired gain | Numerical result |
| --- | ---: | ---: | ---: | --- |
| 4 / 32512 | 2276.728 | 2274.720 | -0.30% | bitwise equal, maximum error 0 |
| 8 / 16376 | 2296.479 | 2302.493 | -0.69% | maximum error 0.000488281, within 0.003 tolerance |

Paired gains and the ratio of separate medians need not agree. Neither cell
shows a robust advantage; do not extrapolate C1/C2's win or enable these routes
globally. Both passed the sampled oracle and left KV unchanged. C8 is a probe
capability, not a new production split-decode admission. No new hardware
counters or sanitizer campaign was collected for these two unchanged kernels.

## Compiler boundary and validation

Changes are offline MoonBit automation only: exact mixed-domain bucket
construction, additive startup records, repeatable ABBA input/vector auditing,
reuse of frozen diagnostic binaries and larger-batch probe contracts. Runtime
selection remains an immutable startup-produced lookup; no new request-path
measurement, filesystem validation, JIT or model-specific branch was added.

Affected scripts pass warning-denied native checks/tests. A separate completed
dispatch trace confirms the actual mixed step at sequence 32: one prefill row
with 1536 queries/history 30976 and one decode row/history 32512. Bucket 4068
selects measured baseline owner 27 (`kind=17`), then actually executes its mixed
companion 28 (`kind=3/4/5`). Previously the same bucket selected split owner 69
and executed companion 70. The candidate-to-owner startup mapping and explicit
phase-companion selection establish the family; owner numbers alone do not.
The diagnostic worker is not a timing sample or deployable binary.

Two failed diagnostic preparations are preserved: a reused-source symlink
violated canonical executable-path admission, then the old trace's child-unit
names collided with an earlier run. The adapter now resolves the frozen source
path and gives reused-source traces experiment-specific names. The isolated
trace completed, drained and left the GPU idle. Serving ABBA did not have those
failures and was not repeated or replaced.

## Result locations

- Serving: `/home/wlc004s/lunaflux-ako-mixed-serving-20261005.kElWPIRZ`.
- Decode frontier: `/home/wlc003s/lunaflux-ako-decode-frontier-20261005.3zDsjBm4`.
- Local frontier archive:
  `/tmp/lunaflux-mixed-serving-20261005.0PsYCpY8/frontier178.tar.gz`, SHA-256
  `267effbe0b99b37c182f52ab32574504111ae1c5b31870e9e7e304984393abf4`;
  all extracted manifest entries verified.
- Local serving archive:
  `/tmp/lunaflux-mixed-serving-20261005.0PsYCpY8/serving179.tar.gz`, SHA-256
  `51694b9a3e77a19b8f359a04d883055038a8d49a8f4d4cb76146781120ab9073`;
  every extracted manifest entry verifies. It includes all four serving runs,
  request bodies/vectors, separate dispatch logs, both preserved preparation
  failures and a copy of the immutable mixed-probe measurement records.

See [the preceding whole-chain measurements](BENCHMARK_AKO_DUAL_SPARK_MIXED_CHAIN_2026-10-05.md)
for mixed-chain correctness/sanitizer scope and the original C1/C2 decode wins.
