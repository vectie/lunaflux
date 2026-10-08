# Workload-specific decode selection versus the previous best

## Question and bounded decision

Can selecting the partition count for the actual long single-request workload
make the new typed decode-binding path beat the previous best complete package,
without sacrificing short or concurrent serving?

The experiment budget was at most three candidate rounds. Two domains were
measured: partition counts for the existing F32 implementation, then the existing
matrix alternative. This is offline selection through the current immutable
compiler plans and AOT exporter, not another IR layer, model-name branch, runtime
JIT, or request-path autotuner. Production deployment/defaults are unchanged.

The four-partition package recovers the previous best 32K result, but its paired
full-service comparison is a tie, not a new overall win. The matrix alternative
does not improve the exact long-context probe. Keep the previous best control;
retain the new package as a verified workload-specific binding candidate.

## Fixed scope

- Spark .179, GB10 sm121, 48 SMs. UUID
  `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`, PCI `0000000F:01:00.0`.
- Candidate exporter: clean source commit `49abc685`. Runtime worker/launcher
  reuse the corresponding tested captured-route source snapshot. Unrelated
  working-tree changes were not uploaded or built.
- Qwen3-0.6B BF16, 16 query heads, 8 KV heads, head dimension 128, page size 8.
  Each service request generates exactly 64 tokens. The existing varied token-ID
  input vectors and frozen reference images/configurations are unchanged.
- CUDA 13.0.88, `-O3 -arch=sm_121 --fmad=false --ftz=false --prec-div=true
  --prec-sqrt=true --maxrregcount=255 -lineinfo`.
- One GPU workload at a time. Serving memory cap 64 GiB, no swap; live
  `MemAvailable` reserve 32 GiB. Observed service availability exceeded 99 GiB.
- Native MoonBit orchestration only. No request-path filesystem, hashing,
  profiling, tuning, or allocation was added.

Pinned tool hashes:

```text
nvcc              fbb111f057786ddd10ba723d993cc7dd43abf978b6baa32fedd3c9d806dc79e1
compute-sanitizer 5cba659e1cd603aa2370e103425f184368e6ed405da90ce85743c4399c324dbc
```

## Round 1: measure the actual partition workload

The earlier joint sweep selected two partitions using C8/history 8192. That is
not a measurement of C1/32K. The current experiment supplies rows=1 and compiler
context=32768, including the current query, to the existing chain search. The
probe uses 32767 previous tokens. Workgroup targets 16/32/64/128 yield three
unique legal realizations: two, four and eight partitions. The current planner
caps this domain at eight; there is no measured sixteen-partition result.

Five alternating pairs per entry, complete partial-plus-merge chain, CUDA event
timing; captured measurements include both launches inside the graph:

| Partitions | Standalone median µs | Captured median µs |
| --- | ---: | ---: |
| 2 | 650.928 | 645.729 |
| 4 | 578.081 | 575.570 |
| 8 | 582.305 | 578.655 |

The implementation remains c468, KV32, two stages, owned8 non-contracting F32
probabilities. Four partitions reduce captured chain time by about 10.9% against
two. This is not a 10.9% service improvement. Standalone observations were bound
with their correct `standalone-partial-merge-v1` timing scope; captured data were
an independent confirmation, not relabeled as standalone measurements.

The selected composed module compiled byte-identically twice. Forty-two fresh
captured row/history measurements select unsplit versus split execution from
that exact module. Startup prepares the four-partition geometry/workspace from
the v15 descriptor; no token-step selection logic was added. The numerical probe
checks output, KV immutability and a sampled FP64 oracle (maximum absolute error
0.003). Partitioned reductions are not claimed bitwise equivalent for all inputs.
Memcheck/full leak check, racecheck, initcheck and synccheck passed.

## Whole-service A/B against the actual previous best

Control is the immutable `materialize-norm-final/norm/long-final` package in the
2026-10-08 gap-repair campaign, not the deliberately unsplit regression control.
Order: control/candidate/candidate/control, fresh process starts, one warmup and
three measured waves per cell/start. Six measured waves per arm:

| Input / concurrency | Previous best ms | Four-partition package ms | Candidate output tok/s | Completion change |
| --- | ---: | ---: | ---: | ---: |
| 512 / C8 | 712.0 | 711.0 | 720.11 | −0.14% |
| 8192 / C8 | 4690.5 | 4696.0 | 109.03 | +0.12% |
| 32512 / C1 | 3901.0 | 3900.0 | 16.41 | −0.03% |

These differences are smaller than start-to-start variation. None proves a new
win over the previous best. The immediately preceding two-partition report's
4077.5 ms at C1 was slower; 3900 ms recovers that regression, but the two figures
are from separate campaigns and are not a paired causal estimate.

Median per-request TTFT / TPOT (ms), control → candidate:

- 512/C8: 83.5 / 9.659 → 83.5 / 9.698.
- 8192/C8: 1396 / 48.357 → 1396 / 48.500.
- 32512/C1: 2437 / 22.968 → 2440.5 / 22.762.

All C1 token vectors match across arms and repetitions. C8 vectors vary within
both arms as well as between arms: respectively 32/40 and 37/40 matching rows
at short context, 30/40 and 31/40 at 8K, compared with each arm's first wave.
This experiment does not establish batch-invariant generation or model-quality
parity. Raw request token IDs and timing vectors are retained; fixed token counts
alone must not be presented as equivalent model output.

## Round 2: matrix alternative on the same C1 workload

This repeats the actual C1/context32768 chain domain with c468 and c482, targets
32/64 (four/eight partitions). c482 has the explicit
`grouped-head-matrix-bf16-probability-v1` law, unlike c468's F32 probability law.
Its probe must pass the numerical bound, but is not a bitwise-preserving rewrite.

| Candidate | Partitions | Standalone median µs | Captured median µs |
| --- | ---: | ---: | ---: |
| c468 | 4 | 593.343 | 580.025 |
| c482 | 4 | 633.481 | 632.948 |
| c468 | 8 | 590.404 | 588.741 |
| c482 | 8 | 598.133 | 594.672 |

All four complete chains pass probe correctness, but none improves on c468/four
in captured timing. The AKO helper's `improved` label is relative to its frozen
*unsplit* control, not relative to the best split candidate. Therefore the matrix
alternative is not bound into the service package or reported as a service gain.

## Preserved preparation failures

- Initial round-1 export accidentally requested page-table capacity beyond the
  admitted page capacity. It failed before GPU testing; the corrected export is
  separately preserved.
- The first sweep harness expected four variants, although the compiler emitted
  three unique counts. All three completed; the nonexistent fourth was rejected.
  Binding consumes only the three complete five-pair records.
- Initial matrix search requested c483 (KV64/two stages), excluded by this
  head-dimension-128, 49152-byte capability policy. The legal c468/c482 search
  was exported separately; no limits were weakened.
- An initial diagnostic tried C8 with a chain exported for C1. The probe rejected
  the incompatible launch scope before producing C8 measurements. The continued
  matrix sweep contains only the correctly scoped C1 comparisons.

These are retained failed attempts, not successful validation records. No new
instruction-counter capture was performed; the partition hypothesis concerns
work distribution, but this campaign does not attribute the gain quantitatively
to occupancy, memory latency or individual instructions.

## Evidence locations

- Main: `/home/wlc004s/lunaflux-overall-best-20261009.b9EitVKQ`.
- Rejected matrix preparation: `/home/wlc004s/lunaflux-overall-best-20261009.x6NamGAR`.
- Legal matrix sweep: `/home/wlc004s/lunaflux-overall-best-20261009.C6tvFz4o`.

## Selected dispatch, not just exported source

A separate Nsight Systems trace reuses the previously isolated diagnostic parent
that retains profiler injection; the serving worker and selected module are
unchanged. This parent is not deployed. Request epoch windows exclude startup and
warmup. Profiled times are not substituted for unprofiled benchmark results.

- C1/32512: 1764 partial and 1764 merge calls, partial grid `1,8,4`, block 64.
  No unsplit decode calls. Partial plus merge GPU duration is 1008.44 ms in this
  one traced request window.
- C8/8192: 1540 unsplit calls, plus 224 partial/merge pairs across row-1 and
  row-8 buckets. Every observed partial uses `gridZ=4`.
- C8/512: 1736 unsplit calls and 28 partial/merge pairs, also `gridZ=4`.

Thus the new selected partition descriptor does reach serving. The remaining
whole-engine tie cannot be explained by the four-partition route never running.

## Fresh reference comparison protocol

Use two reversed fresh-start orders: LunaFlux/vLLM/SGLang, then
SGLang/vLLM/LunaFlux. Each start runs all three input/concurrency cells with one
warmup and three measured waves. No frameworks run concurrently on the device.
Image identity and the 64-GiB no-extra-swap container limits are verified before
starting; the live 32-GiB host reserve guard also covers the reference runs.

- vLLM image:
  `sha256:73e1b0f3230a9377a3109d20be7f8607963a41a715eb3000bc647f5f73c198ff`.
  BF16, model length 40960, max sequences 32, memory utilization 0.4,
  prefix caching disabled.
- SGLang image:
  `sha256:3a9d399434923689565e1a72f5c3fc409323ce039b65bdbe5c2d6ed14abad3a9`.
  BF16, context length 40960, max running requests 32, static memory fraction
  0.4, radix cache disabled, FCFS, TP=1, streaming interval 1.

These are fresh measurements of the existing frozen reference versions, not an
assertion about the newest upstream releases or every possible configuration.
LunaFlux retains the previously tested reference mixed-prefill and vendor
output/down AOT choices; this is not an all-in-house-kernel comparison.

Six measured waves per engine/cell; generated-token throughput includes prefill
and client-visible completion, not decode-only throughput:

| Input / concurrency | LunaFlux tok/s | vLLM tok/s | SGLang tok/s | Luna completion vs vLLM / SGLang |
| --- | ---: | ---: | ---: | ---: |
| 512 / C8 | 719.61 | 785.28 | 762.47 | +9.13% / +5.96% |
| 8192 / C8 | 109.17 | 113.98 | 112.66 | +4.41% / +3.20% |
| 32512 / C1 | 16.47 | 16.20 | 17.38 | −1.66% / +5.50% |

Median wave completion times (LunaFlux / vLLM / SGLang):
711.5 / 652 / 671.5 ms; 4690 / 4492 / 4544.5 ms; and
3885 / 3950.5 / 3682.5 ms. LunaFlux's two fresh-start C1 medians were 3866 and
3902 ms; report this variation rather than treating 3885 as an exact constant.
All three stayed within the memory guard: minimum available memory 99.27,
65.15 and 65.85 GiB, respectively. No GPU processes remained after completion.

Per-request median TTFT / TPOT in milliseconds:

| Input / concurrency | LunaFlux | vLLM | SGLang |
| --- | ---: | ---: | ---: |
| 512 / C8 | 83.5 / 9.690 | 77 / 8.810 | 91 / 8.833 |
| 8192 / C8 | 1391 / 48.286 | 1370 / 45.746 | 1164 / 53.294 |
| 32512 / C1 | 2429 / 22.714 | 2437 / 23.690 | 2120.5 / 24.429 |

The two remaining regimes are different. Short C8 mainly loses after the first
token. At 32K/C1, LunaFlux's TPOT is already lower than both references, while
SGLang's first token arrives about 309 ms earlier. Continuing to tune the C1
decode partition cannot eliminate this prefill-side deficit. At C8, request
medians are not additive critical-path attribution: do not add TTFT and TPOT
medians and call the result an exact hardware breakdown. A subsequent change
needs matched phase/kernel evidence for its target, not another universal
partition heuristic.

The first C1 output vector equals SGLang's but not vLLM's. C8 output vectors
are not equal across engines; LunaFlux and vLLM also vary within repeated
concurrent runs. SGLang is fully repeatable in this short cell and matches 38/40
long-C8 row repetitions. Timing success and bounded kernel correctness do not
resolve this generation-quality/batch-invariance limitation.

## Decision and reproducibility

Two candidate families were enough to reject a universal improvement claim;
the remaining budget was not spent inventing another unmeasured source change.
Retain the previous best full package and the verified four-partition binding
candidate. Do not promote matrix decode or change portable defaults from these
three cells. No runtime/compiler source code changed in this experiment, and
the full native suite was not rerun for the documentation-only repository edits.
The exporter, exact AOT module, numerical probe, sanitizers, route propagation,
serving drain/reap, memory guard and reversed-order serving comparisons were run.

Key identities:

```text
source.tar   ac9f46e54cab57c1322900b15ea257994d438081b5f3a60092981a1bc6474995
decode.cubin 0a0ce21bf82edf8e4c82248f8f0e9ffa1f465d3550c25ac3e653a886948ea051
launch.json  a1b7b77d3650dd546f9cf93ee9a46299ea3adf954a4c6d9318f8149dfd5d5cc0
worker       0511314bbc8c5cf39dcae5b595c85bb33b7817ae1fbbceb5018f07a87883f9a8
```

All three evidence archives were downloaded without overwrite to
`/tmp/lunaflux-overall-best-evidence-20261009.ad9eYv`, under `main`, `matrix`,
and `matrix-rejected`. Each archive hash and every extracted `FILES.sha256`
entry verify locally. Source/dependency/build caches, toolchains and model
payload directories are excluded; source.tar, exact modules, descriptors,
commands, trial vectors, sanitizer logs, traces and failed attempts are retained.

```text
main            94bd9fa3f72d72066de0b5f5440b1fe05c32adbb1061184b18ab0d10c453aa7c
matrix          d093ac3185966613fc6c78db6c1b158a1f7d21396c0a1cc9b2212e667d7f2159
matrix-rejected 9fd4a59f33301dbdbf4fcd1232adfa844023a95a1f12f9b44d91cfb166de7095
```
