# Dual-Spark chunk/logit isolation

## Result and scope

This follows `BENCHMARK_AKO_BATCH_ISOLATION_2026-10-05.md`. It uses .179
for complete-model logit traces and .178 for independent identical-input
attention checks. No two GPU workloads run concurrently on either device.
Frozen model and AOT artifacts are reused; there is no production kernel,
selection-policy, numeric-contract or scheduling-default change in this round.

The full-model mismatch is a reproducible BF16 logit difference, not a
GPU-versus-CPU greedy-sampler disagreement. The same fixed prefill artifact
is bitwise chunk-invariant on all 12 synthetic checks. These findings do not
yet identify the actual model layer/operation that introduces the difference.
They do not justify changing accuracy tolerances or promoting the concurrent
8K serving route as equivalent.

## .179: complete-model margins

GPU: `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6` (GB10, sm121).
Disposable trace root:
`/home/wlc004s/lunaflux-ako-chunk-trace-20261005.foQDKYL8`.

Both arms reuse the source-bound frozen 8K-capacity runtime from
`/home/wlc004s/lunaflux-ako-query-chunks-20261005.PCwRF4kW`, including bundle
SHA-256 `dfb6e35a8940c4e060b8c6d7397f11bbb86360d03b4e6a78dcd72144e4973ac9`.
Only the disposable worker gains synchronous diagnostics. Prepared worker
receipts are regenerated for that binary; no frozen deployment is modified.

The two saved 32,512-input/64-output bodies are replayed sequentially,
prompt 0 then prompt 1, twice per arm. Chunks are 2,048 and 8,192.
All eight complete token vectors and exact request bodies match their
corresponding uninstrumented solo runs. Server readback vectors also match
the client vectors. Each arm's repeats are identical, including all six
tracked logit bit patterns per sample. There are 512 sampled outputs total.

| Observation | 2K chunks | 8K chunks |
| --- | --- | --- |
| GPU/CPU argmax disagreements | 0/256 | 0/256 |
| Exact top-two BF16 ties | 14 | 12 |
| Prompt 1, sample 0, winning token 315 logit | 19.000 | 18.875 |
| Prompt 1, sample 2, token 13 logit | 18.125 | 18.125 |
| Prompt 1, sample 2, token 389 logit | 18.125 | 18.250 |
| Prompt 1, sample 2, selected token | 13 | 389 |

Sample indices are zero-based. At sample 2 both arms have consumed the same
generated prefix `[315,279]` and use the same recorded decode bucket 1779,
executor owner 85, one row/one token and position 32513. The 2K arm selects
the lower token ID on an exact tie; the 8K arm ranks 389 one BF16 step higher.
The later 44/64 differing tokens are not independent errors: changed outputs
change subsequent inputs. Tracked logit differences already occur at sample
0 for both prompts, even though prompt 0's complete greedy vector stays equal.

This trace does not expose per-layer intermediates or prove that both
prefill arms use the same complete executable chain. A matching decode owner
at the first token difference does not prove matching cached KV.
Projection geometry, prefill variant selection and earlier activations/KV
remain candidates; none is established as the sole cause here.

The first report compared whole tracked-logit objects, accidentally including
physical packed-row offsets. The corrected report compares BF16 bits only;
a regression test requires equal bits at different row offsets to compare
equal. `trace-audit.json` is retained, with corrected reports in
`trace-audit-bits.json` and `trace-audit-confirmed.json`, without overwriting
the earlier report. Both corrected comparisons still find real logit
differences at sample 0.

Readbacks intentionally perturb execution. This campaign is a numerical
diagnostic, not a performance measurement or a new vLLM/SGLang comparison.

## .178: same-input attention across query chunks

GPU: `GPU-9c3d3cf0-439a-5da2-67e9-20414255879f` (GB10, sm121).
Root: `/home/wlc003s/lunaflux-ako-chunk-parity-20261005.l0Uje6GS`.
CUDA 13.0.88 compiler SHA-256:
`fbb111f057786ddd10ba723d993cc7dd43abf978b6baa32fedd3c9d806dc79e1`.

The unchanged c322 prefill artifact uses
`lunaflux_attention_prefill_tile_compiler_v1`, query/KV tiles 64/64 and
`strict-natural-exponential-v1`. Each check compares a full submission with
the same query/KV split into 2K or 4K submissions. Dense-current inputs become
paged-history inputs at each chunk boundary; global positions, logical-to-
physical page mapping, causal visibility and immutable KV values are kept.
All launches use the runtime bucket/metadata geometry, not a compact grid.

Finite grid: `(queries,prior_history)` = `(8192,0)`, `(8192,24576)`,
`(7936,24576)`; chunks 2048/4096; two repeats. The last shape includes the
real 32,512-token request's partial final chunk.

All 12 checks exit zero with empty stderr, unchanged KV, **bitwise-equal
complete attention outputs and maximum pairwise error 0**. Sampled FP64
oracle error ranges from 0.000307543 to 0.000347320, within the unchanged
0.003 bound. The synthetic data and sampled oracle do not establish full
model accuracy. In particular, this does not test all possible serving
attention variants or actual per-layer Qwen activations.

The process is externally bounded by user-systemd: MemoryMax 8 GiB, zero
swap, 900 seconds; each invocation requires at least 32 GiB MemAvailable.
The MoonBit runner is built natively on .179 and transferred to .178, which
does not have Moon installed. Production runtime dependencies are unchanged.

Memcheck passes the full `(7936,24576,2048)` fixture with zero errors.
Large-shape race instrumentation was stopped after memcheck rather than
waiting on disproportionate instrumentation cost; that incomplete attempt
is not a sanitizer pass. Separate focused racecheck and synccheck runs use
`(129,1024,64)`, exercising chunk boundaries and a one-token tail. Both pass
with zero hazards/errors and empty stderr, under 8 GiB/zero-swap/120-second
limits. They do not provide long-shape racecheck coverage. The runner now
uses these explicit per-tool fixtures to keep future diagnostics bounded.

## Automation and saved results

New diagnostics: `ako_chunk_trace.mbtx`, `ako_chunk_trace_report.mbtx`, and
`ako_chunk_parity.mbtx`. `install_logit_margin_trace.mbtx` accepts 1–32 unique
explicit tracked token IDs while retaining its previous default.
`selected_policy_probe.cu` adds a check-only chunk-parity mode; it cannot
produce paired timing samples. `seal_domain_trial.mbtx` now retains spec,
geometry-header and bare probe/runner artifacts, with coverage in its test.

Focused native warning-denied checks/tests pass. The fixture compiles with
`-O3 -arch=sm_121 --fmad=false`; unchanged AOT kernels are not regenerated.

.179 archive SHA-256:
`e606734e49c9984d86b543caba7557120f7ecbb04abd14c8829bf1f0fcc7431d`.
Downloaded without overwrite to
`/tmp/lunaflux-chunk-trace-transfer-20261005.mjetUZcQ/chunk-trace.tar.gz`;
the local hash matches. Raw command/result files, complete SSE/client vectors,
diagnostic worker binaries, installer sources and corrected reports are kept.

.178 archive SHA-256:
`76f26e7b2d04c7d28cebe4c4a139cd1486f26d8251bc78e5836b0d8eb65060dc`.
Downloaded without overwrite to the same local transfer directory as
`chunk-attention.tar.gz`, with a matching local hash; extracted separately
under `attention/`. Successful numerical/memcheck/focused sanitizer output
and the incomplete large racecheck command/memory snapshot are retained.

Next: compare actual selected prefill chains and layer intermediates before
altering the compiler. The existing measured 8K timing improvement remains
separate from this round's accuracy investigation; this round claims no new
speed gain and no concurrent numerical-equivalence admission.
