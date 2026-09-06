# Compiler work: implementation and current-source measurement

## Implemented

- `48a94b0`: device/toolchain-scoped, canonical offline projection records,
  consumed by both candidate export and release binding.
- `5e5da56`: terminal output-row compaction in the real reusable descriptor,
  BF16 matrix head and sampler, including captured execution. Original row
  identities, RNG coordinates and upstream KV effects are retained.
- `75423cb`: pure complete-span fusion choice with a real offline ingress
  exporter consumer: full, producer-separated or unfused. No measurements of
  these three serving alternatives are installed yet.
- `9402e1b`: bounded sanitizer mode for the output-row diagnostic. It retains
  all row/mask cases but avoids repeatedly instrumenting full-vocabulary GEMM.

Selection remains an immutable, backend-neutral compiler calculation. File
access and tuning inputs are offline; descriptor storage is allocated at
startup. No request-time tuning, cryptography, filesystem access or additional
allocation/transfer call was introduced by these changes. CUDA details remain
in device lowering and diagnostics.

## Qwen3-0.6B BF16 end-to-end results

Actual runtime, worker and AOT artifacts rebuilt from `9402e1b`. Same input
token vectors, greedy sampling, seed 0, ignore-EOS and disabled prefix reuse
as the saved comparison. Each cell is the arithmetic mean of two timed runs
after one warmup. C8 reports aggregate output tokens/s. These are successive
saved runs, not an interleaved A/B experiment or confidence intervals.

The matching previous run is `lunaflux-vector-new-run2`, not the older figures
of approximately 703/109 tokens/s quoted during execution. This comparison
does not rerun vLLM, SGLang or llama.cpp.

| Input / output tokens | Concurrency | Previous tokens/s | Current tokens/s | Change | Per-request token arrays |
| --- | ---: | ---: | ---: | ---: | --- |
| 59 / 256 | 1 | 221.65 | 224.07 | +1.1% | Identical |
| 59 / 256 | 8 | 738.55 | 833.20 | +12.8% | Not identical; see below |
| 128 / 128 | 1 | 213.87 | 216.95 | +1.4% | Identical |
| 128 / 128 | 8 | 693.06 | 774.01 | +11.7% | Identical |
| 512 / 64 | 1 | 172.74 | 176.56 | +2.2% | Identical |
| 512 / 64 | 8 | 418.47 | 457.96 | +9.4% | Identical |
| 1,528 / 32 | 1 | 84.43 | 86.61 | +2.6% | Identical |
| 1,528 / 32 | 8 | 116.36 | 123.05 | +5.7% | Identical |

All requests produced the requested token count and terminal response. The
59/256 C8 case has three distinct complete sequences across its 24 requests
(warmup included) in both versions. Every current sequence already occurs in
the previous run, but their assignment to request ordinals differs. Identical
concurrent prompts already varied within the previous run; the first observed
differences are at zero-based token 24 or 66. This is not proof of per-request
equivalence or a diagnosis of the cause. Batch/row-dependent numerical behavior
remains to be isolated; the throughput observation must not hide that caveat.

The measured selected-input-residency head record is now installed through
the persisted-record path in both artifact generation and binding. Its source
measurements cover rows 1 through 256, with three trials and bitwise output
checks. The frozen profile key is the admitted maximum token bucket (256),
not a runtime choice among row-specific artifacts. Current serving still uses
full ingress and the synchronous grouped-decode schedule, not async 441.

These end-to-end gains combine the installed head schedule with current-source
compiler/executor changes. They do not isolate each optimization's contribution.

## Changed-boundary correctness and kernel measurements

Real captured descriptor replay passed mixed, one-output, empty and full
output views with stable pointers. Full-vocabulary standalone head checks
passed 24 cases per schedule: rows 1, 2, 8, 17, 32 and 64, ragged token lengths,
and all/none/last/alternating output masks. Retained logits were bitwise equal
to the unprojected view; unobserved output memory remained untouched.

| Full-vocabulary head case | Unprojected → compact (µs, approximate) | Interpretation |
| --- | ---: | --- |
| Baseline schedule, 32 rows / 16 outputs | 4,946 → 2,502 | About 1.98× less head time |
| Baseline schedule, 64 rows / 1 output | 9,807 → 2,466 | About 3.98× less head time |
| Resident schedule, 64 rows / 32 outputs | 2,729 → 1,645 | About 1.66× less head time |
| Resident schedule, 8 rows / 1 output | 1,520 → 1,613 | Regression; not every mask wins |

These are diagnostic kernel timings, not serving throughput. Compaction can
remove matrix tiles, but fixed tile costs and the selected residency schedule
can limit or reverse its benefit. Zero-output views return before head
arithmetic (approximately 2–4 µs launch cost).

On `9402e1b`, memcheck, racecheck, synccheck and initcheck passed for both head
schedules and the real captured executor. The head sanitizer mode limits the
vocabulary grid to four column blocks and skips timing repetitions, while
retaining all row/mask cases. Full-vocabulary memcheck also passed in the
earlier `5e5da56` run. Its full-vocabulary baseline racecheck was deliberately
interrupted after approximately 11 minutes (exit 15); that run is not a full
sanitizer pass and its original logs are preserved.

Focused software tests and the warning-denied native compile passed. A prior
full-suite attempt remains blocked by the separately reproduced, pre-existing
FP8 lowering-test abort; this report does not claim a full-suite pass.

## Reproduction

- Device: RTX 5060 Ti, UUID `GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`,
  PCI `00000000:17:00.0`; CUDA 13.1.115.
- Source archive SHA-256:
  `d9c690d520ab91e485dabdaff41776e88432f7260e92479f57299e9f7459ac7f`.
- Runtime executable SHA-256:
  `dfe0359ea66c800352de224fa699e2f455f3761051d4132db4707b6bc7ee266d`.
- Remote build/results: `/dev/shm/lunaflux-remaining-9402e1b-20260906-r1`.
- Downloaded archive:
  `/private/tmp/lunaflux-remaining-9402e1b-20260906-results.tar.gz`.
  Local and remote SHA-256 agree:
  `0e86b0dbfe31fd55241b588cd9e3ab876c442572a8cd9b62a51635df653c6f1f`.
  The archive includes current request results, previous measured summaries,
  comparison CSV, installed tuning record, artifacts and validation logs.
- Earlier full-vocabulary diagnostic archive:
  `/private/tmp/lunaflux-output-rows-physical-20260906-r1-results.tar.gz`.
  Local and remote SHA-256 agree:
  `256cfe255e0cccdbe97d7ece56a6acceaaec0c7a610a304fd6861cd4fe80f0cd`.
  It retains the deliberately interrupted racecheck with exit 15.

The temporary benchmark server drained and stopped; the GPU was idle after
the campaign. No production deployment or baseline-engine replacement occurred.

## Still open

1. Install distinct measured projection/ingress artifacts per graph bucket;
   whole-profile selection does not implement this.
2. Measure and install a complete-span full/partial/unfused ingress choice.
3. Integrate and compare the measured asynchronous attention schedule in serving.
4. Prune captured upstream pure suffixes, beyond the compact head/sampler view.
5. Isolate the pre-existing C8 greedy-output variation before claiming strict
   batch-invariant generation.

The five workstreams have compiler/executor consumers, but the remaining list
is not declared complete by these measurements.
