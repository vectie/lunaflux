# Typed prefill operand reads: implementation and fresh retest

The measured read-ahead experiment now has a typed compiler implementation in
commit `6d1a8644`. The production exporter reproduces both frozen GPU binaries
byte-for-byte. Fresh kernel trials show long-history gains, but inconsistent
pairwise margins and a short-control regression. Defaults remain unchanged;
this does not establish a new serving throughput or close the framework gap.

## Compiler change and propagation repair

The implementation preserves the functional layering:

`attention strategy → physical read ownership / ordered reads → schedule → CUDA binding → source`

`FragmentReadOwnership` is an immutable provider permutation preserving the
source-word identity. `OrderedOperandReads` is a pure, bounded read/consume
plan: prime slots, consume in ascending order, and refill only after retirement.
Compilation constructs these plans; token-step execution does not allocate or
perform evidence checks. CUDA-specific lane interpretation and PTX remain in
terminal lowering.

Two optional alternatives are enumerated without removing historical routes:
consumer-contiguous reads and consumer-contiguous two-slot read-ahead. The
measured candidate is `330322`; its historical baseline is `30322`. Old IDs,
static defaults, arithmetic order, numerical law, copy publication and barriers
are preserved. Candidate identity includes the read plan, preventing a tuning
record from silently selecting a different implementation.

A new propagation test found that the online-fold source renderer ignored the
read-ahead plan. That renderer now consumes it. Tests assert the helper and
call are present, the obsolete key helper is absent, incompatible lifetimes
are rejected, and measured records cannot cross device/problem identities.
The real exporter, not an experimental text rewrite, generated the retested
artifacts.

## Fresh paired GPU results

Spark GB10, sm121; Q64/K64/D128, block 128, 49,168 shared bytes. The AOT envelope
is 2,048 queries and 32 rows. Each cell uses five alternating pairs, with 30
CUDA-event repetitions per measurement. History is prior KV length; these
are kernel workloads, not complete requests. Median paired gain is the median
of paired ratios, not the ratio of independent median times.

| Queries / rows / history | Baseline median µs | Read-ahead median µs | Median paired gain | Worst pair | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| 1792 / 1 / 30720 | 6301.40 | 5957.77 | 5.58% | 2.92% | Inconclusive |
| 2048 / 2 / 28672 | 6568.75 | 6395.41 | 3.26% | 1.18% | Inconclusive |
| 2048 / 16 / 2048 | 925.25 | 934.00 | −1.02% | −2.23% | Regression |

The unchanged gate requires at least 3% gain in every pair of every cell.
Neither long cell clears that gate, and the short cell regresses. The terminal
result is `contains-regression`, with selection unchanged. The earlier
one-row experiment cleared its gate, but this independent retest did not;
both results remain preserved. No global enablement is justified.

All 15 pairs retain full bitwise output equality (`maxabs=0`). Sampled scalar
oracle error remains below the unchanged 0.003 ceiling. Memcheck, racecheck
and synccheck complete with zero errors/hazards and empty stderr. These are
kernel correctness checks, not whole-model numerical admission.

Jobs were serialized with an 8 GiB memory cap, swap disabled, and a 32 GiB
MemAvailable reserve. The lowest trial observation was 121,990,368 KiB.

## What explains the gain—and what remains

The generated candidate is identical to the previously profiled binary in
[the read-ownership report](BENCHMARK_PREFILL_KEY_READ_OWNERSHIP_2026-10-07.md).
Those earlier matched Q2048/R2/H28672 counters showed total instructions
falling from 891,530,112 to 762,274,496 (14.50%) and register MOVs from
97,258,496 to 4,792,320 (95.07%). Tensor arithmetic, shared matrix reads and
barrier counts were unchanged; NOPs increased from 25,242,624 to 49,550,336.
Allocated registers and resident-block limits were unchanged.

These are earlier counters for the same binary, not a fresh Nsight capture.
They establish removal of physical operand rearrangement, not proportional
completion-time savings. Remaining dependency scheduling, device-inserted
waiting and other kernel work absorb much of the instruction reduction.
Issue-normalized warp metrics must not be treated as elapsed-time shares.

The selected-route, uninstrumented ABBA serving comparison is now complete:
[serving results and remaining gaps](BENCHMARK_TYPED_PREFILL_SERVING_2026-10-07.md).
The recalibrated configuration gains approximately 1% at 32K, but has a losing
short ABBA half and C16 output differences. A kernel-only gain must not be
added directly to token/s. Exact scalar projection replay narrows the separate
numerical investigation without resolving whole-model parity.

## Validation scope

- 241 affected-package tests plus eight CUDA attention AOT tests passed.
- The benchmark driver's contract-rebinding regression passed.
- The full local native suite subsequently passed 4427/4427 with stripping and
  warning exclusions `-79-20-29-25-92-14`; unrelated dirty packages remain
  excluded from an unqualified warning-clean claim.
- Affected native checks passed with legacy warning exclusions
  `-79-20-29-25`; this is not an unqualified warning-clean claim.
- The full warning-denied check with those exclusions found warning 92 in
  unrelated `engine/joint_diffusion_execution/pipeline.mbt` and warning 14 in
  unrelated `runtime/remote_tls/channel.mbt` working-tree changes. Those
  changes were not edited or included in this commit.

## Reproducibility

- Source commit: `6d1a8644`.
- Source archive SHA-256: `fff1c1bb0b5006511eaabd1510347e130afa8c3296140ef003e44f8af23d994f`.
- GPU UUID: `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`.
- NVCC 13.0.88 SHA-256: `fbb111f057786ddd10ba723d993cc7dd43abf978b6baa32fedd3c9d806dc79e1`.
- Probe SHA-256: `78901fc5e3222e4394b3fb35746d60ad36a0181f22472c0397c5dee1e5079536`.
- Baseline cubin: `616ecc0f88c192e59c3134bfc9390a57d59984cb3b31543956b587858e132cdb`.
- Candidate cubin: `301f8fd9db1e0221e4b0b5048d08c03d55ee49f5f50d8ea04e5df33e4582ebf3`.
- Verified evidence archive: `342527b46ccc6b7b0238bb28854685a06422694ad887b27546e0ec71d800d387`.

Remote evidence is sealed at
`/home/wlc004s/lunaflux-typed-key-read-20261007.eUbyLAa5`.
The non-overwriting local copy is
`/tmp/lunaflux-typed-read-verified-20261007.aASGNzvr/verified`;
all 2,398 manifest entries verify. The archive includes generated export
artifacts and the clean source archive, but excludes build/dependency trees.

Two pre-GPU setup failures are preserved separately at roots ending
`qjVe973k` (build CLI argument) and `KriRuVrR` (legacy toolchain warnings).
They are not mislabeled as numerical or performance results.

Automation: `benchmarks/gpu_pipeline/prefill_typed_read_pipeline_20261007.mbtx`.
