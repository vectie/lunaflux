# Prefill ownership and independent MoE branches

## Experiment contract

This follows the actual serving traces in
[the execution-work report](GLM_EXECUTION_WORK_AND_WAITS_2026-10-17.md), not a
new assumption that all routed expert GEMMs are slower than vLLM.

Two independent changes are evaluated, followed by their combination:

1. Expose narrower independent output-column ownership for grouped BF16
   prefill. A 16-row shared group currently produces only 16 gate/up CTAs on
   48 SMs. Test 32 and 16 columns against the retained 64-column schedule.
   Keep the ascending K16 MMA sequence, BF16 boundaries, decode fold and
   collective placement unchanged. Column partitioning belongs to immutable
   grouped-matrix IR; CUDA owns consumer subgroup mapping and staging.
2. Fork the independent shared-expert branch alongside routing/routed experts,
   then join before their existing coalesced partial publication. Use explicit
   fork/join intervals in parallel IR, startup-owned queues/events and distinct
   branch scratch. Preserve numerical combine order, cancellation and resource
   release. Graph replay must retain these dependencies rather than reverting
   to an ordered kernel list.

Budget: at most three schedules per target before revisiting the hypothesis.
First use boundary/deterministic/sanitizer component checks, then frozen
baseline/candidate full-model runs and the matched vLLM prompt corpus. Report
per-request input/output counts, TTFT and completion time; separately report
selected kernel geometry, overlap and stage time. Different output lengths and
different timer boundaries are not silently normalized into a speedup claim.
Comparison and profiling remain offline, outside compiler legality.

Both Sparks are exclusively reserved during GPU tests, at most 100 GiB per
Spark with a separate 16 GiB host MemAvailable reserve and no swap. Stop only
the previously identified GLM head/worker containers, leave NFS running, and
restore those original containers after testing. No payload hashing or
integrity scans. Unrelated toolchain/TLS changes remain outside task commits.

## Results

The first component experiment is complete. H4096/I1024, cap64, one shared
expert; paired AOT kernels on Spark .179, five alternating-order trials with
20 launches per timing. All BF16 activation and F32 partial outputs are bitwise
identical at live rows 0/1/3/16/17/22/32/63/64/65, including inactive tails.
Both 32- and 16-column alternatives pass memcheck/full leak, racecheck and
synccheck. No payload hashes were computed.

| Live rows | 32-column gate/up vs 64 | 32-column down vs 64 |
|---|---:|---:|
| 1 | 0.025552 vs 0.031798 ms | 0.015008 vs 0.013157 ms |
| 16 | 0.134261 vs 0.203419 ms | 0.027701 vs 0.037944 ms |
| 22 | 0.144536 vs 0.199733 ms | 0.031602 vs 0.041733 ms |
| 32 | 0.148229 vs 0.202557 ms | 0.031701 vs 0.042037 ms |
| 64 | 0.165211 vs 0.242888 ms | 0.043371 vs 0.053922 ms |

These are per-kernel medians, not full-model savings. Decode arithmetic is
unchanged, but shared resource declarations and padded launch counts change,
so decode timing is NOT unchanged: down regresses while gate/up improves.
The 16-column route also regresses down at64 rows (0.058322 vs0.053502ms).
Select32 for the next complete-model experiment; retain16 as a nondefault
candidate, not a universal winner.

NCU R22 confirms32 gate/up changes static shared memory17,440→9,248bytes,
registers40→46, active-warps8.87→12.04%; total instructions8.136M→7.890M
and L2 read sectors1.147M→1.241M. More effective parallelism wins despite
more traffic and a higher barrier-stall percentage. Down instructions increase
4.563M→5.022M while time decreases. Counter runs are replay-perturbed and not
substituted for the unprofiled timings above.

Raw roots:
`/tmp/lunaflux-glm-shared-vector-columns32-20261018-v1` and
`/tmp/lunaflux-glm-shared-vector-columns16-20261018-v1`.

The fork/join executor uses existing device events and ordered-launch owners;
no new CUDA ABI is needed. Shared scratch and partial outputs are separately
owned; input/counts are shared reads. Shared producers issue first on the
auxiliary lane, while the primary lane retains routing/routed order. Both join
before the existing F32 TP publication. Scalar index loops preserve the
allocation-free submission contract; iterator allocation caught by the
regression test was removed.

The actual MoonBit GPU queue owner now passes 32 eager and 32 captured replays
with changing input, exact outputs and explicit release, plus partial-submission
cancellation/drain. The same test passes CUDA memcheck/full leak (zero leaked
bytes), racecheck (zero errors/warnings), and synccheck. Evidence:
`/tmp/lunaflux-glm-fork-owner-20261018-v2`; clean source pin `19d4883c`.
The retained v1 failure was a test symbol containing a trailing NUL, rejected
by the real function-loader ABI; the corrected fixture uses plain symbols.
This is a real queue-owner gate, not a replacement CUDA-only harness.

Complete-model AOT variants at columns64/32 are built from `0d614c5f` on ARM;
the owner implementation is unchanged by the fixture correction. Full-model
ablations are running; the historical vLLM baseline is not a new result.

Local phase validation: `moon info --target native` reports 3,669 existing
migration warnings and zero errors; `moon fmt` and native check pass.
The full native suite passes **5,193/5,193** with the existing warning
exclusions `-20-25-29-35-79-92`; no new exclusions or tolerance changes.
This is the current working-tree suite, which includes unrelated ongoing
migration files. Clean pinned ARM builds independently validate the task's
executable and actual fork-owner test; unrelated files are not staged.
