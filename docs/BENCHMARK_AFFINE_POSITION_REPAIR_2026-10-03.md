# Attention affine position repair

This investigation matches actual request rows, query tokens and history before
comparing selected attention kernels. It fixes one measured load-dependency
problem through the functional compiler, rather than treating instruction count,
bank conflicts or occupancy as the performance objective. The isolated kernel
improvement is about 9%; it does not by itself establish that serving is 9%
faster or that the remaining framework gap has disappeared.

## Matched work and the remaining overhead

The workload is Qwen3 0.6B BF16 on the 48-SM DGX Spark GB10, CUDA 13.0.88.
Reference containers are the pinned NVIDIA 26.01 vLLM and SGLang images. Requests
use C16, disabled prefix reuse and identical input/output token vectors.

CPU metadata markers recover sorted `query:past` row vectors and join them to
CUDA launch correlations. They do not read device tensors. Capacity grids alone
are not a work match: inactive capacity CTAs can return immediately. SGLang's
different chunking means its first launch is not interchangeable with the matched
LunaFlux/vLLM launch below.

At one row, 2,048 queries and 2,048 past tokens, the instrumented 28-layer
attention span was 27.049 ms for LunaFlux versus 21.970 ms for vLLM, a 23.1%
time gap. These are attribution runs, not ordinary serving throughput.

| Matched selected invocation before repair | LunaFlux | vLLM |
| --- | ---: | ---: |
| Executed warp instructions | 133.93 M | 53.23 M |
| Tensor MMA instructions | 12.714 M | 12.845 M |
| Register MOV instructions | 13.625 M | 0.939 M |
| Tensor activity | 42.53% | 54.55% |
| Active warp occupancy | 15.58% | 7.95% |
| Registers per thread | 230 | 255 |

The mathematical MMA work is nearly equal; supporting instructions differ
substantially. No register spilling was recorded. LunaFlux had zero
source-correlated excessive shared wavefronts in this probe. Repeated dependent
position loads were among the hot stall locations. A barrier-only or
bank-conflict-only explanation therefore misses part of the problem.

The reference uses live serving operands and LunaFlux's replay uses deterministic
synthetic operands. This is a matched-shape comparison, not an operand-identical
cross-framework correctness comparison.

## Compiler change and numerical behavior

Device-step preflight already constructs each row's positions as
`row_start + local_index`. Previously, attention checked that identity again
inside each current-KV vector load. The new immutable
`AffineDenseCurrentAndPagedHistory` view carries the producer guarantee through
strategy, canonical LunaTile IR, CUDA lowering and source emission. Lowering
removes only the redundant position equality load. Bounds checks, historical
paging, invalid-page handling and zero fill remain.

The ordinary dense-current view remains available for arbitrary positions. The
producer law changes the semantic digest and invalidates incompatible incremental
compiler cache entries. CUDA details remain in lowering; no cryptography,
filesystem checks, profiling or new validation enters token-step execution.

The real frontend exporter produced c322 with Q64/KV64/D128, block128 and 49,168
bytes of dynamic shared memory. Numerical arithmetic, strict `expf`, FMA policy
and output ownership are unchanged. Two independent compiles produced identical
cubins. Six pure, mixed, history and tail comparisons were bitwise identical;
memcheck with leak checking, racecheck and synccheck passed.

| Identical-operand matched probe | Before | After |
| --- | ---: | ---: |
| Unprofiled median CUDA event time | 935.664 us | 853.285 us |
| Paired cold NCU replay time | 1,024.064 us | 927.616 us |
| Executed warp instructions | 133,931,904 | 131,654,528 |
| Register MOV instructions | 13,625,344 | 13,355,008 |
| Tensor activity in paired replay | 42.142% | 47.169% |
| Registers per thread | 230 | 229 |
| Allocated registers per thread | 232 | 232 |
| Resident blocks per SM | 2 | 2 |
| Source correlated excessive shared wavefronts | 0 | 0 |

The unprofiled time reduction is 8.8%. Instruction count falls only about 1.7%:
the result supports removing a dependent load, not a claim that instruction
count alone explains time. Register allocation and residency did not improve.
NCU export units differ between reports; the table normalizes them to
microseconds and keeps replay time separate from ordinary timing.

Other isolated experiments were rejected: compacting probability lifetime gave
no gain, and compact tagged pointer offsets reduced registers but slowed the
probe by about 7%. Approximate exp2 gave a smaller microbenchmark gain but changed
BF16 output; it remains diagnostic-only and is not silently admitted under the
existing arithmetic contract.

## Serving packaging and measurement

An initial isolated package copied the old routing records after replacing the
cubin. That was invalid: route scope binds the entire module set. The worker
failed during startup and no throughput sample from that attempt is valid.
Failed artifacts and logs are preserved.

The preparation workflow now separates kernel qualification from serving
materialization. It re-exports the new module set and measures fresh scoped
attention routes before materialization. Unchanged alternatives use their frozen
recipes, not new frontend defaults that might name different symbols. The worker,
CLI, other cubins, capacities and model files remain frozen. Consequently the
serving experiment measures the affine module **with refreshed route selection**;
it is not a perfectly isolated same-route end-to-end A/B.

Fresh ordinary timing uses three counterbalanced fresh starts, one warm-up and
one measured trial per input/output vector, C16 throughout, serialized GPU use
and a monitored minimum 32 GiB available-memory reserve. Three starts are a small
repeatability check, not a statistical confidence interval.

| Input and output tokens at C16 | LunaFlux output tok/s | vLLM output tok/s | SGLang output tok/s | LunaFlux extra time versus vLLM and SGLang |
| --- | ---: | ---: | ---: | ---: |
| 128 / 32 | 1,316.2 | 1,565.7 | 1,551.5 | 19.0% / 17.9% |
| 4096 / 64 | 207.5 | 243.4 | 242.1 | 17.3% / 16.7% |
| 4096 / 256 | 309.3 | 345.4 | 340.4 | 11.7% / 10.1% |

LunaFlux's long-64 median completion time fell 4,993 to 4,934 ms (1.18% less
time), and long-256 fell 13,356 to 13,244 ms (0.84%). Long-64 ranges were
4,967–5,003 ms before versus 4,921–4,941 ms after. Long-256 ranges overlap;
three samples do not establish statistical significance. The short vector did
not improve. Long-64 median first-token time fell 1,408 to 1,373.5 ms; the new
reference times were 1,066 ms and 965 ms.

All 192 long-vector comparisons agreed with vLLM token for token. Six of 96
short-vector comparisons differed from SGLang at output index 2 (624 versus
382), while within-engine repeats agreed. This unresolved cross-framework
numerical difference prevents an all-workloads token-equivalence claim. It does
not contradict the six bitwise same-law kernel A/B checks.

Minimum monitored available memory was 56,030,112 KiB (53.4 GiB), above the
32 GiB reserve.

### Why nine percent at the kernel becomes about one percent overall

The final instrumented serving capture confirms identical logical-work vector
counts, 96 measured graph steps, all 21,488 kernel calls mapped, and the same
924 normal-prefill calls and geometries. Every observed normal-prefill call now
reports 229 registers rather than 230. The new cubin is executing; it is not an
unused compiler candidate. No wide-prefill kernel executed in either capture.

| Measured window kernel activity | Before | After |
| --- | ---: | ---: |
| Repaired normal prefill | 656.746 ms | 597.462 ms |
| Partitioned decode partial | 1,843.660 ms | 1,847.957 ms |
| Other attention including decode and merge | 456.487 ms | 460.985 ms |
| Other kernels | 1,890.757 ms | 1,904.234 ms |
| Full diagnostic client window | 4,977.433 ms | 4,941.142 ms |

The repaired prefill activity falls 9.03%, but originally occupied only 13.19%
of the request window. Multiplying those figures estimates about 1.19% total
time saved if everything else stayed fixed, consistent with the ordinary
long-64 result. This is a dilution estimate, not an additive critical-path proof:
kernel sums can overlap, profiling has overhead, and other activity varies.

Partitioned decode alone occupies about 37.4% of the new window and was not
changed by this prefill repair. Other projections and attention also remain.
There is therefore no support for the earlier idea that one remaining prefill
pipeline change would remove the complete framework gap. The next investigation
should match executed partitioned-decode work and compare its load-dependency,
copy/readiness and instruction costs against the reference, using the same
whole-chain acceptance criteria.

Native validation passed 137 affected-package tests and the full 4,267-test
suite, with the existing toolchain migration warning exemptions
`-79-20-29-25-92-14`. The additional wide-prefill propagation regressions passed
75 affected tests. Module checks and formatting checks passed. Standalone
benchmark tools use warning-denied native checks without those exemptions.

## Reproduction and scope

Compiler commit: `a3812803`. The physical package overlays the scoped compiler
change on the previously qualified runtime; it is not a deployment of unrelated
dirty working-tree modules.

- Qualification: `benchmarks/gpu_pipeline/prepare_affine_serving.mbtx`.
- Route calibration and materialization:
  `benchmarks/gpu_pipeline/recalibrate_affine_serving.mbtx`.
- Ordinary timing: `run_matched_chain_campaign.mbtx` and
  `compare_matched_campaign.mbtx` in the same directory.
- Work matching: `install_reference_work_rows.mbtx`,
  `summarize_work_shapes.mbtx` and `profile_matched_prefill.mbtx`.
- Paired counters: `profile_prefill_ablation.mbtx`.
- Runtime weighting: `compare_repair_activity.mbtx`, using the measured-window
  summaries and `--work` logical-work tables.

Remote campaign root:
`/home/wlc004s/lunaflux-work-shape-20261003.641b4WHo`.
Qualified compiler-exported prefill cubin SHA256:
`b585bb56d6cabe501ae3217b15a79129c274d1ad9c059b8932bae4257c83e7d5`.
Worker SHA256 remains
`97d6f5d884e48eab93abcb3ea7a3844be3ca7c7798987a908048ae17a4f86aed`.

The completed raw snapshot is downloaded at
`/private/tmp/lunaflux-affine-archive-20261003.BmHkzc/gap-repairs.tar.gz`, SHA256
`8d88986d90337a9c1896e8b6f305f82ecd55adc8f1704648d933771f8f66641a`.
The local archive hash matches the remote hash, and all 8,909 inventoried files
verify locally. `EXCLUDED.txt` records omitted build caches, full source copies,
deployment/model payloads and unreadable files; this is a compact diagnostic
snapshot, not a complete model release. macOS archive extraction consumes
AppleDouble sidecars, so the 17 inventoried metadata sidecars were downloaded
literally before the full file-hash check. Failed attempts are preserved.

Remaining supporting work includes fragment rearrangements, historical page
addressing, copy/readiness dependencies and scalar softmax arithmetic. The data
does not establish a single remaining pipeline problem, and approximate
arithmetic requires a separate declared accuracy contract. Further changes
should target an executed instruction dependency and demonstrate a whole-chain
gain, not merely lower a counter.
