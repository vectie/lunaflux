# Rectangular projection transfer campaign

Scope: the latest explicit sparse/HC/scales/bulk/local-MoE/BF16-TP and
16-site-window path, including the corrected peer dispatch. No overall
benchmark, payload hashing, model comparison inside the compiler, or default
promotion. Stop only the owned GLM service and restore its qualified runtime.
Keep 100 GiB per Spark, 16 GiB host reserve and zero owned swap.

The retained selected vLLM sparse output is
`nvjet_sm121_tst_mma_128x64x64_4_32x64x64_tmaAB_bz_TNNN`.
The previous `CUBLAS_GEMM_DEFAULT_TENSOR_OP` component comparator selected
another CUTLASS kernel; its counters cannot stand in for this entry.

A bounded replay in the installed vLLM image, using its bundled cuBLAS,
`CUBLAS_GEMM_DEFAULT` and 32 MiB workspace, now selects the exact retained
symbol, grid256/block128, 255 registers and 99,328 dynamic shared bytes.
Dispatch-only root: `/tmp/lunaflux-glm-installed-projection-dispatch-20261020-v2`.
That first fixture has zero operands and is not a matched memory comparison.
The repeated root
`/tmp/lunaflux-glm-installed-projection-matched-20261020-v1`
uses the same seed713 BF16 operands as the new physical trials and selects
the same symbol. The table below uses that repeated reference capture.
The changed algorithm flag and workspace are not yet individually isolated.
No full reference model was loaded.

| Actual selected sparse output | Latest native | Installed vLLM symbol replay |
| --- | ---: | ---: |
| Warp instructions | 16,906,240 | 2,293,848 |
| L2 read sectors | 6,407,606 | 3,149,645 |
| Active warps % | 44.24 | 8.33 |
| Long-scoreboard share % | 57.12 | 1.316 |
| Barrier share % | 13.82 | 0.061 |
| Tensor activity % | 7.34 | 16.19 |

Native counters are the retained literal-source fixture, not a new paired
capture. The vendor replay profile duration is not unprofiled timing.
Actual installed linear source dispatches the unquantized projection via
`dispatch_unquantized_gemm`; local `../vlm` resolves the CUDA path to
`torch.nn.functional.linear`. The library owns the selected kernel; the
closed nvjet source is not claimed to be present in the vLLM tree.

Hypothesis: the remaining gap is substantially excessive operand movement
and synchronization work. Repeating the earlier scalar row-copy experiment
would not test the reference mechanism. Test rectangular tensor-map transfers
for both operands with explicit publication and reader-retirement epochs,
preserving the ascending K16 F32 fold and BF16 publication.

Budget: two new physical candidates, then reconsider from measured results.
First isolate true 2D transfer on the existing ownership; only then evaluate
larger useful ownership if justified. Tensor descriptors belong to private
device binding at startup, not model semantics or the token hot path.
No JIT, per-step descriptor creation, or hidden fallback is acceptable.
Boundary correctness, generated-source selection, sanitizer/leak checks and
both-rank execution traces are required before a serving selection claim.

## Trial results: neither earns serving selection

This campaign exercises the large sparse output, rows64 / N4096 / K8192.
It does not qualify every KDA/projection shape by extrapolation.

1. **M32/N32/K64, two rectangular-transfer slots:** all tested active and
   inactive outputs match the retained F32 fold bitwise for rows
   0/1/3/22/36/64/65. Five alternating whole-entry graph pairs at rows64
   regress from a median **410.915 to 526.802 us**, with every paired
   sample slower by 23.56–31.97%. Root:
   `/tmp/lunaflux-glm-projection-staging-rectangular-20261020-v1`.
   Its counters show 27,641,993 warp instructions and 8,392,555 L2 read
   sectors: a transport substitution did not remove the small-tile
   operand-reuse and polling cost. No sanitizer or deployment pass is claimed.
2. **M64/N128/K64, four slots, eight contiguous K partitions:** explicit
   alternative numerical law, F32 partial storage, ascending F32 partial
   reduction, then publication. The one-row check completes, but rows3
   exceed the predeclared F32 comparison bound:
   **1.13248825073e-6 > 1e-6**, with 50 BF16-publication differences.
   This disqualifies the trial under that gate; it does not prove that a
   complete model would produce a wrong answer. No tolerance was relaxed,
   and no full-row timing/quality win is claimed. Roots
   `/tmp/lunaflux-glm-projection-staging-partitioned-20261020-v1` and `v2`
   preserve the initial failure and the diagnostic error report.

Counter-only capture of the second trial's two-entry chain (not an accepted
performance result):

| Entry | Profile duration | Warp instructions | L2 read sectors | Long scoreboard |
| --- | ---: | ---: | ---: | ---: |
| Partition producer | 372.224 us | 7,887,334 | 3,148,998 | 48.694% |
| Partial reduction | 42.912 us | 761,856 | 263,211 | 65.146% |
| Installed reference entry | 339.488 us | 2,293,848 | 3,149,645 | 1.316% |

The producer reaches the reference's operand-traffic level but not its
instruction/issue behavior; the separate reduction also adds a new global
publication boundary. Durations are profiled, not matched unprofiled medians.
Do not advertise their subtraction as a latency win. The producer's compiled
resource report has 99 registers, 8 stack bytes and zero local allocation;
this is not evidence of large register spilling.

### Actual reference disassembly, not an invented internal source

`source-0.stdout` in the matched reference root retains the selected SASS.
It shows:

- `UTMALDG.3D` transfers and uniform-register descriptor/coordinate arithmetic;
- swizzled shared addresses and `LDSM.16.M88.4`;
- batches of independent `HMMA.16816.F32.BF16` using reused A/B fragments;
- matrix loads interleaved with later MMAs, rather than only a
  load-every-fragment/consume/repeat sequence;
- `SYNCS.PHASECHK` plus `NANOSLEEP.RAND.WARP.SYNCS` waiting, rather than our
  tight all-warp try-wait polling loop;
- direct BF16 stores, not our external F32-partial reduction entry.

The 256-CTA launch alone does **not** prove eight useful K partitions. That
partition count was our trial choice, not a demonstrated nvjet algorithm.
Static SASS identifies executable mechanisms; these captures do not assign
all dynamic instruction or stall counts to individual SASS sites. In
particular, a long-scoreboard percentage alone cannot establish which
descriptor, operand, or publication load caused it.

The next implementation must address fragment ownership/layout and issue
schedule together, and either preserve the qualified unsplit numerical law
or independently qualify a changed one. It must not treat “has TMA” or
“same grid size” as proof of equivalence to the reference.

## Compiler/deployment boundary

The two unsuccessful implementations are preserved in
`benchmarks/experiments/glm_tensor_transfer_20261020.patch`, together with
their immutable numerical/physical plans, CUDA lowering, exporters and
bounded harness changes. The patch applies to the retained main source;
`git apply --check` passes. Its public compiler alternatives were withdrawn
from the working source, not enabled merely because they compile.
All raw generated sources and remote evidence remain intact. No user source,
model or evidence was deleted.

The AKO two-candidate budget is exhausted. No production ABI, default
numerical policy, serving artifact or compiler comparison dependency changed.
The **main projection gap is not fixed by this campaign**.

## Trace safety

The latest retained AOT plus peer-dispatch executable, exact-capacity replay
and 16-site windows is used, preserving rows64, eight resident slots,
context8192 and the existing offline eight admission lanes.

- `projection-verified-latest-20261020-v1` fails before launch because the
  owner helper's older SSH defaults were not overridden; no GPU work ran.
- `projection-verified-latest-20261020-v2` supplies the live sockets and starts
  both ranks. Software node tracing reaches the unchanged 17 GiB early
  guard at rank0 host availability **17,762,204 KiB** (16.940 GiB).
  Both owners retire; no complete trace or decoder-time comparison is claimed.
- The tiny approved-child hardware-trace qualification
  `/tmp/lunaflux-nsys-worker-hardware-20261020-v1` captures the actual child
  GPU memset successfully under 2 GiB/no swap. Hardware-assisted tracing
  is an observation mechanism, not an engine change.
- Full latest-path hardware tracing at
  `/tmp/lunaflux-glm-http-projection-hardware-latest-20261020-v1`
  also stops at the 17 GiB guard during loading, with rank0 available
  **17,733,780 KiB** (16.912 GiB). The tiny trace qualification does not prove
  that full-model instrumentation fits. No complete request/cold-warm graph
  exists for this attempt.

The slower owner watcher records at most 96,539 MiB GPU allocation across
the two attempts and zero owned-process swap. Its minima differ from the
faster headroom watcher; use the faster observations above for the abort.
No 100 GiB/GPU or 16 GiB host-floor violation was observed. A valid fresh
two-rank time/overlap allocation is still missing; the older graph must not
be relabeled as this campaign's retrace.

## Required compiler work, still open

Use the existing functional layers rather than inserting baseline comparison
or a model-name shortcut into the compiler:

1. Numerical program: retain the qualified unpartitioned fold/publication
   contract for the next transport/layout trial. A different K reduction
   remains a different numerical program, not a free schedule rewrite.
2. Physical ownership/layout: make larger useful row/column ownership and
   swizzled operand layout explicit, with fragment reuse across independent
   accumulators. Do not infer useful tiles from the reference launch grid.
3. Transfer/issue plan: express publication epochs, reader retirement and
   fragment lookahead together. More pipeline stages alone did not establish
   a win; live registers and dependency distance matter.
4. Device effects/binding: descriptors and scratch are startup-owned and
   budgeted. There must be no per-step allocation/descriptor construction.
   Keep the small-row entry separately compiled so a large resource envelope
   is not imposed on decode.
5. Private lowering: realize the chosen plan with the target's matrix loads,
   transfer instructions and wait mechanism. Validate its emitted source,
   resource use and numerical behavior outside the compiler before selection.

These are outstanding implementation requirements, not work declared complete
by retaining the rejected patch.

## Local gates

After withdrawing the trials, grouped-matrix tests pass **19/19** and
tiled-source tests **14/14** with the existing migration-warning exclusions;
the installed-dispatch runner regression passes. The archive patch passes
`git apply --check`. Both `moon check --target native --deny-warn` and the
full strict test command still stop on pre-existing migration diagnostics,
including implicit derived-method promotion and `try?`; no exclusions were
added. No full-suite or new-kernel sanitizer pass is claimed.

## Restoration complete

The existing qualified deployment was restarted unchanged, not replaced by
either trial or the private diagnostic executable. Both USER units are active
with success/exit0, backend health is healthy, and the complete arithmetic
request returns content `323` with `finish_reason=stop`.

Evidence: `/tmp/lunaflux-glm-projection-restoration-20261020-v2`.
Sampled rank0/rank1 host minima are **17.519 / 19.147 GiB**, GPU maxima
**94.190 / 94.219 GiB**, owned swap zero. The restoration watcher has retired.
No hashing, overall benchmark, or reference-model reload was performed.
