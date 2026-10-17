# Parallel HC and collective-inclusive replay

Continue from `GLM_BULK_EXPERT_2026-10-17.md`. The measured decode HC region
is 12.983 ms; it contains 16,384-add serial control folds. The executor still
submits 91 compute graphs separated by 90 collectives. These profiled durations
are hypotheses for removable dependencies, not an additive attribution of the
unprofiled 36.86% completion gap.

## Implementation contract

* Reuse precision IR's explicit tree square/dot laws. Reassociation defines a
  different finite-precision program; it is not an exact rewrite of ordered
  addition and agreement with vLLM is not a correctness proof.
* Physical HC planning owns reduction partition geometry and legal fusion.
  Fuse collapse/normalization only while preserving the BF16 intermediate
  publication, normalization law and each scalar rounding boundary. Hardware
  instructions belong exclusively in private CUDA lowering.
* Parallel IR owns a fixed-shape collective-inclusive execution graph. Capture
  is startup-only, with stable addresses, ordered rank agreement, explicit
  memory reserve and retained communicator/buffer lifetimes. No live-request
  capture, variable NCCL count inside a fixed graph, or removal of cancellation
  and terminal retirement is permitted. Unsupported shapes retain an explicit
  prepared path, not silently mis-sized collectives.
* Comparison, error measurements, profiler interpretation and quality decisions
  remain outside compiler/runtime. No payload hashing or integrity scans.

## Bounded experiment

At most three HC candidates and two capture variants, followed by separate
HC-only, graph-only and combined full-model ablations against the retained
whole-tile baseline. Use identical literal inputs, output-token vectors,
unprofiled alternating trials, selected-symbol/graph evidence, affected
sanitizer/leak/cancellation tests and a separate vLLM comparison. Retain failures
and regressions. Do not equate microkernel benefit with request benefit.

Keep the original GLM service running during source work. Stop only its exact
head/worker containers for physical tests, leave NFS untouched, and restore the
same containers afterwards through successful health/model responses. Each
task owner stays within 100 GiB and zero swap; independently maintain at least
16 GiB host MemAvailable on each Spark. No concurrent benchmark GPU owners.

## Validation before full-model ablations

Implementation `1251a565` propagates both alternatives through the plans,
lowering, frame binding and explicit checkpoint CLI. Defaults are unchanged.
The clean ARM build peaks at 797.4 MiB with zero swap. The HC experiment
`/tmp/lunaflux-glm-hyper-tree-20261017-v1` passes all four geometry cases and
live-row/tail/identity controls bitwise against an independent staged renderer
of the **same tree arithmetic**. Full GPU memcheck/leak, racecheck and synccheck
report zero errors. This is not bitwise equivalence to ordered reduction or a
vLLM quality claim. Its serial tree oracle is not a production speed baseline.

The first full native suite was 5,182/5,183: the new cancellation fixture lacked
capture API redirects and incorrectly passed fake handles to the real ABI.
The repaired fixture's collective and cancellation tests pass 6/6. Native
ordered-executor and NCCL graph lease/sequence/cancellation ASan/UBSan gates
also pass. Graph teardown now discards executable/raw graph references before
NCCL abort, retaining operand leases until stream drain and executor close;
otherwise NCCL abort can wait for references owned by the graph itself.

After the fixture repair, the complete native suite passes **5,184/5,184**.
Full-model ablations are in progress. The first capture attempt
`/tmp/lunaflux-model-bench-glm-collective-graph-20261017-v1` fails before requests
with `Busy` while NCCL's lazy connection jobs are outstanding. Both ranks close
their CUDA contexts, zero swap. Preserve this failure, not a timing sample.
The startup poll-count cutoff has been replaced by an explicit caller-supplied
monotonic timeout (default 60 seconds); replay performs no clock reads.
vLLM additionally warms up NCCL before capture; whether warmup is necessary
here must be distinguished from an artificially short polling budget.

Fusion is
collapse plus normalization, preserving the BF16 intermediate; it is not yet
vLLM's cross-layer post/pre fusion.

## Separate full-model ablations and capture repair

HC-only runs `glm-parallel-hc-20261017-v1/v2` are repeatable, but change the
generated token IDs for three prompts. The three unchanged-output requests
improve from 885/902/545 ms to 779/812/505 ms in repeat two (7–12%). Aggregate
time is misleading: the first answer now emits eight tokens rather than four.
The tree arithmetic remains an explicit alternative, not a quality-approved
default. Both ranks close cleanly, zero swap, with the host reserve respected.

The longer startup timeout permits capture without adding a warmup. The
capacity-only graph run `glm-collective-graph-20261017-v2` completes inference
but regresses: an eight-token case rises from 902 to 1,307 ms, and a three-token
case from 545 to 672 ms. It communicates 64 rows at every decode site instead
of one. These are provisional request timings from a lifecycle-failed run,
not a successful benchmark gate.

The egress rank then stalls in NCCL teardown. A scoped debugger backtrace
finds `commDestroySync -> ncclCommPollCallbacks -> ncclIntruQueueMpscDequeueAll`:
the checkpoint caller's reverse-order defer aborts its communicator before
closing its graph-owning stage. The exact task unit was terminated after TERM
failed; original service containers and NFS were not modified. Failure logs
are retained in that experiment root. The caller now closes rank/stage before
aborting the communicator, including error paths.

The second schedule is a pure parallel-IR row-bucket plan: one-row decode and
capacity-bounded prefill. Both graphs are prepared at startup. NCCL retains
each graph's operand leases separately from transient submissions; each replay
advances only the logical site count, never the total captured lease count.
Graph references retire before communicator abort. No runtime allocation,
clock, capture or change to kernel arithmetic is added to replay. Alternating
bucket replay/cancellation and native reference-balance/ASan/UBSan tests cover
both graph lifetimes. Full-model validation of this repair is pending.
