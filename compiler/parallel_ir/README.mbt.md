# Parallel IR

This package compiles declared memory effects into an immutable dependency
graph and lowers it to FIFO command queues. It has no device or scheduler
dependency. Build the graph and schedule once, before execution.

The example produces a tensor, reduces it across two ranks, and consumes the
result. Lowering inserts the two cross-queue completion dependencies.

```mbt check
///|
test "lower a compute collective compute chain" {
  let arenas = [@parallel_ir.Arena::new(bytes=8L, initialized=false)]
  let buffers = [
    @parallel_ir.Buffer::new(arena=0, offset_bytes=0L, byte_count=8L),
  ]
  let read = @parallel_ir.Access::new(buffer=0, mode=Read)
  let write = @parallel_ir.Access::new(buffer=0, mode=Write)
  let reduction = @parallel_ir.Collective::new(
    sequence=1,
    kind=SumAllReduce,
    element=BFloat16,
    send_bytes=8L,
    receive_bytes=8L,
  )
  let program = @parallel_ir.compile(rank=0, world_size=2, arenas, buffers, [
    @parallel_ir.Step::new(Compute(0), [write]),
    @parallel_ir.Step::new(Communicate(reduction), [read, write]),
    @parallel_ir.Step::new(Compute(1), [read]),
  ])
  let schedule = program.lower(ComputeCommunicationOverlap)
  assert_true(program.happens_before(0, 2))
  assert_eq(schedule.event_producers().length(), 2)
  assert_eq(schedule.commands(CommunicationLane).length(), 3)
  assert_eq(schedule.completion_steps().length(), 2)
}
```

`Compute` carries the caller's operation ID. `Execute` carries a step index;
`Record` and `Wait` carry indices into `event_producers()`. A wait denotes
producer **completion**, not merely submission. Both queue tails must finish
before memory or event reuse. Group composition checks the collective protocol
across ranks before backend binding.

The TP execution planner exposes `parallel_program()` and `parallel_group()`
to derive these values from admitted physical bindings. Current production
execution defaults to ordered; the TP worker can explicitly bind the overlap
schedule at startup. Physical qualification remains open. See
[the full contract](../../docs/PARALLEL_IR.md).

`ForkJoinSpan` additionally describes two independent compute-producer ranges
inside an ordered submission interval. The binding owner, not this inert
schedule, establishes disjoint writes and shared read-only inputs.
`OrderedSubmissionPlan.validate_forks` rejects overlapping/nested spans and
collective cuts through the producers. The collective-kernel owner binds the
second range to an auxiliary queue and joins it before the common consumer;
graph replay preserves the same dependencies. This is distinct from
compute/communication overlap. The selected GLM shared/routed implementation
and bounded GPU replay/cancellation tests are recorded in
[the October18 measurements](../../docs/GLM_PREFILL_AND_BRANCH_OVERLAP_2026-10-18.md).

`CaptureExactCollectives(rows)` prepares whole effect frames for one row and
exactly `rows`, plus interval graphs for every unmatched count. Unlike padded
bucket replay, a partial prefill never communicates inactive rows. Its three
graph families are budgeted before binding, retain the same collective order,
and use whole-frame retirement/cancellation. This is an explicit alternative,
not a default performance claim; profiler/comparison evidence remains outside
the compiler.

`CaptureWindowedCollectives(rows, maximum_sites)` additionally bounds the
communication horizon of each exact-prefill graph. Windows partition existing
joined collective cuts, preserving reductions and the final compute tail.
Decode stays one graph; partial rows use unchanged interval replay. All windows
share an ordered device stream and whole-frame retirement; no intermediate
host wait or token-path allocation is introduced. The positive site limit is
an explicit resource choice, not a vendor constant or numerical guarantee.
