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
