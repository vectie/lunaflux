///|
fn overlap_program() -> @parallel_ir.Program raise {
  compile(
    [
      compute(0, [access(0, Write)]),
      collective(1, [access(0, ReadWrite)]),
      compute(1, [access(1, Write)]),
      compute(2, [access(0, Read), access(1, Read), access(2, Write)]),
      compute(3, [access(3, Write)]),
    ],
    [view(0L, 4L), view(8L, 4L), view(16L, 4L), view(0L, 4L)],
  )
}

///|
test "overlap lowering fences consumers and reuse but permits independent work" {
  let schedule = overlap_program().lower(ComputeCommunicationOverlap)
  let compute = schedule.commands(ComputeLane)
  assert_eq(compute.length(), 6)
  assert_true(compute[0] == Execute(step=0))
  assert_true(compute[1] == Record(event=0))
  assert_true(compute[2] == Execute(step=2))
  assert_true(compute[3] == Wait(event=1))
  assert_true(compute[4] == Execute(step=3))
  assert_true(compute[5] == Execute(step=4))
  let communication = schedule.commands(CommunicationLane)
  assert_eq(communication.length(), 3)
  assert_true(communication[0] == Wait(event=0))
  assert_true(communication[1] == Execute(step=1))
  assert_true(communication[2] == Record(event=1))
  assert_eq(schedule.event_producers().length(), 2)
  assert_eq(schedule.completion_steps().length(), 2)
}

///|
test "ordered lowering exactly preserves source dispatch with no events" {
  let program = overlap_program()
  let schedule = program.lower(Ordered)
  assert_eq(schedule.event_producers().length(), 0)
  assert_eq(schedule.commands(CommunicationLane).length(), 0)
  for index, command in schedule.commands(ComputeLane) {
    assert_true(command == Execute(step=index))
  }
  assert_eq(schedule.completion_steps()[0], 4)
  let replay = program.lower(ComputeCommunicationOverlap)
  let original = program.lower(ComputeCommunicationOverlap)
  for lane in [@parallel_ir.ComputeLane, @parallel_ir.CommunicationLane] {
    for i, command in original.commands(lane) {
      assert_true(command == replay.commands(lane)[i])
    }
  }
}

///|
// Exhaustive reference execution of the two FIFO command streams. A blocked
// wait makes that stream ineligible; every possible eligible next command is
// explored. This detects both lost dependencies and deadlocking event plans.
fn explore(
  schedule : @parallel_ir.Schedule,
  positions : Array[Int],
  events : Array[Bool],
  memory : Array[Int],
) -> Int raise {
  let lanes = [@parallel_ir.ComputeLane, @parallel_ir.CommunicationLane]
  if positions[0] == schedule.commands(lanes[0]).length() &&
    positions[1] == schedule.commands(lanes[1]).length() {
    assert_eq(memory[0], 7)
    assert_eq(memory[1], 10)
    assert_eq(memory[2], 12)
    return 1
  }
  let mut paths = 0
  for lane in 0..<2 {
    let commands = schedule.commands(lanes[lane])
    if positions[lane] == commands.length() {
      continue
    }
    let command = commands[positions[lane]]
    if command is Wait(event~) && !events[event] {
      continue
    }
    let next_positions = positions.copy()
    let next_events = events.copy()
    let next_memory = memory.copy()
    next_positions[lane] += 1
    match command {
      Wait(..) => ()
      Record(event~) => next_events[event] = true
      Execute(step~) =>
        match step {
          0 => next_memory[0] = 1
          1 => next_memory[0] *= 2
          2 => next_memory[1] = 10
          3 => next_memory[2] = next_memory[0] + next_memory[1]
          4 => next_memory[0] = 7
          _ => fail("unknown fixture step")
        }
    }
    paths += explore(schedule, next_positions, next_events, next_memory)
  }
  if paths == 0 {
    fail("queue dependency deadlock")
  }
  paths
}

///|
test "every legal queue interleaving matches ordered reference output" {
  let program = overlap_program()
  assert_eq(explore(program.lower(Ordered), [0, 0], [], [0, 0, 0]), 1)
  let parallel = program.lower(ComputeCommunicationOverlap)
  assert_true(
    explore(
      parallel,
      [0, 0],
      Array::make(parallel.event_producers().length(), false),
      [0, 0, 0],
    ) >
    1,
  )
}

///|
test "dependencies on an earlier FIFO producer reuse the later completion event" {
  let program = compile(
    [
      compute(0, []),
      compute(1, []),
      @parallel_ir.Step::new(
        Communicate(
          @parallel_ir.Collective::new(
            sequence=1,
            kind=SumAllReduce,
            element=BFloat16,
            send_bytes=4L,
            receive_bytes=4L,
          ),
        ),
        [],
        dependencies=[0, 1],
      ),
    ],
    [],
  )
  let schedule = program.lower(ComputeCommunicationOverlap)
  assert_eq(schedule.event_producers().length(), 1)
  assert_eq(schedule.event_producers()[0], 1)
}

///|
test "a previously satisfied lane dependency does not allocate another event" {
  let program = compile(
    [
      collective(1, []),
      collective(2, []),
      compute(0, [], dependencies=[1]),
      compute(1, [], dependencies=[0]),
    ],
    [],
  )
  let schedule = program.lower(ComputeCommunicationOverlap)
  assert_eq(schedule.event_producers().length(), 1)
  assert_eq(schedule.event_producers()[0], 1)
  assert_eq(schedule.commands(ComputeLane).length(), 3)
  assert_eq(schedule.commands(CommunicationLane).length(), 3)
}
