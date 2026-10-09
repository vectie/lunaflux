///|
// Independent, deliberately quadratic reference: every conflicting source
// pair must remain ordered. This checks the interval-frontier implementation,
// including overwrites that split earlier tracked intervals.
test "interval analysis preserves every source hazard across varied aliased slices" {
  for seed in 1..<17 {
    let buffers : Array[@parallel_ir.Buffer] = []
    let modes : Array[@parallel_ir.AccessMode] = []
    let steps : Array[@parallel_ir.Step] = []
    let mut state = seed.reinterpret_as_uint()
    let mut sequence = 0
    for index in 0..<32 {
      state = state * 1664525U + 1013904223U
      let offset = ((state >> 8) % 12U).reinterpret_as_int().to_int64() * 4L
      state = state * 1664525U + 1013904223U
      let bytes = ((state >> 8) % 4U + 1U).reinterpret_as_int().to_int64() * 4L
      let mode : @parallel_ir.AccessMode = match state % 3U {
        0U => Read
        1U => Write
        _ => ReadWrite
      }
      buffers.push(view(offset, bytes))
      modes.push(mode)
      if index % 5 == 1 {
        sequence += 1
        steps.push(collective(sequence, [access(index, mode)]))
      } else {
        steps.push(compute(index, [access(index, mode)]))
      }
    }
    let program = compile(steps, buffers, initialized=true)
    for later in 0..<steps.length() {
      for earlier in 0..<later {
        let a = buffers[earlier]
        let b = buffers[later]
        if a.offset_bytes() < b.offset_bytes() + b.byte_count() &&
          b.offset_bytes() < a.offset_bytes() + a.byte_count() &&
          (modes[earlier] != Read || modes[later] != Read) {
          assert_true(program.happens_before(earlier, later))
        }
      }
    }
    assert_lowered_dependencies(program.lower(ComputeCommunicationOverlap))
  }
}

///|
// Reconstruct ordering solely from emitted FIFO queues and events, independent
// of the lowering implementation's predecessor selection.
fn assert_lowered_dependencies(schedule : @parallel_ir.Schedule) -> Unit raise {
  let steps = schedule.program().steps()
  let reachable = Array::makei(steps.length(), _ => {
    Array::make(steps.length(), false)
  })
  for lane in [@parallel_ir.ComputeLane, @parallel_ir.CommunicationLane] {
    let mut previous = -1
    let mut waited = -1
    for command in schedule.commands(lane) {
      match command {
        Execute(step~) => {
          if previous >= 0 {
            reachable[previous][step] = true
          }
          if waited >= 0 {
            reachable[waited][step] = true
          }
          previous = step
        }
        Record(event~) => assert_eq(schedule.event_producers()[event], previous)
        Wait(event~) => waited = schedule.event_producers()[event]
      }
    }
  }
  for middle in 0..<steps.length() {
    for from in 0..<steps.length() {
      for to in 0..<steps.length() {
        if reachable[from][middle] && reachable[middle][to] {
          reachable[from][to] = true
        }
      }
    }
  }
  for index, step in steps {
    assert_false(reachable[index][index])
    for dependency in step.dependencies() {
      assert_true(reachable[dependency][index])
    }
  }
}
