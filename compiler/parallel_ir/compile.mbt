///|
/// Pure compilation in topological source order. Physical aliases introduce
/// RAW/WAR/WAW edges; all analysis and allocation are confined to this call.
pub fn compile(
  rank~ : Int,
  world_size~ : Int,
  arenas : ArrayView[Arena],
  buffers : ArrayView[Buffer],
  source : ArrayView[Step],
) -> Program raise ParallelError {
  if world_size <= 0 ||
    rank < 0 ||
    rank >= world_size ||
    arenas.is_empty() ||
    source.is_empty() {
    raise Capacity
  }
  for i, buffer in buffers {
    guard arenas.get(buffer.arena) is Some(arena) else { raise Buffer(i) }
    if buffer.offset + buffer.bytes > arena.bytes {
      raise Buffer(i)
    }
  }
  let frontier : Array[Array[Use]] = Array::makei(arenas.length(), _ => [])
  let mutable_arenas = FixedArray::make(arenas.length(), false)
  for step in source {
    for access in step.accesses {
      guard buffers.get(access.buffer) is Some(buffer) else {
        raise Buffer(access.buffer)
      }
      if access.mode.writes() {
        mutable_arenas[buffer.arena] = true
      }
    }
  }
  let users : Array[Array[Int]] = Array::makei(buffers.length(), _ => [])
  let steps : Array[Step] = []
  let collectives : Array[Int] = []
  for index, step in source {
    let dependencies : Map[Int, Unit] = Map([])
    for previous in step.dependencies {
      if previous < 0 || previous >= index {
        raise Dependency(index)
      }
      dependencies[previous] = ()
    }
    match step.action {
      Compute(operation) => if operation < 0 { raise Step(index) }
      Communicate(collective) => {
        collective.validate(collectives.length() + 1, world_size)
        if collectives.last() is Some(previous) {
          dependencies[previous] = ()
        }
        collectives.push(index)
      }
    }
    // Query every operand before updating any frontier: accesses within one
    // kernel are one indivisible effect, not an artificial self-dependency.
    for access in step.accesses {
      guard buffers.get(access.buffer) is Some(buffer) else {
        raise Buffer(access.buffer)
      }
      if access.mode.reads() &&
        !arenas[buffer.arena].initialized &&
        !initialized(frontier[buffer.arena], buffer) {
        raise Uninitialized(step=index, buffer=access.buffer)
      }
      for previous in frontier[buffer.arena] {
        if overlaps(previous, buffer) &&
          (previous.mode.writes() || access.mode.writes()) {
          dependencies[previous.step] = ()
        }
      }
      if users[access.buffer].last() != Some(index) {
        users[access.buffer].push(index)
      }
    }
    for access in step.accesses {
      let buffer = buffers[access.buffer]
      if mutable_arenas[buffer.arena] {
        frontier[buffer.arena] = update_frontier(
          frontier[buffer.arena],
          buffer,
          access.mode,
          index,
        )
      }
    }
    let ordered = dependencies.keys().to_array()
    ordered.sort()
    steps.push(Step::new(step.action, step.accesses, dependencies=ordered))
  }
  {
    rank,
    world_size,
    arenas: ReadOnlyArray::from_array(arenas),
    buffers: ReadOnlyArray::from_array(buffers),
    steps: ReadOnlyArray::from_array(steps),
    lifetimes: ReadOnlyArray::from_array(
      users.map(users => { users: ReadOnlyArray::from_array(users), }),
    ),
    collective_steps: ReadOnlyArray::from_array(collectives),
  }
}
