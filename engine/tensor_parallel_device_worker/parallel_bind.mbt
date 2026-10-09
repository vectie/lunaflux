///|
priv enum ParallelExecution {
  OrderedExecution
  OverlapExecution(TensorParallelOverlap)
}

///|
priv enum ParallelCommand {
  Kernel(Int)
  Collective(@tensor_parallel_execution_plan.TensorParallelCollectiveBinding)
  Record(Int)
  Wait(Int)
}

///|
priv struct ParallelCommands {
  compute : ReadOnlyArray[ParallelCommand]
  communication : ReadOnlyArray[ParallelCommand]
  event_count : Int
}

///|
priv struct ParallelEvent {
  resource : @device.Event
  mut open : Bool
}

///|
priv struct TensorParallelOverlap {
  commands : ParallelCommands
  communication_stream : TensorParallelOwnedStream
  events : Array[ParallelEvent]
  recorded : FixedArray[Bool]
  mut compute_cursor : Int
  mut communication_cursor : Int
  mut collective_pending : Bool
  mut submission_pending : Bool
  mut completion_recorded : Bool
  mut drained : Bool
}

///|
fn bind_parallel_commands(
  execution : @tensor_parallel_execution_plan.TensorParallelExecutionPlan,
) -> ParallelCommands raise {
  let program = execution.parallel_program()
  let schedule = program.lower(ComputeCommunicationOverlap)
  fn bind(lane : @parallel_ir.Lane) -> ReadOnlyArray[ParallelCommand] raise {
    let result : Array[ParallelCommand] = []
    for command in schedule.commands(lane) {
      result.push(
        match command {
          Execute(step~) =>
            match program.steps()[step].action() {
              Compute(operation) => Kernel(operation)
              Communicate(collective) =>
                Collective(
                  execution.collective_exact(collective.sequence() - 1),
                )
            }
          Record(event~) => Record(event)
          Wait(event~) => Wait(event)
        },
      )
    }
    ReadOnlyArray::from_array(result)
  }
  {
    compute: bind(ComputeLane),
    communication: bind(CommunicationLane),
    event_count: schedule.event_producers().length(),
  }
}

///|
fn prepare_overlap(
  context : @device.Context,
  commands : ParallelCommands,
  resources : TensorParallelPartialResources,
) -> Unit raise TensorParallelDeviceWorkerError {
  let stream = context.create_stream() catch {
    error => raise Device(Stream, error)
  }
  let overlap : TensorParallelOverlap = {
    commands,
    communication_stream: { resource: stream, open: true, },
    events: [],
    recorded: FixedArray::make(commands.event_count, false),
    compute_cursor: 0,
    communication_cursor: 0,
    collective_pending: false,
    submission_pending: false,
    completion_recorded: false,
    drained: false,
  }
  // Publish ownership before the next fallible operation, so partial event
  // creation and failed destruction preserve the existing retry authority.
  resources.overlap = Some(overlap)
  for _ in 0..<commands.event_count {
    let event = context.create_dependency_event() catch {
      error => raise Device(ParallelEvent, error)
    }
    overlap.events.push({ resource: event, open: true, })
  }
}

///|
fn TensorParallelOverlap::reset(self : TensorParallelOverlap) -> Unit {
  self.compute_cursor = 0
  self.communication_cursor = 0
  self.collective_pending = false
  self.submission_pending = false
  self.completion_recorded = false
  self.drained = false
  for index in 0..<self.recorded.length() {
    self.recorded[index] = false
  }
}

///|
fn close_overlap(
  overlap : TensorParallelOverlap,
  compute : TensorParallelOwnedStream?,
) -> Unit raise TensorParallelDeviceWorkerError {
  // NCCL is consumed first by the cleanup chain. Every queued wait already
  // has a recorded producer, so neither stream can wait on an unissued event.
  if !overlap.drained {
    overlap.communication_stream.resource.synchronize() catch {
      error => raise Device(Stream, error)
    }
    if compute is Some(stream) {
      stream.resource.synchronize() catch {
        error => raise Device(Stream, error)
      }
    }
    overlap.drained = true
  }
  for index = overlap.events.length() - 1; index >= 0; index = index - 1 {
    let event = overlap.events[index]
    if event.open {
      event.resource.close() catch {
        error => raise Device(ParallelEvent, error)
      }
      event.open = false
    }
  }
  close_stream_owner(overlap.communication_stream)
}
