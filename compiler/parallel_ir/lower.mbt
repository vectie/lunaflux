///|
fn lane(step : Step, policy : Policy) -> Int {
  match step.action {
    Communicate(_) if policy == ComputeCommunicationOverlap => 1
    _ => 0
  }
}

///|
/// Pure queue lowering. A wait targets completion, not enqueue, of its producer.
/// All events and command storage can be allocated once by a backend adapter.
pub fn Program::lower(self : Program, policy : Policy) -> Schedule {
  let waits = FixedArray::make(self.steps.length(), -1)
  let events = FixedArray::make(self.steps.length(), -1)
  let producers : Array[Int] = []
  let required = FixedArray::make(2, -1)
  for index, step in self.steps {
    let selected_lane = lane(step, policy)
    let mut latest = -1
    for previous in step.dependencies {
      if lane(self.steps[previous], policy) != selected_lane &&
        previous > latest {
        latest = previous
      }
    }
    if latest > required[selected_lane] {
      waits[index] = latest
      events[latest] = 0
      required[selected_lane] = latest
    }
  }
  for index in 0..<events.length() {
    if events[index] == 0 {
      events[index] = producers.length()
      producers.push(index)
    }
  }
  let queues : Array[Array[Command]] = [[], []]
  let tails = Array::make(2, -1)
  for index, step in self.steps {
    let selected_lane = lane(step, policy)
    let queue = queues[selected_lane]
    if waits[index] >= 0 {
      queue.push(Wait(event=events[waits[index]]))
    }
    queue.push(Execute(step=index))
    if events[index] >= 0 {
      queue.push(Record(event=events[index]))
    }
    tails[selected_lane] = index
  }
  {
    program: self,
    policy,
    compute: ReadOnlyArray::from_array(queues[0]),
    communication: ReadOnlyArray::from_array(queues[1]),
    event_producers: ReadOnlyArray::from_array(producers),
    tails: ReadOnlyArray::from_array(tails.filter(index => index >= 0)),
  }
}
