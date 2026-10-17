///|
priv struct BranchQueue {
  queue : @device.OrderedKernelExecutor
  stream : @device.Stream
  ready : @device.Event
  done : @device.Event
}

///|
priv struct Dispatch {
  branch : Int
  index : Int
  fork : Int
  join : Int
}

///|
/// Startup-bound independent branches. No launch packing, stream creation,
/// allocation or host synchronization occurs during enqueue/replay. The main
/// completion event is downstream of every auxiliary producer.
priv struct PreparedQueue {
  main : @device.OrderedKernelExecutor
  stream : @device.Stream
  branches : ReadOnlyArray[BranchQueue]
  dispatch : ReadOnlyArray[Dispatch]
  tail_join : Int
  mut captured : Bool
}

///|
fn PreparedQueue::new(
  context : @device.Context,
  stream : @device.Stream,
  launches : ArrayView[@device.OrderedKernelLaunch],
  spans : ArrayView[@parallel.ForkJoinSpan],
  capture? : Bool = false,
) -> Self raise @device.DeviceError {
  let branches : Array[BranchQueue] = []
  let primary : Array[@device.OrderedKernelLaunch] = []
  let dispatch : Array[Dispatch] = []
  errdefer (for branch in branches.rev() {
    branch.queue.abort() catch {
      _ => ()
    }
    branch.queue.close() catch {
      _ => ()
    }
    branch.done.close() catch {
      _ => ()
    }
    branch.ready.close() catch {
      _ => ()
    }
    branch.stream.close() catch {
      _ => ()
    }
  })
  for span in spans {
    let (_, split, join) = span.range()
    let auxiliary = context.create_stream()
    errdefer (auxiliary.close() catch { _ => () })
    let ready = context.create_dependency_event()
    errdefer (ready.close() catch { _ => () })
    let done = context.create_dependency_event()
    errdefer (done.close() catch { _ => () })
    let queue = context.create_ordered_kernel_executor(
      auxiliary,
      launches[split:join],
    )
    branches.push({ queue, stream: auxiliary, ready, done, })
  }
  let mut tail_join = -1
  for i, launch in launches {
    let mut branch = -1
    let mut local_index = primary.length()
    let mut fork = -1
    let mut join = -1
    for b, span in spans {
      let (first, split, end) = span.range()
      if i >= split && i < end {
        branch = b
        local_index = i - split
      }
      if i == first {
        fork = b
      }
      if i == end {
        join = b
      }
      if end == launches.length() {
        tail_join = b
      }
    }
    if branch < 0 {
      primary.push(launch)
    }
    dispatch.push({ branch, index: local_index, fork, join, })
  }
  let main = context.create_ordered_kernel_executor(stream, primary)
  let result : PreparedQueue = {
    main,
    stream,
    branches: ReadOnlyArray::from_array(branches),
    dispatch: ReadOnlyArray::from_array(dispatch),
    tail_join,
    captured: false,
  }
  errdefer {
    result.abort() catch {
      _ => ()
    }
    result.close() catch {
      _ => ()
    }
  }
  if capture {
    result.begin_effect_capture()
    for i in 0..<launches.length() {
      result.enqueue(i)
    }
    result.end_effect_capture()
  }
  result
}

///|
fn PreparedQueue::enqueue(
  self : Self,
  index : Int,
) -> Unit raise @device.DeviceError {
  let op = self.dispatch[index]
  if op.join >= 0 {
    self.stream.wait_event(self.branches[op.join].done)
  }
  if op.fork >= 0 {
    let branch = self.branches[op.fork]
    branch.ready.record(self.stream)
    branch.stream.wait_event(branch.ready)
    for i in 0..<branch.queue.kernel_count() {
      branch.queue.enqueue(i)
    }
    branch.queue.record_completion()
    branch.done.record(branch.stream)
  }
  if op.branch < 0 {
    self.main.enqueue(op.index)
  }
}

///|
fn PreparedQueue::join_tail(self : Self) -> Unit raise @device.DeviceError {
  if self.tail_join >= 0 {
    self.stream.wait_event(self.branches[self.tail_join].done)
  }
}

///|
fn PreparedQueue::join_before(
  self : Self,
  index : Int,
) -> Unit raise @device.DeviceError {
  if index == self.dispatch.length() {
    self.join_tail()
  } else {
    let join = self.dispatch[index].join
    if join >= 0 {
      self.stream.wait_event(self.branches[join].done)
    }
  }
}

///|
fn PreparedQueue::begin_effect_capture(
  self : Self,
) -> Unit raise @device.DeviceError {
  self.main.begin_effect_capture()
}

///|
fn PreparedQueue::end_effect_capture(
  self : Self,
) -> Unit raise @device.DeviceError {
  self.join_tail()
  self.main.end_effect_capture()
  self.captured = true
}

///|
fn PreparedQueue::launch_captured(
  self : Self,
) -> Unit raise @device.DeviceError {
  self.main.launch_captured()
}

///|
fn PreparedQueue::record_completion(
  self : Self,
) -> Unit raise @device.DeviceError {
  if !self.captured {
    self.join_tail()
  }
  self.main.record_completion()
}

///|
fn PreparedQueue::poll_completion(
  self : Self,
) -> Bool raise @device.DeviceError {
  if !self.main.poll_completion() {
    return false
  }
  if !self.captured {
    for i in 0..<self.branches.length() {
      let branch = self.branches[i]
      if !branch.queue.poll_completion() {
        return false
      }
    }
  }
  true
}

///|
fn PreparedQueue::reset(self : Self) -> Unit raise @device.DeviceError {
  self.main.reset()
  // Auxiliary queues submit only during capture, not graph replay. Their
  // descriptors/events remain leased until the graph is destroyed.
  if !self.captured {
    for i in 0..<self.branches.length() {
      self.branches[i].queue.reset()
    }
  }
}

///|
fn PreparedQueue::discard_effect_graph(
  self : Self,
) -> Unit raise @device.DeviceError {
  self.main.discard_effect_graph()
}

///|
fn PreparedQueue::close(self : Self) -> Unit raise @device.DeviceError {
  self.main.close()
  for branch in self.branches.rev() {
    branch.queue.abort()
    branch.queue.close()
    branch.done.close()
    branch.ready.close()
    branch.stream.close()
  }
}

///|
fn PreparedQueue::abort(self : Self) -> Unit raise @device.DeviceError {
  self.main.discard_effect_graph()
  // Retain both branch operand leases until BOTH streams are drained, even
  // on partial submission. Never release one branch while another can read it.
  self.main.abort()
  for branch in self.branches {
    branch.queue.abort()
  }
}
