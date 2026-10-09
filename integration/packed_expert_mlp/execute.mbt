///|
/// Startup-owned reduction site, not a model-family or backend policy.
pub struct MoeReduction {
  priv owner : @device.TensorCollectiveOwner
  priv generation : UInt64
  priv site_sequence : Int
  priv operation_id : Int
}

///|
pub fn MoeReduction::new(
  owner : @device.TensorCollectiveOwner,
  generation~ : UInt64,
  site_sequence~ : Int,
  operation_id~ : Int,
) -> Self {
  { owner, generation, site_sequence, operation_id, }
}

///|
priv enum MoeExecutionState {
  Idle
  AwaitingSubmission
  AwaitingCompletion
  Failed
  Closed
} derive(Eq)

///|
/// Reusable routed -> reduce -> shared -> finalize execution. Allocation and
/// launch metadata are prepared once. The communicator is borrowed; normal
/// close consumes only this queue, while fault abort also aborts its reduction.
pub struct MoeExecutor {
  priv queue : @device.OrderedKernelExecutor
  priv stream : @device.Stream
  priv routed_output : @device.Allocation
  priv reduction : MoeReduction?
  priv maximum_rows : Int
  priv routed_count : Int
  priv mut state : MoeExecutionState
  priv mut reduction_complete : Bool
}

///|
pub fn prepare_moe(
  context : @device.Context,
  stream : @device.Stream,
  routed : ArrayView[@device.OrderedKernelLaunch],
  shared : ArrayView[@device.OrderedKernelLaunch],
  finalize : @device.OrderedKernelLaunch,
  routed_output~ : @device.Allocation,
  maximum_rows~ : Int,
  reduction? : MoeReduction,
) -> MoeExecutor raise @device.DeviceError {
  guard !routed.is_empty() && !shared.is_empty() && maximum_rows > 0 else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let launches = routed.to_owned()
  launches.append(shared)
  launches.push(finalize)
  let queue = context.create_ordered_kernel_executor(stream, launches)
  {
    queue,
    stream,
    routed_output,
    reduction,
    maximum_rows,
    routed_count: routed.length(),
    state: Idle,
    reduction_complete: false,
  }
}

///|
fn MoeExecutor::enqueue_tail(self : Self) -> Unit raise @device.DeviceError {
  for index in self.routed_count..<self.queue.kernel_count() {
    self.queue.enqueue(index)
  }
  self.queue.record_completion()
  self.state = AwaitingCompletion
}

///|
/// Caller has already updated the device counts and routing buffers. No GPU
/// diagnostic readback or launch-pack allocation occurs here.
pub fn MoeExecutor::submit(
  self : Self,
  live_rows~ : Int,
  plan_sequence~ : UInt64,
  global_sequence~ : UInt64,
) -> Unit raise @device.DeviceError {
  guard self.state == Idle else {
    raise @device.DeviceError::Busy(OrderedExecutorEnqueue)
  }
  guard live_rows > 0 && live_rows <= self.maximum_rows else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
  }
  errdefer {
    self.state = Failed
  }
  for index in 0..<self.routed_count {
    self.queue.enqueue(index)
  }
  match self.reduction {
    None => {
      self.reduction_complete = true
      self.enqueue_tail()
    }
    Some(site) => {
      site.owner.submit(
        generation=site.generation,
        plan_sequence~,
        global_sequence~,
        site_sequence=site.site_sequence,
        operation_id=site.operation_id,
        kind=F32SumAllReduce,
        live_query_tokens=live_rows,
        send=self.routed_output,
        send_offset=0L,
        receive=self.routed_output,
        receive_offset=0L,
        stream=self.stream,
      )
      self.reduction_complete = false
      self.state = AwaitingSubmission
    }
  }
}

///|
/// Nonblocking progress. Shared work is appended only after NCCL has issued
/// the reduction into the same stream; GPU ordering, not a host wait, connects
/// reduction to shared/finalize. Completion rearms the exact startup queue.
pub fn MoeExecutor::poll(self : Self) -> Bool raise @device.DeviceError {
  guard self.state == AwaitingSubmission || self.state == AwaitingCompletion else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorPoll)
  }
  errdefer {
    self.state = Failed
  }
  if self.state == AwaitingSubmission {
    if self.reduction is Some(site) && site.owner.poll_submitted() {
      self.enqueue_tail()
    } else {
      return false
    }
  }
  if !self.reduction_complete && self.reduction is Some(site) {
    self.reduction_complete = site.owner.poll()
  }
  if self.reduction_complete && self.queue.poll_completion() {
    self.queue.reset()
    self.state = Idle
    true
  } else {
    false
  }
}

///|
pub fn MoeExecutor::close(self : Self) -> Unit raise @device.DeviceError {
  if self.state == Closed {
    return
  }
  guard self.state == Idle else {
    raise @device.DeviceError::Busy(OrderedExecutorClose)
  }
  self.queue.close()
  self.state = Closed
}

///|
/// Explicit cancellation/failure drain; borrowed communicator is terminal.
pub fn MoeExecutor::abort(self : Self) -> Unit raise @device.DeviceError {
  if self.state == Closed {
    return
  }
  self.state = Failed
  if self.reduction is Some(site) {
    site.owner.abort()
  }
  self.queue.abort()
  self.queue.close()
  self.state = Closed
}
