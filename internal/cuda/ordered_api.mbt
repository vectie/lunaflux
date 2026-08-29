///|
pub fn prepare_ordered_kernel_executor(
  context : Context,
  stream : Stream,
  functions : FixedArray[Function],
  dimensions : FixedArray[Int],
  argument_starts : FixedArray[Int],
  allocations : FixedArray[Allocation],
  offsets : FixedArray[Int64],
  byte_counts : FixedArray[Int64],
  alignments : FixedArray[Int64],
  policy? : OrderedExecutorPolicy = OrderedExecutorEagerOnly,
) -> OrderedKernelExecutor raise CudaError {
  let policy_code = match policy {
    OrderedExecutorEagerOnly => 0
    OrderedExecutorCaptureRequired => 1
    OrderedExecutorCaptureWithEagerFallback => 2
  }
  let status = Ref(0)
  let executor = raw_ordered_executor_create(
    context, stream, functions, dimensions, argument_starts, allocations, offsets,
    byte_counts, alignments, policy_code, status,
  )
  if status.val != 0 {
    let create_status = status.val
    check_status(raw_ordered_executor_close(executor), OrderedExecutorClose)
    check_status(create_status, OrderedExecutorCreate)
  }
  executor
}

///|
pub fn OrderedKernelExecutor::execution_mode(
  self : OrderedKernelExecutor,
) -> OrderedExecutorMode raise CudaError {
  match raw_ordered_executor_mode(self) {
    0 => OrderedExecutorEager
    1 => OrderedExecutorCaptured
    status => {
      check_status(status, OrderedExecutorCreate)
      raise InvalidOutput(OrderedExecutorCreate)
    }
  }
}

///|
pub fn OrderedKernelExecutor::enqueue(
  self : OrderedKernelExecutor,
  index : Int,
) -> Unit raise CudaError {
  check_status(
    raw_ordered_executor_enqueue(self, index),
    OrderedExecutorEnqueue,
  )
}

///|
pub fn OrderedKernelExecutor::launch_captured(
  self : OrderedKernelExecutor,
) -> Unit raise CudaError {
  check_status(
    raw_ordered_executor_launch_captured(self),
    OrderedExecutorGraphLaunch,
  )
}

///|
pub fn OrderedKernelExecutor::record_completion(
  self : OrderedKernelExecutor,
) -> Unit raise CudaError {
  check_status(raw_ordered_executor_record(self), OrderedExecutorRecord)
}

///|
pub fn OrderedKernelExecutor::poll_completion(
  self : OrderedKernelExecutor,
) -> Bool raise CudaError {
  match raw_ordered_executor_poll(self) {
    0 => false
    1 => true
    status => {
      check_status(status, OrderedExecutorPoll)
      false
    }
  }
}

///|
pub fn OrderedKernelExecutor::wait_completion(
  self : OrderedKernelExecutor,
) -> Unit raise CudaError {
  check_status(raw_ordered_executor_wait(self), OrderedExecutorWait)
}

///|
pub fn OrderedKernelExecutor::abort(
  self : OrderedKernelExecutor,
) -> Unit raise CudaError {
  check_status(raw_ordered_executor_abort(self), OrderedExecutorAbort)
}

///|
pub fn OrderedKernelExecutor::reset(
  self : OrderedKernelExecutor,
) -> Unit raise CudaError {
  check_status(raw_ordered_executor_reset(self), OrderedExecutorReset)
}

///|
pub fn OrderedKernelExecutor::close(
  self : OrderedKernelExecutor,
) -> Unit raise CudaError {
  check_status(raw_ordered_executor_close(self), OrderedExecutorClose)
}
