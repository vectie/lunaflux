///|
/// Binds one already validated function, launch geometry, and immutable
/// argument pack into the startup input for an ordered executor.
pub fn OrderedKernelLaunch::new(
  function : Function,
  dimensions : KernelLaunchDimensions,
  arguments : KernelArguments,
) -> OrderedKernelLaunch {
  { function, dimensions, arguments }
}

///|
/// Builds one reusable ordered queue. The native create boundary revalidates
/// every record, acquires long-lived leases for its stream, functions, and
/// argument allocations, and creates its sole completion event atomically.
/// No launch metadata is constructed in the warmed path.
pub fn Context::create_ordered_kernel_executor(
  self : Context,
  stream : Stream,
  launches : ArrayView[OrderedKernelLaunch],
  policy? : OrderedKernelExecutionPolicy = OrderedKernelEagerOnly,
) -> OrderedKernelExecutor raise DeviceError {
  if launches.is_empty() {
    raise InvalidArgument(OrderedExecutorCreate)
  }
  let functions : Array[@cuda.Function] = []
  let dimensions : Array[Int] = []
  let argument_starts : Array[Int] = [0]
  let allocations : Array[@cuda.Allocation] = []
  let offsets : Array[Int64] = []
  let byte_counts : Array[Int64] = []
  let alignments : Array[Int64] = []
  for launch in launches {
    functions.push(launch.function.raw)
    dimensions.push(launch.dimensions.grid_x)
    dimensions.push(launch.dimensions.grid_y)
    dimensions.push(launch.dimensions.grid_z)
    dimensions.push(launch.dimensions.block_x)
    dimensions.push(launch.dimensions.block_y)
    dimensions.push(launch.dimensions.block_z)
    dimensions.push(launch.dimensions.shared_memory_bytes)
    for index in 0..<launch.arguments.length {
      allocations.push(launch.arguments.allocations[index])
      offsets.push(launch.arguments.offsets[index])
      byte_counts.push(launch.arguments.byte_counts[index])
      alignments.push(launch.arguments.alignments[index])
    }
    argument_starts.push(allocations.length())
  }
  let raw_policy : @cuda.OrderedExecutorPolicy = match policy {
    OrderedKernelEagerOnly => OrderedExecutorEagerOnly
    OrderedKernelCaptureRequired => OrderedExecutorCaptureRequired
    OrderedKernelCaptureWithEagerFallback =>
      OrderedExecutorCaptureWithEagerFallback
  }
  let raw = @cuda.prepare_ordered_kernel_executor(
    self.raw,
    stream.raw,
    FixedArray::from_array(functions),
    FixedArray::from_array(dimensions),
    FixedArray::from_array(argument_starts),
    FixedArray::from_array(allocations),
    FixedArray::from_array(offsets),
    FixedArray::from_array(byte_counts),
    FixedArray::from_array(alignments),
    policy=raw_policy,
  ) catch {
    error => raise error_from_cuda(error)
  }
  let mode = raw.execution_mode() catch {
    error => {
      try raw.close() catch {
        _ => ()
      } noraise {
        _ => ()
      }
      raise error_from_cuda(error)
    }
  }
  {
    raw,
    kernel_count: launches.length(),
    mode: match mode {
      OrderedExecutorEager => OrderedKernelEager
      OrderedExecutorCaptured => OrderedKernelCaptured
    },
  }
}

///|
/// Enqueues exactly the next prevalidated record without synchronizing.
pub fn OrderedKernelExecutor::enqueue(
  self : OrderedKernelExecutor,
  index : Int,
) -> Unit raise DeviceError {
  if index < 0 || index >= self.kernel_count {
    raise InvalidArgument(OrderedExecutorEnqueue)
  }
  self.raw.enqueue(index) catch {
    error => raise error_from_cuda(error)
  }
}

///|
/// Launches the exact graph exec captured during construction. Node updates,
/// geometry changes, and operand replacement are intentionally unavailable.
pub fn OrderedKernelExecutor::launch_captured(
  self : OrderedKernelExecutor,
) -> Unit raise DeviceError {
  if self.mode != OrderedKernelCaptured {
    raise InvalidArgument(OrderedExecutorGraphLaunch)
  }
  self.raw.launch_captured() catch {
    error => raise error_from_cuda(error)
  }
}

///|
/// Records the reusable completion event after the last kernel and any
/// same-stream collectives have been enqueued.
pub fn OrderedKernelExecutor::record_completion(
  self : OrderedKernelExecutor,
) -> Unit raise DeviceError {
  self.raw.record_completion() catch {
    error => raise error_from_cuda(error)
  }
}

///|
/// Nonblocking completion query. False is pending and performs no allocation.
pub fn OrderedKernelExecutor::poll_completion(
  self : OrderedKernelExecutor,
) -> Bool raise DeviceError {
  self.raw.poll_completion() catch {
    error => raise error_from_cuda(error)
  }
}

///|
/// Blocks on the executor's reusable completion event after every launch has
/// been enqueued and recorded. This is one graph-level device barrier; it does
/// not synchronize between individual kernels.
pub fn OrderedKernelExecutor::wait_completion(
  self : OrderedKernelExecutor,
) -> Unit raise DeviceError {
  self.raw.wait_completion() catch {
    error => raise error_from_cuda(error)
  }
}

///|
/// Fault cleanup barrier. Synchronizes the exact retained stream before any
/// leases can be released; retry is safe after a driver failure.
pub fn OrderedKernelExecutor::abort(
  self : OrderedKernelExecutor,
) -> Unit raise DeviceError {
  self.raw.abort() catch {
    error => raise error_from_cuda(error)
  }
}

///|
/// Rearms a completed executor for the next exact plan frame.
pub fn OrderedKernelExecutor::reset(
  self : OrderedKernelExecutor,
) -> Unit raise DeviceError {
  self.raw.reset() catch {
    error => raise error_from_cuda(error)
  }
}

///|
/// Destroys the event and releases all long-lived leases. Active work returns
/// Busy until `poll_completion` succeeds or `abort` synchronizes the stream.
pub fn OrderedKernelExecutor::close(
  self : OrderedKernelExecutor,
) -> Unit raise DeviceError {
  self.raw.close() catch {
    error => raise error_from_cuda(error)
  }
}

///|
pub fn OrderedKernelExecutor::kernel_count(self : OrderedKernelExecutor) -> Int {
  self.kernel_count
}

///|
pub fn OrderedKernelExecutor::execution_mode(
  self : OrderedKernelExecutor,
) -> OrderedKernelExecutionMode {
  self.mode
}
