///|
/// Reusable device-only layout conversion. Owns a fresh output, borrows the
/// input and admitted permutation function through executor leases.
pub struct PermutationBuffer {
  priv output : @device.Allocation
  priv mut queue : @device.OrderedKernelExecutor?
  priv mut complete : Bool
  priv mut started : Bool
}

///|
pub fn PermutationBuffer::allocate(
  context : @device.Context,
  plan : @dense_permutation_plan.Permutation,
  max_bytes~ : Int64,
) -> Self raise @device.DeviceError {
  guard plan.element_count() <= max_bytes / 4L && max_bytes > 0L else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  {
    output: context.allocate(plan.element_count() * 4L),
    queue: None,
    complete: false,
    started: false,
  }
}

///|
/// Bind dense_permutation_cuda_source.render(plan, block_size=256).
pub fn PermutationBuffer::prepare(
  self : Self,
  context : @device.Context,
  stream : @device.Stream,
  function : @device.Function,
  input : @device.Allocation,
  input_offset~ : Int64,
) -> Unit raise @device.DeviceError {
  guard self.queue is None && !self.started else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let bytes = self.output.byte_count()
  let blocks = (bytes / 4L + 255L) / 256L
  guard blocks <= 2147483647L else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let launch = @device.OrderedKernelLaunch::new(
    function,
    @device.KernelLaunchDimensions::new(
      grid_x=blocks.to_int(),
      grid_y=1,
      grid_z=1,
      block_x=256,
      block_y=1,
      block_z=1,
      shared_memory_bytes=0,
    ),
    @device.KernelArguments::new([
      @device.KernelArgument::device_region(
        input,
        offset=input_offset,
        byte_count=bytes,
        alignment=4L,
      ),
      @device.KernelArgument::device_region(
        self.output,
        offset=0L,
        byte_count=bytes,
        alignment=4L,
      ),
    ]),
  )
  self.queue = Some(
    context.create_ordered_kernel_executor(
      stream,
      [launch],
      policy=OrderedKernelEagerOnly,
    ),
  )
}

///|
pub fn PermutationBuffer::execute(
  self : Self,
) -> Unit raise @device.DeviceError {
  guard !self.started && self.queue is Some(queue) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
  }
  self.started = true
  queue.enqueue(0)
  queue.record_completion()
  queue.wait_completion()
  self.complete = true
}

///|
/// Borrow only after completion; consumers must release queue leases before
/// this owner closes. The callback must not close or mutate the allocation.
pub fn[T] PermutationBuffer::with_output(
  self : Self,
  bind : (@device.Allocation, Int64) -> T raise,
) -> T raise {
  guard self.complete else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  bind(self.output, 0L)
}

///|
pub fn PermutationBuffer::close(self : Self) -> Unit raise @device.DeviceError {
  self.complete = false
  self.started = true
  if self.queue is Some(queue) {
    queue.abort()
    queue.close()
    self.queue = None
  }
  self.output.close()
}
