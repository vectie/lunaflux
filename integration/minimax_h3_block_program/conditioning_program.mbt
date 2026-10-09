///|
/// Owns invariant packed text/reference conditioning for the denoising loop.
/// Reference inputs must already contain model-projected BF16 hidden rows.
pub struct ConditioningProgram {
  priv source : @minimax_h3_block_source.ConditioningSource
  priv functions : Array[@device.Function]
  priv mut output : @device.Allocation?
  priv mut queue : @device.OrderedKernelExecutor?
  priv mut phase : ConditioningPhase
}

///|
pub fn ConditioningProgram::new(
  source : @minimax_h3_block_source.ConditioningSource,
) -> Self {
  { source, functions: [], output: None, queue: None, phase: Empty, }
}

///|
/// References follow nonempty sequence segments (audio precedes video within
/// a video reference). Each tuple is allocation plus byte offset. Text must
/// be prepared; execution must follow its successful refinement. All copies
/// target a fresh, nonaliasing allocation, so the whole pipeline can bind first.
/// With latent_weights, references are packed F32 latent rows instead of BF16
/// hidden rows; shared input projections write directly to the final allocation.
pub fn ConditioningProgram::prepare(
  self : Self,
  context : @device.Context,
  stream : @device.Stream,
  aot_module : @device.Module,
  text : TextConditioningProgram,
  references : ArrayView[(@device.Allocation, Int64)],
  max_output_bytes~ : Int64,
  latent_weights? : @device.Allocation,
) -> Unit raise @device.DeviceError {
  guard self.phase == Empty &&
    text.phase is (Prepared | Complete) &&
    text.output is Some(text_output) &&
    text.source.model == self.source.model &&
    self.source.copies.length() == references.length() + 1 &&
    self.source.copies[0].rows == text.source.rows &&
    self.source.byte_count <= max_output_bytes else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.phase = Failed
  let latent_packing = match latent_weights {
    Some(weights) =>
      Some(
        @minimax_h3_weights.input_weight_packing(
          self.source.model,
          max_bytes=weights.byte_count(),
        ) catch {
          _ => raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
        },
      )
    None => None
  }
  acquire_functions(self.source.symbols[:], self.functions, name => {
    aot_module.load_function(@utf8.encode(name))
  })
  let output = context.allocate(self.source.byte_count)
  self.output = Some(output)
  let inputs = [(text_output, 0L)]
  inputs.append(references)
  let launches : Array[@device.OrderedKernelLaunch] = []
  for i, copy in self.source.copies {
    let (input, offset) = inputs[i]
    let elements = copy.byte_count / 2L
    if i > 0 &&
      latent_weights is Some(weights) &&
      latent_packing is Some(packing) {
      launches.push(
        reference_projection_launch(
          self.functions[self.source.copies.length() + i - 1],
          copy,
          hidden=self.source.model.denoiser().hidden_size(),
          input~,
          input_offset=offset,
          weights~,
          packing~,
          output~,
        ),
      )
      continue
    }
    launches.push(
      @device.OrderedKernelLaunch::new(
        self.functions[i],
        @device.KernelLaunchDimensions::new(
          grid_x=((elements + 255L) / 256L).to_int(),
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
            offset~,
            byte_count=copy.byte_count,
            alignment=2L,
          ),
          @device.KernelArgument::device_region(
            output,
            offset=copy.destination_offset,
            byte_count=copy.byte_count,
            alignment=2L,
          ),
        ]),
      ),
    )
  }
  self.queue = Some(
    context.create_ordered_kernel_executor(
      stream,
      launches,
      policy=OrderedKernelEagerOnly,
    ),
  )
  self.phase = Prepared
}

///|
pub fn ConditioningProgram::execute(
  self : Self,
) -> Unit raise @device.DeviceError {
  guard self.phase == Prepared && self.queue is Some(queue) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
  }
  self.phase = Failed
  self.phase = execute_conditioning_once(Prepared, () => {
    for i in 0..<queue.kernel_count() {
      queue.enqueue(i)
    }
    queue.record_completion()
    queue.wait_completion()
  })
}

///|
pub fn ConditioningProgram::bind_input(
  self : Self,
  input : InputProgram,
  packing : @minimax_h3_weights.InputWeightPacking,
  weights~ : @device.Allocation,
) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard self.phase == Complete &&
    self.output is Some(output) &&
    self.source.model == input.source.model &&
    self.source.sequence.segments() == input.source.sequence.segments() &&
    self.source.sequence.assembly().mapping() ==
    input.source.sequence.assembly().mapping() else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  input.bind(
    packing,
    weights~,
    conditioning=Some(output),
    conditioning_offset=0L,
  )
}

///|
/// Release downstream queues first. Partial failures retain owners for retry.
pub fn ConditioningProgram::close(
  self : Self,
) -> Unit raise @device.DeviceError {
  self.phase = Failed
  if self.queue is Some(queue) {
    queue.abort()
    queue.close()
    self.queue = None
  }
  release_functions(self.functions, function => function.close())
  if self.output is Some(output) {
    output.close()
    self.output = None
  }
  self.phase = Empty
}
