///|
fn ordered_launch(
  function : @device.Function,
  launch : (Int32, Int32, Int32),
  arguments : ArrayView[@device.KernelArgument],
) -> @device.OrderedKernelLaunch raise @device.DeviceError {
  let (grid_x, grid_y, block_x) = launch
  @device.OrderedKernelLaunch::new(
    function,
    @device.KernelLaunchDimensions::new(
      grid_x~,
      grid_y~,
      grid_z=1,
      block_x~,
      block_y=1,
      block_z=1,
      shared_memory_bytes=0,
    ),
    @device.KernelArguments::new(arguments),
  )
}

///|
/// Prepare one complete stack once. Modules and borrowed resources must match
/// this program. Caller owns valid counts/CSR, exclusive slots and writable pages.
/// On any partial failure close the frame before releasing those resources.
pub fn Frame::prepare(
  self : Self,
  context : @device.Context,
  modules~ : Modules,
  metadata~ : Metadata,
  layers~ : ArrayView[LayerWeights],
  states~ : StateArenas,
  buffers~ : Buffers,
) -> Unit raise @device.DeviceError {
  guard self.status is Fresh && layers.length() == self.program.layers.length() else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let budget = self.program.budget
  let mutable_buffers = [
    buffers.hidden,
    buffers.normalized,
    buffers.branch_output,
    buffers.branch_workspace,
    buffers.moe_workspace,
    states.convolution,
    states.recurrent,
    states.key_cache,
    states.value_cache,
  ]
  for i in 0..<mutable_buffers.length() {
    for j in 0..<i {
      guard !mutable_buffers[i].same_storage(mutable_buffers[j]) else {
        raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
      }
    }
  }
  for
    (buffer, bytes, alignment) in [
      (buffers.hidden, budget.activation_bytes, 2L),
      (buffers.normalized, budget.activation_bytes, 2L),
      (buffers.branch_output, budget.activation_bytes, 2L),
      (buffers.branch_workspace, budget.branch_workspace_bytes, 16L),
      (buffers.moe_workspace, budget.moe_workspace_bytes, 16L),
      (states.convolution, budget.convolution_state_bytes, 2L),
      (states.recurrent, budget.recurrent_state_bytes, 4L),
      (states.key_cache, budget.key_cache_bytes, 2L),
      (states.value_cache, budget.value_cache_bytes, 2L),
    ] {
    context.validate_allocation_region(
      buffer,
      offset=0L,
      byte_count=bytes,
      alignment~,
    )
  }
  for index, layer in self.program.layers {
    guard (match (layer.kind, layers[index].branch) {
      (GatedDeltaLinearAttention, Recurrent(_))
      | (FullAttention, FullAttention(_)) => true
      _ => false
    }) else {
      raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
    }
  }
  self.status = Failed
  for symbol in [self.program.norm_symbol, self.program.residual_symbol] {
    self.functions.push(modules.envelope.load_function(@utf8.encode(symbol)))
  }
  let counts = @device.KernelArgument::device_region(
    metadata.attention.counts,
    offset=0L,
    byte_count=20L,
    alignment=4L,
  )
  let hidden = @device.KernelArgument::device_region(
    buffers.hidden,
    offset=0L,
    byte_count=budget.activation_bytes,
    alignment=2L,
  )
  let normalized = @device.KernelArgument::device_region(
    buffers.normalized,
    offset=0L,
    byte_count=budget.activation_bytes,
    alignment=2L,
  )
  let branch_output = @device.KernelArgument::device_region(
    buffers.branch_output,
    offset=0L,
    byte_count=budget.activation_bytes,
    alignment=2L,
  )
  let residual = ordered_launch(
    self.functions[1],
    self.program.residual_launch,
    [counts, hidden, branch_output, hidden],
  )
  let recurrent_metadata : @recurrent.Metadata = {
    counts: metadata.attention.counts,
    row_offsets: metadata.attention.row_offsets,
    request_slots: metadata.request_slots,
    reset: metadata.reset,
  }
  let norm_weight_bytes = self.program.budget.activation_bytes /
    self.program.norm_launch.0.to_int64()
  for index, layer in self.program.layers {
    let weights = layers[index]
    let branch = match weights.branch {
      Recurrent(weights) => {
        let frame = @recurrent.Frame::new(self.program.recurrent)
        self.branches.push(Recurrent(frame))
        frame.prepare(
          context,
          modules.recurrent,
          metadata=recurrent_metadata,
          weights~,
          states=@recurrent.States::new(
            states.convolution,
            states.recurrent,
            convolution_offset=layer.convolution_offset.unwrap(),
            recurrent_offset=layer.state_offset,
          ),
          input=buffers.normalized,
          output=buffers.branch_output,
          workspace=buffers.branch_workspace,
        )
        OwnedBranch::Recurrent(frame)
      }
      FullAttention(weights) => {
        let frame = @attention.Frame::new(self.program.attention)
        self.branches.push(FullAttention(frame))
        frame.prepare(
          context,
          modules=modules.attention,
          metadata=metadata.attention,
          weights~,
          caches=@attention.Caches::new(
            states.key_cache,
            states.value_cache,
            key_offset=layer.state_offset,
            value_offset=layer.state_offset,
          ),
          input=buffers.normalized,
          output=buffers.branch_output,
          workspace=buffers.branch_workspace,
        )
        OwnedBranch::FullAttention(frame)
      }
    }
    let expert = @moe.Frame::new(self.program.moe)
    self.experts.push(expert)
    expert.prepare(
      context,
      modules.moe,
      counts=metadata.attention.counts,
      input=buffers.normalized,
      output=buffers.branch_output,
      weights=weights.moe,
      workspace=buffers.moe_workspace,
    )
    let input_norm = ordered_launch(
      self.functions[0],
      self.program.norm_launch,
      [
        counts,
        hidden,
        @device.KernelArgument::device_region(
          weights.input_norm,
          offset=0L,
          byte_count=norm_weight_bytes,
          alignment=2L,
        ),
        normalized,
      ],
    )
    let post_norm = ordered_launch(
      self.functions[0],
      self.program.norm_launch,
      [
        counts,
        hidden,
        @device.KernelArgument::device_region(
          weights.post_attention_norm,
          offset=0L,
          byte_count=norm_weight_bytes,
          alignment=2L,
        ),
        normalized,
      ],
    )
    for phase in [Phase::Prefill, Decode] {
      let queue = match phase {
        Prefill => self.prefill
        Decode => self.decode
      }
      for operation in self.program.operations {
        match operation {
          NormalizeInput => queue.push(input_norm)
          Branch => {
            let launches = match branch {
              Recurrent(frame) => frame.launches()
              FullAttention(frame) => frame.launches(phase)
            }
            for launch in launches {
              queue.push(launch)
            }
          }
          AddBranchResidual | AddMoeResidual => queue.push(residual)
          NormalizeMoe => queue.push(post_norm)
          Moe =>
            for launch in expert.launches() {
              queue.push(launch)
            }
        }
      }
    }
  }
  self.status = Prepared
}
