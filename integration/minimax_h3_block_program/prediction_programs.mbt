///|
priv struct OwnedPrediction {
  block : Program
  timestep : TimestepProgram
  output : OutputProgram
}

///|
priv struct PredictionWorkspace {
  block : Int64
  timestep : Int64
  output : Int64
}

///|
// Pure preflight uses the exact allocators' layouts, not a per-program ceiling
// multiplied implicitly after allocation. No context is touched on failure.
fn prediction_workspace(
  source : @minimax_h3_block_source.PredictionSources,
  maximum : Int64,
) -> PredictionWorkspace raise @device.DeviceError {
  let b = source.block
  let t = source.timestep
  let o = source.output
  guard b.distinct_timesteps == t.rows &&
    b.distinct_timesteps == o.distinct_timesteps &&
    b.model == o.model &&
    b.rows == o.packed_rows &&
    b.packed_mapping == o.packed_mapping &&
    t.input_width == b.shape.timestep_frequency_dimension() &&
    t.hidden_width == b.shape.timestep_embedding_hidden_dimension() &&
    t.output_width == b.shape.final_adaln_conditioning_dimension() else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let block = @joint_diffusion_device.BlockWorkspaceLayout::new(
      rows=b.rows,
      hidden_width=b.shape.hidden_size(),
      head_count=b.shape.attention_head_count(),
      head_dimension=b.shape.attention_head_dimension(),
      intermediate_width=b.shape.feed_forward_dimension(),
      max_bytes=maximum,
    ).byte_count
  let timestep = timestep_workspace(
      t.rows,
      t.input_width,
      t.hidden_width,
      t.output_width,
      maximum,
    ).bytes
  let output = output_layout(
      o.packed_rows,
      o.model.denoiser().hidden_size(),
      o.distinct_timesteps,
      o.video_rows,
      o.audio_rows,
      maximum,
    ).bytes
  { block, timestep, output, }
}

///|
fn prediction_workspaces(
  sources : ArrayView[@minimax_h3_block_source.PredictionSources],
  maximum : Int64,
) -> (Array[PredictionWorkspace], Int64) raise @device.DeviceError {
  guard maximum > 0L && !sources.is_empty() else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  let layouts : Array[PredictionWorkspace] = []
  let counts : Map[Int, Unit] = Map([])
  let mut remaining = maximum
  let first = sources[0].block
  for source in sources {
    let b = source.block
    guard !counts.contains(b.distinct_timesteps) &&
      b.model == first.model &&
      b.rows == first.rows &&
      b.packed_mapping == first.packed_mapping else {
      raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
    }
    counts[b.distinct_timesteps] = ()
    let layout = prediction_workspace(source, maximum)
    for bytes in [layout.block, layout.timestep, layout.output] {
      guard bytes > 0L && bytes <= remaining else {
        raise @device.DeviceError::InvalidArgument(AllocationCreate)
      }
      remaining -= bytes
    }
    layouts.push(layout)
  }
  (layouts, maximum - remaining)
}

///|
/// Owns all schedule-derived AOT prediction programs. Modules/weights are
/// borrowed; scratch is allocated once, and execution performs no selection,
/// source generation or weight upload. Close consuming queues before this owner.
pub struct PredictionPrograms {
  priv sources : ReadOnlyArray[@minimax_h3_block_source.PredictionSources]
  priv layouts : ReadOnlyArray[PredictionWorkspace]
  priv bytes : Int64
  priv programs : Array[OwnedPrediction]
  priv mut started : Bool
  priv mut ready : Bool
}

///|
/// Preflight the cumulative scratch of every variant before any GPU allocation.
/// Input, rotary tables, metadata, weights and AOT modules have separate budgets.
pub fn PredictionPrograms::new(
  sources : ArrayView[@minimax_h3_block_source.PredictionSources],
  max_workspace_bytes~ : Int64,
) -> Self raise @device.DeviceError {
  let (layouts, bytes) = prediction_workspaces(sources, max_workspace_bytes)
  {
    sources: ReadOnlyArray::from_array(sources.to_owned()),
    layouts: ReadOnlyArray::from_array(layouts),
    bytes,
    programs: [],
    started: false,
    ready: false,
  }
}

///|
pub fn PredictionPrograms::workspace_bytes(self : Self) -> Int64 {
  self.bytes
}

///|
/// Resolve each triple of already loaded modules in source order. Retain this
/// owner on a partial failure; close releases every successfully loaded function
/// and allocation. There is no runtime compilation or artifact scan.
pub fn PredictionPrograms::prepare(
  self : Self,
  context : @device.Context,
  modules : ArrayView[(@device.Module, @device.Module, @device.Module)],
) -> Unit raise @device.DeviceError {
  guard !self.started && modules.length() == self.sources.length() else {
    raise @device.DeviceError::InvalidArgument(FunctionLoad)
  }
  self.started = true
  for i, source in self.sources {
    let owned : OwnedPrediction = {
      block: Program::new(source.block),
      timestep: TimestepProgram::new(source.timestep),
      output: OutputProgram::new(source.output),
    }
    self.programs.push(owned)
    let budget = self.layouts[i]
    let module = modules[i]
    owned.block.prepare(context, module.0, max_workspace_bytes=budget.block)
    owned.timestep.prepare(
      context,
      module.1,
      max_workspace_bytes=budget.timestep,
    )
    owned.output.prepare(context, module.2, max_workspace_bytes=budget.output)
  }
  self.ready = true
}

///|
pub fn PredictionPrograms::variants(
  self : Self,
) -> ReadOnlyArray[PredictionVariant] raise @device.DeviceError {
  guard self.ready else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  ReadOnlyArray::from_array(
    self.programs.map(p => PredictionVariant::new(p.block, p.timestep, p.output)),
  )
}

///|
/// Retryable reverse cleanup. Do not close while prepared queues borrow these
/// functions and buffers. A failing close retains its program for retry.
pub fn PredictionPrograms::close(self : Self) -> Unit raise @device.DeviceError {
  self.ready = false
  self.started = true
  while !self.programs.is_empty() {
    let i = self.programs.length() - 1
    let p = self.programs[i]
    p.output.close()
    p.timestep.close()
    p.block.close()
    self.programs.remove(i) |> ignore
  }
}
