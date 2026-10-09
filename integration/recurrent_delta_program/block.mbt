///|
/// Borrowed weights for a complete recurrent branch. They outlive the prepared
/// queue; only the program's request caches and intermediate frames are owned.
pub(all) struct RecurrentBlockWeights {
  query : @device.Allocation
  key : @device.Allocation
  value : @device.Allocation
  forget_low : @device.Allocation
  forget_expansion : @device.Allocation
  beta : @device.Allocation
  dt_bias : @device.Allocation
  a_log : @device.Allocation
  gate_low : @device.Allocation
  gate_expansion : @device.Allocation
  norm : @device.Allocation
  output : @device.Allocation
  query_conv : @device.Allocation
  key_conv : @device.Allocation
  value_conv : @device.Allocation
}

///|
pub fn RecurrentDeltaProgram::with_block(
  block : @block.RecurrentBlockCudaSource,
) -> Self raise @precision.PrecisionPlanError {
  let base = RecurrentDeltaProgram::with_convolution(
    block.delta(),
    block.convolution(),
  )
  let extra = block.plan().frame_bytes()
  guard extra <= 9223372036854775807L - base.workspace_bytes() else {
    raise @precision.PrecisionPlanError::MemoryBudgetExceeded
  }
  {
    source: block.delta(),
    convolution: Some(block.convolution()),
    block: Some(block),
    maximum_rows: base.maximum_rows,
    cache_bytes: base.cache_bytes,
    owned_bytes: base.owned_bytes + extra,
    launch_count: 16,
    status: Fresh,
    cache: None,
    functions: [],
    intermediates: [],
    queue: None,
  }
}

///|
/// Prepare the whole hidden→recurrent→hidden branch once, including all
/// projections and controls. There are no CPU handoffs between its 16 launches.
pub fn RecurrentDeltaProgram::prepare_block(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  stream : @device.Stream,
  weights : RecurrentBlockWeights,
  maximum_state_bytes~ : Int64,
  maximum_workspace_bytes~ : Int64,
  counts~ : @device.Allocation,
  sequence_offsets~ : @device.Allocation,
  sequence_slots~ : @device.Allocation,
  reset_flags~ : @device.Allocation,
  hidden~ : @device.Allocation,
  output~ : @device.Allocation,
) -> Unit raise @device.DeviceError {
  guard self.block is Some(_) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.prepare(
    context,
    aot,
    stream,
    maximum_state_bytes~,
    maximum_workspace_bytes~,
    counts~,
    sequence_offsets~,
    sequence_slots~,
    reset_flags~,
    output~,
    query=hidden,
    key=hidden,
    value=hidden,
    log_decay=hidden,
    beta=hidden,
    hidden~,
    block_weights=weights,
    convolution_weights=FixedArray::from_array([
      weights.query_conv,
      weights.key_conv,
      weights.value_conv,
    ]),
  )
}

///|
fn allocate_block_frame(
  owner : RecurrentDeltaProgram,
  context : @device.Context,
  source : @block.RecurrentBlockCudaSource,
) -> FixedArray[@device.Allocation] raise @device.DeviceError {
  let plan = source.plan()
  let (rows, _, _) = plan.delta().capacity()
  let (heads, _, _) = plan.delta().geometry()
  let (_, rank) = plan.projection_geometry()
  let full = plan.convolution().frame_bytes()
  let low = rows.to_int64() * rank.to_int64() * 2L
  let beta = rows.to_int64() * heads.to_int64() * 2L
  let sizes = [
    full,
    full,
    full,
    low,
    full,
    beta,
    beta,
    full * 2L,
    low,
    full,
    full,
    full,
  ]
  FixedArray::from_array(
    sizes.map(size => {
      let allocation = context.allocate(size)
      owner.intermediates.push(allocation)
      allocation
    }),
  )
}

///|
fn append_block_launch(
  owner : RecurrentDeltaProgram,
  aot : @device.Module,
  source : @block.RecurrentBlockCudaSource,
  launches : Array[@device.OrderedKernelLaunch],
  stage : Int,
  grid_x : Int,
  grid_y : Int,
  block_x : Int,
  arguments : Array[@device.KernelArgument],
) -> Unit raise @device.DeviceError {
  let function = aot.load_function(@utf8.encode(source.symbol(stage)))
  owner.functions.push(function)
  launches.push(
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
    ),
  )
}

///|
fn append_block_projection(
  owner : RecurrentDeltaProgram,
  aot : @device.Module,
  source : @block.RecurrentBlockCudaSource,
  launches : Array[@device.OrderedKernelLaunch],
  stage : Int,
  counts : @device.Allocation,
  input : @device.Allocation,
  weight : @device.Allocation,
  output : @device.Allocation,
  input_width : Int,
  output_width : Int,
) -> Unit raise @device.DeviceError {
  let (rows, _, _) = source.plan().delta().capacity()
  let out_cells = rows.to_int64() * output_width.to_int64()
  append_block_launch(
    owner,
    aot,
    source,
    launches,
    stage,
    ((out_cells + 255L) / 256L).to_int(),
    1,
    256,
    [
      region(counts, 20L, 4L),
      region(input, rows.to_int64() * input_width.to_int64() * 2L, 2L),
      region(weight, output_width.to_int64() * input_width.to_int64() * 2L, 2L),
      region(output, out_cells * 2L, 2L),
    ],
  )
}

///|
fn append_block_prefix(
  owner : RecurrentDeltaProgram,
  aot : @device.Module,
  source : @block.RecurrentBlockCudaSource,
  launches : Array[@device.OrderedKernelLaunch],
  counts : @device.Allocation,
  hidden : @device.Allocation,
  weights : RecurrentBlockWeights,
  frame : FixedArray[@device.Allocation],
) -> Unit raise @device.DeviceError {
  let plan = source.plan()
  let (hidden_width, rank) = plan.projection_geometry()
  let (heads, _, _) = plan.delta().geometry()
  let (channels, _) = plan.convolution().geometry()
  let (rows, _, _) = plan.delta().capacity()
  for index, weight in [weights.query, weights.key, weights.value] {
    append_block_projection(
      owner,
      aot,
      source,
      launches,
      index,
      counts,
      hidden,
      weight,
      frame[index],
      hidden_width,
      channels,
    )
  }
  append_block_projection(
    owner,
    aot,
    source,
    launches,
    3,
    counts,
    hidden,
    weights.forget_low,
    frame[3],
    hidden_width,
    rank,
  )
  append_block_projection(
    owner,
    aot,
    source,
    launches,
    4,
    counts,
    frame[3],
    weights.forget_expansion,
    frame[4],
    rank,
    channels,
  )
  append_block_projection(
    owner,
    aot,
    source,
    launches,
    5,
    counts,
    hidden,
    weights.beta,
    frame[5],
    hidden_width,
    heads,
  )
  let beta_bytes = rows.to_int64() * heads.to_int64() * 2L
  append_block_launch(
    owner,
    aot,
    source,
    launches,
    6,
    ((beta_bytes / 2L + 255L) / 256L).to_int(),
    1,
    256,
    [
      region(counts, 20L, 4L),
      region(frame[5], beta_bytes, 2L),
      region(frame[6], beta_bytes, 2L),
    ],
  )
  let full = plan.convolution().frame_bytes()
  append_block_launch(
    owner,
    aot,
    source,
    launches,
    7,
    ((full / 2L + 255L) / 256L).to_int(),
    1,
    256,
    [
      region(counts, 20L, 4L),
      region(frame[4], full, 2L),
      region(weights.dt_bias, channels.to_int64() * 4L, 4L),
      region(weights.a_log, heads.to_int64() * 4L, 4L),
      region(frame[7], full * 2L, 4L),
    ],
  )
  append_block_projection(
    owner,
    aot,
    source,
    launches,
    8,
    counts,
    hidden,
    weights.gate_low,
    frame[8],
    hidden_width,
    rank,
  )
  append_block_projection(
    owner,
    aot,
    source,
    launches,
    9,
    counts,
    frame[8],
    weights.gate_expansion,
    frame[9],
    rank,
    channels,
  )
}

///|
fn append_block_suffix(
  owner : RecurrentDeltaProgram,
  aot : @device.Module,
  source : @block.RecurrentBlockCudaSource,
  launches : Array[@device.OrderedKernelLaunch],
  counts : @device.Allocation,
  weights : RecurrentBlockWeights,
  frame : FixedArray[@device.Allocation],
  output : @device.Allocation,
) -> Unit raise @device.DeviceError {
  let plan = source.plan()
  let (hidden, _) = plan.projection_geometry()
  let (heads, dimension, _) = plan.delta().geometry()
  let (channels, _) = plan.convolution().geometry()
  let (rows, _, _) = plan.delta().capacity()
  let full = plan.convolution().frame_bytes()
  append_block_launch(owner, aot, source, launches, 14, rows, heads, 128, [
    region(counts, 20L, 4L),
    region(frame[10], full, 2L),
    region(frame[9], full, 2L),
    region(weights.norm, dimension.to_int64() * 2L, 2L),
    region(frame[11], full, 2L),
  ])
  append_block_projection(
    owner,
    aot,
    source,
    launches,
    15,
    counts,
    frame[11],
    weights.output,
    output,
    channels,
    hidden,
  )
}
