///|
priv struct ExpertBuffers {
  map : @device.Allocation
  activation : @device.Allocation
  weighted : @device.Allocation
  output : @device.Allocation
  quantized_input : @device.Allocation?
  quantized_activation : @device.Allocation?
  quantization_error : @device.Allocation?
}

///|
fn MoeProgram::expert_buffers(
  self : Self,
  context : @device.Context,
  source : @source.PackedExpertMlpCudaSource,
) -> ExpertBuffers raise @device.DeviceError {
  let plan = source.plan()
  let sizes = expert_workspace_sizes(source)
  let map = self.allocate(context, sizes[0])
  let activation = self.allocate(context, sizes[1])
  let weighted = self.allocate(context, sizes[2])
  let output = self.allocate(context, sizes[3])
  let quantized_input = if sizes[4] > 0L {
    Some(self.allocate(context, sizes[4]))
  } else {
    None
  }
  let quantized_activation = if sizes[5] > 0L {
    Some(self.allocate(context, sizes[5]))
  } else {
    None
  }
  let quantization_error = if sizes[6] > 0L {
    let error = self.allocate(context, sizes[6])
    context.copy_from_fixed_host(error, [0, 0, 0, 0], 0L, 0L, 4L)
    Some(error)
  } else {
    None
  }
  let entries = plan.expert_map()
  let bytes = FixedArray::make(entries.length() * 4, b'\x00')
  for i, entry in entries {
    for b in 0..<4 {
      bytes[i * 4 + b] = ((entry >> (b * 8)) & 255).to_byte()
    }
  }
  context.copy_from_fixed_host(map, bytes, 0L, 0L, sizes[0])
  {
    map,
    activation,
    weighted,
    output,
    quantized_input,
    quantized_activation,
    quantization_error,
  }
}

///|
fn bind_program_expert(
  source : @source.PackedExpertMlpCudaSource,
  functions : ArrayView[@device.Function],
  buffers : ExpertBuffers,
  counts : @device.Allocation,
  input : @device.Allocation,
  indices : @device.Allocation,
  scores : @device.Allocation,
  bank : @device.Allocation,
) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError {
  bind(
    source,
    functions,
    counts~,
    input~,
    selected_indices=indices,
    selected_weights=scores,
    expert_map=buffers.map,
    bank~,
    activation=buffers.activation,
    weighted=buffers.weighted,
    output=buffers.output,
    quantized_input?=buffers.quantized_input,
    quantized_activation?=buffers.quantized_activation,
    quantization_error?=buffers.quantization_error,
  )
}

///|
/// Prepare once from the exact offline-compiled source. All step storage is
/// reusable. Retain this owner on errors and call close to release partial work.
pub fn MoeProgram::prepare(
  self : Self,
  context : @device.Context,
  aot_module : @device.Module,
  stream : @device.Stream,
  maximum_workspace_bytes~ : Int64,
  counts~ : @device.Allocation,
  input~ : @device.Allocation,
  selected_indices~ : @device.Allocation,
  selected_weights~ : @device.Allocation,
  routed_bank~ : @device.Allocation,
  shared_bank~ : @device.Allocation,
  output~ : @device.Allocation,
  reduction? : MoeReduction,
) -> Unit raise @device.DeviceError {
  guard self.routing is None else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.check_fresh(maximum_workspace_bytes)
  self.prepare_experts(
    context,
    aot_module,
    stream,
    counts~,
    input~,
    selected_indices~,
    selected_weights~,
    routed_bank~,
    shared_bank~,
    output~,
    reduction?,
    prefix=[],
  )
}

///|
fn MoeProgram::check_fresh(
  self : Self,
  maximum_workspace_bytes : Int64,
) -> Unit raise @device.DeviceError {
  guard !self.closed &&
    self.executor is None &&
    self.functions.is_empty() &&
    self.allocations.is_empty() &&
    self.workspace_bytes <= maximum_workspace_bytes else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
}

///|
fn MoeProgram::prepare_experts(
  self : Self,
  context : @device.Context,
  aot_module : @device.Module,
  stream : @device.Stream,
  counts~ : @device.Allocation,
  input~ : @device.Allocation,
  selected_indices~ : @device.Allocation,
  selected_weights~ : @device.Allocation,
  routed_bank~ : @device.Allocation,
  shared_bank~ : @device.Allocation,
  output~ : @device.Allocation,
  reduction? : MoeReduction,
  prefix~ : ArrayView[@device.OrderedKernelLaunch],
) -> Unit raise @device.DeviceError {
  guard self.shared.plan().down_publication() == LocalBFloat16 else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let partial = self.routed.plan().down_publication() ==
    ReduceFloat32BeforeRound
  guard !partial || reduction is Some(_) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  if reduction is Some(site) {
    let (rows, hidden, _, _, selected) = self.routed.plan().geometry()
    let width = hidden.to_int64() *
      (if partial { selected.to_int64() } else { 1L })
    guard width <= 2147483647L else {
      raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
    }
    site.validate_shape(rows, width.to_int())
  }
  let (routed, shared, finalize, routed_output) = self.bind_experts(
    context,
    aot_module,
    counts~,
    input~,
    selected_indices~,
    selected_weights~,
    routed_bank~,
    shared_bank~,
    output~,
    prefix~,
  )
  self.executor = Some(
    prepare_moe(
      context,
      stream,
      routed,
      shared,
      finalize,
      routed_output~,
      maximum_rows=self.routed.plan().geometry().0,
      reduction?,
      post_reduction_routed_kernels=if partial { 1 } else { 0 },
    ),
  )
}

///|
fn MoeProgram::bind_experts(
  self : Self,
  context : @device.Context,
  aot_module : @device.Module,
  counts~ : @device.Allocation,
  input~ : @device.Allocation,
  selected_indices~ : @device.Allocation,
  selected_weights~ : @device.Allocation,
  routed_bank~ : @device.Allocation,
  shared_bank~ : @device.Allocation,
  output~ : @device.Allocation,
  prefix~ : ArrayView[@device.OrderedKernelLaunch],
) -> (
  Array[@device.OrderedKernelLaunch],
  Array[@device.OrderedKernelLaunch],
  @device.OrderedKernelLaunch,
  @device.Allocation,
) raise @device.DeviceError {
  let (rows, _, _, _, _) = self.routed.plan().geometry()
  let routed_functions = self.load_functions(aot_module, self.routed.symbols())
  let shared_functions = self.load_functions(aot_module, self.shared.symbols())
  let finalize_functions = self.load_functions(aot_module, [
    self.combine.symbol(),
  ])
  let routed = self.expert_buffers(context, self.routed)
  let shared = self.expert_buffers(context, self.shared)
  let shared_indices = self.allocate(context, rows.to_int64() * 4L)
  let shared_weights = self.allocate(context, rows.to_int64() * 4L)
  initialize_shared_route(
    context,
    self.shared.plan(),
    indices=shared_indices,
    weights=shared_weights,
    expert_map=shared.map,
  )
  let routed_launches = prefix.to_owned()
  routed_launches.append(
    bind_program_expert(
      self.routed,
      routed_functions,
      routed,
      counts,
      input,
      selected_indices,
      selected_weights,
      routed_bank,
    ),
  )
  let shared_launches = bind_program_expert(
    self.shared,
    shared_functions,
    shared,
    counts,
    input,
    shared_indices,
    shared_weights,
    shared_bank,
  )
  let finalize = bind_combine(
    self.combine,
    finalize_functions[0],
    counts~,
    routed=routed.output,
    shared=shared.output,
    output~,
  )
  if self.collective_region is Some(region_plan) {
    let joined = self.allocate(context, region_plan.storage_bytes())
    let functions = self.load_functions(aot_module, [
      self.combine.symbol() + "_collective_pack",
      self.combine.symbol() + "_collective_unpack",
    ])
    let arguments = @device.KernelArguments::new([
      region(counts, 20L, 4L),
      region(routed.weighted, self.routed.plan().workspace_bytes().1, 4L),
      region(shared.weighted, self.shared.plan().workspace_bytes().1, 4L),
      region(joined, region_plan.storage_bytes(), 4L),
    ])
    let dimensions = @device.KernelLaunchDimensions::new(
      grid_x=((region_plan.storage_bytes() / 4L + 255L) / 256L).to_int(),
      grid_y=1,
      grid_z=1,
      block_x=256,
      block_y=1,
      block_z=1,
      shared_memory_bytes=0,
    )
    // Neither consumer runs until both partial producers and the single
    // concatenated reduction complete. No sum crosses the region boundary.
    let routed_publish = routed_launches.pop().unwrap()
    let shared_publish = shared_launches.pop().unwrap()
    routed_launches.append(shared_launches)
    routed_launches.push(
      @device.OrderedKernelLaunch::new(functions[0], dimensions, arguments),
    )
    let consumers = [
      @device.OrderedKernelLaunch::new(functions[1], dimensions, arguments),
      routed_publish,
      shared_publish,
    ]
    return (routed_launches, consumers, finalize, joined)
  }
  if self.shared.plan().down_publication() == ReduceFloat32BeforeRound {
    let (_, hidden, _, _, _) = self.shared.plan().geometry()
    self.shared_tensor_boundary = Some({
      before_kernel: routed_launches.length() + shared_launches.length() - 1,
      partials: shared.weighted,
      width: hidden,
    })
  }
  (
    routed_launches,
    shared_launches,
    finalize,
    if self.routed.plan().down_publication() == ReduceFloat32BeforeRound {
      routed.weighted
    } else {
      routed.output
    },
  )
}
