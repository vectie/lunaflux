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
  plan : @precision.ExpertMlpPrecision,
) -> ExpertBuffers raise @device.DeviceError {
  let sizes = expert_workspace_sizes(plan)
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
  guard !self.closed &&
    self.executor is None &&
    self.functions.is_empty() &&
    self.allocations.is_empty() &&
    self.workspace_bytes <= maximum_workspace_bytes else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let (rows, _, _, _, _) = self.routed.plan().geometry()
  let routed_functions = self.load_functions(aot_module, self.routed.symbols())
  let shared_functions = self.load_functions(aot_module, self.shared.symbols())
  let finalize_functions = self.load_functions(aot_module, [
    self.combine.symbol(),
  ])
  let routed = self.expert_buffers(context, self.routed.plan())
  let shared = self.expert_buffers(context, self.shared.plan())
  let shared_indices = self.allocate(context, rows.to_int64() * 4L)
  let shared_weights = self.allocate(context, rows.to_int64() * 4L)
  initialize_shared_route(
    context,
    self.shared.plan(),
    indices=shared_indices,
    weights=shared_weights,
    expert_map=shared.map,
  )
  let routed_launches = bind_program_expert(
    self.routed,
    routed_functions,
    routed,
    counts,
    input,
    selected_indices,
    selected_weights,
    routed_bank,
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
  self.executor = Some(
    prepare_moe(
      context,
      stream,
      routed_launches,
      shared_launches,
      finalize,
      routed_output=routed.output,
      maximum_rows=rows,
      reduction?,
    ),
  )
}
