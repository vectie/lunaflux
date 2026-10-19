///|
/// Startup owner of a complete routed/shared AOT MoE program. Weight banks,
/// input, output, routing, stream and live counts remain caller-owned. The
/// owner allocates only compact intermediate state, never a decoded weight bank.
pub struct MoeProgram {
  priv routed : @source.PackedExpertMlpCudaSource
  priv shared : @source.PackedExpertMlpCudaSource
  priv combine : @combine.MoeCombineSource
  priv routing : @router.MoeRoutingCudaSource?
  priv workspace_bytes : Int64
  priv functions : Array[@device.Function]
  priv allocations : Array[@device.Allocation]
  priv startup_streams : Array[@device.Stream]
  priv launches : Array[@device.OrderedKernelLaunch]
  priv frontiers : Array[(String, @device.Allocation)]
  priv mut executor : MoeExecutor?
  priv mut closed : Bool
  priv mut shared_tensor_boundary : ExpertTensorBoundary?
  priv collective_region : @parallel.CoalescedSumPlan?
  priv reduction_placement : @precision.MoeReductionPlacement
  priv mut producer_fork : @parallel.ForkJoinSpan?
}

///|
fn expert_workspace_sizes(
  source : @source.PackedExpertMlpCudaSource,
) -> Array[Int64] {
  let plan = source.plan()
  let (rows, hidden, _, experts, _) = plan.geometry()
  let weighted = plan.workspace_bytes().1
  let (qx, qa) = source.quantized_workspace_bytes()
  [
    experts.to_int64() * 4L,
    source.activation_bytes(),
    weighted,
    rows.to_int64() * hidden.to_int64() * 4L,
    qx,
    qa,
    if plan.activation_quantization() is Some(_) {
      4L
    } else {
      0L
    },
  ]
}

///|
pub fn MoeProgram::new(
  routed : @source.PackedExpertMlpCudaSource,
  shared : @source.PackedExpertMlpCudaSource,
  combine : @combine.MoeCombineSource,
  routing? : @router.MoeRoutingCudaSource,
  coalesce_reductions~ : Bool = false,
  reduction_placement~ : @precision.MoeReductionPlacement = ExpertPartialsFloat32,
) -> Self raise @device.DeviceError {
  let (rows, hidden, _, experts, selected) = routed.plan().geometry()
  let (sr, sh, _, se, ss) = shared.plan().geometry()
  if reduction_placement == LocalCombinedBFloat16 {
    guard !coalesce_reductions &&
      routed.plan().down_publication() == ReduceFloat32BeforeRound &&
      shared.plan().down_publication() == ReduceFloat32BeforeRound else {
      raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
    }
  }
  guard rows == sr &&
    hidden == sh &&
    se == 1 &&
    ss == 1 &&
    (
      shared.plan().down_publication() == LocalBFloat16 ||
      routed.plan().down_publication() == ReduceFloat32BeforeRound
    ) &&
    shared.plan().local_experts()[:] == [0] &&
    combine.plan().geometry() == (rows, hidden) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let symbols : Set[String] = Set([])
  let entries = [routed.symbols(), shared.symbols()]
  if routing is Some(router) {
    guard router.plan().geometry() == (rows, hidden, experts, selected) else {
      raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
    }
    entries.push(router.symbols())
  }
  for names in entries {
    for name in names {
      guard !symbols.contains(name) else {
        raise @device.DeviceError::InvalidArgument(FunctionLoad)
      }
      symbols.add(name)
    }
  }
  guard !symbols.contains(combine.symbol()) else {
    raise @device.DeviceError::InvalidArgument(FunctionLoad)
  }
  let sizes = expert_workspace_sizes(routed)
  for source in [routed, shared] {
    if source.bank_preparation() is Some(_) {
      sizes.push(source.plan().expert_stride())
    }
  }
  let collective_region = if coalesce_reductions {
    guard shared.plan().down_publication() == ReduceFloat32BeforeRound &&
      routed.plan().down_publication() == ReduceFloat32BeforeRound else {
      raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
    }
    let region = @parallel.CoalescedSumPlan::new(
      maximum_rows=rows,
      left_width=hidden * selected,
      right_width=hidden,
    ) catch {
      _ => raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
    }
    sizes.push(region.storage_bytes())
    Some(region)
  } else {
    None
  }
  sizes.append(expert_workspace_sizes(shared))
  // The shared map is included above; these are its unit indices and scores.
  sizes.push(rows.to_int64() * 8L)
  if routing is Some(router) {
    sizes.push(router.plan().workspace_bytes())
  }
  let workspace_bytes = for size in sizes; total = 0L {
    guard size <= 9223372036854775807L - total else {
      raise @device.DeviceError::InvalidArgument(AllocationCreate)
    }
    continue total + size
  } nobreak {
    total
  }
  {
    routed,
    shared,
    combine,
    routing,
    workspace_bytes,
    functions: [],
    allocations: [],
    startup_streams: [],
    launches: [],
    frontiers: [],
    executor: None,
    closed: false,
    shared_tensor_boundary: None,
    collective_region,
    reduction_placement,
    producer_fork: None,
  }
}

///|
/// Offline AOT compilation input. Never compile this in submit/poll.
pub fn MoeProgram::source_bytes(self : Self) -> Bytes raise {
  let routing = match self.routing {
    None => ""
    Some(router) => @utf8.decode(router.source_bytes()) + "\n"
  }
  @utf8.encode(
    routing +
    @utf8.decode(self.routed.source_bytes()) +
    "\n" +
    @utf8.decode(self.shared.source_bytes()) +
    "\n" +
    self.combine.source() +
    (match self.collective_region {
      None => ""
      Some(region) =>
        "\n" +
        @coalesced.source(region, symbol=self.combine.symbol() + "_collective")
    }),
  )
}

///|
/// Additional compact device memory, excluding caller-owned weight banks and
/// input/output/routing. Can be included in a whole-model startup memory plan.
pub fn MoeProgram::workspace_bytes(self : Self) -> Int64 {
  self.workspace_bytes
}

///|
/// Routed activation and packed-row transport die after this program's joined
/// consumers. A caller with serialized programs may lend this region; shared
/// producer storage, persistent maps and quantization flags remain separate.
pub fn MoeProgram::transient_scratch_bytes(self : Self) -> Int64 {
  self.routed.activation_bytes()
}

///|
pub fn MoeProgram::coalesced_tensor_width(self : Self) -> Int32? {
  self.collective_region.map(region => region.width())
}

///|
/// A single collective for either concatenated F32 lanes or combined BF16.
/// The width does not conflate their distinct numerical programs.
pub fn MoeProgram::combined_tensor_width(self : Self) -> Int32? {
  if self.reduction_placement == LocalCombinedBFloat16 {
    Some(self.routed.plan().geometry().1)
  } else {
    self.coalesced_tensor_width()
  }
}

///|
pub fn MoeProgram::tensor_kind(self : Self) -> @device.TensorCollectiveKind {
  match self.reduction_placement {
    ExpertPartialsFloat32 => F32SumAllReduce
    LocalCombinedBFloat16 => Bf16SumAllReduce
  }
}

///|
fn MoeProgram::allocate(
  self : Self,
  context : @device.Context,
  bytes : Int64,
) -> @device.Allocation raise @device.DeviceError {
  let allocation = context.allocate(bytes)
  self.allocations.push(allocation)
  allocation
}

///|
fn MoeProgram::load_functions(
  self : Self,
  aot_module : @device.Module,
  symbols : ArrayView[String],
) -> Array[@device.Function] raise @device.DeviceError {
  let functions = []
  for symbol in symbols {
    let function = aot_module.load_function(@utf8.encode(symbol))
    self.functions.push(function)
    functions.push(function)
  }
  functions
}

///|
/// Stop the executor first, then release function and intermediate owners.
/// A failed release stays in the owner so close can be retried.
pub fn MoeProgram::close(self : Self) -> Unit raise @device.DeviceError {
  if self.closed {
    return
  }
  if self.executor is Some(executor) {
    executor.close()
    self.executor = None
  }
  self.launches.clear()
  while self.startup_streams.last() is Some(stream) {
    stream.close()
    ignore(self.startup_streams.pop())
  }
  while !self.functions.is_empty() {
    self.functions[self.functions.length() - 1].close()
    ignore(self.functions.pop())
  }
  while !self.allocations.is_empty() {
    self.allocations[self.allocations.length() - 1].close()
    ignore(self.allocations.pop())
  }
  self.closed = true
}

///|
/// Explicit cancellation of active work, followed by the ordinary release path.
pub fn MoeProgram::abort(self : Self) -> Unit raise @device.DeviceError {
  if self.executor is Some(executor) {
    executor.abort()
  }
  self.close()
}

///|
pub fn MoeProgram::submit(
  self : Self,
  live_rows~ : Int32,
  plan_sequence~ : UInt64,
  global_sequence~ : UInt64,
) -> Unit raise @device.DeviceError {
  guard !self.closed && self.executor is Some(executor) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
  }
  executor.submit(live_rows~, plan_sequence~, global_sequence~)
}

///|
pub fn MoeProgram::poll(self : Self) -> Bool raise @device.DeviceError {
  guard !self.closed && self.executor is Some(executor) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorPoll)
  }
  executor.poll()
}
