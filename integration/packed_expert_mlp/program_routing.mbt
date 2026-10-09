///|
/// Prepare projection -> score/correction -> selection -> expert execution on
/// the same queue. Selected routing buffers are owned, never copied to the host.
/// Correction bias is an F32 expert vector; supply a zero vector if absent.
pub fn MoeProgram::prepare_routed(
  self : Self,
  context : @device.Context,
  aot_module : @device.Module,
  stream : @device.Stream,
  maximum_workspace_bytes~ : Int64,
  counts~ : @device.Allocation,
  input~ : @device.Allocation,
  router_weight~ : @device.Allocation,
  correction_bias~ : @device.Allocation,
  routed_bank~ : @device.Allocation,
  shared_bank~ : @device.Allocation,
  output~ : @device.Allocation,
  reduction? : MoeReduction,
) -> Unit raise @device.DeviceError {
  self.check_fresh(maximum_workspace_bytes)
  guard self.routing is Some(router) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let functions = self.load_functions(aot_module, router.symbols())
  let plan = router.plan()
  let (rows, hidden, experts, _) = plan.geometry()
  let sizes = plan.buffer_bytes()
  let buffers = sizes.map(bytes => self.allocate(context, bytes))
  let counts_arg = region(counts, 20L, 4L)
  let logits = region(buffers[0], sizes[0], 4L)
  let scores = region(buffers[1], sizes[1], 4L)
  let choice = region(buffers[2], sizes[2], 4L)
  let indices = region(buffers[3], sizes[3], 4L)
  let weights = region(buffers[4], sizes[4], 4L)
  let arguments = [
    [
      counts_arg,
      region(input, rows.to_int64() * hidden.to_int64() * 2L, 2L),
      region(
        router_weight,
        plan.weight().representation().payload_bytes(),
        if plan.weight().representation().format() == Bf16 {
          2L
        } else {
          4L
        },
      ),
      logits,
    ],
    [
      counts_arg,
      logits,
      region(correction_bias, experts.to_int64() * 4L, 4L),
      scores,
      choice,
    ],
    [counts_arg, scores, choice, indices, weights],
  ]
  let prefix = []
  for i, launch in router.launches() {
    prefix.push(
      @device.OrderedKernelLaunch::new(
        functions[i],
        @device.KernelLaunchDimensions::new(
          grid_x=launch.0,
          grid_y=1,
          grid_z=1,
          block_x=launch.1,
          block_y=1,
          block_z=1,
          shared_memory_bytes=0,
        ),
        @device.KernelArguments::new(arguments[i]),
      ),
    )
  }
  self.prepare_experts(
    context,
    aot_module,
    stream,
    counts~,
    input~,
    selected_indices=buffers[3],
    selected_weights=buffers[4],
    routed_bank~,
    shared_bank~,
    output~,
    reduction?,
    prefix~,
  )
}
