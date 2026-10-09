///|
/// Prepare projection -> score/correction -> selection -> expert execution on
/// the same queue. Selected routing buffers are owned, never copied to the host.
/// Grouped selection borrows an F32 bias vector (zero if absent in the model).
/// Token-hash selection borrows token IDs and an already-narrowed I32 table.
pub fn MoeProgram::prepare_routed(
  self : Self,
  context : @device.Context,
  aot_module : @device.Module,
  stream : @device.Stream,
  maximum_workspace_bytes~ : Int64,
  counts~ : @device.Allocation,
  input~ : @device.Allocation,
  router_weight~ : @device.Allocation,
  correction_bias? : @device.Allocation,
  token_ids? : @device.Allocation,
  token_table? : @device.Allocation,
  token_table_offset? : Int64 = 0L,
  routed_bank~ : @device.Allocation,
  shared_bank~ : @device.Allocation,
  output~ : @device.Allocation,
  reduction? : MoeReduction,
) -> Unit raise @device.DeviceError {
  self.check_fresh(maximum_workspace_bytes)
  let (indices, weights, prefix) = self.bind_router(
    context,
    aot_module,
    counts~,
    input~,
    router_weight~,
    correction_bias?,
    token_ids?,
    token_table?,
    token_table_offset~,
  )
  self.prepare_experts(
    context,
    aot_module,
    stream,
    counts~,
    input~,
    selected_indices=indices,
    selected_weights=weights,
    routed_bank~,
    shared_bank~,
    output~,
    reduction?,
    prefix~,
  )
}

///|
fn MoeProgram::bind_router(
  self : Self,
  context : @device.Context,
  aot_module : @device.Module,
  counts~ : @device.Allocation,
  input~ : @device.Allocation,
  router_weight~ : @device.Allocation,
  correction_bias? : @device.Allocation,
  token_ids? : @device.Allocation,
  token_table? : @device.Allocation,
  token_table_offset~ : Int64,
) -> (
  @device.Allocation,
  @device.Allocation,
  Array[@device.OrderedKernelLaunch],
) raise @device.DeviceError {
  guard self.routing is Some(router) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let functions = self.load_functions(aot_module, router.symbols())
  let plan = router.plan()
  let (rows, hidden, experts, selected) = plan.geometry()
  let sizes = plan.buffer_bytes()
  let buffers = sizes.map(bytes => {
    if bytes > 0L {
      Some(self.allocate(context, bytes))
    } else {
      None
    }
  })
  let counts_arg = region(counts, 20L, 4L)
  let logits = region(buffers[0].unwrap(), sizes[0], 4L)
  let scores = region(buffers[1].unwrap(), sizes[1], 4L)
  let indices = region(buffers[3].unwrap(), sizes[3], 4L)
  let weights = region(buffers[4].unwrap(), sizes[4], 4L)
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
  ]
  match plan.selection() {
    GroupedTopK => {
      guard correction_bias is Some(bias) else {
        raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
      }
      let choice = region(buffers[2].unwrap(), sizes[2], 4L)
      arguments.append([
        [
          counts_arg,
          logits,
          region(bias, experts.to_int64() * 4L, 4L),
          scores,
          choice,
        ],
        [counts_arg, scores, choice, indices, weights],
      ])
    }
    TokenHash(vocabulary_size~) => {
      guard token_ids is Some(tokens) && token_table is Some(table) else {
        raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
      }
      arguments.append([
        [counts_arg, logits, scores],
        [
          counts_arg,
          region(tokens, rows.to_int64() * 4L, 4L),
          @device.KernelArgument::device_region(
            table,
            offset=token_table_offset,
            byte_count=vocabulary_size.to_int64() * selected.to_int64() * 4L,
            alignment=4L,
          ),
          indices,
        ],
        [counts_arg, scores, indices, weights],
      ])
    }
  }
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
  (buffers[3].unwrap(), buffers[4].unwrap(), prefix)
}
