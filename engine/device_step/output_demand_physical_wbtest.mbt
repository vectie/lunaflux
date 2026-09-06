///|
/// Opt-in physical test of the real startup/queue/cleanup consumers. Compile
/// tests/fixtures/output_demand.cu offline and supply its cubin path. Ordinary
/// package tests do not require CUDA or compile device code at runtime.
async test "physical eager output demand preserves effects and releases shared leases" {
  guard @env.get_env_var("LUNAFLUX_OUTPUT_DEMAND_FIXTURE") is Some(path) else {
    return
  }
  let image = @fs.read_file(path).binary()
  let context = @device.open_context(0)
  defer (try! context.close())
  let stream = context.create_stream()
  defer (try! stream.close())
  let allocation = context.allocate(12L)
  defer (try! allocation.close())
  allocation.copy_from_host(
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00", 0L, 0L, 12L,
  )
  let loaded = context.load_module(image)
  defer (try! loaded.close())
  let effect = loaded.load_function(b"effect")
  defer (try! effect.close())
  let head = loaded.load_function(b"head")
  defer (try! head.close())
  let sample = loaded.load_function(b"sample")
  defer (try! sample.close())
  let dimensions = @device.KernelLaunchDimensions::new(
    grid_x=1,
    grid_y=1,
    grid_z=1,
    block_x=1,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
  let arguments = @device.KernelArguments::new([
    @device.KernelArgument::device_region(
      allocation,
      offset=0L,
      byte_count=12L,
      alignment=4L,
    ),
  ])
  let launches = [effect, head, sample].map(function => {
    @device.OrderedKernelLaunch::new(function, dimensions, arguments)
  })
  let operations = ReadOnlyArray::from_array(
    [0, 1, 2].map(value => try! @plan.OperationId::new(value)),
  )
  let resources = empty_paged_resources()
  resources.stream = Some({ resource: stream, open: true })
  resources.effect_operation_prefix = 1
  let failure : Ref[DeviceStepError?] = Ref(None)
  defer {
    close_paged_ordered_executors(resources, failure)
    try! assert_eq(failure.val, None)
    try! assert_false(paged_ordered_executor_open(resources))
  }
  let full = prepare_one_paged_ordered_executor(
    context,
    launches[:],
    operations,
    resources,
    OrderedKernelEagerOnly,
  )
  let prefix = resources.ordered_executors[full].effect_only_owner_index
  assert_true(prefix >= 0)
  assert_eq(resources.ordered_executors[prefix].resource.kernel_count(), 1)
  for index, owner_index in [full, prefix, full, prefix] {
    let executor = resources.ordered_executors[owner_index].resource
    for kernel in 0..<executor.kernel_count() {
      executor.enqueue(kernel)
    }
    executor.record_completion()
    executor.wait_completion()
    let result = allocation.copy_to_host(0L, 12L)
    assert_eq(result[0].to_int(), index + 1)
    assert_eq(result[4].to_int(), index / 2 + 1)
    assert_eq(result[8].to_int(), index / 2 + 1)
    executor.reset()
  }
  // Captured execution retains its full list and does not allocate an
  // unaccounted second graph when an effect-only suffix exists.
  let captured = prepare_one_paged_ordered_executor(
    context,
    launches[:],
    operations,
    resources,
    OrderedKernelCaptureRequired,
  )
  assert_eq(resources.ordered_executors[captured].effect_only_owner_index, -1)
  assert_eq(resources.ordered_executors.length(), 3)
  let graph = resources.ordered_executors[captured].resource
  graph.launch_captured()
  graph.record_completion()
  graph.wait_completion()
  let result = allocation.copy_to_host(0L, 12L)
  assert_eq(result[0].to_int(), 5)
  assert_eq(result[4].to_int(), 3)
  assert_eq(result[8].to_int(), 3)
  graph.reset()
  println(
    "output_demand_physical=passed eager_effects=preserved head_sampling=omitted capture=unchanged",
  )
}
