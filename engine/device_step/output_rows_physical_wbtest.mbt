///|
async test "physical captured descriptor projection preserves mixed and empty frames" {
  guard @env.get_env_var("LUNAFLUX_OUTPUT_DEMAND_FIXTURE") is Some(path) else {
    return
  }
  let image = @fs.read_file(path).binary()
  let context = @device.open_context(0)
  defer (try! context.close())
  let stream = context.create_stream()
  defer (try! stream.close())
  let prepared = prepare_validated(
    context,
    step_test_model_plan(),
    @worker_protocol.ModelPlanGeneration::from_wire(1UL),
    step_test_layout(),
    step_device_limits(max_tokens=16),
    step_required_capabilities(),
    false,
    0UL,
  )
  guard prepared is Ready(descriptor) else { fail("descriptor preparation") }
  defer (try! descriptor.close())
  descriptor.host.project_output_rows = true
  let observed = context.allocate(36L)
  defer (try! observed.close())
  let loaded = context.load_module(image)
  defer (try! loaded.close())
  let function = loaded.load_function(b"observe_output_rows")
  defer (try! function.close())
  let arguments = @device.KernelArguments::new([
    descriptor.descriptor_argument(Counts, byte_count=20L, alignment=4L),
    @device.KernelArgument::device_region(
      descriptor.resources.counts,
      offset=20L,
      byte_count=20L,
      alignment=4L,
    ),
    @device.KernelArgument::device_region(
      descriptor.resources.counts,
      offset=40L,
      byte_count=20L,
      alignment=4L,
    ),
    @device.KernelArgument::device_region(
      observed,
      offset=0L,
      byte_count=36L,
      alignment=4L,
    ),
  ])
  let dimensions = @device.KernelLaunchDimensions::new(
    grid_x=1,
    grid_y=1,
    grid_z=1,
    block_x=1,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
  let graph = context.create_ordered_kernel_executor(
    stream,
    [@device.OrderedKernelLaunch::new(function, dimensions, arguments)],
    policy=OrderedKernelCaptureRequired,
  )
  defer (try! graph.close())
  for
    index, produces in [
      [true, false, true, false],
      [false, true, false, false],
      [false, false, false, false],
      [true, true, true, true],
    ] {
    let plan = output_rows_fixture(produces, sequence=index.to_uint64() + 1UL)
    let staged = descriptor.stage(plan)
    graph.launch_captured()
    graph.record_completion()
    graph.wait_completion()
    let result = FixedArray::from_array(
      observed.copy_to_host(0L, 36L).to_array(),
    )
    assert_eq(read_descriptor_i32(result, 0), 4)
    assert_eq(read_descriptor_i32(result, 1), 10)
    assert_eq(read_descriptor_i32(result, 2), 4)
    let rows = output_physical_rows(descriptor.host, descriptor.staged_summary)
    assert_eq(read_descriptor_i32(result, 3), rows)
    for row in 0..<(rows + 1) {
      assert_eq(
        read_descriptor_i32(result, 4 + row),
        read_descriptor_i32(descriptor.host.counts, OUTPUT_ROW_ENDS_CELL + row),
      )
    }
    graph.reset()
    descriptor.finish(staged)
  }
  println(
    "output_rows_physical=passed captured=stable_pointer mixed=preserved empty=zero single=matrix_padded",
  )
}
