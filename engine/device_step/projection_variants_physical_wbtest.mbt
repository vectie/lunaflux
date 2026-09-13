///|
/// Opt-in differential check of two compiler schedules in one AOT module.
/// The synthetic tuning record only chooses this test's second schedule.
async test "physical row variants preserve dense values and graph fallback" {
  guard @env.get_env_var("LUNAFLUX_ROW_VARIANTS_FIXTURE") is Some(path) else {
    return
  }
  let context = @device.open_context(0)
  defer (try! context.close())
  let stream = context.create_stream()
  defer (try! stream.close())
  let loaded = context.load_module(@fs.read_file(path).binary())
  defer (try! loaded.close())
  let primary = loaded.load_function(b"projection")
  defer (try! primary.close())
  let small = loaded.load_function(b"projection_rows8")
  defer (try! small.close())
  let allocations : Array[@device.Allocation] = []
  defer (for allocation in allocations.rev() {
    try! allocation.close()
  })
  let sizes = [16, 256 * 1024 * 2, 1024 * 1024 * 2, 256 * 1024 * 2]
  for size in sizes {
    allocations.push(context.allocate(size.to_int64()))
  }
  let input = FixedArray::make(sizes[1], b'\x00')
  for index in 0..<(sizes[1] / 2) {
    input[index * 2 + 1] = b'\x3f'
    input[index * 2] = b'\x80'
  }
  let weight = FixedArray::make(sizes[2], b'\x00')
  for index in 0..<(sizes[2] / 2) {
    weight[index * 2 + 1] = b'\x3a'
    weight[index * 2] = b'\x80'
  }
  allocations[1].copy_from_host(
    Bytes::from_array(input),
    0L,
    0L,
    sizes[1].to_int64(),
  )
  allocations[2].copy_from_host(
    Bytes::from_array(weight),
    0L,
    0L,
    sizes[2].to_int64(),
  )
  let arguments = @device.KernelArguments::new(
    allocations.mapi((index, allocation) => {
      @device.KernelArgument::device_region(
        allocation,
        offset=0L,
        byte_count=sizes[index].to_int64(),
        alignment=4L,
      )
    }),
  )
  let benchmark = @env.get_env_var("LUNAFLUX_ROW_VARIANTS_BENCHMARK") ==
    Some("1")
  let start = context.create_event()
  defer (try! start.close())
  let finish = context.create_event()
  defer (try! finish.close())
  for tokens in [1, 2, 7, 8, 9, 16, 32, 256] {
    let counts = FixedArray::make(16, b'\x00')
    counts[12] = (tokens & 255).to_byte()
    counts[13] = (tokens >> 8).to_byte()
    allocations[0].copy_from_host(Bytes::from_array(counts), 0L, 0L, 16L)
    for groups in (if tokens > 1 && tokens <= 8 { [8, 2] } else { [8] }) {
      let source = @launch_contract.AotLaunchDimensions::new(
        grid_x=1024,
        grid_y=1,
        grid_z=1,
        block_x=groups * 32,
        block_y=1,
        block_z=1,
        shared_memory_bytes=groups * 1024,
      )
      let dimensions = paged_device_launch_dimensions(
        paged_bucket_launch(ProjectionTensorCoreGridX(source, 1024), tokens),
      )
      let zeros = Bytes::make(sizes[3], 0)
      allocations[3].copy_from_host(zeros, 0L, 0L, sizes[3].to_int64())
      let resources = empty_paged_resources()
      resources.functions.push({
        resource: primary,
        module_index: 0,
        open: true,
      })
      resources.functions.push({ resource: small, module_index: 0, open: true, })
      if groups == 2 {
        resources.projection_variants.push({
          base_function_index: 0,
          maximum_token_rows: 8,
          function_index: 1,
          launch: ProjectionTensorCoreGridX(source, 1024),
        })
      }
      let launches = paged_ordered_launches(
        [
          {
            operation_id: @plan.OperationId::new(0),
            function_index: 0,
            dimensions,
            arguments,
          },
        ],
        resources,
        query_token_bound=Some(tokens),
      )
      let executor = context.create_ordered_kernel_executor(
        stream,
        Array::make(if benchmark { 64 } else { 1 }, launches[0]),
        policy=OrderedKernelCaptureRequired,
      )
      defer (try! executor.close())
      executor.launch_captured()
      executor.record_completion()
      executor.wait_completion()
      let result = allocations[3].copy_to_host(0L, sizes[3].to_int64())
      for index in 0..<(tokens * 1024) {
        assert_eq(result[index * 2], b'\x80')
        assert_eq(result[index * 2 + 1], b'\x3f')
      }
      for index in (tokens * 2048)..<sizes[3] {
        assert_eq(result[index], b'\x00')
      }
      executor.reset()
      if benchmark {
        for trial in 0..<7 {
          start.record(stream)
          executor.launch_captured()
          finish.record(stream)
          executor.record_completion()
          executor.wait_completion()
          executor.reset()
          println(
            "rows=\{tokens} groups=\{groups} trial=\{trial} microseconds=\{start.elapsed_milliseconds(finish) * 1000.0 / 64.0}",
          )
        }
      }
    }
  }
  println(
    "row_variants_physical=passed tokens=1,2,7,8,9,16,32,256 captured=true fallback=true tail_untouched=true",
  )
}
