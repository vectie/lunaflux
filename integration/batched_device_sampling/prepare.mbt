///|
pub fn BatchedDeviceSampler::new(
  context : @device.Context,
  aot : @device.Module,
  logits : @device.Allocation,
  program : @physical.FullVocabularySamplingProgram,
  maximum_device_bytes~ : Int64,
  maximum_host_bytes~ : Int64,
) -> Self raise {
  let (vocabulary, padded, rows) = program.geometry()
  let control_bytes = 8L + rows.to_int64() * 40L
  let result_bytes = rows.to_int64() * 8L
  let stage_bytes = program.stages().length().max(1).to_int64() * 8L
  guard control_bytes <= 2147483647L &&
    result_bytes <= 2147483647L &&
    (control_bytes + result_bytes) * 2L <= maximum_host_bytes &&
    program.workspace_bytes() <=
    maximum_device_bytes - control_bytes - result_bytes - stage_bytes &&
    logits.byte_count() >= rows.to_int64() * vocabulary.to_int64() * 4L else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let control = context.allocate(control_bytes)
  errdefer (control.close() catch { _ => () })
  let workspace = context.allocate(program.workspace_bytes())
  errdefer (workspace.close() catch { _ => () })
  let stages = context.allocate(stage_bytes)
  errdefer (stages.close() catch { _ => () })
  let results = context.allocate(result_bytes)
  errdefer (results.close() catch { _ => () })
  let parameters = FixedArray::make(control_bytes.to_int(), b'\x00')
  context.copy_from_fixed_host(control, parameters, 0L, 0L, control_bytes)
  let stage_data = FixedArray::make(stage_bytes.to_int(), b'\x00')
  for i, (width, distance) in program.stages() {
    put_i32(stage_data, i * 8, width)
    put_i32(stage_data, i * 8 + 4, distance)
  }
  context.copy_from_fixed_host(stages, stage_data, 0L, 0L, stage_bytes)
  let upload = context.prepare_transfer_pool(
    control,
    host_bytes=control_bytes,
    lanes=1,
    max_regions_per_lane=1,
  )
  errdefer (upload.close() catch { _ => () })
  upload.check_prepared()
  let download = context.prepare_transfer_pool(
    results,
    host_bytes=result_bytes,
    lanes=1,
    max_regions_per_lane=1,
  )
  errdefer (download.close() catch { _ => () })
  download.check_prepared()
  let stream = context.create_stream()
  errdefer (stream.close() catch { _ => () })
  let functions : Array[@device.Function] = []
  errdefer (for f in functions.rev_iter() {
    f.close() catch {
      _ => ()
    }
  })
  for name in ["init", "sort", "weights", "finish"] {
    functions.push(
      aot.load_function(@utf8.encode("luna_full_sampling_" + name)),
    )
  }
  let control_arg = argument(control)
  let workspace_arg = argument(workspace)
  let result_arg = argument(results)
  let launches = [
    launch(functions[0], rows, 1, [
      control_arg,
      argument(logits),
      workspace_arg,
      result_arg,
    ]),
  ]
  for i in 0..<program.stages().length() {
    launches.push(
      launch(functions[1], (padded + 255) / 256, rows, [
        control_arg,
        workspace_arg,
        argument(stages, offset=i.to_int64() * 8L, bytes=8L),
        result_arg,
      ]),
    )
  }
  launches.push(
    launch(functions[2], (vocabulary + 255) / 256, rows, [
      control_arg, workspace_arg, result_arg,
    ]),
  )
  launches.push(
    launch(functions[3], rows, 1, [control_arg, workspace_arg, result_arg]),
  )
  let executor = context.create_ordered_kernel_executor(
    stream,
    launches,
    policy=OrderedKernelCaptureRequired,
  )
  {
    program,
    control,
    workspace,
    stages,
    results,
    upload,
    download,
    stream,
    functions,
    executor,
    parameters,
    output: FixedArray::make(result_bytes.to_int(), b'\x00'),
    count: 0,
    state: Ready,
  }
}

///|
fn launch(
  function : @device.Function,
  x : Int32,
  y : Int32,
  args : Array[@device.KernelArgument],
) -> @device.OrderedKernelLaunch raise {
  @device.OrderedKernelLaunch::new(
    function,
    @device.KernelLaunchDimensions::new(
      grid_x=x,
      grid_y=y,
      grid_z=1,
      block_x=256,
      block_y=1,
      block_z=1,
      shared_memory_bytes=0,
    ),
    @device.KernelArguments::new(args),
  )
}
