///|
struct Fp8ProbeBuffer {
  allocation : @device.Allocation
  bytes : Int64
}

///|
fn fp8_probe_allocate(
  context : @device.Context,
  bytes : Int64,
  balance : Fp8V2ProbeBalance,
) -> Fp8ProbeBuffer raise {
  let allocation = context.allocate(bytes)
  balance.allocations += 1
  balance.device_bytes += bytes
  { allocation, bytes }
}

///|
fn fp8_probe_close_buffer(
  buffer : Fp8ProbeBuffer,
  balance : Fp8V2ProbeBalance,
) -> Unit {
  try buffer.allocation.close() catch {
    _ => balance.cleanup += 1
  } noraise {
    _ => {
      balance.allocations -= 1
      balance.device_bytes -= buffer.bytes
    }
  }
}

///|
fn fp8_probe_upload(
  context : @device.Context,
  buffer : Fp8ProbeBuffer,
  bytes : FixedArray[Byte],
) -> Unit raise {
  require_fp8_probe(bytes.length().to_int64() == buffer.bytes, "upload size")
  context.copy_from_fixed_host(buffer.allocation, bytes, 0L, 0L, buffer.bytes)
}

///|
fn fp8_probe_region(
  buffer : Fp8ProbeBuffer,
  alignment : Int64,
) -> @device.KernelArgument raise {
  @device.KernelArgument::device_region(
    buffer.allocation,
    offset=0L,
    byte_count=buffer.bytes,
    alignment~,
  )
}

///|
fn fp8_probe_dimensions(
  source : @luna_cuda_fp8_projection_aot.Fp8ProjectionCudaLoweringV2,
) -> @device.KernelLaunchDimensions raise {
  let dimensions = source.launch_dimensions()
  @device.KernelLaunchDimensions::new(
    grid_x=dimensions.grid_x(),
    grid_y=dimensions.grid_y(),
    grid_z=dimensions.grid_z(),
    block_x=dimensions.block_x(),
    block_y=dimensions.block_y(),
    block_z=dimensions.block_z(),
    shared_memory_bytes=dimensions.shared_memory_bytes(),
  )
}

///|
fn fp8_probe_execute_repeated(
  context : @device.Context,
  stream : @device.Stream,
  cubin : Bytes,
  source : @luna_cuda_fp8_projection_aot.Fp8ProjectionCudaLoweringV2,
  arguments : @device.KernelArguments,
  cycles : Int,
  output : Fp8ProbeBuffer,
  output_host : FixedArray[Byte],
  workspace : Fp8ProbeBuffer,
  workspace_host : FixedArray[Byte],
  expected : FixedArray[Float],
  expected_scales : FixedArray[Float],
  label : String,
  balance : Fp8V2ProbeBalance,
) -> Float raise {
  let loaded = context.load_module(cubin)
  balance.modules += 1
  defer (try loaded.close() catch {
    _ => balance.cleanup += 1
  } noraise {
    _ => balance.modules -= 1
  })
  let function = loaded.load_function(@utf8.encode(source.function_symbol()))
  balance.functions += 1
  defer (try function.close() catch {
    _ => balance.cleanup += 1
  } noraise {
    _ => balance.functions -= 1
  })
  let launch = @device.OrderedKernelLaunch::new(
    function,
    fp8_probe_dimensions(source),
    arguments,
  )
  let executor = context.create_ordered_kernel_executor(
    stream,
    [launch],
    policy=OrderedKernelEagerOnly,
  )
  balance.executors += 1
  defer (try executor.close() catch {
    _ => balance.cleanup += 1
  } noraise {
    _ => balance.executors -= 1
  })
  require_fp8_probe(
    executor.execution_mode() == OrderedKernelEager,
    label + " diagnostic executor is not eager",
  )
  let output_poison = FixedArray::make(output_host.length(), b'\xa5')
  let workspace_poison = FixedArray::make(workspace_host.length(), b'\x00')
  for index in 0..<(workspace_poison.length() / 4) {
    fp8_probe_write_f32(
      workspace_poison,
      index,
      Float::reinterpret_from_uint(0x7fc00000U),
    )
  }
  let mut maximum_error = 0.0F
  for _ in 0..<cycles {
    fp8_probe_upload(context, output, output_poison)
    fp8_probe_upload(context, workspace, workspace_poison)
    executor.enqueue(0)
    balance.pending += 1
    executor.record_completion()
    executor.wait_completion()
    balance.pending -= 1
    context.copy_to_fixed_host(
      output.allocation,
      output_host,
      0L,
      0L,
      output.bytes,
    )
    context.copy_to_fixed_host(
      workspace.allocation,
      workspace_host,
      0L,
      0L,
      workspace.bytes,
    )
    maximum_error = maximum_error.max(
      fp8_probe_validate_output(output_host, expected, label),
    )
    for index, expected_scale in expected_scales {
      let observed = fp8_probe_read_f32(workspace_host, index)
      require_fp8_probe(
        !observed.is_nan() &&
        !observed.is_inf() &&
        observed > 0.0F &&
        (observed - expected_scale).abs() <=
        0.000001F.max(expected_scale.abs() * 0.00001F),
        label + " production scale evidence is invalid",
      )
    }
    executor.reset()
  }
  maximum_error
}

///|
struct Fp8ProbeObservation {
  simple_error : Float
  gated_error : Float
  device_name_sha256 : String
  device_total_memory_bytes : Int64
}

///|
fn run_fp8_probe_device(
  simple_cubin : Bytes,
  gated_cubin : Bytes,
  candidates : Fp8V2ProbeCandidates,
  cycles : Int,
  balance : Fp8V2ProbeBalance,
) -> Fp8ProbeObservation raise {
  let devices = match @device.probe() {
    Available(devices~, ..) => devices
    Unavailable(_) => raise Fp8V2CudaProbeError::Rejected("CUDA unavailable")
  }
  require_fp8_probe(devices.length() > 0, "no CUDA device")
  let capability = devices[0].capability()
  require_fp8_probe(
    capability.compute_major() == 12 && capability.compute_minor() == 0,
    "physical FP8 v2 qualification requires exact sm120 device 0",
  )
  require_fp8_probe(
    capability.supports_bf16() && capability.supports_cublas_lt(),
    "sm120 device lacks authenticated BF16 tensor-math capability",
  )
  let context = @device.open_context(0)
  balance.contexts += 1
  defer (try context.close() catch {
    _ => balance.cleanup += 1
  } noraise {
    _ => balance.contexts -= 1
  })
  require_fp8_probe(
    context.capability().compute_major() == 12 &&
    context.capability().compute_minor() == 0,
    "opened context does not match probed sm120 device",
  )
  let stream = context.create_stream()
  balance.streams += 1
  defer (try stream.close() catch {
    _ => balance.cleanup += 1
  } noraise {
    _ => balance.streams -= 1
  })
  let sizes : Array[Int64] = [
    20L, 32L, 64L, 4L, 32L, 4L, 32L, 4L, 64L, 4L, 128L, 4L, 128L, 4L, 128L, 4L, 32L,
    8L,
  ]
  let buffers : Array[Fp8ProbeBuffer] = []
  for size in sizes {
    buffers.push(fp8_probe_allocate(context, size, balance))
  }
  defer (for buffer in buffers.rev() {
    fp8_probe_close_buffer(buffer, balance)
  })
  let counts = fp8_probe_counts()
  let input = fp8_probe_input_bytes()
  fp8_probe_upload(context, buffers[0], counts)
  fp8_probe_upload(context, buffers[1], input)
  fp8_probe_upload(context, buffers[2], fp8_probe_weight_bytes(64, 1))
  fp8_probe_upload(context, buffers[3], fp8_probe_scale_bytes(0.03125F))
  fp8_probe_upload(context, buffers[4], fp8_probe_weight_bytes(32, 2))
  fp8_probe_upload(context, buffers[5], fp8_probe_scale_bytes(0.0625F))
  fp8_probe_upload(context, buffers[6], fp8_probe_weight_bytes(32, 3))
  fp8_probe_upload(context, buffers[7], fp8_probe_scale_bytes(0.015625F))
  fp8_probe_upload(context, buffers[10], fp8_probe_weight_bytes(128, 4))
  fp8_probe_upload(context, buffers[11], fp8_probe_scale_bytes(0.00390625F))
  fp8_probe_upload(context, buffers[12], fp8_probe_weight_bytes(128, 5))
  fp8_probe_upload(context, buffers[13], fp8_probe_scale_bytes(0.0078125F))
  fp8_probe_upload(context, buffers[14], fp8_probe_weight_bytes(128, 6))
  fp8_probe_upload(context, buffers[15], fp8_probe_scale_bytes(0.015625F))
  let simple_arguments = @device.KernelArguments::new([
    fp8_probe_region(buffers[0], 4L),
    fp8_probe_region(buffers[1], 16L),
    fp8_probe_region(buffers[2], 16L),
    fp8_probe_region(buffers[3], 16L),
    fp8_probe_region(buffers[4], 16L),
    fp8_probe_region(buffers[5], 16L),
    fp8_probe_region(buffers[6], 16L),
    fp8_probe_region(buffers[7], 16L),
    fp8_probe_region(buffers[8], 16L),
    fp8_probe_region(buffers[9], 4L),
  ])
  let gated_arguments = @device.KernelArguments::new([
    fp8_probe_region(buffers[0], 4L),
    fp8_probe_region(buffers[1], 16L),
    fp8_probe_region(buffers[10], 16L),
    fp8_probe_region(buffers[11], 16L),
    fp8_probe_region(buffers[12], 16L),
    fp8_probe_region(buffers[13], 16L),
    fp8_probe_region(buffers[14], 16L),
    fp8_probe_region(buffers[15], 16L),
    fp8_probe_region(buffers[16], 16L),
    fp8_probe_region(buffers[17], 4L),
  ])
  let (simple_expected, simple_scale) = fp8_probe_qkv_expected()
  let simple_error = fp8_probe_execute_repeated(
    context,
    stream,
    simple_cubin,
    candidates.simple,
    simple_arguments,
    cycles,
    buffers[8],
    FixedArray::make(64, b'\x00'),
    buffers[9],
    FixedArray::make(4, b'\x00'),
    simple_expected,
    [simple_scale],
    "QKV",
    balance,
  )
  let (gated_expected, external_scale, post_scale) = fp8_probe_gated_expected()
  let gated_error = fp8_probe_execute_repeated(
    context,
    stream,
    gated_cubin,
    candidates.gated,
    gated_arguments,
    cycles,
    buffers[16],
    FixedArray::make(32, b'\x00'),
    buffers[17],
    FixedArray::make(8, b'\x00'),
    gated_expected,
    [external_scale, post_scale],
    "gated MLP",
    balance,
  )
  {
    simple_error,
    gated_error,
    device_name_sha256: fp8_probe_sha256(@utf8.encode(devices[0].name())),
    device_total_memory_bytes: capability.total_memory_bytes(),
  }
}

///|
async fn run_fp8_probe(args : Array[String]) -> Unit {
  require_fp8_probe(
    args.length() == 7,
    "usage: fp8_v2_cuda_probe run SIMPLE_CUBIN GATED_CUBIN TOOLCHAIN_SHA256 CYCLES sm_120",
  )
  require_fp8_probe(args[6] == "sm_120", "probe target must be exact sm_120")
  let cycles = @string.parse_int(args[5], base=10) catch {
    _ => raise Fp8V2CudaProbeError::Rejected("cycles is not an integer")
  }
  require_fp8_probe(cycles > 0 && cycles <= 32, "cycles outside 1..32")
  let simple_cubin = @fs.read_file(args[2]).binary()
  let gated_cubin = @fs.read_file(args[3]).binary()
  require_fp8_probe(
    simple_cubin.length() > 0 && gated_cubin.length() > 0,
    "CUBIN is empty",
  )
  let candidates = fp8_probe_candidates(args[4])
  let balance = fp8_probe_balance()
  let observation = run_fp8_probe_device(
    simple_cubin, gated_cubin, candidates, cycles, balance,
  )
  require_fp8_probe(balance.is_zero(), "CUDA resource balance is nonzero")
  println(
    "outcome=fp8-v2-sm120-physical-pass cycles=\{cycles} families=qkv,gated-mlp launches=\{cycles * 2} numeric_values=\{cycles * 48} qkv_max_abs_error=\{observation.simple_error} gated_max_abs_error=\{observation.gated_error} absolute_tolerance=0.03125 relative_tolerance=0.015625 cpu_oracle=independent-ordered-f32-v1 abi=production-paged-v4-operands-v2 diagnostic=scale-cells-finite-positive target=sm_120 device_ordinal=0 device_name_sha256=\{observation.device_name_sha256} device_total_memory_bytes=\{observation.device_total_memory_bytes} resources=context0,stream0,executor0,allocation0,module0,function0,device_bytes0,pending0,cleanup0 manifest_bindable=false readiness=false",
  )
}
