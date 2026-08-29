///|
fn tensor_core_arguments(
  left : @device.Allocation,
  right : @device.Allocation,
  accumulator : @device.Allocation,
  output : @device.Allocation,
) -> @device.KernelArguments raise {
  @device.KernelArguments::new([
    @device.KernelArgument::device_region(
      left,
      offset=0L,
      byte_count=TENSOR_CORE_BF16_BYTES,
      alignment=32L,
    ),
    @device.KernelArgument::device_region(
      right,
      offset=0L,
      byte_count=TENSOR_CORE_BF16_BYTES,
      alignment=32L,
    ),
    @device.KernelArgument::device_region(
      accumulator,
      offset=0L,
      byte_count=TENSOR_CORE_ACCUMULATOR_BYTES,
      alignment=32L,
    ),
    @device.KernelArgument::device_region(
      output,
      offset=0L,
      byte_count=TENSOR_CORE_F32_BYTES,
      alignment=32L,
    ),
  ])
}

///|
struct LunaTensorCoreProbeObservation {
  cpu_vs_serial_max : Float
  cpu_vs_tensor_core_max : Float
  serial_vs_tensor_core_max : Float
  device_name_sha256 : String
  device_total_memory_bytes : Int64
}

///|
fn run_tensor_core_device_probe(
  serial_cubin : Bytes,
  tensor_core_cubin : Bytes,
  cycles : Int,
  balance : LunaTensorCoreProbeBalance,
) -> LunaTensorCoreProbeObservation raise {
  let (serial, candidate) = tensor_core_probe_candidates()
  let devices = match @device.probe() {
    Available(devices~, ..) => devices
    Unavailable(_) =>
      raise LunaTensorCoreProbeRejected("CUDA driver unavailable")
  }
  require_tensor_core_probe(devices.length() > 0, "no CUDA device")
  let capability = devices[0].capability()
  require_tensor_core_probe(
    capability.compute_major() == 12 && capability.compute_minor() == 0,
    "tensor-core qualification requires exact sm120 device 0",
  )
  require_tensor_core_probe(capability.supports_bf16(), "sm120 lacks BF16")
  let context = @device.open_context(0)
  balance.contexts += 1
  defer (try context.close() catch {
    _ => balance.cleanup_failures += 1
  } noraise {
    _ => balance.contexts -= 1
  })
  let stream = context.create_stream()
  balance.streams += 1
  defer (try stream.close() catch {
    _ => balance.cleanup_failures += 1
  } noraise {
    _ => balance.streams -= 1
  })
  let allocations : Array[(@device.Allocation, Int64)] = []
  for
    size in [
      TENSOR_CORE_BF16_BYTES,
      TENSOR_CORE_BF16_BYTES,
      TENSOR_CORE_ACCUMULATOR_BYTES,
      TENSOR_CORE_F32_BYTES,
      TENSOR_CORE_F32_BYTES,
    ] {
    let allocation = context.allocate(size)
    allocations.push((allocation, size))
    balance.allocations += 1
    balance.device_bytes += size
  }
  defer (for pair in allocations.rev() {
    try pair.0.close() catch {
      _ => balance.cleanup_failures += 1
    } noraise {
      _ => {
        balance.allocations -= 1
        balance.device_bytes -= pair.1
      }
    }
  })
  let serial_module = context.load_module(serial_cubin)
  balance.modules += 1
  defer (try serial_module.close() catch {
    _ => balance.cleanup_failures += 1
  } noraise {
    _ => balance.modules -= 1
  })
  let serial_function = serial_module.load_function(
    @utf8.encode(serial.entry_point()),
  )
  balance.functions += 1
  defer (try serial_function.close() catch {
    _ => balance.cleanup_failures += 1
  } noraise {
    _ => balance.functions -= 1
  })
  let tensor_module = context.load_module(tensor_core_cubin)
  balance.modules += 1
  defer (try tensor_module.close() catch {
    _ => balance.cleanup_failures += 1
  } noraise {
    _ => balance.modules -= 1
  })
  let tensor_function = tensor_module.load_function(
    @utf8.encode(candidate.entry_point()),
  )
  balance.functions += 1
  defer (try tensor_function.close() catch {
    _ => balance.cleanup_failures += 1
  } noraise {
    _ => balance.functions -= 1
  })
  let (left, _) = allocations[0]
  let (right, _) = allocations[1]
  let (accumulator, _) = allocations[2]
  let (serial_output, _) = allocations[3]
  let (tensor_output, _) = allocations[4]
  let serial_dimensions = @device.KernelLaunchDimensions::new(
    grid_x=1,
    grid_y=1,
    grid_z=1,
    block_x=1,
    block_y=1,
    block_z=1,
    shared_memory_bytes=serial.memory_plan().shared_memory_bytes().to_int(),
  )
  let tensor_dimensions = @device.KernelLaunchDimensions::new(
    grid_x=64,
    grid_y=1,
    grid_z=1,
    block_x=32,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
  let poison = Bytes::from_array(
    FixedArray::make(TENSOR_CORE_F32_BYTES.to_int(), b'\xa5')[:],
  )
  let mut cpu_vs_serial_max = 0.0F
  let mut cpu_vs_tensor_core_max = 0.0F
  let mut serial_vs_tensor_core_max = 0.0F
  for cycle in 0..<cycles {
    let left_host = tensor_core_hostile_bf16(cycle, false)
    let right_host = tensor_core_hostile_bf16(cycle, true)
    let accumulator_host = tensor_core_accumulator(cycle)
    left.copy_from_host(
      Bytes::from_array(left_host[:]),
      0L,
      0L,
      TENSOR_CORE_BF16_BYTES,
    )
    right.copy_from_host(
      Bytes::from_array(right_host[:]),
      0L,
      0L,
      TENSOR_CORE_BF16_BYTES,
    )
    accumulator.copy_from_host(
      Bytes::from_array(accumulator_host[:]),
      0L,
      0L,
      TENSOR_CORE_ACCUMULATOR_BYTES,
    )
    serial_output.copy_from_host(poison, 0L, 0L, TENSOR_CORE_F32_BYTES)
    tensor_output.copy_from_host(poison, 0L, 0L, TENSOR_CORE_F32_BYTES)
    let serial_args = tensor_core_arguments(
      left, right, accumulator, serial_output,
    )
    serial_function.launch_synchronous(serial_dimensions, serial_args, stream)
    let serial_actual = serial_output.copy_to_host(0L, TENSOR_CORE_F32_BYTES)
    let expected = tensor_core_cpu_oracle(cycle)
    cpu_vs_serial_max = cpu_vs_serial_max.max(
      validate_tensor_core_output(serial_actual, expected, "serial CUDA oracle"),
    )
    let tensor_args = tensor_core_arguments(
      left, right, accumulator, tensor_output,
    )
    tensor_function.launch_synchronous(tensor_dimensions, tensor_args, stream)
    let tensor_actual = tensor_output.copy_to_host(0L, TENSOR_CORE_F32_BYTES)
    cpu_vs_tensor_core_max = cpu_vs_tensor_core_max.max(
      validate_tensor_core_output(
        tensor_actual, expected, "tensor-core candidate",
      ),
    )
    for index in 0..<16384 {
      let difference = (tensor_core_read_f32(tensor_actual, index) -
      tensor_core_read_f32(serial_actual, index)).abs()
      require_tensor_core_probe(
        difference <= 0.001F + 0.0001F * expected[index].abs(),
        "tensor-core candidate differs from serial CUDA oracle",
      )
      serial_vs_tensor_core_max = serial_vs_tensor_core_max.max(difference)
    }
  }
  {
    cpu_vs_serial_max,
    cpu_vs_tensor_core_max,
    serial_vs_tensor_core_max,
    device_name_sha256: tensor_core_probe_sha256(
      @utf8.encode(devices[0].name()),
    ),
    device_total_memory_bytes: capability.total_memory_bytes(),
  }
}

///|
async fn run_tensor_core_probe(args : Array[String]) -> Unit {
  require_tensor_core_probe(
    args.length() == 5,
    "usage: luna_tensor_core_cuda_probe run SERIAL_CUBIN TENSOR_CUBIN CYCLES",
  )
  let cycles = @string.parse_int(args[4], base=10) catch {
    _ => raise LunaTensorCoreProbeRejected("CYCLES must be an integer")
  }
  require_tensor_core_probe(cycles > 0 && cycles <= 32, "CYCLES outside 1..32")
  let serial_cubin = @fs.read_file(args[2]).binary()
  let tensor_cubin = @fs.read_file(args[3]).binary()
  require_tensor_core_probe(
    serial_cubin.length() > 0 && tensor_cubin.length() > 0,
    "compiled module is empty",
  )
  let balance = luna_tensor_core_probe_balance()
  let observation = run_tensor_core_device_probe(
    serial_cubin, tensor_cubin, cycles, balance,
  )
  require_tensor_core_probe(
    balance.is_zero(),
    "CUDA resource balance is nonzero",
  )
  let (serial, candidate) = tensor_core_probe_candidates()
  println(
    "outcome=lunatile-tensor-core-sm120-qualification-pass cycles=\{cycles} numeric_case_count=\{cycles * 16384} launches=\{cycles * 2} cpu_vs_serial_max_abs_error=\{observation.cpu_vs_serial_max} cpu_vs_tensor_core_max_abs_error=\{observation.cpu_vs_tensor_core_max} serial_vs_tensor_core_max_abs_error=\{observation.serial_vs_tensor_core_max} absolute_tolerance=0.001 relative_tolerance=0.0001 cpu_oracle=independent-ordered-f32-v1 serial_cuda_oracle=passed serial_source_sha256=\{serial.source_digest().as_hex()} tensor_core_source_sha256=\{candidate.source_digest().as_hex()} tensor_core_candidate_sha256=\{candidate.digest().as_hex()} target=sm_120 device_ordinal=0 device_name_sha256=\{observation.device_name_sha256} device_total_memory_bytes=\{observation.device_total_memory_bytes} resources=context0,stream0,allocation0,module0,function0,device_bytes0,pending0,cleanup0 manifest_bindable=false promotion_authority=absent",
  )
}
