///|
const LUNA_TILE_PROBE_BYTES : Int64 = 256L

///|
fn luna_tile_probe_input(cycle : Int) -> Bytes {
  let bytes = FixedArray::make(LUNA_TILE_PROBE_BYTES.to_int(), b'\x00')
  for index in 0..<bytes.length() {
    bytes[index] = ((index * 37 + cycle * 19 + 11) & 255).to_byte()
  }
  Bytes::from_array(bytes[:])
}

///|
fn luna_tile_probe_poison() -> Bytes {
  Bytes::from_array(
    FixedArray::make(LUNA_TILE_PROBE_BYTES.to_int(), b'\xa5')[:],
  )
}

///|
fn luna_tile_probe_arguments(
  input : @device.Allocation,
  output : @device.Allocation,
) -> @device.KernelArguments raise {
  @device.KernelArguments::new([
    @device.KernelArgument::device_region(
      input,
      offset=0L,
      byte_count=LUNA_TILE_PROBE_BYTES,
      alignment=16L,
    ),
    @device.KernelArgument::device_region(
      output,
      offset=0L,
      byte_count=LUNA_TILE_PROBE_BYTES,
      alignment=16L,
    ),
  ])
}

///|
fn luna_tile_probe_serial_dimensions(
  serial : @luna_tile_ir.LunaTileCudaTranslationUnit,
) -> @device.KernelLaunchDimensions raise {
  @device.KernelLaunchDimensions::new(
    grid_x=1,
    grid_y=1,
    grid_z=1,
    block_x=1,
    block_y=1,
    block_z=1,
    shared_memory_bytes=serial.memory_plan().shared_memory_bytes().to_int(),
  )
}

///|
fn luna_tile_probe_parallel_dimensions(
  parallel : @luna_tile_ir.LunaTileParallelCandidate,
) -> @device.KernelLaunchDimensions raise {
  let mapping = parallel.mapping()
  @device.KernelLaunchDimensions::new(
    grid_x=mapping.block_count(),
    grid_y=1,
    grid_z=1,
    block_x=mapping.block_threads(),
    block_y=1,
    block_z=1,
    shared_memory_bytes=parallel
      .shared_plan()
      .total_shared_memory_bytes()
      .to_int(),
  )
}

///|
fn run_luna_tile_parallel_device_probe(
  serial_module_bytes : Bytes,
  parallel_module_bytes : Bytes,
  cycles : Int,
  balance : LunaTileParallelProbeBalance,
) -> Unit raise {
  let (serial, parallel) = luna_tile_probe_candidates()
  let devices = match @device.probe() {
    Available(devices~, ..) => devices
    Unavailable(_) =>
      raise LunaTileParallelCudaProbeRejected("CUDA driver is unavailable")
  }
  require_luna_tile_parallel_probe(devices.length() > 0, "no CUDA device")
  let capability = devices[0].capability()
  require_luna_tile_parallel_probe(
    capability.compute_major() == 12 && capability.compute_minor() == 0,
    "LunaTile parallel qualification requires exact sm120 device 0",
  )
  require_luna_tile_parallel_probe(
    capability.supports_bf16(),
    "sm120 device lacks BF16 support",
  )
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
  let input = context.allocate(LUNA_TILE_PROBE_BYTES)
  balance.allocations += 1
  balance.device_bytes += LUNA_TILE_PROBE_BYTES
  defer (try input.close() catch {
    _ => balance.cleanup_failures += 1
  } noraise {
    _ => {
      balance.allocations -= 1
      balance.device_bytes -= LUNA_TILE_PROBE_BYTES
    }
  })
  let serial_output = context.allocate(LUNA_TILE_PROBE_BYTES)
  balance.allocations += 1
  balance.device_bytes += LUNA_TILE_PROBE_BYTES
  defer (try serial_output.close() catch {
    _ => balance.cleanup_failures += 1
  } noraise {
    _ => {
      balance.allocations -= 1
      balance.device_bytes -= LUNA_TILE_PROBE_BYTES
    }
  })
  let parallel_output = context.allocate(LUNA_TILE_PROBE_BYTES)
  balance.allocations += 1
  balance.device_bytes += LUNA_TILE_PROBE_BYTES
  defer (try parallel_output.close() catch {
    _ => balance.cleanup_failures += 1
  } noraise {
    _ => {
      balance.allocations -= 1
      balance.device_bytes -= LUNA_TILE_PROBE_BYTES
    }
  })
  let serial_module = context.load_module(serial_module_bytes)
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
  let parallel_module = context.load_module(parallel_module_bytes)
  balance.modules += 1
  defer (try parallel_module.close() catch {
    _ => balance.cleanup_failures += 1
  } noraise {
    _ => balance.modules -= 1
  })
  let parallel_function = parallel_module.load_function(
    @utf8.encode(parallel.entry_point()),
  )
  balance.functions += 1
  defer (try parallel_function.close() catch {
    _ => balance.cleanup_failures += 1
  } noraise {
    _ => balance.functions -= 1
  })
  let poison = luna_tile_probe_poison()
  for cycle in 0..<cycles {
    let expected = luna_tile_probe_input(cycle)
    input.copy_from_host(expected, 0L, 0L, LUNA_TILE_PROBE_BYTES)
    serial_output.copy_from_host(poison, 0L, 0L, LUNA_TILE_PROBE_BYTES)
    parallel_output.copy_from_host(poison, 0L, 0L, LUNA_TILE_PROBE_BYTES)
    serial_function.launch_synchronous(
      luna_tile_probe_serial_dimensions(serial),
      luna_tile_probe_arguments(input, serial_output),
      stream,
    )
    let serial_actual = serial_output.copy_to_host(0L, LUNA_TILE_PROBE_BYTES)
    require_luna_tile_parallel_probe(
      serial_actual == expected,
      "serial CUDA oracle differs from exact host bytes",
    )
    parallel_function.launch_synchronous(
      luna_tile_probe_parallel_dimensions(parallel),
      luna_tile_probe_arguments(input, parallel_output),
      stream,
    )
    let parallel_actual = parallel_output.copy_to_host(
      0L,
      LUNA_TILE_PROBE_BYTES,
    )
    require_luna_tile_parallel_probe(
      parallel_actual == expected && parallel_actual == serial_actual,
      "parallel CUDA output differs from host/serial oracle",
    )
  }
}

///|
async fn run_luna_tile_parallel_probe(args : Array[String]) -> Unit {
  require_luna_tile_parallel_probe(
    args.length() == 5,
    "usage: luna_tile_parallel_cuda_probe run SERIAL_CUBIN PARALLEL_CUBIN CYCLES",
  )
  let cycles = @string.parse_int(args[4], base=10) catch {
    _ => raise LunaTileParallelCudaProbeRejected("CYCLES must be an integer")
  }
  require_luna_tile_parallel_probe(
    cycles > 0 && cycles <= 32,
    "CYCLES must be in 1..32",
  )
  let serial_module = @fs.read_file(args[2]).binary()
  let parallel_module = @fs.read_file(args[3]).binary()
  require_luna_tile_parallel_probe(
    serial_module.length() > 0 && parallel_module.length() > 0,
    "compiled module is empty",
  )
  let balance = luna_tile_parallel_probe_balance()
  run_luna_tile_parallel_device_probe(
    serial_module, parallel_module, cycles, balance,
  )
  require_luna_tile_parallel_probe(
    balance.is_zero(),
    "CUDA resource balance is nonzero",
  )
  let (serial, parallel) = luna_tile_probe_candidates()
  println(
    "outcome=lunatile-parallel-sm120-qualification-pass cycles=\{cycles} launches=\{cycles * 2} bytes_per_launch=256 serial_source_sha256=\{serial.source_digest().as_hex()} parallel_source_sha256=\{parallel.source_digest().as_hex()} parallel_candidate_sha256=\{parallel.digest().as_hex()} target=sm_120 resources=context0,stream0,allocation0,module0,function0,device_bytes0,pending0,cleanup0 manifest_bindable=false promotion_authority=absent",
  )
}
