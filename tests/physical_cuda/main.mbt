///|
suberror PhysicalCudaFailure {
  PhysicalCudaFailure(String)
}

///|
fn require(
  condition : Bool,
  message : String,
) -> Unit raise PhysicalCudaFailure {
  if !condition {
    raise PhysicalCudaFailure(message)
  }
}

///|
fn write_u16_le(bytes : FixedArray[Byte], index : Int, value : Int) -> Unit {
  let offset = index * 2
  bytes[offset] = value.to_byte()
  bytes[offset + 1] = (value >> 8).to_byte()
}

///|
fn write_u32_le(bytes : FixedArray[Byte], index : Int, value : Int) -> Unit {
  let offset = index * 4
  bytes[offset] = value.to_byte()
  bytes[offset + 1] = (value >> 8).to_byte()
  bytes[offset + 2] = (value >> 16).to_byte()
  bytes[offset + 3] = (value >> 24).to_byte()
}

///|
fn read_u32_le(bytes : Bytes, index : Int) -> Int {
  let offset = index * 4
  bytes[offset].to_int() |
  (bytes[offset + 1].to_int() << 8) |
  (bytes[offset + 2].to_int() << 16) |
  (bytes[offset + 3].to_int() << 24)
}

///|
fn require_bf16_output(output : Bytes) -> Unit raise PhysicalCudaFailure {
  let expected : FixedArray[Int] = [0x4268, 0x4280, 0x430b, 0x431a]
  require(output.length() == 8, "BF16 GEMM returned the wrong byte count")
  for index in 0..<expected.length() {
    let offset = index * 2
    let value = output[offset].to_int() | (output[offset + 1].to_int() << 8)
    require(
      value == expected[index],
      "BF16 GEMM numerical mismatch at output index \{index}: \{value}",
    )
  }
}

///|
fn run_transfer_and_gemm_cycle(ordinal : Int, cycle : Int) -> Unit raise {
  let context = @cuda.open_context(ordinal)
  defer (try! context.close())
  let stream = context.create_stream()
  defer (try! stream.close())
  let start = context.create_event()
  defer (try! start.close())
  let finish = context.create_event()
  defer (try! finish.close())

  let transfer = context.allocate(4096L)
  defer (try! transfer.close())
  let source = FixedArray::makei(4096, index => {
    ((index + cycle) % 251).to_byte()
  })
  context.copy_from_fixed_host(transfer, source, 0L, 0L, 4096L)
  let readback = transfer.copy_to_host(4096L)
  for index in 0..<source.length() {
    require(
      readback[index] == source[index],
      "host/device transfer mismatch at byte \{index} in cycle \{cycle}",
    )
  }

  let cublas = context.create_cublas_lt()
  defer (try! cublas.close())
  let plan = cublas.create_bf16_gemm_plan(2L, 3L, 2L, 4096L)
  defer (try! plan.close())
  let left = context.allocate(16L)
  defer (try! left.close())
  let right = context.allocate(16L)
  defer (try! right.close())
  let output = context.allocate(16L)
  defer (try! output.close())
  let workspace = context.allocate(4096L)
  defer (try! workspace.close())

  let left_host = FixedArray::make(16, b'\x00')
  for index, value in [0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0] {
    write_u16_le(left_host, index, value)
  }
  let right_host = FixedArray::make(16, b'\x00')
  for index, value in [0x40e0, 0x4110, 0x4130, 0x4100, 0x4120, 0x4140] {
    write_u16_le(right_host, index, value)
  }
  context.copy_from_fixed_host(left, left_host, 0L, 0L, 12L)
  context.copy_from_fixed_host(right, right_host, 0L, 0L, 12L)
  start.record(stream)
  plan.run(left, 0L, right, 0L, output, 0L, workspace, 0L, stream)
  finish.record(stream)
  finish.synchronize()
  require(
    start.elapsed_milliseconds(finish) >= 0.0,
    "CUDA event timing returned a negative duration",
  )
  require_bf16_output(output.copy_to_host(8L))
}

///|
fn require_add_one(
  output : Bytes,
  operation : String,
) -> Unit raise PhysicalCudaFailure {
  require(output.length() == 16, "\{operation} returned the wrong byte count")
  for index in 0..<4 {
    require(
      read_u32_le(output, index) == index + 2,
      "\{operation} numerical mismatch at index \{index}",
    )
  }
}

///|
fn require_residual_add(output : Bytes) -> Unit raise PhysicalCudaFailure {
  require(output.length() == 32, "residual add returned the wrong byte count")
  for index in 0..<16 {
    let offset = index * 2
    let value = output[offset].to_int() | (output[offset + 1].to_int() << 8)
    require(
      value == 0x4188,
      "residual add numerical mismatch at index \{index}: \{value}",
    )
  }
}

///|
fn run_residual_aot_cycle(ordinal : Int, image : Bytes) -> Unit raise {
  let context = @cuda.open_context(ordinal)
  defer (try! context.close())
  let stream = context.create_stream()
  defer (try! stream.close())
  let left = context.allocate(32L)
  defer (try! left.close())
  let right = context.allocate(32L)
  defer (try! right.close())
  let output = context.allocate(32L)
  defer (try! output.close())
  let module_ = context.load_module(image)
  defer (try! module_.close())
  let function_ = module_.load_function(b"lunaflux_luna_residual_add_v1_ep_7")
  defer (try! function_.close())

  let left_host = FixedArray::make(32, b'\x00')
  let right_host = FixedArray::make(32, b'\x00')
  let left_values : FixedArray[Int] = [
    0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0, 0x40e0, 0x4100, 0x4110, 0x4120,
    0x4130, 0x4140, 0x4150, 0x4160, 0x4170, 0x4180,
  ]
  let right_values : FixedArray[Int] = [
    0x4180, 0x4170, 0x4160, 0x4150, 0x4140, 0x4130, 0x4120, 0x4110, 0x4100, 0x40e0,
    0x40c0, 0x40a0, 0x4080, 0x4040, 0x4000, 0x3f80,
  ]
  for index in 0..<16 {
    write_u16_le(left_host, index, left_values[index])
    write_u16_le(right_host, index, right_values[index])
  }
  context.copy_from_fixed_host(left, left_host, 0L, 0L, 32L)
  context.copy_from_fixed_host(right, right_host, 0L, 0L, 32L)
  let executor = @cuda.prepare_ordered_kernel_executor(
    context,
    stream,
    FixedArray::make(1, function_),
    [1, 1, 1, 256, 1, 1, 0],
    [0, 3],
    [left, right, output],
    [0L, 0L, 0L],
    [32L, 32L, 32L],
    [2L, 2L, 2L],
  )
  defer (try! executor.close())
  executor.enqueue(0)
  executor.record_completion()
  executor.wait_completion()
  require_residual_add(output.copy_to_host(32L))
}

///|
fn run_aot_cycle(ordinal : Int, image : Bytes) -> Unit raise {
  let context = @cuda.open_context(ordinal)
  defer (try! context.close())
  let stream = context.create_stream()
  defer (try! stream.close())
  let allocation = context.allocate(16L)
  defer (try! allocation.close())
  let module_ = context.load_module(image)
  defer (try! module_.close())
  let function_ = module_.load_function(b"lunaflux_physical_add_one")
  defer (try! function_.close())

  let input = FixedArray::make(16, b'\x00')
  for index in 0..<4 {
    write_u32_le(input, index, index + 1)
  }
  context.copy_from_fixed_host(allocation, input, 0L, 0L, 16L)
  function_.launch_synchronous(
    stream,
    1,
    1,
    1,
    4,
    1,
    1,
    0,
    FixedArray::make(1, allocation),
    [0L],
    [16L],
    [4L],
  )
  require_add_one(allocation.copy_to_host(16L), "direct AOT launch")

  for index in 0..<4 {
    write_u32_le(input, index, index + 1)
  }
  context.copy_from_fixed_host(allocation, input, 0L, 0L, 16L)
  let executor = @cuda.prepare_ordered_kernel_executor(
    context,
    stream,
    FixedArray::make(1, function_),
    [1, 1, 1, 4, 1, 1, 0],
    [0, 1],
    FixedArray::make(1, allocation),
    [0L],
    [16L],
    [4L],
    policy=OrderedExecutorCaptureRequired,
  )
  defer (try! executor.close())
  require(
    executor.execution_mode() == OrderedExecutorCaptured,
    "capture-required executor did not select CUDA graph mode",
  )
  executor.launch_captured()
  executor.record_completion()
  let mut complete = false
  let mut polls = 0
  while !complete && polls < 1000000 {
    complete = executor.poll_completion()
    polls += 1
  }
  require(
    complete, "ordered AOT executor did not complete within the poll bound",
  )
  require_add_one(allocation.copy_to_host(16L), "captured AOT graph launch")
}

///|
async fn main {
  let args = @env.args()
  require(
    args.length() == 3 || args.length() == 4,
    "usage: physical_cuda PTX_PATH CYCLES [RESIDUAL_CUBIN_PATH]",
  )
  let cycles = @string.parse_int(args[2], base=10) catch {
    _ => raise PhysicalCudaFailure("CYCLES must be an integer")
  }
  require(cycles > 0 && cycles <= 10000, "CYCLES must be in 1..10000")
  let image = @fs.read_file(args[1]).binary()
  let residual_image = if args.length() == 4 {
    Some(@fs.read_file(args[3]).binary())
  } else {
    None
  }
  let inventory = @cuda.probe()
  require(
    inventory.availability() == Available,
    "CUDA driver inventory is unavailable",
  )
  require(inventory.device_count() >= 1, "no CUDA device was discovered")
  require(inventory.cublas_lt_available(), "matrix library is unavailable")
  let device = @cuda.device_info(0)
  require(device.supports_bf16(), "CUDA device 0 does not support BF16")
  println(
    "physical_cuda device=0 name=\{device.name()} compute=\{device.compute_major()}.\{device.compute_minor()} memory=\{device.total_memory_bytes()} driver=\{inventory.driver_version()}",
  )
  if inventory.device_count() > 1 {
    let second = @cuda.device_info(1)
    let forward = @cuda.device_can_access_peer(0, 1)
    let reverse = @cuda.device_can_access_peer(1, 0)
    println(
      "physical_cuda peer device=1 name=\{second.name()} compute=\{second.compute_major()}.\{second.compute_minor()} bf16=\{second.supports_bf16()} access_0_1=\{forward} access_1_0=\{reverse}",
    )
  }
  for cycle in 0..<cycles {
    run_transfer_and_gemm_cycle(0, cycle)
    run_aot_cycle(0, image)
    if residual_image is Some(residual) {
      run_residual_aot_cycle(0, residual)
    }
  }
  println(
    "physical_cuda outcome=pass cycles=\{cycles} graph_capture=required-selected residual_aot=\{residual_image is Some(_)} resources=closed",
  )
}
