///|
fn ingress_launch(executor : @device.OrderedKernelExecutor) -> Unit raise {
  executor.launch_captured()
  executor.record_completion()
  executor.wait_completion()
  executor.reset()
}

///|
fn ingress_readback(
  allocations : ArrayView[@device.Allocation],
) -> Array[Bytes] raise {
  [
    allocations[12].copy_to_host(0L, 256L * 4096L * 2L),
    allocations[13].copy_to_host(0L, 33L * 8192L * 2L),
    allocations[14].copy_to_host(0L, 33L * 8192L * 2L),
  ]
}

///|
fn ingress_run_pair(
  old_bytes : Bytes,
  new_bytes : Bytes,
  balance : Qwen3ProbeBalance,
  benchmark : Bool,
) -> Unit raise {
  let context = @device.open_context(0)
  defer (context.close() catch { _ => balance.cleanup += 1 })
  let identity = context.device_identity()
  assert_eq(identity.uuid(), "GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6")
  assert_eq(identity.pci(), "00000000:17:00.0")
  let stream = context.create_stream()
  defer (stream.close() catch { _ => balance.cleanup += 1 })
  let modules : Array[@device.Module] = []
  defer (for module_ in modules.rev() {
    module_.close() catch {
      _ => balance.cleanup += 1
    }
  })
  modules.push(context.load_module(old_bytes))
  modules.push(context.load_module(new_bytes))
  let functions : Array[@device.Function] = []
  defer (for function_ in functions.rev() {
    function_.close() catch {
      _ => balance.cleanup += 1
    }
  })
  for module_ in modules {
    functions.push(
      module_.load_function(
        b"lunaflux_fused_qwen_qkv_qknorm_rope_kvwrite_bf16_head128_production_v2",
      ),
    )
  }
  let input = ingress_dense(256, 1024, 2)
  let value_weight = ingress_dense(1024, 1024, 5)
  let output_sentinel = Bytes::from_array(
    FixedArray::make(256 * 4096 * 2, b'\x00')[:],
  )
  let cache_bytes = FixedArray::make(33 * 8192 * 2, b'\x00')
  for index in 0..<(33 * 8192) {
    qwen3_write_bf16(cache_bytes, index, 0.25F)
  }
  let cache_sentinel = Bytes::from_array(cache_bytes[:])
  let host = [
    ingress_i32([1, 0, 1, 1, 1]),
    ingress_i32(Array::makei(256, index => index + 1)),
    ingress_i32([0, 1]),
    ingress_i32([2]),
    ingress_i32([0, 1]),
    ingress_i32(Array::makei(33, index => 32 - index)),
    input,
    ingress_dense(2048, 1024, 3),
    ingress_dense(1024, 1024, 4),
    value_weight,
    qwen3_norm_weight(128, 1),
    qwen3_norm_weight(128, 3),
    output_sentinel,
    cache_sentinel,
    cache_sentinel,
  ]
  let allocations : Array[@device.Allocation] = []
  defer (for allocation in allocations.rev() {
    allocation.close() catch {
      _ => balance.cleanup += 1
    }
  })
  for bytes in host {
    let allocation = context.allocate(bytes.length().to_int64())
    allocations.push(allocation)
    allocation.copy_from_host(bytes, 0L, 0L, bytes.length().to_int64())
  }
  let arguments = @device.KernelArguments::new(
    Array::makei(allocations.length(), index => {
      qwen3_region(
        allocations[index],
        host[index].length(),
        if index < 6 {
          4L
        } else {
          16L
        },
      )
    }),
  )
  let dimensions = @device.KernelLaunchDimensions::new(
    grid_x=16,
    grid_y=32,
    grid_z=1,
    block_x=128,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
  let executors : Array[@device.OrderedKernelExecutor] = []
  let repetitions = if benchmark { 64 } else { 1 }
  defer (for executor in executors.rev() {
    executor.close() catch {
      _ => balance.cleanup += 1
    }
  })
  for function_ in functions {
    executors.push(
      context.create_ordered_kernel_executor(
        stream,
        Array::make(
          repetitions,
          @device.OrderedKernelLaunch::new(function_, dimensions, arguments),
        ),
        policy=OrderedKernelCaptureRequired,
      ),
    )
  }
  let start = context.create_event()
  defer (start.close() catch { _ => balance.cleanup += 1 })
  let finish = context.create_event()
  defer (finish.close() catch { _ => balance.cleanup += 1 })
  for tokens in [1, 2, 15, 16, 17, 32, 64, 128, 256] {
    let pages = (tokens + 1 + 7) / 8
    let updates = [
      (
        0,
        ingress_i32(
          if tokens == 1 {
            [0, 1, 1, tokens, pages]
          } else {
            [1, 0, 1, tokens, pages]
          },
        ),
      ),
      (2, ingress_i32([0, tokens])),
      (3, ingress_i32([tokens + 1])),
      (4, ingress_i32([0, pages])),
    ]
    for update in updates {
      let (index, bytes) = update
      allocations[index].copy_from_host(
        bytes,
        0L,
        0L,
        bytes.length().to_int64(),
      )
    }
    let results : Array[Array[Bytes]] = []
    for executor in executors {
      for index in 12..<15 {
        let bytes = host[index]
        allocations[index].copy_from_host(
          bytes,
          0L,
          0L,
          bytes.length().to_int64(),
        )
      }
      ingress_launch(executor)
      results.push(ingress_readback(allocations))
    }
    ingress_validate(results[1], results[0], input, value_weight, tokens)
    // Interleave old/new and reverse order each trial; captured repetition
    // amortizes host launch overhead. This is kernel-only, not serving speed.
    for trial in 0..<(if benchmark { 7 } else { 1 }) {
      for turn in 0..<2 {
        let index = (trial + turn) % 2
        let executor = executors[index]
        start.record(stream)
        executor.launch_captured()
        finish.record(stream)
        executor.record_completion()
        executor.wait_completion()
        executor.reset()
        let micros = start.elapsed_milliseconds(finish) *
          1000.0 /
          repetitions.to_double()
        println(
          "tokens=\{tokens} trial=\{trial} kernel=\{if index == 0 { "old" } else { "new" }} microseconds=\{micros}",
        )
      }
    }
    println(
      "tokens=\{tokens} dense_bit_exact=true sampled_v_oracle=true positioned_kv=true",
    )
  }
}

///|
async fn qwen3_ingress_benchmark(args : ArrayView[String]) -> Unit {
  qwen3_require(
    args.length() == 4,
    "usage: ingress-benchmark OLD_CUBIN NEW_CUBIN",
  )
  let old_bytes = @fs.read_file(args[2]).binary()
  let new_bytes = @fs.read_file(args[3]).binary()
  println(
    "old_sha256=\{qwen3_sha256(old_bytes)} new_sha256=\{qwen3_sha256(new_bytes)}",
  )
  let balance = qwen3_probe_balance()
  ingress_run_pair(
    old_bytes,
    new_bytes,
    balance,
    args[1] == "ingress-benchmark",
  )
  assert_eq(balance.cleanup, 0)
  println("outcome=ingress-microbenchmark-pass resources_closed=true")
}
