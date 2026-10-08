///|
fn projection_fixture_value(index : Int, seed : Int) -> Double {
  ((index * 13 + seed * 7) % 17 - 8).to_double() / 8.0
}

///|
fn projection_fixture_bytes(count : Int, seed : Int) -> Bytes {
  let bytes = FixedArray::make(count * 2, b'\x00')
  for i in 0..<count {
    let bits = projection_fixture_value(i, seed)
      .to_float()
      .reinterpret_as_uint() >>
      16
    bytes[i * 2] = bits.to_byte()
    bytes[i * 2 + 1] = (bits >> 8).to_byte()
  }
  Bytes::from_fixedarray(bytes)
}

///|
/// Real capture/replay, guarded tail and deterministic release. The fixture
/// function is never launched: explicit projection records replace it.
async test "physical ordered vendor projection matches FP64 and preserves tail" {
  guard @env.get_env_var("LUNAFLUX_VENDOR_PROJECTION_FIXTURE") is Some(path) else {
    return
  }
  let context = @device.open_context(0)
  defer (try! context.close())
  let stream = context.create_stream()
  defer (try! stream.close())
  let loaded = context.load_module(@fs.read_file(path).binary())
  defer (try! loaded.close())
  let function = loaded.load_function(b"effect")
  defer (try! function.close())
  let dims = @device.KernelLaunchDimensions::new(
    grid_x=1,
    grid_y=1,
    grid_z=1,
    block_x=1,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
  for
    (m, n, k) in [
      (17, 33, 65),
      (128, 128, 256),
      (2048, 1024, 2048),
      (2048, 1024, 3072),
    ] {
    let x = context.allocate((m * k * 2).to_int64())
    defer (try! x.close())
    let w = context.allocate((n * k * 2).to_int64())
    defer (try! w.close())
    let y = context.allocate((m * n * 2 + 32).to_int64())
    defer (try! y.close())
    x.copy_from_host(
      projection_fixture_bytes(m * k, 1),
      0L,
      0L,
      (m * k * 2).to_int64(),
    )
    w.copy_from_host(
      projection_fixture_bytes(n * k, 2),
      0L,
      0L,
      (n * k * 2).to_int64(),
    )
    y.copy_from_host(
      Bytes::make(m * n * 2 + 32, b'\x55'),
      0L,
      0L,
      (m * n * 2 + 32).to_int64(),
    )
    let args = @device.KernelArguments::new(
      [x, w, y].map(a => {
        @device.KernelArgument::device_region(
          a,
          offset=0L,
          byte_count=a.byte_count(),
          alignment=16L,
        )
      }),
    )
    let graph = context.create_ordered_kernel_executor(
      stream,
      [@device.OrderedKernelLaunch::new(function, dims, args)],
      policy=OrderedKernelCaptureRequired,
      projections=[
        @device.OrderedBf16Projection::new(
          launch_index=0,
          rows=m,
          columns=n,
          inner=k,
          input_argument=0,
          weight_argument=1,
          output_argument=2,
        ),
      ],
    )
    defer (try! graph.close())
    for _ in 0..<3 {
      graph.launch_captured()
      graph.record_completion()
      graph.wait_completion()
      let bytes = y.copy_to_host(0L, y.byte_count())
      // Exhaustive on tail shapes; deterministic strided samples on the two
      // production geometries keep the CPU oracle bounded under sanitizers.
      let stride = if m * n > 65536 { 2053 } else { 1 }
      for pos = 0; pos < m * n; pos = pos + stride {
        let row = pos / n
        let column = pos % n
        let mut expected = 0.0
        for inner in 0..<k {
          expected += projection_fixture_value(row * k + inner, 1) *
            projection_fixture_value(column * k + inner, 2)
        }
        let bits = (
            bytes[pos * 2].to_uint() | (bytes[pos * 2 + 1].to_uint() << 8)
          ) <<
          16
        let actual = bits.reinterpret_as_float().to_double()
        assert_true((actual - expected).abs() <= 0.004 * expected.abs() + 0.02)
      }
      for pos in (m * n * 2)..<(m * n * 2 + 32) {
        assert_eq(bytes[pos], b'\x55')
      }
      graph.reset()
    }
  }
  println(
    "vendor_projection_physical=passed oracle=fp64 replay=3 tail=preserved explicit_release=true",
  )
}
