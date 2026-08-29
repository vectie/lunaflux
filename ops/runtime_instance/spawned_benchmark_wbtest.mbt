///|
fn spawned_benchmark_contract_for_test() -> RuntimeSpawnedServingValidationContract raise {
  let limits = spawned_serving_wb_limits()
  let framed = @framed_wire.FramedWireLimits::new(limits, max_frame_bytes=4096)
  let encoded = @framed_wire.RequestFrameBuffer::new(framed).encode(
    spawned_serving_wb_request(),
  )
  let source = FixedArray::make(encoded.length(), b'\x00')
  ignore(encoded.copy_to(source, destination_offset=0))
  RuntimeSpawnedServingValidationContract::new(
    source,
    spawned_validation_wb_identity(),
    1031,
    2185,
    1,
    8,
    8,
    limits,
    4096,
    1024,
    5000,
  )
}

///|
test "spawned benchmark contract fixes LunaFlux trial coordinates" {
  let serving = spawned_benchmark_contract_for_test()
  ignore(RuntimeSpawnedBenchmarkContract::new(serving, Latency, 1, 0))
  for coordinates in [(0, 0), (4, 0), (1, -1), (1, 3)] {
    let (ordinal, order_position) = coordinates
    try
      RuntimeSpawnedBenchmarkContract::new(
        serving,
        Latency,
        ordinal,
        order_position,
      )
    catch {
      RuntimeSpawnedBenchmarkFailed(SpawnedBenchmarkContract) => ()
      _ => abort("unexpected spawned benchmark contract error")
    } noraise {
      _ => abort("invalid spawned benchmark coordinate was accepted")
    }
  }
}
