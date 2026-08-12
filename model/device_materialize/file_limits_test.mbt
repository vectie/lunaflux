///|
fn file_header_limits() -> @safetensors.HeaderLimits raise {
  @safetensors.HeaderLimits::new(
    max_header_bytes=4096,
    max_tensor_count=32,
    max_rank=2,
    max_name_length=128,
    max_payload_bytes=8192L,
    max_materialized_tensor_bytes=8192L,
  )
}

///|
test "streaming file limits are explicit immutable bounds" {
  let limits = @device_materialize.DeviceWeightFileLimits::new(
    max_file_bytes=16384L,
    transfer_chunk_bytes=65536,
    safetensors=file_header_limits(),
  )
  assert_eq(limits.max_file_bytes(), 16384L)
  assert_eq(limits.transfer_chunk_bytes(), 65536)
  assert_eq(limits.safetensors(), file_header_limits())
}

///|
test "streaming file limits reject unbounded host chunks" {
  for chunk in [0, -1, 16 * 1024 * 1024 + 1] {
    try
      @device_materialize.DeviceWeightFileLimits::new(
        max_file_bytes=16384L,
        transfer_chunk_bytes=chunk,
        safetensors=file_header_limits(),
      )
    catch {
      @device_materialize.DeviceWeightError::InvalidFileLimit(
        TransferChunkBytes
      ) => ()
      _ => fail("unexpected chunk limit error")
    } noraise {
      _ => fail("expected invalid chunk limit")
    }
  }
  try
    @device_materialize.DeviceWeightFileLimits::new(
      max_file_bytes=8L,
      transfer_chunk_bytes=4096,
      safetensors=file_header_limits(),
    )
  catch {
    @device_materialize.DeviceWeightError::InvalidFileLimit(FileBytes) => ()
    _ => fail("unexpected file limit error")
  } noraise {
    _ => fail("expected invalid file limit")
  }
}
