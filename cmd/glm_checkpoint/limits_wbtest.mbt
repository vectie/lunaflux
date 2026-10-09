///|
async test "checkpoint accepts concentrated 37152 scalar scales before binding" {
  let count = 37152
  let header = StringBuilder()
  header <+ "{"
  for ordinal in 0..<count {
    if ordinal > 0 {
      header <+ ","
    }
    let start = ordinal * 4
    header <+
      "\"scale_\{ordinal}\":{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[\{start},\{start + 4}]}"
  }
  header <+ "}"
  let encoded = @utf8.encode(header.to_string())
  let bytes = Array::make(8 + encoded.length() + count * 4, b'\x00')
  for byte in 0..<8 {
    bytes[byte] = ((encoded.length().to_uint64() >> (byte * 8)) & 255UL).to_byte()
  }
  for ordinal, byte in encoded {
    bytes[8 + ordinal] = byte
  }
  let file_bytes = Bytes::from_array(bytes)
  let directory = @fs.tmpdir(prefix="lunaflux-concentrated-scales")
  defer @fs.rmdir(directory, recursive=true)
  let canonical = @fs.realpath(directory)
  @fs.write_file(
    canonical + "/scales.safetensors",
    file_bytes,
    create_mode=CreateNew,
  )
  let root = @approved_fs.ApprovedRoot::open_absolute(canonical)
  defer root.close()
  let inspection = @streaming.inspect_shards(
    root,
    [
      @streaming.StreamingSafetensorsShardInput::new(
        locator=@approved_fs.ApprovedRelativeLocator::new("scales.safetensors"),
        expected_digest=@streaming.StreamingSafetensorsFileDigest::from_sha256(
          @crypto.bytes_to_hex_string(@crypto.sha256(file_bytes)),
        ),
      ),
    ],
    dtype_specs=[@streaming.StreamingSafetensorsDTypeSpec::float32()],
    limits=checkpoint_limits(1048576L),
  )
  assert_eq(inspection.total_tensor_count(), count)
}
