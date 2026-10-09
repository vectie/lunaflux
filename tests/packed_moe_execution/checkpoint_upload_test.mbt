///|
async test "checkpoint device owner uploads original dense rank and float control bytes" {
  @async.with_task_group() <| group => {
    let directory = @fs.realpath(@fs.tmpdir(prefix="lunaflux-dense-upload"))
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let header = @utf8.encode(
      (
        #|{"conv":{"dtype":"BF16","shape":[2,1,4],"data_offsets":[0,16]},"control":{"dtype":"F32","shape":[2],"data_offsets":[16,24]}}
      ),
    )
    let payload = b"\x80\x3f\x00\x40\x40\x40\x80\x40\xa0\x40\xc0\x40\xe0\x40\x00\x41\x00\x00\x80\x3f\x00\x00\x00\x40"
    let file : Array[Byte] = []
    for i in 0..<8 {
      file.push((header.length().to_uint64() >> (8 * i)).to_byte())
    }
    for byte in header {
      file.push(byte)
    }
    for byte in payload {
      file.push(byte)
    }
    let bytes = Bytes::from_array(file)
    @fs.write_file(directory + "/weights", bytes, create_mode=CreateNew)
    let root = @approved_fs.ApprovedRoot::open_absolute(directory)
    defer root.close()
    let limits = @streaming.StreamingSafetensorsLimits::new(
      maximum_shards=1,
      maximum_file_bytes=4096L,
      maximum_header_bytes=1024,
      maximum_tensors_per_shard=2,
      maximum_total_tensors=2,
      maximum_rank=3,
      maximum_name_bytes=64,
      maximum_dtype_specs=2,
      authentication_chunk_bytes=32,
      maximum_copy_requests=2,
      maximum_copy_bytes=24L,
    )
    let inspection = @streaming.inspect_shards(
      root,
      [
        @streaming.StreamingSafetensorsShardInput::new(
          locator=@approved_fs.ApprovedRelativeLocator::new("weights"),
          expected_digest=@streaming.StreamingSafetensorsFileDigest::from_sha256(
            @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
          ),
        ),
      ],
      dtype_specs=[
        @streaming.StreamingSafetensorsDTypeSpec::bfloat16(),
        @streaming.StreamingSafetensorsDTypeSpec::float32(),
      ],
      limits~,
    )
    let checkpoint = @materialize.PackedCheckpoint::new(inspection)
    let weights = [
      checkpoint.bind_dense_tensor(
        representation=@format.Representation::new(
          format=Bf16,
          rows=2,
          columns=4,
        ),
        payload_name="conv",
        source_shape=[2L, 1L, 4L],
      ),
      checkpoint.bind_vector(
        representation=@format.Representation::new(
          format=F32,
          rows=1,
          columns=2,
        ),
        payload_name="control",
      ),
    ]
    let baseline = live_resources()
    let context = @device.open_context(0)
    defer (context.close() catch { _ => () })
    let owner = @upload.CheckpointDeviceWeights::new()
    defer (owner.close() catch { _ => () })
    try
      owner.upload(
        context,
        checkpoint,
        root,
        weights,
        maximum_device_bytes=23L,
        chunk_bytes=3,
        limits~,
      )
    catch {
      @device.DeviceError::InvalidArgument(_) => ()
      e => raise e
    } noraise {
      _ => fail("underbudget upload accepted")
    }
    assert_eq(owner.device_bytes(), 0L)
    owner.upload(
      context,
      checkpoint,
      root,
      weights,
      maximum_device_bytes=24L,
      chunk_bytes=3,
      limits~,
    )
    assert_eq(owner.device_bytes(), 24L)
    let convolution = FixedArray::make(16, b'?')
    let controls = FixedArray::make(8, b'?')
    context.copy_to_fixed_host(owner.allocation(0), convolution, 0L, 0L, 16L)
    context.copy_to_fixed_host(owner.allocation(1), controls, 0L, 0L, 8L)
    assert_eq(Bytes::from_array(convolution), payload[:16].to_owned())
    assert_eq(Bytes::from_array(controls), payload[16:].to_owned())
    owner.close()
    owner.close()
    assert_eq(owner.device_bytes(), 0L)
    context.close()
    assert_eq(live_resources(), baseline)
  }
}
