///|
async test "packed checkpoint column slices copy only selected payload and scales" {
  @async.with_task_group() <| group => {
    let directory = @fs.realpath(@fs.tmpdir(prefix="lunaflux-shard-transfer"))
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let header = @utf8.encode(
      (
        #|{"w":{"dtype":"U8","shape":[4,16],"data_offsets":[0,64]},"s":{"dtype":"F8_E4M3","shape":[4,2],"data_offsets":[64,72]},"g":{"dtype":"F32","shape":[],"data_offsets":[72,76]}}
      ),
    )
    let file : Array[Byte] = []
    for i in 0..<8 {
      file.push((header.length().to_uint64() >> (8 * i)).to_byte())
    }
    file.append(header.to_array())
    for i in 0..<76 {
      file.push(i.to_byte())
    }
    @fs.write_file(
      directory + "/weights",
      Bytes::from_array(file),
      create_mode=CreateNew,
    )
    let root = @approved_fs.ApprovedRoot::open_absolute(directory)
    defer root.close()
    let limits = @streaming.StreamingSafetensorsLimits::new(
      maximum_shards=1,
      maximum_file_bytes=4096L,
      maximum_header_bytes=2048,
      maximum_tensors_per_shard=3,
      maximum_total_tensors=3,
      maximum_rank=2,
      maximum_name_bytes=32,
      maximum_dtype_specs=3,
      maximum_copy_requests=10,
      maximum_copy_bytes=1024L,
    )
    let checkpoint = @precision_materialize.PackedCheckpoint::new(
      @streaming.inspect_shards(
        root,
        [
          @streaming.StreamingSafetensorsShardInput::new(
            locator=@approved_fs.ApprovedRelativeLocator::new("weights"),
            // A declared inventory label only. No payload digest is computed.
            expected_digest=@streaming.StreamingSafetensorsFileDigest::from_sha256(
              "0000000000000000000000000000000000000000000000000000000000000000",
            ),
          ),
        ],
        dtype_specs=[
          @streaming.StreamingSafetensorsDTypeSpec::float32(),
          @streaming.StreamingSafetensorsDTypeSpec::opaque_one_byte("U8"),
          @streaming.StreamingSafetensorsDTypeSpec::opaque_one_byte("F8_E4M3"),
        ],
        limits~,
      ),
    )
    let rep = @format.Representation::new(format=NvFp4, rows=4, columns=32)
    let whole = checkpoint.bind_weight(
      representation=rep,
      payload_name="w",
      scales_name="s",
      global_scale_name="g",
    )
    // Use the public compiler-owned view consumed by the startup loader.
    let view = @precision.PackedMatrixSlice::new(
      rep,
      row_start=0,
      rows=4,
      column_start=16,
      columns=16,
    )
    let shard = whole.slice(view)
    assert_eq(shard.buffer_bytes(), 40L)
    for scratch_size in [1, 3, 7, 16, 27, 64, 128] {
      let result = FixedArray::make(40, b'?')
      let mut copied = 0
      checkpoint.transfer_weights(
        root,
        [shard],
        scratch=FixedArray::make(scratch_size, b'?'),
        limits~,
        consume=(_, offset, chunk, count) => {
          assert_true(count <= scratch_size) catch {
            _ => return false
          }
          copied += count
          for i in 0..<count {
            result[offset.to_int() + i] = chunk[i]
          }
          true
        },
      )
      assert_eq(copied, 40)
      for row in 0..<4 {
        for column in 0..<8 {
          assert_eq(result[row * 8 + column], (row * 16 + 8 + column).to_byte())
        }
        assert_eq(result[32 + row], (65 + row * 2).to_byte())
      }
      for i in 0..<4 {
        assert_eq(result[36 + i], (72 + i).to_byte())
      }
    }
    try shard.slice(view) catch {
      _ => ()
    } noraise {
      _ => fail("ambiguous nested slice")
    }
  }
}
