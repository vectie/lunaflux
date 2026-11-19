///|
async test "startup plan streams tensor and sliced bank bytes through one odd-size scratch" {
  @async.with_task_group() <| group => {
    let directory = @fs.realpath(@fs.tmpdir(prefix="lunaflux-v47-weight-copy"))
    group.add_defer(() => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    })
    let header_text =
      #|{"x":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},"y":{"dtype":"BF16","shape":[2],"data_offsets":[8,12]}}
    let header = @utf8.encode(header_text)
    let file = @buffer.Buffer()
    file.write_uint64_le(header.length().to_uint64())
    file.write_bytes(header)
    file.write_bytes(b"\x00\x80\xc1\x7f\x00\x00\x81\xff\x80\x3f\x80\xbf")
    @fs.write_file(
      directory + "/weights",
      file.to_bytes(),
      create_mode=CreateNew,
    )
    let root = @approved_fs.ApprovedRoot::open_absolute(directory)
    defer root.close()
    let limits = @streaming.StreamingSafetensorsLimits::new(
      maximum_shards=1,
      maximum_file_bytes=4096L,
      maximum_header_bytes=1024,
      maximum_tensors_per_shard=2,
      maximum_total_tensors=2,
      maximum_rank=2,
      maximum_name_bytes=32,
      maximum_dtype_specs=1,
      maximum_copy_requests=4,
      maximum_copy_bytes=32L,
    )
    let inspection = @streaming.inspect_shards(
      root,
      [
        @streaming.StreamingSafetensorsShardInput::new(
          locator=@approved_fs.ApprovedRelativeLocator::new("weights"),
          expected_digest=@streaming.StreamingSafetensorsFileDigest::from_sha256(
            "b".repeat(64),
          ),
        ),
      ],
      dtype_specs=[@streaming.StreamingSafetensorsDTypeSpec::bfloat16()],
      limits~,
    )
    let checkpoint = @checkpoint.PackedCheckpoint::new(inspection)
    let x_rep = @format.Representation::new(format=Bf16, rows=2, columns=2)
    let x = checkpoint.bind_dense_tensor(
      representation=x_rep,
      payload_name="x",
      source_shape=[2L, 2L],
    )
    let y = checkpoint.bind_vector(
      representation=@format.Representation::new(format=Bf16, rows=1, columns=2),
      payload_name="y",
    )
    let first = x.slice(
      @precision.PackedMatrixSlice::new(
        x_rep,
        row_start=0,
        rows=1,
        column_start=0,
        columns=2,
      ),
    )
    let second = x.slice(
      @precision.PackedMatrixSlice::new(
        x_rep,
        row_start=1,
        rows=1,
        column_start=0,
        columns=2,
      ),
    )
    // Private transport fixture: small arbitrary tensors exercise the same
    // transfer path; this is not an alternate selected-model geometry.
    let plan : Plan = {
      checkpoint,
      allocations: [
        {
          label: "y",
          sources: ["y"],
          bytes: 4L,
          weights: [y],
          destinations: [(0, 0L)],
        },
        {
          label: "slices",
          sources: ["x"],
          bytes: 12L,
          weights: [second, first],
          destinations: [(0, 0L), (0, 8L)],
        },
      ],
      layers: [],
      text: { embedding: 0, final_norm: 0, language_head: 1, },
      bytes: 16L,
    }
    let outputs = [FixedArray::make(4, b'!'), FixedArray::make(12, b'!')]
    let scratch = FixedArray::make(3, b'\x00')
    let mut chunks = 0
    plan.transfer(root, scratch~, limits~, consume=(
      ordinal,
      offset,
      source,
      count,
    ) => {
      if count > 3 {
        return false
      }
      for i in 0..<count {
        outputs[ordinal][offset.to_int() + i] = source[i]
      }
      chunks += 1
      true
    })
    assert_eq(Bytes::from_array(outputs[0]), b"\x80\x3f\x80\xbf")
    assert_eq(
      Bytes::from_array(outputs[1]),
      b"\x00\x00\x81\xff!!!!\x00\x80\xc1\x7f",
    )
    assert_eq(chunks, 6)
    let rejected = try
      plan.transfer(root, scratch~, limits~, consume=(_, _, _, _) => false)
    catch {
      _ => true
    } noraise {
      _ => false
    }
    assert_true(rejected)
    // The failed scoped session has closed its handles; a new startup can run.
    plan.transfer(root, scratch~, limits~, consume=(_, _, _, _) => true)
  }
}
