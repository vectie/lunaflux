///|
async test "GLM official-sized expert partition streams all three packed projections" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-glm-packed-expert")
    let canonical = @fs.realpath(directory)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(canonical, recursive=true))
    }
    let projections = @glm53_weights.nvfp4_expert_weights(layer_index=10, expert_indices=[
      287,
    ])
    let payloads : Array[(Int64, Bytes)] = []
    let entries : Array[String] = []
    let mut cursor = 0L
    for i, weight in projections {
      let rep = weight.representation()
      let (rows, _) = rep.shape()
      let (_, scale_columns) = rep.scale_shape()
      let payload_bytes = rep.payload_bytes()
      let scale_bytes = rep.scale_bytes()
      let payload_name = weight.payload_name()
      let scale_name = weight.scales_name()
      let global_name = weight.global_scale_name()
      entries.push(
        "\"\{payload_name}\":{\"dtype\":\"U8\",\"shape\":[\{rows},\{rep.row_bytes()}],\"data_offsets\":[\{cursor},\{cursor + payload_bytes}]}",
      )
      payloads.push(
        (
          cursor,
          Bytes::make(payload_bytes.to_int(), (0x21 + 0x22 * i).to_byte()),
        ),
      )
      cursor += payload_bytes
      entries.push(
        "\"\{scale_name}\":{\"dtype\":\"F8_E4M3\",\"shape\":[\{rows},\{scale_columns}],\"data_offsets\":[\{cursor},\{cursor + scale_bytes}]}",
      )
      payloads.push((cursor, Bytes::make(scale_bytes.to_int(), b'\x38')))
      cursor += scale_bytes
      entries.push(
        "\"\{global_name}\":{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[\{cursor},\{cursor + 4L}]}",
      )
      payloads.push((cursor, b"\x00\x00\x80\x3f"))
      cursor += 4L
    }
    let header = @utf8.encode("{" + entries.join(",") + "}")
    let file = FixedArray::make(8 + header.length() + cursor.to_int(), b'\x00')
    for i in 0..<8 {
      file[i] = (header.length().to_uint64() >> (8 * i)).to_byte()
    }
    for i, byte in header {
      file[8 + i] = byte
    }
    for (offset, bytes) in payloads {
      for i, byte in bytes {
        file[8 + header.length() + offset.to_int() + i] = byte
      }
    }
    let bytes = Bytes::from_array(file)
    @fs.write_file(canonical + "/expert", bytes, create_mode=CreateNew)
    let root = @approved_fs.ApprovedRoot::open_absolute(canonical)
    defer root.close()
    let limits = @streaming_safetensors.StreamingSafetensorsLimits::new(
      maximum_shards=1,
      maximum_file_bytes=16777216L,
      maximum_header_bytes=8192,
      maximum_tensors_per_shard=9,
      maximum_total_tensors=9,
      maximum_rank=2,
      maximum_name_bytes=256,
      maximum_dtype_specs=3,
      maximum_copy_requests=9,
      maximum_copy_bytes=16777216L,
    )
    let inspection = @streaming_safetensors.inspect_shards(
      root,
      [
        @streaming_safetensors.StreamingSafetensorsShardInput::new(
          locator=@approved_fs.ApprovedRelativeLocator::new("expert"),
          expected_digest=@streaming_safetensors.StreamingSafetensorsFileDigest::from_sha256(
            @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
          ),
        ),
      ],
      dtype_specs=[
        @streaming_safetensors.StreamingSafetensorsDTypeSpec::float32(),
        @streaming_safetensors.StreamingSafetensorsDTypeSpec::opaque_one_byte(
          "U8",
        ),
        @streaming_safetensors.StreamingSafetensorsDTypeSpec::opaque_one_byte(
          "F8_E4M3",
        ),
      ],
      limits~,
    )
    let checkpoint = @precision_materialize.PackedCheckpoint::new(inspection)
    let weights = @glm53_packed_expert_upload.bind_experts(
      checkpoint,
      layer_index=10,
      expert_indices=[287],
    )
    let output = weights.map(weight => {
      FixedArray::make(weight.buffer_bytes().to_int(), b'!')
    })
    let mut maximum_chunk = 0
    let mut copied = 0L
    checkpoint.transfer_weights(
      root,
      weights,
      scratch=FixedArray::make(4093, b'?'),
      limits~,
      consume=(destination, offset, buffer, count) => {
        maximum_chunk = maximum_chunk.max(count)
        copied += count.to_int64()
        for i in 0..<count {
          output[destination][offset.to_int() + i] = buffer[i]
        }
        true
      },
    )
    assert_eq(weights.length(), 3)
    assert_eq(maximum_chunk, 4093)
    assert_eq(copied, 14155788L)
    for i, weight in weights {
      assert_eq(weight.buffer_bytes(), 4718596L)
      let (rows, columns) = weight.representation().shape()
      let tensor = @precision_format.PackedTensor::new(
        weight.representation(),
        payload=Bytes::from_array(output[i][:4194304]),
        scales=Bytes::from_array(output[i][4194304:4718592]),
      )
      let expected = [0.5F, 1.5F, 3.0F][i]
      assert_eq(tensor.decode(0, 0), expected)
      assert_eq(tensor.decode(rows - 1, columns - 2), expected)
      assert_eq(
        Bytes::from_array(output[i][4718592:4718596]),
        b"\x00\x00\x80\x3f",
      )
    }
  }
}
