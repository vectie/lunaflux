///|
async test "packed checkpoint transfers preserve NVFP4 MXFP4 and BF16 without expansion" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-packed-weights")
    let canonical = @fs.realpath(directory)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(canonical, recursive=true))
    }
    let header = @utf8.encode(
      (
        #|{"w":{"dtype":"U8","shape":[2,8],"data_offsets":[0,16]},"s":{"dtype":"F8_E4M3","shape":[2,1],"data_offsets":[16,18]},"g":{"dtype":"F32","shape":[],"data_offsets":[18,22]},"bf":{"dtype":"BF16","shape":[1,2],"data_offsets":[22,26]},"mx":{"dtype":"I8","shape":[2,16],"data_offsets":[26,58]},"ms":{"dtype":"F8_E8M0","shape":[2,1],"data_offsets":[58,60]},"bias":{"dtype":"F32","shape":[1],"data_offsets":[60,64]},"conv":{"dtype":"BF16","shape":[2,1,4],"data_offsets":[64,80]}}
      ),
    )
    let payload = FixedArray::make(80, b'\x21')
    payload[16] = b'\x38'
    payload[17] = b'\x38'
    for i, byte in b"\x00\x00\x00\x40\x80\x3f\x00\x40" {
      payload[18 + i] = byte
    }
    payload[58] = b'\x7f'
    payload[59] = b'\x7f'
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
    @fs.write_file(canonical + "/weights", bytes, create_mode=CreateNew)
    let root = @approved_fs.ApprovedRoot::open_absolute(canonical)
    defer root.close()
    let limits = @streaming.StreamingSafetensorsLimits::new(
      maximum_shards=2,
      maximum_file_bytes=65536L,
      maximum_header_bytes=4096,
      maximum_tensors_per_shard=10,
      maximum_total_tensors=10,
      maximum_rank=3,
      maximum_name_bytes=128,
      maximum_dtype_specs=6,
      authentication_chunk_bytes=32,
      maximum_copy_requests=10,
      maximum_copy_bytes=1024L,
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
        @streaming.StreamingSafetensorsDTypeSpec::opaque_one_byte("U8"),
        @streaming.StreamingSafetensorsDTypeSpec::opaque_one_byte("F8_E4M3"),
        @streaming.StreamingSafetensorsDTypeSpec::opaque_one_byte("I8"),
        @streaming.StreamingSafetensorsDTypeSpec::opaque_one_byte("F8_E8M0"),
      ],
      limits~,
    )
    let checkpoint = @precision_materialize.PackedCheckpoint::new(inspection)
    let dense = @format.Representation::new(format=Bf16, rows=2, columns=4)
    let conv = checkpoint.bind_dense_tensor(
      representation=dense,
      payload_name="conv",
      source_shape=[2L, 1L, 4L],
    )
    assert_eq(conv.buffer_bytes(), 16L)
    for shape in [[2L, 4L, 1L], [2L, 4L], [2L, 1L, 3L], [0L, 1L, 4L]] {
      try
        checkpoint.bind_dense_tensor(
          representation=dense,
          payload_name="conv",
          source_shape=shape,
        )
      catch {
        TensorStorageMismatch("conv") => ()
        _ => fail("wrong dense shape error")
      } noraise {
        _ => fail("rank/layout mismatch accepted")
      }
    }
    let convolution_bytes = FixedArray::make(16, b'?')
    checkpoint.transfer_weights(
      root,
      [conv],
      scratch=FixedArray::make(3, b'?'),
      limits~,
      consume=(_, offset, chunk, count) => {
        for i in 0..<count {
          convolution_bytes[offset.to_int() + i] = chunk[i]
        }
        true
      },
    )
    assert_eq(
      Bytes::from_array(convolution_bytes),
      Bytes::from_array(payload[64:80]),
    )
    let vector = checkpoint.bind_vector(
      representation=@format.Representation::new(format=F32, rows=1, columns=1),
      payload_name="bias",
    )
    assert_eq(vector.buffer_bytes(), 4L)
    let escaped : Ref[@precision_materialize.PackedCheckpoint?] = Ref(None)
    checkpoint.with_transfer_session(root, limits, scoped => {
      escaped.val = Some(scoped)
      for _ in 0..<2 {
        scoped.transfer_weights(
          root,
          [vector],
          scratch=FixedArray::make(3, b'?'),
          limits~,
          consume=(_, _, _, _) => true,
        )
      }
    })
    try
      escaped.val
      .unwrap()
      .transfer_weights(
        root,
        [vector],
        scratch=FixedArray::make(3, b'?'),
        limits~,
        consume=(_, _, _, _) => true,
      )
    catch {
      @precision_materialize.PackedCheckpointError::Streaming(
        @streaming.StreamingSafetensorsError::InvalidLimit
      ) => ()
      e => raise e
    } noraise {
      _ => fail("checkpoint retained live file effects after startup scope")
    }
    let vector_bytes = FixedArray::make(4, b'?')
    checkpoint.transfer_weights(
      root,
      [vector],
      scratch=FixedArray::make(3, b'?'),
      limits~,
      consume=(_, offset, chunk, count) => {
        for i in 0..<count {
          vector_bytes[offset.to_int() + i] = chunk[i]
        }
        true
      },
    )
    assert_eq(Bytes::from_array(vector_bytes), b"\x21\x21\x21\x21")
    try
      checkpoint.bind_weight(
        representation=@format.Representation::new(
          format=F32,
          rows=1,
          columns=1,
        ),
        payload_name="bias",
      )
    catch {
      @precision_materialize.PackedCheckpointError::TensorStorageMismatch(
        "bias"
      ) => ()
      _ => fail("wrong rank error")
    } noraise {
      _ => fail("rank-one vector disguised as matrix")
    }
    let nv = checkpoint.bind_weight(
      representation=@format.Representation::new(
        format=NvFp4,
        rows=2,
        columns=16,
      ),
      payload_name="w",
      scales_name="s",
      global_scale_name="g",
    )
    let bf = checkpoint.bind_weight(
      representation=@format.Representation::new(format=Bf16, rows=1, columns=2),
      payload_name="bf",
    )
    let mx = checkpoint.bind_weight(
      representation=@format.Representation::new(
        format=MxFp4,
        rows=2,
        columns=32,
      ),
      payload_name="mx",
      scales_name="ms",
    )
    assert_eq(nv.buffer_bytes(), 24L)
    assert_eq(nv.scales_offset(), Some(16L))
    assert_eq(nv.global_scale_offset(), Some(20L))
    assert_eq(bf.buffer_bytes(), 4L)
    assert_eq(bf.scales_offset(), None)
    assert_eq(mx.buffer_bytes(), 34L)
    assert_eq(mx.global_scale_offset(), None)
    let destinations = [
      FixedArray::make(24, b'!'),
      FixedArray::make(4, b'!'),
      FixedArray::make(34, b'!'),
    ]
    let mut largest_chunk = 0
    checkpoint.transfer_weights(
      root,
      [nv, bf, mx],
      scratch=FixedArray::make(3, b'?'),
      limits~,
      consume=(destination, offset, chunk, count) => {
        largest_chunk = largest_chunk.max(count)
        for i in 0..<count {
          destinations[destination][offset.to_int() + i] = chunk[i]
        }
        true
      },
    )
    assert_eq(largest_chunk, 3)
    assert_eq(
      Bytes::from_array(destinations[0][:16]),
      Bytes::from_array(payload[:16]),
    )
    assert_eq(Bytes::from_array(destinations[0][16:18]), b"\x38\x38")
    assert_eq(Bytes::from_array(destinations[0][18:20]), b"!!")
    assert_eq(Bytes::from_array(destinations[0][20:24]), b"\x00\x00\x00\x40")
    assert_eq(Bytes::from_array(destinations[1]), b"\x80\x3f\x00\x40")
    assert_eq(
      Bytes::from_array(destinations[2]),
      Bytes::from_array(payload[26:60]),
    )
    let decoded = @format.PackedTensor::new(
      nv.representation(),
      payload=Bytes::from_array(destinations[0][:16]),
      scales=Bytes::from_array(destinations[0][16:18]),
      global_scale=2.0F,
    )
    assert_eq(decoded.decode(0, 0), 1.0F)
    assert_eq(decoded.decode(0, 1), 2.0F)
    assert_eq(decoded.decode(1, 15), 2.0F)
    // Several physical planes share one expert bank without touching padding.
    let bank = FixedArray::make(80, b'!')
    checkpoint.transfer_weights_to(
      root,
      [nv, bf, mx],
      destinations=[(0, 0L), (0, 32L), (0, 40L)],
      destination_bytes=[80L],
      scratch=FixedArray::make(3, b'?'),
      limits~,
      consume=(_, offset, chunk, count) => {
        for i in 0..<count {
          bank[offset.to_int() + i] = chunk[i]
        }
        true
      },
    )
    assert_eq(Bytes::from_array(bank[:24]), Bytes::from_array(destinations[0]))
    assert_eq(Bytes::from_array(bank[24:32]), b"!!!!!!!!")
    assert_eq(
      Bytes::from_array(bank[32:36]),
      Bytes::from_array(destinations[1]),
    )
    assert_eq(Bytes::from_array(bank[36:40]), b"!!!!")
    assert_eq(
      Bytes::from_array(bank[40:74]),
      Bytes::from_array(destinations[2]),
    )
    try
      checkpoint.bind_weight(
        representation=nv.representation(),
        payload_name="w",
        scales_name="s",
      )
    catch {
      InvalidPlaneNames => ()
      _ => fail("wrong missing global scale error")
    } noraise {
      _ => fail("NVFP4 lost its global scale")
    }
    let wrong_shape = @format.Representation::new(
      format=NvFp4,
      rows=1,
      columns=32,
    )
    try
      checkpoint.bind_weight(
        representation=wrong_shape,
        payload_name="w",
        scales_name="s",
        global_scale_name="g",
      )
    catch {
      TensorStorageMismatch("w") => ()
      _ => fail("wrong packed shape error")
    } noraise {
      _ => fail("equal byte count hid a shape mismatch")
    }
    let mut calls = 0
    try
      checkpoint.transfer_weights(
        root,
        [nv],
        scratch=FixedArray::make(3, b'?'),
        limits~,
        consume=(_, _, _, _) => {
          calls += 1
          false
        },
      )
    catch {
      Streaming(VisitorRejected(0)) => ()
      _ => fail("wrong failed upload error")
    } noraise {
      _ => fail("failed upload continued")
    }
    assert_eq(calls, 1)
  }
}
