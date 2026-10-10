///|
fn host_test_append_u64(output : Array[Byte], value : UInt64) -> Unit {
  for shift in 0..<8 {
    output.push(((value >> (shift * 8)) & 0xffUL).to_byte())
  }
}

///|
fn host_test_file(header : String, payload : BytesView) -> Bytes {
  let encoded = @utf8.encode(header)
  let output : Array[Byte] = []
  host_test_append_u64(output, encoded.length().to_uint64())
  for byte in encoded {
    output.push(byte)
  }
  for byte in payload {
    output.push(byte)
  }
  Bytes::from_array(output)
}

///|
fn host_test_digest(
  bytes : Bytes,
) -> @streaming_safetensors.StreamingSafetensorsFileDigest {
  try! @streaming_safetensors.StreamingSafetensorsFileDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
  )
}

///|
fn host_test_input(
  name : StringView,
  bytes : Bytes,
) -> @streaming_safetensors.StreamingSafetensorsShardInput {
  @streaming_safetensors.StreamingSafetensorsShardInput::new(
    locator=try! @approved_fs.ApprovedRelativeLocator::new(name),
    expected_digest=host_test_digest(bytes),
  )
}

///|
fn host_test_streaming_limits() -> @streaming_safetensors.StreamingSafetensorsLimits {
  try! @streaming_safetensors.StreamingSafetensorsLimits::new(
    maximum_shards=4,
    maximum_file_bytes=4096L,
    maximum_header_bytes=2048,
    maximum_tensors_per_shard=8,
    maximum_total_tensors=16,
    maximum_rank=4,
    maximum_name_bytes=256,
    maximum_dtype_specs=8,
    maximum_copy_requests=16,
    maximum_copy_bytes=1024L,
  )
}

///|
async test "tiny shards stream exact raw dtypes into bounded final arenas" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-deepseek-v4-host")
    let canonical = @fs.realpath(directory)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(canonical, recursive=true))
    }
    let first = host_test_file(
      (
        #|{"x":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}
      ),
      b"abcd",
    )
    let second = host_test_file(
      (
        #|{"y":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"z":{"dtype":"F8_E4M3","shape":[3],"data_offsets":[4,7]}}
      ),
      b"WXYZ123",
    )
    @fs.write_file(canonical + "/s0", first, create_mode=CreateNew)
    @fs.write_file(canonical + "/s1", second, create_mode=CreateNew)
    let root = @approved_fs.ApprovedRoot::open_absolute(canonical)
    let streaming_limits = host_test_streaming_limits()
    let inspection = @streaming_safetensors.inspect_shards(
      root,
      [host_test_input("s0", first), host_test_input("s1", second)],
      dtype_specs=deepseek_dtype_specs(),
      limits=streaming_limits,
    )
    let host_limits = try! DeepSeekV4HostMaterializationLimits::new(
      maximum_total_host_bytes=256L,
      maximum_arena_bytes=68,
      maximum_arenas=4,
      maximum_index_entries=16,
    )
    let sources = [
      { shard_ordinal: 0, tensor_ordinal: 0, byte_count: 4L, },
      { shard_ordinal: 1, tensor_ordinal: 0, byte_count: 4L, },
      { shard_ordinal: 1, tensor_ordinal: 1, byte_count: 3L, },
    ]
    let (layouts, arena_sizes, tensor_bytes, arena_bytes) = build_layout(
      sources, host_limits,
    )
    assert_eq(arena_sizes, ReadOnlyArray::from_array([68, 3]))
    assert_eq(tensor_bytes, 11L)
    assert_eq(arena_bytes, 71L)
    let owners = copy_layout_into_final_arenas(
      root, inspection, layouts, arena_sizes, streaming_limits,
    )
    guard owners[0] is Some(first_arena) && owners[1] is Some(second_arena) else {
      fail("final arenas missing")
    }
    assert_eq(Bytes::from_array(first_arena[0:4]), b"abcd")
    assert_eq(Bytes::from_array(first_arena[64:68]), b"WXYZ")
    assert_eq(Bytes::from_array(second_arena[:]), b"123")
    drop_arenas(owners)
    assert_eq(Bytes::from_array(first_arena[0:4]), b"abcd")
    assert_eq(owners[0], None)
    root.close()
  }
}

///|
async test "index membership rejects duplicate missing and wrong-shard claims" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-deepseek-v4-index")
    let canonical = @fs.realpath(directory)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(canonical, recursive=true))
    }
    let first = host_test_file(
      (
        #|{"x":{"dtype":"I64","shape":[1],"data_offsets":[0,8]}}
      ),
      b"12345678",
    )
    let second = host_test_file(
      (
        #|{"y":{"dtype":"I8","shape":[2],"data_offsets":[0,2]}}
      ),
      b"ab",
    )
    @fs.write_file(canonical + "/s0", first, create_mode=CreateNew)
    @fs.write_file(canonical + "/s1", second, create_mode=CreateNew)
    let root = @approved_fs.ApprovedRoot::open_absolute(canonical)
    let inspection = @streaming_safetensors.inspect_shards(
      root,
      [host_test_input("s0", first), host_test_input("s1", second)],
      dtype_specs=deepseek_dtype_specs(),
      limits=host_test_streaming_limits(),
    )
    validate_index_membership(
      inspection,
      [
        DeepSeekV4TensorIndexInput::new(tensor_name="x", shard_ordinal=0),
        DeepSeekV4TensorIndexInput::new(tensor_name="y", shard_ordinal=1),
      ],
      16,
    )
    try
      validate_index_membership(
        inspection,
        [
          DeepSeekV4TensorIndexInput::new(tensor_name="x", shard_ordinal=0),
          DeepSeekV4TensorIndexInput::new(tensor_name="x", shard_ordinal=1),
        ],
        16,
      )
    catch {
      DuplicateIndexEntry(0, 1) => ()
      _ => fail("unexpected duplicate-index rejection")
    } noraise {
      _ => fail("duplicate index entry was admitted")
    }
    try
      validate_index_membership(
        inspection,
        [
          DeepSeekV4TensorIndexInput::new(tensor_name="x", shard_ordinal=1),
          DeepSeekV4TensorIndexInput::new(tensor_name="y", shard_ordinal=1),
        ],
        16,
      )
    catch {
      IndexShardMismatch(0) => ()
      _ => fail("unexpected wrong-shard rejection")
    } noraise {
      _ => fail("wrong index shard was admitted")
    }
    try
      validate_index_membership(
        inspection,
        [DeepSeekV4TensorIndexInput::new(tensor_name="x", shard_ordinal=0)],
        16,
      )
    catch {
      InvalidIndexCount(expected=2, actual=1) => ()
      _ => fail("unexpected missing-index rejection")
    } noraise {
      _ => fail("missing index entry was admitted")
    }
    root.close()
  }
}

///|
test "layout bounds reject oversized tensors budgets and arena exhaustion" {
  let one_arena = try! DeepSeekV4HostMaterializationLimits::new(
    maximum_total_host_bytes=128L,
    maximum_arena_bytes=64,
    maximum_arenas=1,
    maximum_index_entries=4,
  )
  try
    build_layout(
      [{ shard_ordinal: 0, tensor_ordinal: 0, byte_count: 65L, }],
      one_arena,
    )
  catch {
    TensorTooLarge(0) => ()
    _ => fail("unexpected oversized-tensor rejection")
  } noraise {
    _ => fail("tensor larger than an arena was admitted")
  }
  try
    build_layout(
      [
        { shard_ordinal: 0, tensor_ordinal: 0, byte_count: 2L, },
        { shard_ordinal: 0, tensor_ordinal: 1, byte_count: 2L, },
      ],
      one_arena,
    )
  catch {
    ArenaCountExceeded => ()
    _ => fail("unexpected arena-exhaustion rejection")
  } noraise {
    _ => fail("layout exceeded its arena count")
  }
  let low_budget = try! DeepSeekV4HostMaterializationLimits::new(
    maximum_total_host_bytes=5L,
    maximum_arena_bytes=5,
    maximum_arenas=2,
    maximum_index_entries=4,
  )
  try
    build_layout(
      [
        { shard_ordinal: 0, tensor_ordinal: 0, byte_count: 3L, },
        { shard_ordinal: 0, tensor_ordinal: 1, byte_count: 3L, },
      ],
      low_budget,
    )
  catch {
    HostByteBudgetExceeded => ()
    _ => fail("unexpected host-budget rejection")
  } noraise {
    _ => fail("layout exceeded its total host budget")
  }
}

///|
test "quantized scale pairing follows exact role and scope" {
  let spec = try! @deepseek_v4_spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash",
  )
  let rows = spec.q_lora_rank().to_int64()
  let columns = spec.hidden_size().to_int64()
  let weight = try! @deepseek_v4_weights.DeepSeekV4SafetensorsTensor::new(
    name="layers.0.attn.wq_a.weight",
    dtype="F8_E4M3",
    shape=[rows, columns],
    shard=0,
    data_start=0L,
    data_end=rows * columns,
  )
  let scale_rows = (rows + 127L) / 128L
  let scale_columns = (columns + 127L) / 128L
  let scale = try! @deepseek_v4_weights.DeepSeekV4SafetensorsTensor::new(
    name="layers.0.attn.wq_a.scale",
    dtype="F8_E8M0",
    shape=[scale_rows, scale_columns],
    shard=0,
    data_start=0L,
    data_end=scale_rows * scale_columns,
  )
  let other_layer = try! @deepseek_v4_weights.DeepSeekV4SafetensorsTensor::new(
    name="layers.1.attn.wq_a.weight",
    dtype="F8_E4M3",
    shape=[rows, columns],
    shard=0,
    data_start=0L,
    data_end=rows * columns,
  )
  let weight_binding = try! @deepseek_v4_weights.validate_tensor_semantic(
    spec, weight,
  )
  let scale_binding = try! @deepseek_v4_weights.validate_tensor_semantic(
    spec, scale,
  )
  let other_binding = try! @deepseek_v4_weights.validate_tensor_semantic(
    spec, other_layer,
  )
  assert_true(
    quantized_parameter_pair(
      scale_binding.semantic(),
      weight_binding.semantic(),
    ),
  )
  assert_false(
    quantized_parameter_pair(scale_binding.semantic(), other_binding.semantic()),
  )
}
