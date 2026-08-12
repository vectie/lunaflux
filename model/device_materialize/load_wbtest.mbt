///|
fn wb_test_model() -> @spec.LlamaModelMetadata raise {
  let spec = @spec.LlamaModelSpec::new(
    vocabulary_size=8,
    hidden_size=4,
    intermediate_size=6,
    layer_count=1,
    attention_head_count=2,
    kv_head_count=1,
    context_length=16,
    rope_theta=10000.0,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
  )
  let content = @spec.ContentDigest::from_sha256(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  @spec.LlamaModelMetadata::from_verified_content(spec, content)
}

///|
fn wb_append_raw(
  entries : Array[@safetensors.RawTensorMetadata],
  name : String,
  shape : ArrayView[Int64],
  offset : Ref[Int64],
) -> Unit {
  let start = offset.val
  let byte_length = shape.fold(init=2L, (size, dimension) => size * dimension)
  let end = start + byte_length
  entries.push(
    @safetensors.RawTensorMetadata::new(
      name~,
      dtype="BF16",
      shape~,
      data_start=start,
      data_end=end,
    ),
  )
  offset.val = end
}

///|
fn wb_test_bindings() -> @llama_weights.LlamaWeightBindings raise {
  let entries : Array[@safetensors.RawTensorMetadata] = []
  let offset = Ref(0L)
  wb_append_raw(entries, "model.embed_tokens.weight", [8L, 4L], offset)
  wb_append_raw(entries, "model.layers.0.input_layernorm.weight", [4L], offset)
  wb_append_raw(
    entries,
    "model.layers.0.self_attn.q_proj.weight",
    [4L, 4L],
    offset,
  )
  wb_append_raw(
    entries,
    "model.layers.0.self_attn.k_proj.weight",
    [2L, 4L],
    offset,
  )
  wb_append_raw(
    entries,
    "model.layers.0.self_attn.v_proj.weight",
    [2L, 4L],
    offset,
  )
  wb_append_raw(
    entries,
    "model.layers.0.self_attn.o_proj.weight",
    [4L, 4L],
    offset,
  )
  wb_append_raw(
    entries,
    "model.layers.0.post_attention_layernorm.weight",
    [4L],
    offset,
  )
  wb_append_raw(
    entries,
    "model.layers.0.mlp.gate_proj.weight",
    [6L, 4L],
    offset,
  )
  wb_append_raw(entries, "model.layers.0.mlp.up_proj.weight", [6L, 4L], offset)
  wb_append_raw(
    entries,
    "model.layers.0.mlp.down_proj.weight",
    [4L, 6L],
    offset,
  )
  wb_append_raw(entries, "model.norm.weight", [4L], offset)
  wb_append_raw(entries, "lm_head.weight", [8L, 4L], offset)
  let limits = @safetensors.HeaderLimits::new(
    max_header_bytes=4096,
    max_tensor_count=32,
    max_rank=2,
    max_name_length=128,
    max_payload_bytes=4096L,
    max_materialized_tensor_bytes=4096L,
  )
  let metadata = @safetensors.validate_metadata(
    entries,
    encoded_header_bytes=16,
    payload_bytes=offset.val,
    limits~,
  )
  @llama_weights.bind(wb_test_model(), metadata)
}

///|
fn wb_test_file(payload_bytes : Int) -> Bytes {
  Bytes::makei(24 + payload_bytes, index => {
    if index == 0 {
      b'\x10'
    } else if index < 8 {
      b'\x00'
    } else if index == 8 {
      b'{'
    } else if index == 9 {
      b'}'
    } else if index < 24 {
      b' '
    } else {
      ((index - 24) % 251).to_byte()
    }
  })
}

///|
fn wb_test_limits() -> @materialize.MaterializationLimits raise {
  @materialize.MaterializationLimits::new(
    max_header_bytes=1024,
    max_tensor_count=32,
    max_payload_bytes=4096L,
    max_host_bytes=4096L,
  )
}

///|
test "malformed envelope does not open a destination resource" {
  let bindings = wb_test_bindings()
  let layout = plan_layout(bindings, alignment_bytes=64L, max_arena_bytes=4096L)
  let opens = Ref(0)
  try
    load_with_resource(
      bindings,
      b"short",
      layout,
      wb_test_limits(),
      () => {
        opens.val = opens.val + 1
        7
      },
      (_resource, _source, _destination, _count) => (),
      _resource => (),
    )
  catch {
    MaterializationFailed(MissingHeaderLength) => ()
    _ => fail("unexpected malformed-envelope error")
  } noraise {
    _ => fail("expected malformed-envelope failure")
  }
  assert_eq(opens.val, 0)
}

///|
test "transfer failure closes the partially loaded resource exactly once" {
  let bindings = wb_test_bindings()
  let file = wb_test_file(bindings.materialized_bytes().to_int())
  let layout = plan_layout(bindings, alignment_bytes=64L, max_arena_bytes=4096L)
  let opens = Ref(0)
  let copies = Ref(0)
  let closes = Ref(0)
  try
    load_with_resource(
      bindings,
      file,
      layout,
      wb_test_limits(),
      () => {
        opens.val = opens.val + 1
        7
      },
      (_resource, _source, _destination, _count) => {
        copies.val = copies.val + 1
        if copies.val == 3 {
          raise DriverFailure(CopyToDevice)
        }
      },
      _resource => closes.val = closes.val + 1,
    )
  catch {
    DeviceFailed(DriverFailure(CopyToDevice)) => ()
    _ => fail("unexpected transfer error")
  } noraise {
    _ => fail("expected transfer failure")
  }
  assert_eq(opens.val, 1)
  assert_eq(copies.val, 3)
  assert_eq(closes.val, 1)
}

///|
test "cleanup failure retains the primary transfer stage" {
  let bindings = wb_test_bindings()
  let file = wb_test_file(bindings.materialized_bytes().to_int())
  let layout = plan_layout(bindings, alignment_bytes=64L, max_arena_bytes=4096L)
  try
    load_with_resource(
      bindings,
      file,
      layout,
      wb_test_limits(),
      () => 7,
      (_resource, _source, _destination, _count) => {
        raise DriverFailure(CopyToDevice)
      },
      _resource => raise Busy(AllocationClose),
    )
  catch {
    CleanupFailed(stage=Transfer, Busy(AllocationClose)) => ()
    _ => fail("unexpected cleanup error")
  } noraise {
    _ => fail("expected cleanup failure")
  }
}
