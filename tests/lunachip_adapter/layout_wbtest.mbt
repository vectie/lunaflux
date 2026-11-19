///|
test "all Qwen3 tensors and FP32 KV fit with tied head and explicit reserves" {
  let model = metadata()
  let layout = @layout.prepare(@device.parse_contract(contract_text()), model)
  assert_eq(layout.context(), 2048)
  assert_eq(layout.weight_count(), 310)
  assert_eq(layout.weight_bytes(), 1192099840UL)
  assert_eq(layout.kv_bytes(), 469762048UL)
  assert_eq(layout.free_bytes(), 351141888UL)
  assert_eq(layout.lm_head().address, layout.weight(0).unwrap().address)
  assert_eq(layout.lm_head().bytes, 151936UL * 1024UL * 2UL)
  let mut previous_end = 0UL
  let mut norm_bytes = 0UL
  for ordinal in 0..<310 {
    let weight = layout.weight(ordinal).unwrap()
    assert_eq(weight.address, previous_end)
    assert_eq(weight.address % 128UL, 0UL)
    previous_end = weight.address + weight.bytes
  }
  for ordinal in 0..<310 {
    if layout.expanded_norm(ordinal) is Some(norm) {
      assert_eq(norm.address, previous_end)
      previous_end = norm.address + norm.bytes
      norm_bytes += norm.bytes
    }
  }
  assert_eq(norm_bytes, 262144UL)
  for layer in 0..<28 {
    for value in [false, true] {
      let first = layout.kv_head(layer, value, 0, 0)
      let last = layout.kv_head(layer, value, 2047, 7)
      assert_eq(first.address, previous_end)
      assert_eq(last.address + last.bytes - first.address, 8388608UL)
      previous_end = last.address + last.bytes
    }
  }
  assert_eq(previous_end, layout.workspace().address)
  assert_eq(layout.reserve().address, layout.workspace().address + 67108864UL)
  assert_eq(
    layout.reserve().address + layout.reserve().bytes + layout.free_bytes(),
    2147483648UL,
  )
  // The original CUDA source spec is deliberately unchanged.
  assert_true(model.spec().kv_dtype() == BFloat16)
}

///|
test "1 KiB row reads align all matrices while preserving payload lengths and tied head" {
  let device = @device.parse_contract(contract_text())
  let model = metadata()
  let compact = @layout.prepare(device, model)
  let aligned = @layout.prepare(device, model, align_matrices_1k=true)
  let mut end_address = 0UL
  let mut padding = 0UL
  for i in 0..<aligned.weight_count() {
    let tensor = aligned.weight(i).unwrap()
    let rank = aligned.dimensions(i).unwrap().length()
    let expected = if rank == 2 {
      (end_address + 1023UL) / 1024UL * 1024UL
    } else {
      end_address
    }
    assert_eq(tensor.address, expected)
    assert_eq(tensor.bytes, compact.weight(i).unwrap().bytes)
    padding += tensor.address - end_address
    end_address = tensor.address + tensor.bytes
  }
  assert_eq(padding, 14336UL)
  assert_eq(aligned.weight_bytes(), compact.weight_bytes())
  assert_eq(aligned.lm_head().address, aligned.weight(0).unwrap().address)
  assert_true(aligned.free_bytes() < compact.free_bytes())
}

///|
test "model budget rejects oversized context and overflowing reservations" {
  let device = @device.parse_contract(contract_text())
  for context in [0, -1, 2049, 4096] {
    try @layout.prepare(device, metadata(), context~) catch {
      @layout.UnsupportedContext => ()
      _ => fail("wrong context error")
    } noraise {
      _ => fail("context outside target admitted")
    }
  }
  for bytes in [0UL, 0xffffffffffffffffUL, 1073741824UL] {
    try @layout.prepare(device, metadata(), workspace_bytes=bytes) catch {
      @layout.InsufficientMemory => ()
      _ => fail("wrong budget error")
    } noraise {
      _ => fail("oversized workspace admitted")
    }
  }
  let layout = @layout.prepare(device, metadata())
  for
    (layer, position, head) in [
      (28, 0, 0),
      (27, 2048, 7),
      (0, 0, 8),
      (-1, 0, 0),
    ] {
    try layout.kv_head(layer, false, position, head) catch {
      @layout.InvalidIndex => ()
      _ => fail("wrong KV index error")
    } noraise {
      _ => fail("bad KV index admitted")
    }
  }
}

///|
fn source_metadata(
  layout : @layout.Layout,
) -> @safetensors.SafetensorsMetadata raise {
  let raw = []
  let names = [
    "input_layernorm", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.q_norm", "self_attn.k_norm", "self_attn.o_proj", "post_attention_layernorm",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
  ]
  for ordinal in 0..<layout.weight_count() {
    let name = if ordinal == 0 {
      "model.embed_tokens.weight"
    } else if ordinal == 309 {
      "model.norm.weight"
    } else {
      "model.layers.\{(ordinal - 1) / 11}.\{names[(ordinal - 1) % 11]}.weight"
    }
    let region = layout.weight(ordinal).unwrap()
    let dimensions = layout.dimensions(ordinal).unwrap().iter().to_array()
    raw.push(
      @safetensors.RawTensorMetadata::new(
        name~,
        dtype="BF16",
        shape=dimensions,
        data_start=region.address.reinterpret_as_int64(),
        data_end=(region.address + region.bytes).reinterpret_as_int64(),
      ),
    )
  }
  @safetensors.validate_metadata(
    raw,
    encoded_header_bytes=65536,
    payload_bytes=layout.weight_bytes().reinterpret_as_int64(),
    limits=@safetensors.HeaderLimits::new(
      max_header_bytes=1048576,
      max_tensor_count=512,
      max_rank=4,
      max_name_length=128,
      max_payload_bytes=2147483648L,
      max_materialized_tensor_bytes=2147483648L,
    ),
  )
}

///|
test "existing Qwen binder joins all 310 tensors and streams bounded ranges" {
  let model = metadata()
  let layout = @layout.prepare(@device.parse_contract(contract_text()), model)
  let bindings = @weights.bind(model, source_metadata(layout))
  let transfers = @transfer.prepare_transfers(layout, bindings)
  assert_eq(transfers.length(), 310)
  let mut total = 0L
  for transfer in transfers {
    let mut offset = 0L
    while offset < transfer.source_bytes {
      let chunk = transfer.chunk(offset, 1048576).unwrap()
      assert_true(chunk.source_bytes <= 1048576 && chunk.source_bytes > 0)
      assert_eq(chunk.source_offset, transfer.source_offset + offset)
      assert_eq(
        chunk.destination_address,
        transfer.destination.address + offset.reinterpret_as_uint64(),
      )
      offset += chunk.source_bytes.to_int64()
    }
    assert_true(transfer.chunk(offset, 1048576) is None)
    total += offset
  }
  assert_eq(total, 1192099840L)
  let output = Array::make(8, b'\xff')
  @transfer.expand_bf16(b"\x80\x3f\x00\xc0"[:], output)
  assert_eq(Bytes::from_array(output), b"\x00\x00\x80\x3f\x00\x00\x00\xc0")
  try transfers[0].chunk(1L, 1024) catch {
    @transfer.InvalidChunk => ()
    _ => fail("wrong chunk error")
  } noraise {
    _ => fail("split BF16 element admitted")
  }
}
