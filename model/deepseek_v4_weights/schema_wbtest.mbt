///|
fn official_spec(id : StringView) -> @deepseek_v4_spec.DeepSeekV4ModelSpec {
  try! @deepseek_v4_spec.DeepSeekV4ModelSpec::from_official_model_id(id)
}

///|
fn header_tensor(
  name : String,
  dtype : StringView,
  shape : ArrayView[Int64],
) -> DeepSeekV4SafetensorsTensor {
  let width = match dtype {
    "BF16" => 2L
    "F32" => 4L
    "I64" => 8L
    _ => 1L
  }
  let mut bytes = width
  for dimension in shape {
    bytes *= dimension
  }
  try! DeepSeekV4SafetensorsTensor::new(
    name~,
    dtype~,
    shape~,
    shard=0,
    data_start=0L,
    data_end=bytes,
  )
}

///|
test "official DeepSeek V4 artifact cardinalities are exact" {
  let cases = [
    ("deepseek-ai/DeepSeek-V4-Flash", 69187),
    ("deepseek-ai/DeepSeek-V4-Flash-Base", 69189),
    ("deepseek-ai/DeepSeek-V4-Pro", 145116),
    ("deepseek-ai/DeepSeek-V4-Pro-Base", 145118),
    ("deepseek-ai/DeepSeek-V4-Flash-0731", 72317),
  ]
  for pair in cases {
    let (id, count) = pair
    assert_eq(expected_tensor_count(official_spec(id)), count)
  }
}

///|
test "semantic binder recognizes routing compression mHC and FP4 roles" {
  let spec = official_spec("deepseek-ai/DeepSeek-V4-Flash-0731")
  let probes = [
    header_tensor("layers.0.ffn.gate.tid2eid", "I64", [129280L, 6L]),
    header_tensor("layers.0.ffn.experts.255.w1.weight", "I8", [2048L, 2048L]),
    header_tensor("layers.0.ffn.experts.255.w1.scale", "F8_E8M0", [2048L, 128L]),
    header_tensor("layers.2.attn.compressor.ape", "F32", [4L, 1024L]),
    header_tensor("layers.2.attn.indexer.wq_b.scale", "F8_E8M0", [64L, 8L]),
    header_tensor("layers.3.ffn.gate.bias", "F32", [256L]),
    header_tensor("layers.3.hc_attn_fn", "F32", [24L, 16384L]),
    header_tensor("mtp.0.main_proj.weight", "F8_E4M3", [4096L, 12288L]),
    header_tensor("mtp.2.confidence_head.proj.weight", "BF16", [1L, 4352L]),
    header_tensor("mtp.2.markov_head.markov_w1.weight", "BF16", [129280L, 256L]),
  ]
  for tensor in probes {
    let binding = validate_tensor_semantic(spec, tensor)
    assert_eq(binding.tensor(), tensor)
  }
  let fp4 = validate_tensor_semantic(spec, probes[1])
  assert_eq(fp4.encoding(), PackedFp4E2M1Block32)
  assert_eq(fp4.semantic().scope(), DecoderLayer(0))
  assert_eq(fp4.semantic().role(), RoutedExpert(255, ExpertW1, QuantizedWeight))
}

///|
test "all official profiles bind exact block and head mHC F32 layouts" {
  let ids = [
    "deepseek-ai/DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4-Flash-Base", "deepseek-ai/DeepSeek-V4-Pro",
    "deepseek-ai/DeepSeek-V4-Pro-Base", "deepseek-ai/DeepSeek-V4-Flash-0731",
  ]
  for id in ids {
    let spec = official_spec(id)
    let h = Int64::from_int(spec.hidden_size())
    let probes = [
      header_tensor("layers.3.hc_attn_base", "F32", [24L]),
      header_tensor("layers.3.hc_attn_fn", "F32", [24L, 4L * h]),
      header_tensor("layers.3.hc_attn_scale", "F32", [3L]),
      header_tensor("layers.3.hc_ffn_base", "F32", [24L]),
      header_tensor("layers.3.hc_ffn_fn", "F32", [24L, 4L * h]),
      header_tensor("layers.3.hc_ffn_scale", "F32", [3L]),
      header_tensor("hc_head_base", "F32", [4L]),
      header_tensor("hc_head_fn", "F32", [4L, 4L * h]),
      header_tensor("hc_head_scale", "F32", [1L]),
    ]
    for tensor in probes {
      assert_eq(validate_tensor_semantic(spec, tensor).encoding(), Float32)
    }
  }
}

///|
test "all official profiles bind exact block-scaled query A and B layouts" {
  let cases = [
    ("deepseek-ai/DeepSeek-V4-Flash", 4096L, 1024L, 64L),
    ("deepseek-ai/DeepSeek-V4-Flash-Base", 4096L, 1024L, 64L),
    ("deepseek-ai/DeepSeek-V4-Pro", 7168L, 1536L, 128L),
    ("deepseek-ai/DeepSeek-V4-Pro-Base", 7168L, 1536L, 128L),
    ("deepseek-ai/DeepSeek-V4-Flash-0731", 4096L, 1024L, 64L),
  ]
  for entry in cases {
    let (id, hidden, rank, heads) = entry
    let spec = official_spec(id)
    let output = heads * 512L
    let probes = [
      header_tensor("layers.0.attn.wq_a.weight", "F8_E4M3", [rank, hidden]),
      header_tensor("layers.0.attn.wq_a.scale", "F8_E8M0", [
        rank / 128L,
        hidden / 128L,
      ]),
      header_tensor("layers.0.attn.wq_b.weight", "F8_E4M3", [output, rank]),
      header_tensor("layers.0.attn.wq_b.scale", "F8_E8M0", [
        output / 128L,
        rank / 128L,
      ]),
    ]
    assert_eq(
      validate_tensor_semantic(spec, probes[0]).encoding(),
      Fp8E4M3Block128,
    )
    assert_eq(validate_tensor_semantic(spec, probes[1]).encoding(), Ue8M0Scale)
    assert_eq(
      validate_tensor_semantic(spec, probes[2]).encoding(),
      Fp8E4M3Block128,
    )
    assert_eq(validate_tensor_semantic(spec, probes[3]).encoding(), Ue8M0Scale)
  }
}

///|
test "base checkpoints require full FP8 expert matrices and no FP4 codebook" {
  let spec = official_spec("deepseek-ai/DeepSeek-V4-Flash-Base")
  let weight = header_tensor("layers.0.ffn.experts.0.w1.weight", "F8_E4M3", [
    2048L, 4096L,
  ])
  let scale = header_tensor("layers.0.ffn.experts.0.w1.scale", "F8_E8M0", [
    16L, 32L,
  ])
  assert_eq(validate_tensor_semantic(spec, weight).encoding(), Fp8E4M3Block128)
  assert_eq(validate_tensor_semantic(spec, scale).encoding(), Ue8M0Scale)
}

///|
test "hostile tensor shape dtype scope and names fail closed" {
  let spec = official_spec("deepseek-ai/DeepSeek-V4-Flash-0731")
  let hostile = [
    header_tensor("layers.0.ffn.gate.bias", "F32", [256L]),
    header_tensor("layers.3.ffn.gate.tid2eid", "I64", [129280L, 6L]),
    header_tensor("layers.2.attn.indexer.wq_b.weight", "F8_E4M3", [4096L, 1024L]),
    header_tensor("layers.3.attn.indexer.weights_proj.weight", "BF16", [
      64L, 4096L,
    ]),
    header_tensor("mtp.1.main_proj.weight", "F8_E4M3", [4096L, 12288L]),
    header_tensor("layers.0.ffn.experts.256.w1.weight", "I8", [2048L, 2048L]),
    header_tensor("layers.0.ffn.experts.0.w4.weight", "I8", [2048L, 2048L]),
  ]
  for tensor in hostile {
    try validate_tensor_semantic(spec, tensor) catch {
      UnknownTensor(_) | InvalidTensor(_) => ()
      _ => fail("unexpected tensor-schema rejection")
    } noraise {
      _ => fail("hostile semantic tensor was admitted")
    }
  }
  let wrong_dtype = header_tensor("layers.0.attn.q_norm.weight", "F32", [1024L])
  try validate_tensor_semantic(spec, wrong_dtype) catch {
    InvalidTensor(0) => ()
    _ => fail("unexpected dtype rejection")
  } noraise {
    _ => fail("wrong dtype was admitted")
  }
}

///|
test "header constructor rejects unsupported dtype and byte layouts" {
  try
    DeepSeekV4SafetensorsTensor::new(
      name="layers.0.ffn.gate.tid2eid",
      dtype="I32",
      shape=[129280L, 6L],
      shard=0,
      data_start=0L,
      data_end=3102720L,
    )
  catch {
    InvalidDTypeTag => ()
    _ => fail("unexpected runtime token-hash dtype rejection")
  } noraise {
    _ => fail("runtime I32 was admitted as checkpoint storage")
  }
  try
    DeepSeekV4SafetensorsTensor::new(
      name="embed.weight",
      dtype="F4_E2M1",
      shape=[1L],
      shard=0,
      data_start=0L,
      data_end=1L,
    )
  catch {
    InvalidDTypeTag => ()
    _ => fail("unexpected dtype-tag error")
  } noraise {
    _ => fail("unsupported physical FP4 tag was admitted")
  }
  try
    DeepSeekV4SafetensorsTensor::new(
      name="embed.weight",
      dtype="BF16",
      shape=[2L],
      shard=0,
      data_start=0L,
      data_end=3L,
    )
  catch {
    ByteLengthMismatch => ()
    _ => fail("unexpected byte-layout error")
  } noraise {
    _ => fail("truncated tensor byte range was admitted")
  }
}

///|
test "aggregate binder rejects incomplete official manifests before identity" {
  let spec = official_spec("deepseek-ai/DeepSeek-V4-Flash")
  let metadata = @deepseek_v4_spec.DeepSeekV4ModelMetadata::from_verified_content(
    spec,
    try! @spec.ContentDigest::from_sha256(
      "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    ),
  )
  let plan = @deepseek_v4.build(metadata)
  try bind(plan, []) catch {
    IncorrectTensorCount(expected=69187, actual=0) => ()
    _ => fail("unexpected incomplete-manifest error")
  } noraise {
    _ => fail("empty official manifest was admitted")
  }
}

///|
test "recognized metadata never widens current materialization authority" {
  let fp4 = official_spec("deepseek-ai/DeepSeek-V4-Flash")
  let fp8 = official_spec("deepseek-ai/DeepSeek-V4-Flash-Base")
  assert_eq(
    materialization_readiness(fp4),
    UnsupportedByteLayouts([
      BFloat16,
      Float32,
      Int64,
      Fp8E4M3Block128,
      Ue8M0Scale,
      PackedFp4E2M1Block32,
    ]),
  )
  assert_eq(
    materialization_readiness(fp8),
    UnsupportedByteLayouts([
      BFloat16,
      Float32,
      Int64,
      Fp8E4M3Block128,
      Ue8M0Scale,
    ]),
  )
}
