///|
async fn source() -> (@model.Spec, Array[@safetensors.SafetensorsMetadata]) {
  let spec = @model.Spec::parse(
    @fs.read_file("model/minicpm_v47/config.fixture.json").binary()[:],
  )
  let dir = @fs.opendir("model/minicpm_v47_weights/fixtures")
  defer dir.close()
  let files = dir.read_all()
  files.sort()
  let limits = @safetensors.HeaderLimits::new(
    max_header_bytes=1048576,
    max_tensor_count=4096,
    max_rank=4,
    max_name_length=256,
    max_payload_bytes=80000000000L,
    max_materialized_tensor_bytes=80000000000L,
  )
  let shards = []
  for file in files {
    let bytes = @fs.read_file("model/minicpm_v47_weights/fixtures/" + file).binary()
    guard @json.parse(@utf8.decode(bytes)) is Object(root) else {
      fail("header fixture")
    }
    let mut payload = 0L
    for name, entry in root {
      if name == "__metadata__" {
        continue
      }
      guard entry is { "data_offsets": [_, Number(end, ..)], .. } else {
        fail("offset fixture")
      }
      if end.to_int64() > payload {
        payload = end.to_int64()
      }
    }
    let prefix = @buffer.Buffer()
    prefix.write_uint64_le(bytes.length().to_uint64())
    prefix.write_bytes(bytes)
    shards.push(
      @safetensors_reader.read_header(
        prefix.to_bytes()[:],
        payload_bytes=payload,
        limits~,
      ),
    )
  }
  (spec, shards)
}

///|
async test "all 1152 real checkpoint headers bind without reading weight payloads" {
  let (spec, shards) = source()
  let weights = @minicpm_v47_weights.bind(spec, shards[:])
  assert_eq(shards.length(), 16)
  assert_eq(weights.tensor_count(), 1152)
  assert_eq(weights.total_bytes(), 70425751648L)
  let query = weights
    .find_tensor("model.language_model.layers.3.self_attn.q_proj.weight")
    .unwrap()
  assert_eq(query.shape, ReadOnlyArray::from_array([8192L, 2048L]))
  let experts = weights
    .find_tensor("model.language_model.layers.0.mlp.experts.gate_up_proj")
    .unwrap()
  assert_eq(experts.shape, ReadOnlyArray::from_array([256L, 1024L, 2048L]))
  assert_true(experts.component is Decoder(0))
  assert_true(query.file_offset >= 8L)
  assert_true(
    weights.find_tensor("model.vision_tower.vit_merger.linear_1.weight")
    is Some({ component: EarlyVisionMerger, .. }),
  )
  assert_true(
    try @minicpm_v47_weights.bind(spec, shards[1:]) catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  let duplicated = shards.copy()
  duplicated.push(shards[0])
  assert_true(
    try @minicpm_v47_weights.bind(spec, duplicated[:]) catch {
      _ => true
    } noraise {
      _ => false
    },
  )
}

///|
async test "packed expert axes and BF16 control parameters cannot be reinterpreted" {
  let (spec, original) = source()
  let limits = @safetensors.HeaderLimits::new(
    max_header_bytes=1048576,
    max_tensor_count=4096,
    max_rank=4,
    max_name_length=256,
    max_payload_bytes=80000000000L,
    max_materialized_tensor_bytes=80000000000L,
  )
  for packed_rank in [true, false] {
    let shards = original.copy()
    let shard = shards[0]
    let entries = []
    let mut changed = false
    for i in 0..<shard.tensor_count() {
      let tensor = shard.tensor(i).unwrap()
      let rewrite = if packed_rank {
        tensor.name() ==
        "model.language_model.layers.0.mlp.experts.gate_up_proj"
      } else {
        tensor.name() == "model.language_model.layers.0.linear_attn.A_log"
      }
      let shape = if rewrite {
        if packed_rank {
          [256L, 2L, 512L, 2048L]
        } else {
          [16L]
        }
      } else {
        tensor.shape().iter().to_array()
      }
      let dtype = if rewrite && !packed_rank { "F32" } else { "BF16" }
      if rewrite {
        changed = true
      }
      entries.push(
        @safetensors.RawTensorMetadata::new(
          name=tensor.name(),
          dtype~,
          shape~,
          data_start=tensor.data_start(),
          data_end=tensor.data_end(),
        ),
      )
    }
    assert_true(changed)
    shards[0] = @safetensors.validate_metadata(
      entries[:],
      encoded_header_bytes=shard.encoded_header_bytes(),
      payload_bytes=shard.payload_bytes(),
      limits~,
    )
    assert_true(
      try @minicpm_v47_weights.bind(spec, shards[:]) catch {
        _ => true
      } noraise {
        _ => false
      },
    )
  }
}
