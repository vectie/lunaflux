///|
struct MistralFixtureEntry {
  name : String
  shape : Array[Int64]
}

///|
fn mistral_fixture_metadata(
  content? : String = "1111111111111111111111111111111111111111111111111111111111111111",
  window? : Int = 16,
) -> @mistral_spec.MistralModelMetadata {
  try! @mistral_spec.MistralModelMetadata::from_verified_content(
    @mistral_spec.MistralModelSpec::new(
      vocabulary_size=32,
      hidden_size=8,
      intermediate_size=16,
      layer_count=2,
      attention_head_count=4,
      kv_head_count=2,
      max_position_embeddings=32,
      sliding_window=window,
      rope_theta=10000.0,
      weight_dtype=BFloat16,
      kv_dtype=BFloat16,
    ),
    @spec.ContentDigest::from_sha256(content),
  )
}

///|
fn mistral_fixture_entries() -> Array[MistralFixtureEntry] {
  let entries : Array[MistralFixtureEntry] = [
    { name: "model.embed_tokens.weight", shape: [32L, 8L] },
  ]
  for layer in 0..<2 {
    let prefix = "model.layers.\{layer}"
    entries.push({ name: "\{prefix}.input_layernorm.weight", shape: [8L] })
    entries.push({ name: "\{prefix}.self_attn.q_proj.weight", shape: [8L, 8L] })
    entries.push({ name: "\{prefix}.self_attn.k_proj.weight", shape: [4L, 8L] })
    entries.push({ name: "\{prefix}.self_attn.v_proj.weight", shape: [4L, 8L] })
    entries.push({ name: "\{prefix}.self_attn.o_proj.weight", shape: [8L, 8L] })
    entries.push({
      name: "\{prefix}.post_attention_layernorm.weight",
      shape: [8L],
    })
    entries.push({ name: "\{prefix}.mlp.gate_proj.weight", shape: [16L, 8L] })
    entries.push({ name: "\{prefix}.mlp.up_proj.weight", shape: [16L, 8L] })
    entries.push({ name: "\{prefix}.mlp.down_proj.weight", shape: [8L, 16L] })
  }
  entries.push({ name: "model.norm.weight", shape: [8L] })
  entries.push({ name: "lm_head.weight", shape: [32L, 8L] })
  entries
}

///|
fn mistral_fixture_elements(shape : ArrayView[Int64]) -> Int64 {
  let mut count = 1L
  for dimension in shape {
    count = count * dimension
  }
  count
}

///|
fn mistral_fixture_safetensors(
  rename? : Int = -1,
  shape_change? : Int = -1,
  extra? : Bool = false,
) -> @safetensors.SafetensorsMetadata {
  let raw : Array[@safetensors.RawTensorMetadata] = []
  let mut offset = 0L
  for index, entry in mistral_fixture_entries() {
    let name = if index == rename {
      "foreign.\{index}.weight"
    } else {
      entry.name
    }
    let shape = if index == shape_change {
      [entry.shape[0] + 1L]
    } else {
      entry.shape
    }
    let length = mistral_fixture_elements(shape) * 2L
    raw.push(
      @safetensors.RawTensorMetadata::new(
        name~,
        dtype="BF16",
        shape~,
        data_start=offset,
        data_end=offset + length,
      ),
    )
    offset += length
  }
  if extra {
    raw.push(
      @safetensors.RawTensorMetadata::new(
        name="unexpected.weight",
        dtype="BF16",
        shape=[1L],
        data_start=offset,
        data_end=offset + 2L,
      ),
    )
    offset += 2L
  }
  try! @safetensors.validate_metadata(
    raw,
    encoded_header_bytes=4096,
    payload_bytes=offset,
    limits=@safetensors.HeaderLimits::new(
      max_header_bytes=16384,
      max_tensor_count=32,
      max_rank=2,
      max_name_length=128,
      max_payload_bytes=65536L,
      max_materialized_tensor_bytes=65536L,
    ),
  )
}

///|
fn mistral_append_u64(output : Array[Byte], value : UInt64) -> Unit {
  for shift in 0..<8 {
    output.push(((value >> (shift * 8)) & 0xffUL).to_byte())
  }
}

///|
fn mistral_numeric_shape_json(shape : @numeric_contract.NumericShape) -> String {
  let dimensions : Array[String] = []
  for dimension in shape.dimensions() {
    dimensions.push(dimension.to_string())
  }
  "[\{dimensions.join(",")}]"
}

///|
fn mistral_fixture_numeric_file(
  plan : @plan.ModelPlan,
  omit? : Int = -1,
  extra? : Bool = false,
  shape_change? : Int = -1,
) -> Bytes {
  let tensors = plan.numeric_binding().schema().tensors()
  let header = StringBuilder()
  header.write_string(
    "{\"__metadata__\":{\"format\":\"lunaflux.numeric-safetensors.v1\",\"model_content_sha256\":\"\{plan.identity().content().as_hex()}\",\"model_plan_sha256\":\"\{plan.identity().plan().as_hex()}\",\"numeric_schema_sha256\":\"\{plan.numeric_binding().digest().as_hex()}\"}",
  )
  let mut offset = 0L
  for ordinal, tensor in tensors {
    let length = try! @numeric_contract.storage_byte_length(
      tensor.storage().dtype(),
      tensor.shape(),
    )
    if ordinal != omit {
      let shape = if ordinal == shape_change {
        "[999]"
      } else {
        mistral_numeric_shape_json(tensor.shape())
      }
      header.write_string(
        ",\"lunaflux.tensor.v1.\{ordinal}\":{\"dtype\":\"BF16\",\"shape\":\{shape},\"data_offsets\":[\{offset},\{offset + length}]}",
      )
    }
    offset += length
  }
  if extra {
    header.write_string(
      ",\"unexpected\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[\{offset},\{offset + 2L}]}",
    )
    offset += 2L
  }
  header.write_string("}")
  let header_bytes = @utf8.encode(header.to_string())
  let output : Array[Byte] = []
  mistral_append_u64(output, header_bytes.length().to_uint64())
  for byte in header_bytes {
    output.push(byte)
  }
  for index in 0L..<offset {
    output.push(((index * 17L + 3L) & 0xffL).to_byte())
  }
  Bytes::from_array(output)
}

///|
fn mistral_fixture_artifact_digest(
  file : BytesView,
) -> @numeric_weight_file.NumericWeightArtifactDigest {
  try! @numeric_weight_file.NumericWeightArtifactDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(file)),
  )
}

///|
fn mistral_fixture_limits() -> @numeric_weight_file.NumericWeightFileLimits {
  try! @numeric_weight_file.NumericWeightFileLimits::new(
    max_file_bytes=131072L,
    max_header_bytes=65536,
    transfer_chunk_bytes=257,
    max_tensor_count=32,
  )
}
