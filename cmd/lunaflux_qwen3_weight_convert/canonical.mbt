///|
fn append_u64_le(output : Array[Byte], value : UInt64) -> Unit {
  for byte in value.to_le_bytes() {
    output.push(byte)
  }
}

///|
fn numeric_shape_json(shape : @numeric_contract.NumericShape) -> String {
  let dimensions : Array[String] = []
  for dimension in shape.dimensions() {
    dimensions.push(dimension.to_string())
  }
  "[" + dimensions.join(",") + "]"
}

///|
fn prepare_qwen3_numeric_conversion(
  plan : @plan.ModelPlan,
  bindings : @qwen3_weights.Qwen3WeightBindings,
) -> Qwen3NumericConversionPlan raise Qwen3WeightConvertError {
  let tensors = plan.numeric_binding().tensors()
  if tensors.length() != bindings.length() || tensors.length() != 310 {
    raise ConversionPlan
  }
  let header = StringBuilder()
  header.write_string(
    "{\"__metadata__\":{\"format\":\"lunaflux.numeric-safetensors.v1\",\"model_content_sha256\":\"\{plan.identity().content().as_hex()}\",\"model_plan_sha256\":\"\{plan.identity().plan().as_hex()}\",\"numeric_schema_sha256\":\"\{plan.numeric_binding().digest().as_hex()}\"}",
  )
  let regions : Array[Byte] = []
  let mut offset = 0L
  for ordinal, tensor in tensors {
    let reference = @plan.TensorRef::new(ordinal) catch {
      _ => raise ConversionPlan
    }
    guard bindings.binding(reference) is Some(binding) else {
      raise ConversionPlan
    }
    let source = binding.tensor()
    let length = @numeric_contract.storage_byte_length(
      tensor.storage().dtype(),
      tensor.shape(),
    ) catch {
      _ => raise ConversionPlan
    }
    if tensor.reference().as_int() != ordinal ||
      tensor.storage().dtype() != @numeric_contract.StorageDType::bfloat16() ||
      source.byte_length() != length {
      raise ConversionPlan
    }
    header.write_string(
      ",\"lunaflux.tensor.v1.\{ordinal}\":{\"dtype\":\"BF16\",\"shape\":\{numeric_shape_json(tensor.shape())},\"data_offsets\":[\{offset},\{offset + length}]}",
    )
    append_u64_le(regions, source.data_start().reinterpret_as_uint64())
    append_u64_le(regions, length.reinterpret_as_uint64())
    offset += length
  }
  header.write_string("}")
  // Safetensors requires the JSON header to end on an eight-byte boundary.
  // ASCII space is the only canonical padding byte emitted by this tool.
  let padding = (8 - header.to_string().length() % 8) % 8
  for _ in 0..<padding {
    header.write_char(' ')
  }
  {
    header: @utf8.encode(header.to_string()),
    source_regions: Bytes::from_array(regions),
    payload_bytes: offset,
  }
}

///|
fn qwen3_route_manifest_sha256(
  plan : @plan.ModelPlan,
  bindings : @qwen3_weights.Qwen3WeightBindings,
  artifact_sha256 : StringView,
) -> String {
  let output = StringBuilder()
  output.write_string(
    "schema=lunaflux.qwen3-route-manifest.v1\ncontent=" +
    plan.identity().content().as_hex() +
    "\nplan=" +
    plan.identity().plan().as_hex() +
    "\nnumeric_schema=" +
    plan.numeric_binding().digest().as_hex() +
    "\nartifact=" +
    artifact_sha256.to_owned() +
    "\ntied_lm_head_tensor_ref=0\n",
  )
  for ordinal in 0..<bindings.length() {
    let binding = bindings.binding(try! @plan.TensorRef::new(ordinal)).unwrap()
    output.write_string(
      "tensor=" + ordinal.to_string() + "|" + binding.tensor().name() + "\n",
    )
  }
  @crypto.bytes_to_hex_string(@crypto.sha256(@utf8.encode(output.to_string())))
}
