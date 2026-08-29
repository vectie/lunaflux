///|
fn mistral_checked_tensor_count(layer_count : Int) -> Int {
  let count = 3L + Int64::from_int(layer_count) * 9L
  if count > 2147483647L {
    2147483647
  } else {
    count.to_int()
  }
}

///|
fn mistral_expected_plan(
  metadata : @mistral_spec.MistralModelMetadata,
  supplied : @plan.ModelPlan,
) -> @plan.ModelPlan raise MistralWeightsError {
  match supplied.kv_execution() {
    StatelessFullContext =>
      @mistral.build(metadata) catch {
        error => raise InvalidModelPlan(error)
      }
    PagedKeyValue => {
      let batch = @mistral.MistralPagedBatchEnvelope::new(
        supplied.shape_constraints().max_batch_rows(),
      ) catch {
        error => raise InvalidModelPlan(error)
      }
      @mistral.build_paged(metadata, batch~) catch {
        error => raise InvalidModelPlan(error)
      }
    }
  }
}

///|
fn mistral_dimensions_match(
  actual : ReadOnlyArray[Int64],
  expected : ArrayView[Int64],
  reference : Int,
) -> Unit raise MistralWeightsError {
  if actual.length() != expected.length() {
    raise InvalidTensor(reference~, TensorRank)
  }
  for axis, dimension in expected {
    if actual[axis] != dimension {
      raise InvalidTensor(reference~, TensorDimension)
    }
  }
}

///|
fn mistral_append_binding(
  output : Array[MistralWeightBinding],
  tensors : @safetensors.SafetensorsMetadata,
  name : StringView,
  shape : ArrayView[Int64],
) -> Unit raise MistralWeightsError {
  let reference = output.length()
  guard tensors.find_tensor(name) is Some(tensor) else {
    raise MissingTensor(reference~)
  }
  if tensor.name() != name.to_owned() {
    raise InvalidTensor(reference~, TensorName)
  }
  if tensor.dtype() != BFloat16 {
    raise InvalidTensor(reference~, TensorDType)
  }
  mistral_dimensions_match(tensor.shape(), shape, reference)
  output.push({
    reference: @plan.TensorRef::new(reference) catch {
      _ => raise InvalidTensor(reference~, TensorCount)
    },
    tensor,
  })
}

///|
fn mistral_manifest_bytes(
  identity : @spec.ModelIdentity,
  bindings : ArrayView[MistralWeightBinding],
  materialized_bytes : Int64,
) -> Bytes {
  let encoded = StringBuilder()
  encoded.write_string(
    "schema=lunaflux.mistral-weight-manifest.v1\nmodel_content_sha256=" +
    identity.content().as_hex() +
    "\nmodel_plan_sha256=" +
    identity.plan().as_hex() +
    "\ntensor_count=" +
    bindings.length().to_string() +
    "\nmaterialized_bytes=" +
    materialized_bytes.to_string() +
    "\n",
  )
  for binding in bindings {
    let tensor = binding.tensor()
    let dimensions : Array[String] = []
    for dimension in tensor.shape() {
      dimensions.push(dimension.to_string())
    }
    encoded.write_string(
      "tensor=" +
      binding.reference().as_int().to_string() +
      "|" +
      tensor.name() +
      "|BF16|" +
      dimensions.join(",") +
      "|" +
      tensor.data_start().to_string() +
      "|" +
      tensor.data_end().to_string() +
      "\n",
    )
  }
  @utf8.encode(encoded.to_string())
}

///|
/// Binds the exact Hugging Face Mistral tensor vocabulary to the existing
/// dense 21-operation semantic tensor order. Name translation ends here;
/// scheduler, KV, device execution, and kernel selection remain family-free.
pub fn bind(
  metadata : @mistral_spec.MistralModelMetadata,
  model_plan : @plan.ModelPlan,
  tensors : @safetensors.SafetensorsMetadata,
) -> MistralWeightManifest raise MistralWeightsError {
  if metadata.content_digest() != model_plan.identity().content() {
    raise Mismatch(ModelContent)
  }
  let expected_plan = mistral_expected_plan(metadata, model_plan)
  if model_plan != expected_plan {
    raise Mismatch(ModelPlan)
  }
  let spec = metadata.spec()
  let expected_count = mistral_checked_tensor_count(spec.layer_count())
  if tensors.tensor_count() != expected_count {
    raise Mismatch(TensorCount)
  }
  let hidden = Int64::from_int(spec.hidden_size())
  let intermediate = Int64::from_int(spec.intermediate_size())
  let vocabulary = Int64::from_int(spec.vocabulary_size())
  let kv_width = Int64::from_int(spec.kv_head_count() * spec.head_dimension())
  let bindings : Array[MistralWeightBinding] = []
  mistral_append_binding(bindings, tensors, "model.embed_tokens.weight", [
    vocabulary, hidden,
  ])
  for layer in 0..<spec.layer_count() {
    let prefix = "model.layers.\{layer}"
    mistral_append_binding(
      bindings,
      tensors,
      "\{prefix}.input_layernorm.weight",
      [hidden],
    )
    mistral_append_binding(
      bindings,
      tensors,
      "\{prefix}.self_attn.q_proj.weight",
      [hidden, hidden],
    )
    mistral_append_binding(
      bindings,
      tensors,
      "\{prefix}.self_attn.k_proj.weight",
      [kv_width, hidden],
    )
    mistral_append_binding(
      bindings,
      tensors,
      "\{prefix}.self_attn.v_proj.weight",
      [kv_width, hidden],
    )
    mistral_append_binding(
      bindings,
      tensors,
      "\{prefix}.self_attn.o_proj.weight",
      [hidden, hidden],
    )
    mistral_append_binding(
      bindings,
      tensors,
      "\{prefix}.post_attention_layernorm.weight",
      [hidden],
    )
    mistral_append_binding(
      bindings,
      tensors,
      "\{prefix}.mlp.gate_proj.weight",
      [intermediate, hidden],
    )
    mistral_append_binding(bindings, tensors, "\{prefix}.mlp.up_proj.weight", [
      intermediate, hidden,
    ])
    mistral_append_binding(
      bindings,
      tensors,
      "\{prefix}.mlp.down_proj.weight",
      [hidden, intermediate],
    )
  }
  mistral_append_binding(bindings, tensors, "model.norm.weight", [hidden])
  mistral_append_binding(bindings, tensors, "lm_head.weight", [
    vocabulary, hidden,
  ])
  let manifest = mistral_manifest_bytes(
    model_plan.identity(),
    bindings,
    tensors.materialized_tensor_bytes(),
  )
  {
    identity: model_plan.identity(),
    bindings: ReadOnlyArray::from_array(bindings),
    materialized_bytes: tensors.materialized_tensor_bytes(),
    manifest_sha256: @crypto.bytes_to_hex_string(@crypto.sha256(manifest)),
  }
}
