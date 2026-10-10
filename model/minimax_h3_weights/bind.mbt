///|
fn expected_variant(
  component : MiniMaxH3WeightComponent,
) -> @minimax_h3_spec.MiniMaxH3Variant? {
  match component {
    Fl2vaDenoiser => Some(Fl2va)
    Ref2vaDenoiser => Some(Ref2va)
    TextConditioner | VideoVae | AudioVae => None
  }
}

///|
fn validate_rank(
  component : MiniMaxH3WeightComponent,
  role : MiniMaxH3TensorRole,
  rank : Int,
) -> Bool {
  match (component, role) {
    (Fl2vaDenoiser | Ref2vaDenoiser, DenoiserProjection) =>
      rank == 1 || rank == 2 || rank == 5
    (Fl2vaDenoiser | Ref2vaDenoiser, _) => rank == 1 || rank == 2
    (TextConditioner, VisionPatchEmbedding) =>
      rank == 1 || rank == 2 || rank == 5
    (TextConditioner, _) => rank == 1 || rank == 2
    (VideoVae, _) => rank >= 1 && rank <= 5
    (AudioVae, _) => rank >= 1 && rank <= 4
  }
}

///|
fn shape_is(actual : ReadOnlyArray[Int64], expected : ArrayView[Int64]) -> Bool {
  actual == ReadOnlyArray::from_array(expected)
}

///|
fn validate_denoiser_shape(name : String, shape : ReadOnlyArray[Int64]) -> Bool {
  match name {
    "proj_in.weight" => shape_is(shape, [5376L, 96L])
    "proj_in.bias" => shape_is(shape, [5376L])
    "audio_proj_in.weight" => shape_is(shape, [5376L, 32L])
    "audio_proj_in.bias" => shape_is(shape, [5376L])
    "context_embedder.weight" => shape_is(shape, [5376L, 5120L])
    "context_embedder.bias" => shape_is(shape, [5376L])
    "time_embedder.linear_1.weight" => shape_is(shape, [5376L, 256L])
    "time_embedder.linear_1.bias" => shape_is(shape, [5376L])
    "time_embedder.linear_2.weight" => shape_is(shape, [2688L, 5376L])
    "time_embedder.linear_2.bias" => shape_is(shape, [2688L])
    "norm_out.linear.weight" => shape_is(shape, [10752L, 2688L])
    "norm_out.linear.bias" => shape_is(shape, [10752L])
    "norm_out.norm.weight" | "token_refiner.final_norm.weight" =>
      shape_is(shape, [5376L])
    "proj_out.weight" => shape_is(shape, [96L, 5376L])
    "proj_out.bias" => shape_is(shape, [96L])
    "audio_proj_out.weight" => shape_is(shape, [32L, 5376L])
    "audio_proj_out.bias" => shape_is(shape, [32L])
    _ => {
      let parts = name.split(".").to_array()
      let offset = if parts[0] == "transformer_blocks" { 2 } else { 3 }
      if parts_match(parts, offset, ["adaln_proj", "linear", "weight"]) {
        shape_is(shape, [96768L, 2688L])
      } else if parts_match(parts, offset, ["adaln_proj", "linear", "bias"]) {
        shape_is(shape, [96768L])
      } else if parts_match(parts, offset, ["attn", "norm_k", "weight"]) ||
        parts_match(parts, offset, ["attn", "norm_q", "weight"]) {
        shape_is(shape, [128L])
      } else if parts_match(parts, offset, ["attn", "to_k", "weight"]) ||
        parts_match(parts, offset, ["attn", "to_q", "weight"]) ||
        parts_match(parts, offset, ["attn", "to_v", "weight"]) {
        shape_is(shape, [7168L, 5376L])
      } else if parts_match(parts, offset, ["attn", "to_out", "0", "weight"]) {
        shape_is(shape, [5376L, 7168L])
      } else if parts_match(parts, offset, ["ff", "net", "0", "proj", "weight"]) {
        shape_is(shape, [28672L, 5376L])
      } else if parts_match(parts, offset, ["ff", "net", "2", "weight"]) {
        shape_is(shape, [5376L, 14336L])
      } else if parts_match(parts, offset, ["norm1", "weight"]) ||
        parts_match(parts, offset, ["norm2", "weight"]) {
        shape_is(shape, [5376L])
      } else {
        false
      }
    }
  }
}

///|
fn denoiser_dtype_is_exact(
  name : String,
  dtype : @safetensors.TensorDType,
) -> Bool {
  let expected = if name == "proj_in.weight" ||
    name == "proj_in.bias" ||
    name == "audio_proj_in.weight" ||
    name == "audio_proj_in.bias" ||
    name == "proj_out.weight" ||
    name == "proj_out.bias" ||
    name == "audio_proj_out.weight" ||
    name == "audio_proj_out.bias" ||
    name.has_prefix("time_embedder.") {
    @safetensors.TensorDType::Float32
  } else {
    BFloat16
  }
  dtype == expected
}

///|
/// Binds one already bounds-checked safetensors shard to the exact schema-v1
/// MiniMax H3 semantic vocabulary.
///
/// This proves content/plan identity, partition compatibility, and a closed
/// operator role for every tensor in the shard. It intentionally does not
/// materialize payload bytes or assert that this shard is a complete component
/// artifact.
pub fn bind(
  metadata : @minimax_h3_spec.MiniMaxH3ModelMetadata,
  plan : @joint_diffusion_plan.JointDiffusionPlan,
  component : MiniMaxH3WeightComponent,
  tensors : @safetensors.SafetensorsMetadata,
) -> MiniMaxH3PartialShardBinding raise MiniMaxH3WeightsError {
  if tensors.tensor_count() == 0 {
    raise EmptyComponentShard(component)
  }
  if plan.content_digest() != metadata.content_digest() {
    raise PlanContentMismatch
  }
  let expected_plan = try! @minimax_h3.build(metadata)
  if plan.plan_digest() != expected_plan.plan_digest() ||
    plan.schema_version() != expected_plan.schema_version() {
    raise PlanSemanticMismatch
  }
  match expected_variant(component) {
    Some(variant) if variant != metadata.spec().variant() =>
      raise ComponentVariantMismatch(component)
    _ => ()
  }
  let bound : Array[MiniMaxH3SemanticTensorBinding] = []
  for index in 0..<tensors.tensor_count() {
    guard tensors.tensor(index) is Some(tensor) else {
      raise UnknownSemanticTensor(index~)
    }
    if tensor.name().contains("transformer.") ||
      tensor.name().contains("text_encoder.") ||
      tensor.name().contains("video_vae.") ||
      tensor.name().contains("audio_vae.") {
      raise UnsupportedRawLayout(index~)
    }
    guard classify_tensor(component, tensor.name()) is Some(role) else {
      raise UnknownSemanticTensor(index~)
    }
    let shape_valid = match component {
      Fl2vaDenoiser | Ref2vaDenoiser =>
        validate_denoiser_shape(tensor.name(), tensor.shape())
      VideoVae =>
        match video_vae_shape(tensor.name()) {
          Some(expected) => video_vae_shape_is(tensor.shape(), expected)
          None =>
            match original_video_shape(tensor.name()) {
              Some(expected) => video_vae_shape_is(tensor.shape(), expected)
              None => false
            }
        }
      AudioVae =>
        match audio_vae_shape(tensor.name()) {
          Some(expected) => video_vae_shape_is(tensor.shape(), expected)
          None => false
        }
      _ => validate_rank(component, role, tensor.shape().length())
    }
    if !shape_valid {
      raise InvalidSemanticRank(index~)
    }
    let dtype_valid = match component {
      Fl2vaDenoiser | Ref2vaDenoiser =>
        denoiser_dtype_is_exact(tensor.name(), tensor.dtype())
      TextConditioner => tensor.dtype() == BFloat16
      VideoVae => tensor.dtype() == Float32
      AudioVae => tensor.dtype() == Float32
    }
    if !dtype_valid {
      raise InvalidSemanticDType(index~)
    }
    bound.push({ source_index: index, role, tensor, })
  }
  let identity = try! @spec.ModelIdentity::new(
    plan.content_digest(),
    plan.plan_digest(),
  )
  {
    component,
    identity,
    schema_version: plan.schema_version(),
    tensors: ReadOnlyArray::from_array(bound),
  }
}

///|
/// Device materialization remains a typed fail-closed boundary until a native
/// MiniMax H3 component loader and dtype-complete payload reader exist.
pub fn MiniMaxH3PartialShardBinding::require_materialization(
  self : Self,
) -> Unit raise MiniMaxH3WeightsError {
  raise MaterializationUnavailable(self.component)
}
