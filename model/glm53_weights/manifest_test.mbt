///|
fn manifest_full_metadata() -> @glm53_spec.Glm53FullMetadata {
  @glm53_spec.Glm53FullMetadata::from_verified_content(
    @glm53_spec.Glm53FullSpec::official(),
    try! @spec.ContentDigest::from_sha256(
      "4123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
  )
}

///|
fn manifest_flash_metadata() -> @glm53_spec.Glm53FlashMetadata {
  @glm53_spec.Glm53FlashMetadata::from_verified_content(
    @glm53_spec.Glm53FlashSpec::official(),
    try! @spec.ContentDigest::from_sha256(
      "5123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
  )
}

///|
fn observed_from_manifest(
  manifest : @glm53_weights.Glm53TensorManifest,
) -> Array[@glm53_weights.Glm53ObservedTensor] {
  let observed = []
  for ordinal in 0..<manifest.tensor_count() {
    let tensor = manifest.tensor(ordinal).unwrap()
    observed.push(
      @glm53_weights.Glm53ObservedTensor::new(
        name=tensor.name(),
        dtype=tensor.dtype(),
        shape=tensor.shape(),
      ),
    )
  }
  observed
}

///|
test "full logical manifest exactly enumerates non-scale official tensors" {
  let manifest = @glm53_weights.build_full_tensor_manifest(
    manifest_full_metadata(),
    BFloat16WithFloat32Routing,
  )
  assert_eq(manifest.tensor_count(), 59585)
  assert_eq(manifest.tensor(0).unwrap().name(), "model.embed_tokens.weight")
  assert_eq(
    manifest.tensor(1).unwrap().name(),
    "model.layers.0.input_layernorm.weight",
  )
  assert_eq(
    manifest.tensor(3).unwrap().name(),
    "model.layers.0.self_attn.q_a_proj.weight",
  )
  assert_eq(
    manifest.tensor(14).unwrap().name(),
    "model.layers.0.self_attn.indexer.weights_proj.weight",
  )
  assert_eq(
    manifest.tensor(15).unwrap().name(),
    "model.layers.0.mlp.gate_proj.weight",
  )
  assert_eq(manifest.tensor(59584).unwrap().name(), "lm_head.weight")
  assert_eq(manifest.control_token_ids().length(), 0)
}

///|
test "Flash manifest exactly covers official text MTP vision and token contract" {
  let manifest = @glm53_weights.build_flash_tensor_manifest(
    manifest_flash_metadata(),
    BFloat16WithFloat32Routing,
  )
  assert_eq(manifest.tensor_count(), 38770)
  assert_eq(
    manifest.tensor(0).unwrap().name(),
    "model.language_model.embed_tokens.weight",
  )
  assert_eq(
    manifest.tensor(3).unwrap().name(),
    "model.language_model.layers.0.hc_attn_fn",
  )
  assert_eq(
    manifest.tensor(9).unwrap().name(),
    "model.language_model.layers.0.self_attn.A_log",
  )
  assert_eq(manifest.tensor(9).unwrap().dtype(), Float32)
  assert_eq(manifest.tensor(9).unwrap().shape(), [64L])
  assert_eq(
    manifest.tensor(38769).unwrap().name(),
    "model.visual.merger.post_projection_norm.bias",
  )
  assert_eq(manifest.control_token_ids(), [
    154830, 154831, 154832, 154833, 154854, 154855,
  ])
}

///|
test "manifest identity is deterministic and binds content plus execution plan" {
  let first = @glm53_weights.build_flash_tensor_manifest(
    manifest_flash_metadata(),
    BFloat16WithFloat32Routing,
  )
  let second = @glm53_weights.build_flash_tensor_manifest(
    manifest_flash_metadata(),
    BFloat16WithFloat32Routing,
  )
  assert_eq(first, second)
  assert_eq(
    first.execution_identity(),
    @glm53.build_flash_execution_plan(manifest_flash_metadata()).identity(),
  )
  let changed_metadata = @glm53_spec.Glm53FlashMetadata::from_verified_content(
    @glm53_spec.Glm53FlashSpec::official(),
    try! @spec.ContentDigest::from_sha256(
      "6123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
  )
  let changed = @glm53_weights.build_flash_tensor_manifest(
    changed_metadata,
    BFloat16WithFloat32Routing,
  )
  assert_true(first.execution_identity() != changed.execution_identity())
  assert_true(first.layout_digest() != changed.layout_digest())
}

///|
test "observed safetensors binding rejects count order duplicate dtype and shape" {
  let manifest = @glm53_weights.build_flash_tensor_manifest(
    manifest_flash_metadata(),
    BFloat16WithFloat32Routing,
  )
  let exact = observed_from_manifest(manifest)
  let binding = @glm53_weights.bind_observed_tensor_manifest(manifest, exact)
  assert_eq(binding.tensor_count(), 38770)
  assert_eq(binding.execution_identity(), manifest.execution_identity())
  let reordered = exact.copy()
  let first = reordered[0]
  reordered[0] = reordered[1]
  reordered[1] = first
  try @glm53_weights.bind_observed_tensor_manifest(manifest, reordered) catch {
    TensorNameMismatch(ordinal=0) => ()
    _ => fail("unexpected order failure")
  } noraise {
    _ => fail("reordered tensors admitted")
  }
  let duplicate = exact.copy()
  let tensor0 = manifest.tensor(0).unwrap()
  duplicate[1] = @glm53_weights.Glm53ObservedTensor::new(
    name=tensor0.name(),
    dtype=tensor0.dtype(),
    shape=tensor0.shape(),
  )
  try @glm53_weights.bind_observed_tensor_manifest(manifest, duplicate) catch {
    DuplicateTensor(ordinal=1) => ()
    _ => fail("unexpected duplicate failure")
  } noraise {
    _ => fail("duplicate tensor admitted")
  }
  let wrong_dtype = exact.copy()
  wrong_dtype[0] = @glm53_weights.Glm53ObservedTensor::new(
    name=tensor0.name(),
    dtype=Float32,
    shape=tensor0.shape(),
  )
  try
    @glm53_weights.bind_observed_tensor_manifest(manifest, wrong_dtype)
  catch {
    TensorDTypeMismatch(ordinal=0) => ()
    _ => fail("unexpected dtype failure")
  } noraise {
    _ => fail("wrong dtype admitted")
  }
  let wrong_shape = exact.copy()
  wrong_shape[0] = @glm53_weights.Glm53ObservedTensor::new(
    name=tensor0.name(),
    dtype=tensor0.dtype(),
    shape=[154879L, 4096L],
  )
  try
    @glm53_weights.bind_observed_tensor_manifest(manifest, wrong_shape)
  catch {
    TensorShapeMismatch(ordinal=0) => ()
    _ => fail("unexpected shape failure")
  } noraise {
    _ => fail("wrong shape admitted")
  }
  try
    @glm53_weights.bind_observed_tensor_manifest(
      manifest,
      exact[0:exact.length() - 1],
    )
  catch {
    TensorCountMismatch(expected=38770, actual=38769) => ()
    _ => fail("unexpected count failure")
  } noraise {
    _ => fail("missing tensor admitted")
  }
}

///|
test "unmodeled full FP8 and Flash FP8 layouts fail closed" {
  try
    @glm53_weights.build_full_tensor_manifest(
      manifest_full_metadata(),
      Fp8E4M3Block128Dynamic,
    )
  catch {
    UnsupportedPhysicalEncoding(Fp8E4M3Block128Dynamic) => ()
    _ => fail("unexpected full FP8 failure")
  } noraise {
    _ => fail("unmodeled full FP8 layout admitted")
  }
  try
    @glm53_weights.build_flash_tensor_manifest(
      manifest_flash_metadata(),
      Fp8E4M3Block128Dynamic,
    )
  catch {
    UnsupportedPhysicalEncoding(Fp8E4M3Block128Dynamic) => ()
    _ => fail("unexpected Flash FP8 failure")
  } noraise {
    _ => fail("non-official Flash FP8 layout admitted")
  }
}
