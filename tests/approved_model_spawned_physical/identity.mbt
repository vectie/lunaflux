///|
const PINNED_RUNTIME_RECIPE : String = "dense_llama_paged_aot_v5"

///|
const PINNED_MODEL_CONTENT_SHA256 : String = "852db1b39acb2336abc997440c6f6d6e4ab640f91e5e2aa9e2488d5794159d30"

///|
const PINNED_TOKENIZER_SHA256 : String = "fbcdbe15960e43ef351662e7b77a319ceb294b3c5dc2569c23b729fb87e13d7b"

///|
const PINNED_UPSTREAM_REVISION : String = "3579d71fd57e04f5a364d824d3a2ec3e913dbb67"

///|
fn pinned_tiny_model_identity() -> @spec.ModelIdentity raise {
  let model = @spec.LlamaModelSpec::new(
    vocabulary_size=3000,
    hidden_size=16,
    intermediate_size=64,
    layer_count=2,
    attention_head_count=4,
    kv_head_count=4,
    context_length=256,
    rope_theta=10000.0,
    rms_norm_epsilon=0.00001,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
  )
  let metadata = @spec.LlamaModelMetadata::from_verified_content(
    model,
    @spec.ContentDigest::from_sha256(PINNED_MODEL_CONTENT_SHA256),
  )
  @llama.build_paged(
    metadata,
    batch=@llama.LlamaPagedBatchEnvelope::single_row(),
  ).identity()
}

///|
fn foreign_tiny_model_identity() -> @spec.ModelIdentity raise {
  let model = @spec.LlamaModelSpec::new(
    vocabulary_size=3000,
    hidden_size=16,
    intermediate_size=64,
    layer_count=2,
    attention_head_count=4,
    kv_head_count=4,
    context_length=256,
    rope_theta=10000.0,
    rms_norm_epsilon=0.00001,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
  )
  let metadata = @spec.LlamaModelMetadata::from_verified_content(
    model,
    @spec.ContentDigest::from_sha256(
      "052db1b39acb2336abc997440c6f6d6e4ab640f91e5e2aa9e2488d5794159d30",
    ),
  )
  @llama.build_paged(
    metadata,
    batch=@llama.LlamaPagedBatchEnvelope::single_row(),
  ).identity()
}

///|
fn require_pinned_release_values(
  runtime_recipe : String,
  model_content_sha256 : String,
  model_plan_sha256 : String,
  tokenizer_sha256 : String,
  worker_executable_sha256 : String,
  expected_worker_sha256 : String,
) -> Unit raise ApprovedModelSpawnedPhysicalError {
  let identity = pinned_tiny_model_identity() catch {
    _ => raise PinnedModelIdentityRejected
  }
  if runtime_recipe != PINNED_RUNTIME_RECIPE {
    raise PinnedReleaseMismatch
  }
  if model_content_sha256 != identity.content().as_hex() {
    raise PinnedReleaseMismatch
  }
  if model_plan_sha256 != identity.plan().as_hex() {
    raise PinnedReleaseMismatch
  }
  if tokenizer_sha256 != PINNED_TOKENIZER_SHA256 {
    raise PinnedReleaseMismatch
  }
  if worker_executable_sha256 != expected_worker_sha256 {
    raise PinnedReleaseMismatch
  }
}

///|
fn require_pinned_release(
  release : @runtime_instance.RuntimeReleaseAdmission,
  expected_worker_sha256 : String,
) -> Unit raise ApprovedModelSpawnedPhysicalError {
  require_pinned_release_values(
    release.runtime_recipe(),
    release.model_content_sha256(),
    release.model_plan_sha256(),
    release.tokenizer_sha256(),
    release.worker_executable_sha256(),
    expected_worker_sha256,
  )
}
