///|
fn deepseek_flash_plan() -> @deepseek_v4.DeepSeekV4SemanticPlan raise {
  let spec = @deepseek_v4_spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash-0731",
  )
  let metadata = @deepseek_v4_spec.DeepSeekV4ModelMetadata::from_verified_content(
    spec,
    @spec.ContentDigest::from_sha256(
      "4444444444444444444444444444444444444444444444444444444444444444",
    ),
  )
  @deepseek_v4.build(metadata)
}

///|
test "DeepSeek V4 publishes the exact modular family contract" {
  let required = @deepseek_v4.requirements()
  assert_eq(required.family(), DeepSeekV4)
  assert_eq(required.architecture(), "DeepseekV4ForCausalLM")
  assert_eq(required.model_type(), "deepseek_v4")
  assert_eq(required.schema_version(), 1)
  assert_eq(required.workload(), DecoderOnlyText)
  assert_eq(required.capabilities(), [
    DenseCausalAttention,
    QueryKeyRmsNorm,
    MixtureOfExperts,
    MultiTokenPrediction,
    HyperConnections,
    CompressedSparseAttention,
    HeavilyCompressedAttention,
  ])
}

///|
test "DeepSeek V4 refuses a partial dense backend" {
  let plan = deepseek_flash_plan()
  let partial = @family_contract.ModelBackendCapabilities::new([
    DenseCausalAttention,
    QueryKeyRmsNorm,
  ])
  match plan.admit_backend(partial) {
    MissingCapabilities(profile, missing) => {
      assert_eq(profile.family(), DeepSeekV4)
      assert_eq(missing, [
        MixtureOfExperts,
        MultiTokenPrediction,
        HyperConnections,
        CompressedSparseAttention,
        HeavilyCompressedAttention,
      ])
    }
    SemanticCapabilitiesAvailable(_) =>
      fail("partial dense backend admitted DeepSeek V4")
  }
}

///|
test "DeepSeek V4 admits only the complete semantic backend set" {
  let plan = deepseek_flash_plan()
  let complete = @family_contract.ModelBackendCapabilities::new([
    DenseCausalAttention,
    QueryKeyRmsNorm,
    MixtureOfExperts,
    MultiTokenPrediction,
    HyperConnections,
    CompressedSparseAttention,
    HeavilyCompressedAttention,
  ])
  match plan.admit_backend(complete) {
    SemanticCapabilitiesAvailable(profile) =>
      assert_eq(profile, plan.requirements())
    MissingCapabilities(_, _) => fail("complete DeepSeek backend was rejected")
  }
}

///|
test "DeepSeek V4 hybrid attention remains layer-specific" {
  let plan = deepseek_flash_plan()
  assert_eq(plan.attention_mode(0), Some(SlidingWindowCausal))
  assert_eq(plan.attention_mode(2), Some(CompressedSparse))
  assert_eq(plan.attention_mode(3), Some(HeavilyCompressed))
  assert_eq(plan.attention_mode(-1), None)
  assert_eq(plan.attention_mode(43), None)
  assert_eq(
    plan.content_digest().as_hex(),
    "4444444444444444444444444444444444444444444444444444444444444444",
  )
}

///|
test "DeepSeek V4 emits a complete typed advanced decoder plan" {
  let family = deepseek_flash_plan()
  let execution = family.execution_plan()
  let shape = execution.shape()
  assert_eq(execution.schema_version(), 1)
  assert_eq(
    execution.profile_identity(),
    "deepseek-ai/DeepSeek-V4-Flash-0731/v1",
  )
  assert_eq(shape.vocabulary_size(), 129280)
  assert_eq(shape.hidden_size(), 4096)
  assert_eq(shape.decoder_layer_count(), 43)
  assert_eq(shape.attention_head_count(), 64)
  assert_eq(shape.kv_head_count(), 1)
  assert_eq(shape.attention_head_dimension(), 512)
  assert_eq(shape.rope_head_dimension(), 64)
  assert_eq(shape.query_low_rank(), 1024)
  assert_eq(shape.output_low_rank(), 1024)
  assert_eq(shape.output_group_count(), 8)
  assert_eq(shape.maximum_position_count(), 1048576)
  let hyper = execution.hyper_connection()
  assert_eq(hyper.width(), 4)
  assert_eq(hyper.epsilon(), 0.000001)
  assert_eq(hyper.normalization_epsilon(), 0.000001)
  assert_eq(hyper.sinkhorn_iterations(), 20)
  let moe = execution.moe()
  assert_eq(moe.routed_expert_count(), 256)
  assert_eq(moe.shared_expert_count(), 1)
  assert_eq(moe.experts_per_token(), 6)
  assert_eq(moe.intermediate_size(), 2048)
  assert_eq(moe.routed_scaling_factor(), 1.5)
  assert_true(moe.normalize_selected_probabilities())
  assert_eq(moe.score_function(), SqrtSoftplus)
  assert_eq(
    moe.selection_method(),
    TokenHashThenNoAuxiliaryLossTokenChoice(hash_layer_count=3),
  )
  assert_eq(moe.hash_routed_layer_count(), 3)
  assert_eq(moe.activation(), ClampedSwiGlu(10.0))
  assert_eq(execution.layers().length(), 43)
  assert_eq(
    execution.layers()[0].attention(),
    SlidingWindowCausal(window_size=128),
  )
  assert_eq(
    execution.layers()[2].attention(),
    IndexedCompressed(
      compression_ratio=4,
      index_head_count=64,
      index_head_dimension=128,
      index_top_k=512,
    ),
  )
  assert_eq(
    execution.layers()[3].attention(),
    DeterministicCompressed(compression_ratio=128, window_size=128),
  )
  match execution.auxiliary_prediction() {
    MultiTokenPredictionWithDspark(layer_count~, dspark~) => {
      assert_eq(layer_count, 3)
      assert_eq(dspark.block_size(), 5)
      assert_eq(dspark.noise_token_id(), 128799)
      assert_eq(dspark.target_layer_ids(), [40, 41, 42])
      assert_eq(dspark.markov_rank(), 256)
    }
    MultiTokenPrediction(..) => fail("Flash-0731 lost its DSpark module")
  }
  inspect(
    execution.plan_digest().as_hex(),
    content=(
      #|3a3b35e9b7c576aff409f2350dfdd77f4e077134fe82f23a18bf8cce80cb9c5e
    ),
  )
}

///|
test "DeepSeek V4 projects exact identity into decoder workload envelope" {
  let semantic = deepseek_flash_plan()
  let execution = semantic.execution_plan()
  let workload = semantic.workload_execution_plan()
  assert_eq(workload.kind(), DecoderOnlyText)
  assert_eq(workload.schema_version(), execution.schema_version())
  assert_eq(workload.decoder_layer_count(), Some(43))
  assert_eq(workload.maximum_token_context(), Some(1048576))
  assert_eq(workload.identity().content(), execution.content_digest())
  assert_eq(workload.identity().plan(), execution.plan_digest())
}

///|
test "DeepSeek V4 profile and content are bound into plan identity" {
  let flash = deepseek_flash_plan().execution_plan()
  let spec = @deepseek_v4_spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Pro",
  )
  let metadata = @deepseek_v4_spec.DeepSeekV4ModelMetadata::from_verified_content(
    spec,
    @spec.ContentDigest::from_sha256(
      "4444444444444444444444444444444444444444444444444444444444444444",
    ),
  )
  let pro = @deepseek_v4.build(metadata).execution_plan()
  assert_true(flash.plan_digest() != pro.plan_digest())
  assert_eq(pro.shape().decoder_layer_count(), 61)
  assert_eq(pro.moe().routed_expert_count(), 384)
  assert_eq(
    pro.layers()[0].attention(),
    DeterministicCompressed(compression_ratio=128, window_size=128),
  )
  assert_true(pro.auxiliary_prediction() is MultiTokenPrediction(layer_count=1))
}
