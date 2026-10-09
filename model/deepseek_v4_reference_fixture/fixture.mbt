///|
fn[T] reference(
  operation : () -> T raise @advanced_decoder_reference.AdvancedDecoderReferenceError,
) -> T raise DeepSeekV4ReferenceFixtureError {
  operation() catch {
    error => raise ReferenceFailure(error)
  }
}

///|
fn validate_flash_0731_plan(
  semantic : @deepseek_v4.DeepSeekV4SemanticPlan,
) -> Unit raise DeepSeekV4ReferenceFixtureError {
  let spec = semantic.metadata().spec()
  if spec.profile() != Flash0731 {
    raise UnsupportedFixtureProfile
  }
  let plan = semantic.execution_plan()
  let shape = plan.shape()
  let moe = plan.moe()
  let hyper = plan.hyper_connection()
  if plan.schema_version() != 1 ||
    plan.profile_identity() != "deepseek-ai/DeepSeek-V4-Flash-0731/v1" ||
    plan.content_digest() != semantic.content_digest() ||
    shape.decoder_layer_count() != 43 ||
    shape.vocabulary_size() != 129280 ||
    moe.routed_expert_count() != 256 ||
    moe.shared_expert_count() != 1 ||
    moe.experts_per_token() != 6 ||
    moe.score_function() != SqrtSoftplus ||
    !moe.normalize_selected_probabilities() ||
    moe.routed_scaling_factor() != 1.5 ||
    moe.selection_method() !=
    TokenHashThenNoAuxiliaryLossTokenChoice(hash_layer_count=3) ||
    hyper.width() != 4 ||
    hyper.sinkhorn_iterations() != 20 ||
    plan.layers().length() != 43 {
    raise InvalidFixturePlan
  }
  match plan.layers()[0].attention() {
    SlidingWindowCausal(window_size=128) => ()
    _ => raise InvalidFixturePlan
  }
  match plan.layers()[2].attention() {
    IndexedCompressed(
      compression_ratio=4,
      index_head_count=64,
      index_head_dimension=128,
      index_top_k=512
    ) => ()
    _ => raise InvalidFixturePlan
  }
  match plan.layers()[3].attention() {
    DeterministicCompressed(compression_ratio=128, window_size=128) => ()
    _ => raise InvalidFixturePlan
  }
  match plan.auxiliary_prediction() {
    MultiTokenPredictionWithDspark(layer_count=3, dspark~) =>
      if dspark.block_size() != 5 ||
        dspark.noise_token_id() != 128799 ||
        dspark.target_layer_ids() != [40, 41, 42] ||
        dspark.markov_rank() != 256 {
        raise InvalidFixturePlan
      }
    _ => raise InvalidFixturePlan
  }
}

///|
fn fixture_logits(expert_count : Int) -> Array[Float] {
  let logits = Array::make(expert_count, -10.0F)
  logits[5] = 6.0F
  logits[7] = 5.0F
  logits[11] = 4.0F
  logits[13] = 3.0F
  logits[17] = 2.0F
  logits[19] = 1.0F
  logits
}

///|
/// Runs a bounded, deterministic oracle fixture driven exclusively by the
/// immutable Flash-0731 execution plan. This allocates tiny Float tensors and
/// must never be substituted for production model execution.
pub fn run_flash_0731(
  semantic : @deepseek_v4.DeepSeekV4SemanticPlan,
  limits~ : @advanced_decoder_reference.ReferenceLimits,
) -> DeepSeekV4ReferenceFixtureResult raise DeepSeekV4ReferenceFixtureError {
  validate_flash_0731_plan(semantic)
  let plan = semantic.execution_plan()
  let moe = plan.moe()
  let logits = fixture_logits(moe.routed_expert_count())
  let top_k_routing = reference(() => {
    @advanced_decoder_reference.route_top_k(
      logits,
      [],
      top_k=moe.experts_per_token(),
      score_function=moe.score_function(),
      normalize=moe.normalize_selected_probabilities(),
      scaling=Float::from_double(moe.routed_scaling_factor()),
      limits~,
    )
  })
  let hash_routing = reference(() => {
    @advanced_decoder_reference.route_token_hash(
      logits,
      [19, 17, 13, 11, 7, 5],
      vocabulary_size=1,
      experts_per_token=moe.experts_per_token(),
      token_id=0,
      score_function=moe.score_function(),
      normalize=moe.normalize_selected_probabilities(),
      scaling=Float::from_double(moe.routed_scaling_factor()),
      limits~,
    )
  })
  let routed_outputs : Array[Float] = []
  for route in 0..<hash_routing.selected_count() {
    let value = Float::from_int(route + 1)
    routed_outputs.push(value)
    routed_outputs.push(value * 2.0F)
  }
  let expert_output = reference(() => {
    @advanced_decoder_reference.combine_expert_outputs(
      hash_routing,
      routed_outputs,
      [0.25F, 0.5F],
      width=2,
      limits~,
    )
  })
  let (index_head_count, index_head_dimension) = match
    plan.layers()[2].attention() {
    IndexedCompressed(index_head_count~, index_head_dimension~, ..) =>
      (index_head_count, index_head_dimension)
    _ => raise InvalidFixturePlan
  }
  let query = Array::make(index_head_count * index_head_dimension, 0.0F)
  let keys = Array::make(2 * index_head_dimension, 0.0F)
  let head_weights = Array::make(index_head_count, 0.0F)
  query[0] = 1.0F
  keys[0] = 1.0F
  keys[index_head_dimension] = 2.0F
  head_weights[0] = 1.0F
  let learned_indices = reference(() => {
    @advanced_decoder_reference.learned_index_top_k(
      query,
      keys,
      head_weights,
      head_count=index_head_count,
      head_dimension=index_head_dimension,
      top_k=2,
      limits~,
    )
  })
  let compression_ratio = match plan.layers()[3].attention() {
    DeterministicCompressed(compression_ratio~, window_size=128) =>
      compression_ratio
    _ => raise InvalidFixturePlan
  }
  let compressed_indices = reference(() => {
    @advanced_decoder_reference.deterministic_compressed_indices(
      compression_ratio~,
      query_position=compression_ratio * 2 - 1,
      cache_offset=7,
      limits~,
    )
  })
  let hyper = plan.hyper_connection()
  let stream_values : Array[Float] = [
    1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F,
  ]
  let control_count = 6 * hyper.width()
  let controls = reference(() => {
    @advanced_decoder_reference.hyper_connection_block_controls_v1(
      stream_values,
      Array::make(control_count * stream_values.length(), 0.0F),
      Array::make(control_count, 0.0F),
      [1.0F, 1.0F, 1.0F],
      row_count=1,
      stream_count=hyper.width(),
      feature_width=2,
      normalization_epsilon=Float::from_double(hyper.normalization_epsilon()),
      control_epsilon=Float::from_double(hyper.control_epsilon()),
      sinkhorn_iterations=hyper.sinkhorn_iterations(),
      limits~,
    )
  })
  let reduced = reference(() => {
    @advanced_decoder_reference.hyper_connection_pre_reduce_v1(
      stream_values,
      controls.pre(),
      row_count=1,
      stream_count=hyper.width(),
      feature_width=2,
      limits~,
    )
  })
  let mixed_streams = reference(() => {
    @advanced_decoder_reference.hyper_connection_post_combine_v1(
      reduced,
      stream_values,
      controls.post(),
      controls.combination(),
      row_count=1,
      stream_count=hyper.width(),
      feature_width=2,
      limits~,
    )
  })
  let auxiliary_stages = reference(() => {
    @advanced_decoder_reference.auxiliary_stage_order(
      plan.auxiliary_prediction(),
      limits~,
    )
  })
  {
    plan_digest: plan.plan_digest(),
    top_k_routing,
    hash_routing,
    expert_output,
    learned_indices,
    compressed_indices,
    mixed_streams,
    auxiliary_stages,
  }
}
