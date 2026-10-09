///|
/// Converts the immutable family plan into a correctness-reference traversal.
/// This performs no model math; it verifies layer order and IndexShare state
/// dependencies before tiny numeric DSA/KDA/MoE/mHC fixtures are dispatched.
pub fn build_reference_schedule(
  plan : @glm_hybrid_execution_plan.GlmHybridExecutionPlan,
) -> GlmHybridReferenceSchedule raise GlmHybridReferenceError {
  let steps = []
  let mut dsa_compute_count = 0
  let mut dsa_reuse_count = 0
  let mut kda_count = 0
  let mut dense_count = 0
  let mut moe_count = 0
  let mut last_dsa_compute : Int? = None
  for expected_index, layer in plan.layers() {
    if layer.layer_index() != expected_index {
      raise InvalidLayerOrder
    }
    let attention : GlmReferenceAttentionStep = match layer.attention() {
      DynamicSparse(ComputeTopK) => {
        dsa_compute_count += 1
        last_dsa_compute = Some(expected_index)
        DsaCompute
      }
      DynamicSparse(ReusePreviousTopK) => {
        guard last_dsa_compute is Some(source) else { raise MissingDsaState }
        dsa_reuse_count += 1
        DsaReuse(source_layer=source)
      }
      KimiDelta => {
        kda_count += 1
        last_dsa_compute = None
        Kda
      }
    }
    let feed_forward : GlmReferenceFeedForwardStep = match
      layer.feed_forward() {
      DenseSwiGlu => {
        dense_count += 1
        Dense
      }
      RoutedMoe => {
        moe_count += 1
        Moe
      }
    }
    steps.push({
      layer_index: expected_index,
      attention,
      feed_forward,
      hyper_connection: plan.hyper_connection() is Some(_),
    })
  }
  let hyper_connection_site_count = if plan.hyper_connection() is Some(_) {
    steps.length() * 2
  } else {
    0
  }
  let vision_layer_count = match plan.vision_conditioning() {
    Some(vision) => vision.layer_count()
    None => 0
  }
  {
    steps: ReadOnlyArray::from_array(steps),
    dsa_compute_count,
    dsa_reuse_count,
    kda_count,
    dense_count,
    moe_count,
    hyper_connection_site_count,
    vision_layer_count,
  }
}
