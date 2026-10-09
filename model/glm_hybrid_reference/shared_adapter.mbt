///|
/// Executes bounded deterministic micro-fixtures through the shared advanced
/// decoder MoE and hyper-connection reference primitives using the GLM plan's
/// exact routing scale, score function, normalization, width, epsilon, and
/// Sinkhorn iteration count.
pub fn exercise_shared_advanced_primitives(
  plan : @glm_hybrid_execution_plan.GlmHybridExecutionPlan,
  limits~ : @advanced_decoder_reference.ReferenceLimits,
) -> GlmSharedAdvancedFixture raise @advanced_decoder_reference.AdvancedDecoderReferenceError {
  let selected_count = plan.moe().experts_per_token()
  let logits : Array[Float] = [
    for index in 0..<selected_count => {
      Float::from_int(index - selected_count / 2) / 4.0F
    }
  ]
  let bias = Array::make(selected_count, 0.0F)
  let routing = @advanced_decoder_reference.route_top_k(
    logits,
    bias,
    top_k=selected_count,
    score_function=plan.moe().score_function(),
    normalize=plan.moe().normalize_selected_probabilities(),
    scaling=Float::from_double(plan.moe().routed_scaling_factor()),
    limits~,
  )
  let routed_outputs : Array[Float] = [
    for index in 0..<selected_count => Float::from_int(index + 1)
  ]
  let shared_outputs = Array::make(plan.moe().shared_expert_count(), 0.5F)
  let moe = @advanced_decoder_reference.combine_expert_outputs(
    routing,
    routed_outputs,
    shared_outputs,
    width=1,
    limits~,
  )
  let mixed_hyper_streams = match plan.hyper_connection() {
    None => ReadOnlyArray::from_array([])
    Some(hyper) => {
      let width = hyper.width()
      let logits : Array[Float] = []
      for row in 0..<width {
        for column in 0..<width {
          logits.push(if row == column { 1.0F } else { 0.0F })
        }
      }
      let matrix = @advanced_decoder_reference.sinkhorn_normalize(
        logits,
        width~,
        iterations=hyper.sinkhorn_iterations(),
        epsilon=Float::from_double(hyper.epsilon()),
        limits~,
      )
      let streams : Array[Float] = [
        for index in 0..<width => Float::from_int(index + 1)
      ]
      @advanced_decoder_reference.mix_hyper_connections(
        streams,
        matrix,
        stream_count=width,
        feature_width=1,
        limits~,
      )
    }
  }
  { moe_output: moe[0], mixed_hyper_streams, }
}
