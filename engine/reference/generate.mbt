///|
/// Generates greedy tokens through bounded full-prefill recomputation.
///
/// Each step reinterprets the complete prompt plus generated suffix, matching
/// the explicit Phase 1 reference algorithm. The prepared weights are reused;
/// no tokenizer, KV cache, scheduler, device, or model-family branch is used.
pub fn generate(
  plan : @plan.ModelPlan,
  model : PreparedModel,
  prompt : ArrayView[Int],
  max_new_tokens~ : Int,
  limits~ : ExecutionLimits,
) -> GenerationResult raise ReferenceExecutionError {
  if max_new_tokens <= 0 {
    raise InvalidGraph(InvalidGenerationLength)
  }
  if prompt.is_empty() || prompt.length() > limits.max_input_tokens() {
    raise InvalidGraph(InputLimitExceeded)
  }
  if max_new_tokens > limits.max_input_tokens() - prompt.length() {
    raise InvalidGraph(InputLimitExceeded)
  }
  if max_new_tokens >
    plan.shape_constraints().max_sequence_tokens() - prompt.length() {
    raise InvalidGraph(InputContextExceeded)
  }
  let tokens = prompt.to_owned()
  let generated : Array[Int] = []
  for _ in 0..<max_new_tokens {
    let result = execute(plan, model, tokens, limits~)
    let token = result.selection().token_id()
    tokens.push(token)
    generated.push(token)
  }
  {
    identity: plan.identity(),
    input_tokens: prompt.length(),
    generated_tokens: ReadOnlyArray::from_array(generated),
  }
}
