///|
/// Caller-authorized, loopback-only OpenAI Responses qualification inputs.
/// The authentication policy is already admitted by the caller and remains
/// opaque. This owner performs no environment or filesystem lookup.
pub struct RuntimeOpenAiQualificationPolicy {
  priv server : RuntimeOpenAiServerPolicy
}

///|
/// Shared immutable server inputs. Public authority is owned by the distinct
/// qualification and embedding-production wrappers, never by this private
/// value.
priv struct RuntimeOpenAiServerPolicy {
  http_limits : @http1.LunaHttp1Limits
  http_budget : @http1.LunaHttp1StepBudget
  response_budget : @http1.LunaHttp1ResponseStepBudget
  auth_policy : @api_auth.LunaApiAuthPolicy
  inbound_budget : @openai_compat.LunaOpenAICompatStepBudget
  outbound_budget : @openai_compat.LunaOpenAICompatStepBudget
  inbound_limits : @openai_compat.LunaOpenAIInboundLimits
  template : @openai_compat.LunaOpenAIChatTemplate
  model_alias : FixedArray[Byte]
  response_id_prefix : FixedArray[Byte]
  cache_scope : @inference.CacheScope
  max_new_tokens : Int
  context_ceiling : Int
  sampling : @inference.SamplingParameters
  sampling_seed : @inference.SamplingSeed
  deadline : @inference.DeadlineBudget
}

///|
/// Captures explicit pre-admitted compatibility capabilities without deriving
/// a credential or endpoint from process-global state.
pub fn RuntimeOpenAiQualificationPolicy::new(
  http_limits : @http1.LunaHttp1Limits,
  http_budget : @http1.LunaHttp1StepBudget,
  response_budget : @http1.LunaHttp1ResponseStepBudget,
  auth_policy : @api_auth.LunaApiAuthPolicy,
  inbound_budget : @openai_compat.LunaOpenAICompatStepBudget,
  outbound_budget : @openai_compat.LunaOpenAICompatStepBudget,
  inbound_limits : @openai_compat.LunaOpenAIInboundLimits,
  template : @openai_compat.LunaOpenAIChatTemplate,
  model_alias : FixedArray[Byte],
  response_id_prefix : FixedArray[Byte],
  cache_scope : @inference.CacheScope,
  max_new_tokens : Int,
  context_ceiling : Int,
  sampling : @inference.SamplingParameters,
  sampling_seed : @inference.SamplingSeed,
  deadline : @inference.DeadlineBudget,
) -> RuntimeOpenAiQualificationPolicy raise RuntimeInstanceError {
  {
    server: new_openai_server_policy(
      http_limits,
      http_budget,
      response_budget,
      auth_policy,
      inbound_budget,
      outbound_budget,
      inbound_limits,
      template,
      model_alias,
      response_id_prefix,
      cache_scope,
      max_new_tokens,
      context_ceiling,
      sampling,
      sampling_seed,
      deadline,
      OpenAiQualification,
    ),
  }
}

///|
fn new_openai_server_policy(
  http_limits : @http1.LunaHttp1Limits,
  http_budget : @http1.LunaHttp1StepBudget,
  response_budget : @http1.LunaHttp1ResponseStepBudget,
  auth_policy : @api_auth.LunaApiAuthPolicy,
  inbound_budget : @openai_compat.LunaOpenAICompatStepBudget,
  outbound_budget : @openai_compat.LunaOpenAICompatStepBudget,
  inbound_limits : @openai_compat.LunaOpenAIInboundLimits,
  template : @openai_compat.LunaOpenAIChatTemplate,
  model_alias : FixedArray[Byte],
  response_id_prefix : FixedArray[Byte],
  cache_scope : @inference.CacheScope,
  max_new_tokens : Int,
  context_ceiling : Int,
  sampling : @inference.SamplingParameters,
  sampling_seed : @inference.SamplingSeed,
  deadline : @inference.DeadlineBudget,
  failure_stage : RuntimeInstanceStage,
) -> RuntimeOpenAiServerPolicy raise RuntimeInstanceError {
  if model_alias.length() == 0 ||
    response_id_prefix.length() == 0 ||
    max_new_tokens <= 0 ||
    context_ceiling <= 0 {
    raise fail(failure_stage)
  }
  {
    http_limits,
    http_budget,
    response_budget,
    auth_policy,
    inbound_budget,
    outbound_budget,
    inbound_limits,
    template,
    model_alias,
    response_id_prefix,
    cache_scope,
    max_new_tokens,
    context_ceiling,
    sampling,
    sampling_seed,
    deadline,
  }
}
