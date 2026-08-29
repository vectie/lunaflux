///|
const OPENAI_EMBEDDING_INGRESS_APPROVAL_MARKER : UInt64 = 0x4c554e414f414950UL

///|
/// Explicit embedding-owner attestation for an authenticated deployment
/// ingress that reaches LunaFlux only through the configured loopback socket.
///
/// This capability owns no credential, TLS material, listener, endpoint, or
/// readiness state. It is deliberately absent from the one-argument CLI path.
/// The embedding caller remains responsible for external TLS, network policy,
/// and control endpoints.
pub struct RuntimeOpenAiLoopbackIngressApproval {
  priv marker : UInt64
}

///|
/// Creates the explicit caller-owned approval needed by the embedding-only
/// production binding. Calling this constructor is an authority assertion by
/// the embedding deployment owner; LunaFlux does not infer it from policy,
/// environment, qualification evidence, or listener address.
pub fn RuntimeOpenAiLoopbackIngressApproval::approve_authenticated_loopback() -> RuntimeOpenAiLoopbackIngressApproval {
  { marker: OPENAI_EMBEDDING_INGRESS_APPROVAL_MARKER }
}

///|
fn RuntimeOpenAiLoopbackIngressApproval::is_valid(
  self : RuntimeOpenAiLoopbackIngressApproval,
) -> Bool {
  self.marker == OPENAI_EMBEDDING_INGRESS_APPROVAL_MARKER
}

///|
/// Embedding-only production OpenAI Responses authority. The caller supplies
/// both an opaque startup-owned Bearer policy and an explicit loopback ingress
/// approval. This type is intentionally distinct from qualification policy and
/// has no conversion from qualification evidence.
pub struct RuntimeOpenAiProductionPolicy {
  priv server : RuntimeOpenAiServerPolicy
  priv ingress : RuntimeOpenAiLoopbackIngressApproval
}

///|
/// Captures exact caller-supplied OpenAI server inputs for the embedding API.
/// No credential, endpoint, or transport value is read from global state.
pub fn RuntimeOpenAiProductionPolicy::new(
  ingress : RuntimeOpenAiLoopbackIngressApproval,
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
) -> RuntimeOpenAiProductionPolicy raise RuntimeInstanceError {
  if !ingress.is_valid() {
    raise fail(OpenAiProduction)
  }
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
      OpenAiProduction,
    ),
    ingress,
  }
}
