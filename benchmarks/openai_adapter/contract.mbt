///|
pub fn OpenAIBaselineIdentity::new(
  engine : @evidence.BenchmarkEngine,
  revision : @evidence.BenchmarkDigest,
  image : @evidence.BenchmarkDigest,
  configuration : @evidence.BenchmarkDigest,
) -> OpenAIBaselineIdentity {
  { engine, revision, image, configuration }
}

///|
pub fn OpenAIBaselineInputs::new(
  profile : @evidence.BenchmarkProfile,
  tokenizer : @evidence.BenchmarkDigest,
  corpus : @evidence.BenchmarkDigest,
  protocol : @evidence.BenchmarkDigest,
  workload : @evidence.BenchmarkDigest,
  request_input_tokens : FixedArray[Int64],
) -> OpenAIBaselineInputs raise OpenAIBaselineAdapterError {
  if request_input_tokens.length() <= 0 ||
    request_input_tokens.length() >
    @evidence.maximum_request_events_per_summary() {
    raise OpenAIBaselineInvalidInputs
  }
  let owned = FixedArray::make(request_input_tokens.length(), 0L)
  for index, count in request_input_tokens {
    if count < 0L {
      raise OpenAIBaselineInvalidInputs
    }
    owned[index] = count
  }
  {
    profile,
    tokenizer,
    corpus,
    protocol,
    workload,
    request_input_tokens: owned,
  }
}

///|
fn openai_baseline_token_counts_equal(
  left : FixedArray[Int64],
  right : FixedArray[Int64],
) -> Bool {
  if left.length() != right.length() {
    return false
  }
  for index, count in left {
    if count != right[index] {
      return false
    }
  }
  true
}

///|
/// Requires the shared tokenizer, corpus, protocol, profile configuration,
/// request count, and tokenizer-verified per-request token lengths to match.
pub fn require_identical_openai_baseline_inputs(
  left : OpenAIBaselineInputs,
  right : OpenAIBaselineInputs,
) -> Unit raise OpenAIBaselineAdapterError {
  if left.profile != right.profile ||
    left.tokenizer != right.tokenizer ||
    left.corpus != right.corpus ||
    left.protocol != right.protocol ||
    left.workload != right.workload ||
    !openai_baseline_token_counts_equal(
      left.request_input_tokens,
      right.request_input_tokens,
    ) {
    raise OpenAIBaselineInputMismatch
  }
}

///|
/// Establishes one exact Chat Completions input declaration across LunaFlux,
/// vLLM, and SGLang before any comparison trial starts.
pub fn require_identical_openai_comparison_inputs(
  lunaflux : OpenAIBaselineInputs,
  vllm : OpenAIBaselineInputs,
  sglang : OpenAIBaselineInputs,
) -> Unit raise OpenAIBaselineAdapterError {
  require_identical_openai_baseline_inputs(lunaflux, vllm)
  require_identical_openai_baseline_inputs(lunaflux, sglang)
}

///|
pub fn OpenAIBaselineTrialContract::new(
  identity : OpenAIBaselineIdentity,
  inputs : OpenAIBaselineInputs,
  ordinal : Int,
  order_position : Int,
  maximum_event_bytes : Int,
  maximum_json_depth : Int,
) -> OpenAIBaselineTrialContract raise OpenAIBaselineAdapterError {
  if ordinal < 1 || ordinal > 3 || order_position < 0 || order_position > 2 {
    raise OpenAIBaselineInvalidInputs
  }
  if maximum_event_bytes < 64 ||
    maximum_event_bytes > 1048576 ||
    maximum_json_depth <= 0 ||
    maximum_json_depth > 32 {
    raise OpenAIBaselineInvalidLimits
  }
  {
    identity,
    inputs,
    ordinal,
    order_position,
    maximum_event_bytes,
    maximum_json_depth,
  }
}
