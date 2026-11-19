///|
pub struct Backend {
  priv model : @model.Spec
  priv tokenizer : @tokenizer.TokenizerSpec
  priv driver : Driver
  priv vision : @vision.Program?
  priv limits : @inference.InferenceLimits
  priv output : @output.LunaIncrementalOutputWorkspace
  priv delta : FixedArray[Byte]
  priv mut busy : Bool
  priv mut healthy : Bool
}

///|
pub fn inference_limits() -> @inference.InferenceLimits raise {
  @inference.InferenceLimits::new(
    max_text_bytes=32768,
    max_input_tokens=4096,
    max_new_tokens=4096,
    max_context_tokens=4096,
    max_token_id=248143,
    max_stop_token_ids=16,
    max_stop_strings=8,
    max_stop_string_bytes=128,
    max_trace_bytes=128,
    max_cache_scope_bytes=64,
    max_decoded_delta_bytes=4096,
    max_deadline_millis=300000UL,
    max_top_k=248144,
    max_temperature=10.0,
  )
}

///|
/// Borrow a prepared resident runner. Close/cancel the HTTP service before
/// releasing that session. No actor or robot state belongs to this adapter.
pub fn Backend::new(
  model : @model.Spec,
  tokenizer : @tokenizer.TokenizerSpec,
  runner : @request.Runner,
) -> Self raise {
  new_backend(model, tokenizer, Text(runner), None)
}

///|
/// Borrow a prepared combined resident execution. The same endpoint accepts
/// text strings and inline PNG/JPEG image content lists, without file/URL fetch.
pub fn Backend::with_images(
  model : @model.Spec,
  tokenizer : @tokenizer.TokenizerSpec,
  vision : @vision.Program,
  execution : @image_runtime.Execution,
) -> Self raise {
  new_backend(model, tokenizer, Image(execution), Some(vision))
}

///|
fn new_backend(
  model : @model.Spec,
  tokenizer : @tokenizer.TokenizerSpec,
  driver : Driver,
  vision : @vision.Program?,
) -> Backend raise {
  let limits = inference_limits()
  {
    model,
    tokenizer,
    driver,
    vision,
    limits,
    output: @output.LunaIncrementalOutputWorkspace::new(
      tokenizer,
      limits,
      @output.LunaIncrementalOutputStepBudget::new(4096),
    ),
    delta: FixedArray::make(4096, b'\x00'),
    busy: false,
    healthy: true,
  }
}

///|
pub fn Backend::is_ready(self : Self) -> Bool {
  self.healthy && !self.busy
}

///|
pub fn Backend::service(
  self : Self,
  policy : @auth.LunaApiAuthPolicy,
  model_alias~ : String = "MiniCPM-V-4.7-35B-A3B",
  timeout_ms~ : Int32 = 30000,
) -> @session.Service raise {
  let model_name = @utf8.encode(model_alias).to_fixedarray()
  let prefix = @utf8.encode("minicpm_").to_fixedarray()
  let identity = @spec.ModelIdentity::new(
    @spec.ContentDigest::from_sha256("0".repeat(64)),
    @spec.PlanDigest::from_sha256("0".repeat(64)),
  )
  let inbound = @compat.LunaOpenAIInboundWorkspace::new(
    self.limits,
    @compat.LunaOpenAICompatStepBudget::new(4096),
    @compat.LunaOpenAIInboundLimits::new(
      max_messages=2,
      required_stop_tokens=@prompt.stop_token_ids(),
    ),
    @prompt.chat_template(self.limits, max_rendered_prompt_bytes=32768),
    model_name,
    model_alias_offset=0,
    model_alias_length=model_name.length(),
    identity,
    @inference.CacheScope::from_ascii(b"minicpm", self.limits),
    max_new_tokens=256,
    context_ceiling=4096,
    @inference.SamplingParameters::greedy(),
    @inference.SamplingSeed::new(1UL),
    @inference.DeadlineBudget::from_milliseconds(
      timeout_ms.to_uint64(),
      self.limits,
    ),
  )
  let presentation = @compat.LunaOpenAICompatWorkspace::new(
    self.limits,
    @compat.LunaOpenAICompatStepBudget::new(4096),
    model_name,
    model_alias_offset=0,
    model_alias_length=model_name.length(),
    prefix,
    response_id_prefix_offset=0,
    response_id_prefix_length=prefix.length(),
    buffered=true,
  )
  @session.Service::with_handler(
    self.limits,
    policy,
    presentation,
    timeout_ms~,
    maximum_body_bytes=if self.vision is Some(_) { 16777216 } else { 1048576 },
    handle=(incoming, request_id, events, select_response, emit) => {
      self.generate_http(
        inbound, incoming, request_id, events, select_response, emit,
      )
    },
    ready=() => self.is_ready(),
  )
}
