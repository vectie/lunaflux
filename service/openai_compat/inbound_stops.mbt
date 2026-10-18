///|
fn LunaOpenAIInboundWorkspace::begin_render(
  self : Self,
) -> Unit raise LunaOpenAICompatError {
  // Merge before opening the typed handoff: its declared stop count must be
  // identical to the number subsequently published. Empty user stops cannot
  // disable the deployed model's EOS set.
  for token in self.limits.required_stop_tokens {
    let mut present = false
    for i in 0..<self.stop_token_count { if self.stop_tokens[i] == token { present = true } }
    if !present {
      if self.stop_token_count == self.stop_tokens.length() { raise LunaOpenAICompatInvalid(LunaOpenAICompatCapacity) }
      self.stop_tokens[self.stop_token_count] = token
      self.stop_token_count += 1
    }
  }
  let sampling = if self.sampling_mode == 0 {
    @inference.LunaValidatedSampling::greedy()
  } else {
    // Zero temperature is an explicit outer-API translation to native Greedy.
    // Validate filters even in greedy mode; malformed values are not ignored.
    let checked = @inference.LunaValidatedSampling::sample(
      temperature=if self.temperature == 0.0 {
        self.inference.max_temperature()
      } else {
        self.temperature
      },
      has_top_k=self.has_top_k,
      top_k=self.top_k,
      has_top_p=self.has_top_p,
      top_p=self.top_p,
      self.inference,
    ) catch {
      _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
    }
    if self.temperature == 0.0 {
      @inference.LunaValidatedSampling::greedy()
    } else {
      checked
    }
  }
  let prompt_limit = if self.route == 2 {
    self.template.max_rendered_prompt_bytes
  } else {
    self.inference.max_text_bytes()
  }
  // Chat's modern field takes precedence regardless of JSON key order, like
  // the existing vLLM request adapter. Responses uses max_output_tokens.
  let requested = if self.requested_max_completion_tokens > 0 {
    self.requested_max_completion_tokens
  } else {
    self.requested_max_tokens
  }
  let effective = if requested > 0 {
    @inference.EffectiveLimits::new(
      max_new_tokens=requested,
      context_ceiling=self.effective.context_ceiling(),
      self.inference,
    ) catch {
      _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
    }
  } else {
    self.effective
  }
  self.handoff_writes.push(
    self.handoff_workspace.begin(
      prompt_byte_limit=prompt_limit,
      stop_token_count=self.stop_token_count,
      stop_string_count=self.stop_string_count,
      request_id=self.request_ids[0],
      effective~,
      sampling~,
      sampling_seed=@inference.SamplingSeed::new(self.sampling_seed_value) catch {
        _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
      },
      stream_preference=Streaming,
      deadline=self.deadline,
    ) catch {
      _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
    },
  )
  if self.route == 1 {
    self.render_source_start = self.input_start
    self.render_source_length = self.input_length
    self.render_source_index = 0
    self.escape_state = 0
    self.utf8_remaining = 0
    self.high_surrogate = 0
    self.phase = LUNA_INBOUND_RENDER_STRING
  } else {
    self.render_message = 0
    self.render_segment = if self.template.segment_length(7) > 0 { 7 } else {
      match self.message_cells[0] { 1 => 0; 2 => 2; _ => 4 }
    }
    self.render_segment_index = 0
    self.phase = LUNA_INBOUND_RENDER_SEGMENT
  }
}

///|
