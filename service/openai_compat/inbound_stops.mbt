///|
fn LunaOpenAIInboundWorkspace::begin_render(
  self : Self,
) -> Unit raise LunaOpenAICompatError {
  let sampling = if self.sampling_mode == 0 {
    @inference.LunaValidatedSampling::greedy()
  } else {
    @inference.LunaValidatedSampling::sample(
      temperature=self.temperature,
      has_top_k=self.has_top_k,
      top_k=self.top_k,
      has_top_p=self.has_top_p,
      top_p=self.top_p,
      self.inference,
    ) catch {
      _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
    }
  }
  let prompt_limit = if self.route == 2 {
    self.template.max_rendered_prompt_bytes
  } else {
    self.inference.max_text_bytes()
  }
  self.handoff_writes.push(
    self.handoff_workspace.begin(
      prompt_byte_limit=prompt_limit,
      stop_token_count=self.stop_token_count,
      stop_string_count=self.stop_string_count,
      request_id=self.request_ids[0],
      effective=self.effective,
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
    self.render_segment = match self.message_cells[0] {
      1 => 0
      2 => 2
      _ => 4
    }
    self.render_segment_index = 0
    self.phase = LUNA_INBOUND_RENDER_SEGMENT
  }
}

///|
