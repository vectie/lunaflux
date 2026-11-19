///|
async fn publish(
  events : @event.LunaEventOwner,
  emit : async (@event.LunaEventView) -> Unit,
) -> Unit {
  let view = events.view()
  defer events.retire(view)
  emit(view)
}

///|
/// Handoff already contains the model's rendered prompt. Encode it once; never
/// apply the chat template again or retokenize during decode.
pub async fn Backend::generate(
  self : Self,
  handoff : @inference.LunaTextRequestHandoffLease,
  events : @event.LunaEventOwner,
  emit : async (@event.LunaEventView) -> Unit,
) -> Unit {
  guard self.healthy && !self.busy else {
    raise @session.SessionChatError::InvalidLifecycle
  }
  let prompt_capacity = prompt_capacity(handoff)
  let bytes = Bytes::makei(handoff.prompt_length(), index => {
    handoff.prompt_byte_at(index)
  })
  let encoded = self.tokenizer.encode_text(
    @utf8.decode(bytes),
    special_policy=Allow,
    output_limit=prompt_capacity,
    truncation=RejectOverflow,
  )
  let prompt = @prompt.prepare_token_prompt(
    self.model,
    encoded.token_ids,
    maximum_tokens=prompt_capacity,
  )
  self.generate_prepared(handoff, prompt, None, events, emit)
}

///|
fn prompt_capacity(
  handoff : @inference.LunaTextRequestHandoffLease,
) -> Int32 raise {
  // HTTP buffering is owned by Service. Canonical ingress deliberately keeps
  // engine-side token events streaming even for a buffered HTTP response.
  guard handoff.validated_sampling().materialize().mode() is Greedy else {
    raise @session.SessionChatError::UnsupportedRequest
  }
  let effective = handoff.effective_limits()
  let prompt_capacity = effective.context_ceiling() - effective.max_new_tokens()
  guard effective.context_ceiling() <= 4096 && prompt_capacity > 0 else {
    raise @session.SessionChatError::UnsupportedRequest
  }
  prompt_capacity
}

///|
async fn Backend::generate_prepared(
  self : Self,
  handoff : @inference.LunaTextRequestHandoffLease,
  prompt : @prompt.PreparedTextPrompt,
  image : (@multimodal.ImagePrompt, @vision.Input)?,
  events : @event.LunaEventOwner,
  emit : async (@event.LunaEventView) -> Unit,
) -> Unit {
  guard self.healthy && !self.busy else {
    raise @session.SessionChatError::InvalidLifecycle
  }
  self.busy = true
  defer {
    self.busy = false
  }
  ignore(prompt_capacity(handoff))
  let effective = handoff.effective_limits()
  let semantics = handoff.semantic_view()
  let stops = ReadOnlyArray::from_array(
    Array::makei(semantics.stop_token_count(), index => {
      semantics.stop_token_at(index)
    }),
  )
  let setup = self.output.begin_luna_request_semantics(
    semantics,
    special_policy=Preserve,
  )
  defer (setup.abort() catch { _ => () })
  while setup.progress() is LunaIncrementalOutputPending {
    @async.pause()
  }
  let output = setup.take_lease()
  defer output.release()
  self.healthy = false
  let mut complete = false
  defer (if !complete {
    self.driver.cancel()
    self.healthy = true
  })
  self.driver.begin(
    prompt,
    image,
    maximum_new_tokens=effective.max_new_tokens(),
    stop_token_ids=stops,
    context_ceiling=effective.context_ceiling(),
  )
  events.publish_accepted(handoff.model_identity(), effective)
  publish(events, emit)
  if image is Some(_) {
    while !self.driver.poll() {
      @async.pause()
    }
  }
  let mut published = 0
  let mut string_stopped = false
  while self.driver.submit_next() {
    while !self.driver.poll() {
      @async.pause()
    }
    let tokens = self.driver.emitted_tokens()
    while published < tokens.length() {
      let token = tokens[published]
      let status = output.push_token_status(
        token,
        self.delta,
        destination_offset=0,
      )
      guard status.is_ready() else {
        raise @session.SessionChatError::InvalidLifecycle
      }
      guard events.publish_token_status(
          token_id=token,
          position=published,
          self.delta,
          source_offset=0,
          length=status.copied_bytes(),
        )
        is LunaEventPublishReady else {
        raise @session.SessionChatError::InvalidLifecycle
      }
      publish(events, emit)
      published += 1
      if status.stop_matched() {
        string_stopped = true
        break
      }
    }
    if string_stopped {
      self.driver.cancel()
      break
    }
    @async.pause()
  }
  let reason = if string_stopped {
    @inference.FinishReason::StopSequence
  } else {
    self.driver.finish_reason().unwrap()
  }
  let tail_length = if output.is_stopped() {
    0
  } else {
    let status = output.finish_into_status(
      self.delta,
      destination_offset=0,
      boundary=if reason is TokenLimit { TokenLimit } else { CompleteText },
    )
    guard status.is_ready() else {
      raise @session.SessionChatError::InvalidLifecycle
    }
    status.copied_bytes()
  }
  events.publish_completed(
    reason,
    input_tokens=prompt.tokens().length(),
    cached_input_tokens=0,
    output_tokens=self.driver.sampled_tokens(),
    self.delta,
    tail_offset=0,
    tail_length~,
  )
  publish(events, emit)
  complete = true
  self.healthy = true
}
