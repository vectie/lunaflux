///|
async fn Backend::generate_http(
  self : Self,
  workspace : @compat.LunaOpenAIInboundWorkspace,
  incoming : @http1.LunaHttp1View,
  request_id : @inference.RequestId,
  events : @event.LunaEventOwner,
  select_response : (Bool) -> Unit raise,
  emit : async (@event.LunaEventView) -> Unit,
) -> Unit {
  let body = Bytes::makei(incoming.body_length(), i => incoming.body_byte_at(i))
  let normalized = @media.normalize_chat(
    self.model,
    body,
    allow_images=self.vision is Some(_),
  ) catch {
    _ => raise @session.SessionChatError::UnsupportedRequest
  }
  @async.pause()
  let work = workspace.begin_normalized(incoming, request_id, normalized.body())
  defer (work.abort() catch { _ => () })
  while work.state() is LunaOpenAIInboundPending {
    ignore(work.progress())
    @async.pause()
  }
  if work.state() is LunaOpenAIInboundFailed {
    raise work.failure()
  }
  let admitted = work.take_view()
  defer admitted.release()
  select_response(admitted.streaming_response())
  let handoff = admitted.handoff()
  guard self.vision is Some(vision) else {
    // Text uses the same model-owned normalization, without vision preparation.
    self.generate(handoff, events, emit)
    return
  }
  let prepared = @media.prepare(
    self.model,
    self.tokenizer,
    vision,
    normalized.system(),
    normalized.content()[:],
    maximum_tokens=prompt_capacity(handoff),
    maximum_prepared_pixel_bytes=33554432L,
    system_present=normalized.system_present(),
  ) catch {
    _ => raise @session.SessionChatError::UnsupportedRequest
  }
  // Deliver deadline/disconnect cancellation before any device submission.
  @async.pause()
  self.generate_prepared(
    handoff,
    prepared.prompt(),
    prepared.vision_input(),
    events,
    emit,
  )
}
