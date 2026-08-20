///|
struct FoundationSession {
  instance : @online_session.LunaOnlineInstance
  ticket : @online_session.LunaOnlineRequestTicket
}

///|
fn prepare_foundation_instance(
  executable : Bytes,
  expected_tokenizer : @tokenizer.TokenizerDigest,
) -> @online_session.LunaOnlineInstance raise {
  let model = @approved_fs.ApprovedRoot::open_absolute("/private/tmp")
  let kernel = @approved_fs.ApprovedRoot::open_absolute("/private/tmp")
  let prepared = @online_session.prepare_owned_luna_online_instance(
    expected_tokenizer,
    identity(),
    inference_limits(),
    @framed_wire.FramedWireLimits::new(inference_limits(), max_frame_bytes=4096),
    scheduler_blueprint(),
    service_binding(),
    executable,
    startup(),
    bootstrap_source(),
    process_limits(),
    model,
    kernel,
  )
  model.close()
  kernel.close()
  assert_eq(prepared.state(), LunaOnlineInstanceReady)
  prepared.take_ready()
}

///|
fn prepare_default_foundation_instance(
  executable : Bytes,
) -> @online_session.LunaOnlineInstance raise {
  prepare_foundation_instance(executable, session_tokenizer().digest())
}

///|
fn prepare_foundation_request(
  tokenizer : @tokenizer.TokenizerSpec,
  received : @request_admission.ReceivedRequest,
) -> @request_admission.LunaPreparedRequest raise {
  @request_admission.prepare_luna_request(
    received,
    tokenizer,
    tokenizer.digest(),
    identity(),
    inference_limits(),
    @monotonic_clock.MonotonicClock::system(),
  )
}

///|
fn begin_foundation_session(
  instance : @online_session.LunaOnlineInstance,
  prepared : @request_admission.LunaPreparedRequest,
) -> FoundationSession raise {
  match instance.begin(prepared) {
    LunaOnlineInstanceAdmitted(ticket) => { instance, ticket }
    LunaOnlineInstanceBusy =>
      raise Failure("idle online instance rejected a request as busy")
    LunaOnlineInstanceDraining =>
      raise Failure("live online instance rejected a request as draining")
  }
}

///|
fn begin_default_foundation_session(
  instance : @online_session.LunaOnlineInstance,
  received : @request_admission.ReceivedRequest,
) -> FoundationSession raise {
  begin_foundation_session(
    instance,
    prepare_foundation_request(session_tokenizer(), received),
  )
}

///|
fn FoundationSession::has_event(self : FoundationSession) -> Bool raise {
  self.instance.has_event(self.ticket)
}

///|
fn FoundationSession::event_length(self : FoundationSession) -> Int raise {
  self.instance.event_length(self.ticket)
}

///|
fn FoundationSession::copy_event_to(
  self : FoundationSession,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> Int raise {
  self.instance.copy_event_to(self.ticket, destination, destination_offset~)
}

///|
fn FoundationSession::ack_event(self : FoundationSession) -> Unit raise {
  self.instance.ack_event(self.ticket)
}

///|
fn FoundationSession::progress(
  self : FoundationSession,
) -> @online_session.LunaOnlineInstanceProgress raise {
  self.instance.progress(self.ticket)
}

///|
fn FoundationSession::progress_terminalization(
  self : FoundationSession,
) -> @online_session.LunaOnlineInstanceProgress raise {
  self.instance.progress_terminalization(self.ticket)
}

///|
fn FoundationSession::request_cancel(self : FoundationSession) -> Unit raise {
  self.instance.request_cancel(self.ticket)
}

///|
fn FoundationSession::check_deadline(self : FoundationSession) -> Unit raise {
  self.instance.check_deadline(self.ticket)
}

///|
fn FoundationSession::begin_abort(self : FoundationSession) -> Unit raise {
  self.instance.begin_abort(self.ticket)
}

///|
fn FoundationSession::progress_retirement(
  self : FoundationSession,
) -> @online_session.LunaOnlineInstanceRetirementProgress raise {
  self.instance.progress_request_retirement(self.ticket)
}

///|
fn drain_foundation_instance(
  instance : @online_session.LunaOnlineInstance,
) -> Unit raise {
  instance.begin_drain()
  let mut attempts = 0
  while attempts < 1000000 {
    match instance.progress_shutdown() {
      LunaOnlineInstanceShutdownClosed => return
      LunaOnlineInstanceShutdownPending => attempts += 1
    }
  }
  raise Failure("online instance did not complete bounded shutdown")
}

///|
fn progress_foundation_to_idle(session : FoundationSession) -> Unit raise {
  let mut attempts = 0
  while attempts < 1000000 {
    match session.progress_retirement() {
      LunaOnlineInstanceIdle => return
      LunaOnlineInstanceRetirementPending => attempts += 1
      LunaOnlineInstanceClosed =>
        raise Failure("healthy online request closed its persistent instance")
    }
  }
  raise Failure("online request did not complete bounded retirement")
}
