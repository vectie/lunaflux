///|
pub fn RemoteGroup::new(
  remote : @remote_startup.RemoteGroupStartup,
  admitted : @bootstrap.TensorParallelGroupBootstrap,
  identities : ArrayView[@worker_protocol.PlanBufferIdentity],
  limits : @wire.RankGroupWireLimits,
  timeout_millis~ : UInt64,
) -> RemoteGroup raise {
  if identities.length() != 2 ||
    identities[0].same_owner(identities[1]) ||
    admitted.template_digest_sha256() != remote.digest_sha256() ||
    timeout_millis == 0UL ||
    timeout_millis > 3600000UL {
    raise Configuration
  }
  let first = remote.rank(0).unwrap().worker()
  let bindings = FixedArray::makei(2, rank => {
    let startup = remote.rank(rank).unwrap()
    let envelope = admitted.rank_envelope(rank).unwrap()
    if envelope.rank() != rank ||
      envelope.world_size() != 2 ||
      envelope.group_generation_value() != startup.group_generation_value() ||
      envelope.worker_digest() !=
      @bootstrap.worker_contract_digest(startup.worker()) {
      raise Binding
    }
    binding(envelope)
  })
  let protocol = @protocol.RankGroupOwner::new(
    identity=first.identity(),
    model_plan_generation=@worker_protocol.ModelPlanGeneration::from_wire(
      first.model_generation_value(),
    ),
    generation=@protocol.RankGroupGeneration::from_wire(
      bindings[0].group_generation_value(),
    ),
    world_size=2,
    previous_plan_sequence_value=first.predecessor_value(),
    plan_buffer_identities=identities,
  )
  let plan_frame = @worker_wire.PlanFrameBuffer::new(
    first.worker_limits(),
    first.inference_limits(),
  )
  let completion_frame = @worker_wire.CompletionFrameBuffer::new(
    first.worker_limits(),
    first.inference_limits(),
  )
  if plan_frame.capacity() > limits.max_payload_bytes() ||
    completion_frame.capacity() > limits.max_payload_bytes() ||
    @remote_startup.remote_startup_frame_bytes() > limits.max_payload_bytes() {
    raise Configuration
  }
  let group : RemoteGroup = {
    startup: remote,
    released: FixedArray::make(2, false),
    failure: None,
    failure_taken: false,
    mailboxes: FixedArray::makei(2, _ => {
      @remote_mailbox.RemoteMailbox::new(limits.max_frame_bytes())
    }),
    bindings,
    identities: FixedArray::makei(2, i => identities[i]),
    protocol,
    transcript: @wire.RankGroupWireTranscript::new(
      bindings,
      previous_sequence_value=first.predecessor_value(),
    ),
    limits,
    plan_frame,
    completion_frame,
    tx: FixedArray::make(limits.max_frame_bytes(), b'\x00'),
    rx: FixedArray::make(limits.max_frame_bytes(), b'\x00'),
    payload: FixedArray::make(limits.max_payload_bytes(), b'\x00'),
    clock: @monotonic_clock.MonotonicClock::system().prepare_poller(),
    phases: FixedArray::make(2, 0),
    reports: FixedArray::makei(2, _ => {
      @worker_wire.WorkerGraphTelemetrySlot::new()
    }),
    timeout: timeout_millis,
    deadline: 0UL,
    last_tick: 0UL,
    lifecycle: Starting,
    slot: A,
    sequence: 0UL,
    epoch: 0UL,
    cursor: 0,
  }
  group.start_deadline()
  for rank in 0..<2 {
    let length = remote.rank(rank).unwrap().copy_to(group.payload)
    let length = @wire.encode_rank_group_wire_payload_into(
      group.tx,
      limits,
      bindings[rank],
      Configure,
      ParentToChild,
      None,
      0UL,
      group.payload,
      length,
    )
    group.send(rank, length)
  }
  group
}

///|
fn RemoteGroup::send(
  self : RemoteGroup,
  rank : Int,
  length : Int,
) -> Unit raise {
  ignore(self.transcript.accept_bytes(self.tx, length, self.limits))
  self.mailboxes[rank].begin_write(self.tx, 0, length)
}

///|
fn RemoteGroup::receive(
  self : RemoteGroup,
  rank : Int,
  length : Int,
) -> @wire.RankGroupWireDecodedHeader raise {
  ignore(
    @wire.validate_rank_group_wire_binding(
      self.rx,
      length,
      self.limits,
      self.bindings[rank],
    ),
  )
  self.transcript.accept_bytes(self.rx, length, self.limits)
}

///|
pub fn RemoteGroup::poll_startup(self : RemoteGroup) -> Bool raise {
  if self.lifecycle != Starting {
    raise State
  }
  errdefer {
    self.protocol.report_idle_host_failure() catch {
      _ => ()
    }
    self.poison()
  }
  self.check_live()
  let rank = self.cursor
  self.cursor = 1 - rank
  match self.phases[rank] {
    0 =>
      if self.mailboxes[rank].poll_write() {
        self.mailboxes[rank].begin_read()
        self.phases[rank] = 1
      }
    1 =>
      match self.mailboxes[rank].poll_read(self.rx) {
        Some(length) => {
          if self.receive(rank, length).kind() != Ready {
            raise Protocol
          }
          self.phases[rank] = 2
        }
        None => ()
      }
    _ => ()
  }
  if self.phases[0] == 2 && self.phases[1] == 2 {
    self.lifecycle = Ready
    true
  } else {
    false
  }
}

///|
pub fn RemoteGroup::probe_idle_health(self : RemoteGroup) -> Unit raise {
  if self.lifecycle != Ready {
    raise State
  }
  for rank in 0..<2 {
    if self.mailboxes[rank].is_failed() {
      self.protocol.report_idle_rank_loss(rank)
      self.poison()
      raise RankLost(rank)
    }
  }
}
