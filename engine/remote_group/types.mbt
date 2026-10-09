///|
pub(all) enum RemoteGroupLifecycle {
  Starting
  Ready
  Executing
  CompletionReady
  Accepted
  Faulted
  Closing
  Closed
} derive(Eq)

///|
pub extend RemoteGroupLifecycle with Eq::{equal, not_equal}

///|
pub(all) suberror RemoteGroupError {
  Configuration
  State
  Binding
  Deadline
  RankLost(Int)
  Protocol
}

///|
/// One scheduler flight, two fixed transport mailboxes, and the existing
/// rank-group semantic owner. No device, model-family or socket execution here.
pub struct RemoteGroup {
  priv startup : @remote_startup.RemoteGroupStartup
  priv released : FixedArray[Bool]
  priv mut failure : @protocol.RankGroupFailureReport?
  priv mut failure_taken : Bool
  priv mailboxes : FixedArray[@remote_mailbox.RemoteMailbox]
  priv bindings : FixedArray[@wire.RankGroupWireBinding]
  priv identities : FixedArray[@worker_protocol.PlanBufferIdentity]
  priv protocol : @protocol.RankGroupOwner
  priv transcript : @wire.RankGroupWireTranscript
  priv limits : @wire.RankGroupWireLimits
  priv plan_frame : @worker_wire.PlanFrameBuffer
  priv completion_frame : @worker_wire.CompletionFrameBuffer
  priv tx : FixedArray[Byte]
  priv rx : FixedArray[Byte]
  priv payload : FixedArray[Byte]
  priv clock : @monotonic_clock.ReusableMonotonicRead
  priv phases : FixedArray[Int]
  priv reports : FixedArray[@worker_wire.WorkerGraphTelemetrySlot]
  priv timeout : UInt64
  priv mut deadline : UInt64
  priv mut last_tick : UInt64
  priv mut lifecycle : RemoteGroupLifecycle
  priv mut slot : @wire.RankGroupWireSlot
  priv mut sequence : UInt64
  priv mut epoch : UInt64
  priv mut cursor : Int
}

///|
pub fn RemoteGroup::lifecycle(self : RemoteGroup) -> RemoteGroupLifecycle {
  self.lifecycle
}

///|
pub fn RemoteGroup::previous_plan_sequence_value(self : RemoteGroup) -> UInt64 {
  self.protocol.previous_plan_sequence_value()
}

///|
pub fn RemoteGroup::mailbox(
  self : RemoteGroup,
  rank : Int,
) -> @remote_mailbox.RemoteMailbox raise RemoteGroupError {
  if rank < 0 || rank >= 2 {
    raise Configuration
  }
  self.mailboxes[rank]
}

///|
fn binding(
  envelope : @bootstrap.TensorParallelRankBootstrapEnvelope,
) -> @wire.RankGroupWireBinding raise {
  @wire.RankGroupWireBinding::new(
    identity=envelope.identity(),
    model_generation_value=envelope.model_generation_value(),
    group_generation_value=envelope.group_generation_value(),
    rank=envelope.rank(),
    world_size=envelope.world_size(),
    group_digest=envelope.group_digest().as_sha256(),
    topology_digest=envelope.topology_digest().as_sha256(),
    worker_digest=envelope.worker_digest().as_sha256(),
    device_plan_digest=envelope.device_plan_digest().as_sha256(),
    execution_plan_digest=envelope.execution_plan_digest_sha256(),
    collective_digest=envelope.collective_digest().as_hex(),
    kv_digest=envelope.kv_digest().as_sha256(),
  )
}

///|
fn RemoteGroup::now(self : RemoteGroup) -> UInt64 raise {
  if !self.clock.poll() {
    raise Deadline
  }
  let now = self.clock.value()
  if now < self.last_tick {
    raise Deadline
  }
  self.last_tick = now
  now
}

///|
fn RemoteGroup::start_deadline(self : RemoteGroup) -> Unit raise {
  let now = self.now()
  if now > 0xffffffffffffffffUL - self.timeout {
    raise Deadline
  }
  self.deadline = now + self.timeout
}

///|
fn RemoteGroup::check_live(self : RemoteGroup) -> Unit raise {
  for rank in 0..<2 {
    if self.mailboxes[rank].is_failed() {
      raise RankLost(rank)
    }
  }
  if self.now() >= self.deadline {
    raise Deadline
  }
}

///|
fn RemoteGroup::poison(self : RemoteGroup) -> Unit {
  match self.protocol.lifecycle() {
    Ready => self.protocol.report_idle_host_failure() catch { _ => () }
    _ => self.protocol.report_active_host_failure() catch { _ => () }
  }
  self.lifecycle = Faulted
  self.transcript.abandon_on_transport_loss()
  for mailbox in self.mailboxes {
    mailbox.fail()
  }
}

///|
pub fn RemoteGroup::take_failure(
  self : RemoteGroup,
) -> @protocol.RankGroupFailureReport raise {
  if self.lifecycle != Faulted || self.failure_taken {
    raise State
  }
  let failure = self.protocol.take_failure()
  self.failure = Some(failure)
  self.failure_taken = true
  failure
}

///|
pub fn RemoteGroup::taken_failure(
  self : RemoteGroup,
) -> @protocol.RankGroupFailureReport raise RemoteGroupError {
  guard self.failure is Some(failure) else { raise State }
  failure
}

///|
pub fn RemoteGroup::confirm_release(
  self : RemoteGroup,
  rank : Int,
  receipt : @remote_release.RemoteReleaseReceipt,
) -> Unit raise {
  if rank < 0 || rank >= 2 || self.lifecycle != Faulted {
    raise State
  }
  receipt.verify(self.startup.rank(rank).unwrap())
  self.released[rank] = true
}

///|
pub fn RemoteGroup::recovery_cleanup_complete(self : RemoteGroup) -> Bool {
  self.lifecycle == Faulted && self.released[0] && self.released[1]
}

///|
pub fn RemoteGroup::generation(self : RemoteGroup) -> UInt64 {
  self.bindings[0].group_generation_value()
}

///|
pub fn RemoteGroup::startup(
  self : RemoteGroup,
) -> @remote_startup.RemoteGroupStartup {
  self.startup
}

///|
pub fn RemoteGroup::transport_lost(self : RemoteGroup) -> Unit {
  if self.lifecycle == Closed || self.lifecycle == Faulted {
    return
  }
  // An accepted completion may be waiting for scheduler publication capacity.
  // Its compute result is already final. Preserve that retirement authority;
  // the loss becomes an idle group failure immediately after scheduler commit.
  if self.lifecycle == Accepted {
    for mailbox in self.mailboxes {
      mailbox.fail()
    }
    return
  }
  match self.protocol.lifecycle() {
    Ready => self.protocol.report_idle_host_failure() catch { _ => () }
    _ => self.protocol.report_active_host_failure() catch { _ => () }
  }
  self.poison()
}
