///|
fn[T] RankChildControl::reject(
  self : RankChildControl,
  rule : RankChildControlRule,
) -> T raise RankChildControlError {
  if self.failed_value {
    raise Failed(self.failure_rule_value)
  }
  raise Invalid(rule)
}

///|
fn[T] RankChildControl::poison(
  self : RankChildControl,
  rule : RankChildControlRule,
) -> T raise RankChildControlError {
  if !self.failed_value {
    self.failed_value = true
    self.failure_rule_value = rule
    self.receive_active = false
    self.receive_held = false
    self.receive_held_eof = false
    self.receive_held_length = 0
    self.inbound_active = false
    self.send_active = false
    self.transcript.abandon_on_transport_loss()
  }
  raise Failed(self.failure_rule_value)
}

///|
fn[T] RankChildControl::map_transport(
  self : RankChildControl,
  _error : @process.ProcessError,
) -> T raise RankChildControlError {
  self.poison(Transport)
}

///|
fn[T] RankChildControl::map_wire(
  self : RankChildControl,
  _error : @rank_group_wire.RankGroupWireError,
) -> T raise RankChildControlError {
  self.poison(Wire)
}

///|
/// Irrevocably faults the channel after local artifact, manifest, bootstrap,
/// or execution admission fails. It emits no synthetic protocol success.
pub fn RankChildControl::fail_local_admission(
  self : RankChildControl,
) -> Unit raise RankChildControlError {
  self.poison(LocalAdmission)
}

///|
fn RankChildControl::require_usable(
  self : RankChildControl,
) -> Unit raise RankChildControlError {
  if self.failed_value {
    raise Failed(self.failure_rule_value)
  }
}
