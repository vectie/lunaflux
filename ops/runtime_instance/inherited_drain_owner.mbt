///|
priv enum RuntimeInheritedDrainAction {
  RuntimeInheritedDrainContinue
  RuntimeInheritedDrainAccept
  RuntimeInheritedDrainIdempotent
  RuntimeInheritedDrainClosedResponse
  RuntimeInheritedDrainAuthorityLost
}

///|
fn runtime_inherited_drain_action(
  phase : RuntimeInstancePhase,
  progress : @inherited_drain.InheritedDrainProgress,
) -> RuntimeInheritedDrainAction {
  match progress {
    InheritedDrainPending
    | InheritedDrainResponseConfirmed
    | InheritedDrainClosed => RuntimeInheritedDrainContinue
    InheritedDrainRequested =>
      match phase {
        Starting | HealthyNotReady | Ready | OpenAiQualificationReady =>
          RuntimeInheritedDrainAccept
        Draining => RuntimeInheritedDrainIdempotent
        Closed | Failed => RuntimeInheritedDrainClosedResponse
      }
    InheritedDrainChannelLost =>
      match phase {
        Draining | Closed | Failed => RuntimeInheritedDrainContinue
        _ => RuntimeInheritedDrainAuthorityLost
      }
  }
}

///|
pub fn RuntimeInstanceOwner::attach_inherited_drain_v1(
  self : RuntimeInstanceOwner,
  preparation : @inherited_drain.InheritedDrainPreparation,
) -> Unit raise RuntimeInstanceError {
  if self.phase != HealthyNotReady ||
    self.inherited_drain_required ||
    self.inherited_drains.length() != 0 ||
    self.servers.length() != 0 ||
    self.openai_servers.length() != 0 ||
    self.controls.length() != 0 {
    raise fail(InheritedDrainBinding)
  }
  let channel = preparation.activate() catch {
    _ => raise fail(InheritedDrainBinding)
  }
  self.inherited_drains.push(channel)
  self.inherited_drain_required = true
}

///|
fn RuntimeInstanceOwner::inherited_drain_live(
  self : RuntimeInstanceOwner,
) -> Bool {
  !self.inherited_drain_required ||
  (self.inherited_drains.length() == 1 && self.inherited_drains[0].is_live())
}

///|
fn RuntimeInstanceOwner::progress_inherited_drain(
  self : RuntimeInstanceOwner,
) -> Bool {
  if !self.inherited_drain_required {
    return self.inherited_drains.length() == 0
  }
  if self.inherited_drains.length() != 1 {
    return false
  }
  let channel = self.inherited_drains[0]
  let action = runtime_inherited_drain_action(self.phase, channel.progress())
  match action {
    RuntimeInheritedDrainContinue => true
    RuntimeInheritedDrainAuthorityLost => false
    RuntimeInheritedDrainAccept => {
      self.begin_drain() catch {
        _ => return false
      }
      channel.begin_response(InheritedDrainAccepted) catch {
        _ => return false
      }
      true
    }
    RuntimeInheritedDrainIdempotent => {
      channel.begin_response(InheritedDrainAlreadyDraining) catch {
        _ => return false
      }
      true
    }
    RuntimeInheritedDrainClosedResponse => {
      channel.begin_response(InheritedDrainAlreadyClosed) catch {
        _ => return false
      }
      true
    }
  }
}

///|
fn RuntimeInstanceOwner::close_inherited_drain(
  self : RuntimeInstanceOwner,
  wait_for_response~ : Bool,
) -> Bool {
  if self.inherited_drains.length() == 0 {
    return true
  }
  if self.inherited_drains.length() != 1 {
    return false
  }
  let channel = self.inherited_drains[0]
  if wait_for_response &&
    channel.request_received() &&
    channel.is_live() &&
    !channel.response_confirmed() {
    return false
  }
  channel.close() catch {
    _ => ()
  }
  if channel.is_closed() {
    self.inherited_drains.clear()
    true
  } else {
    false
  }
}
