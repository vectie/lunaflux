///|
fn luna_online_observation_code(kind : LunaOnlineFramedObservationKind) -> Int {
  return match kind {
    LunaOnlineFramedAdmissionObservation => 1
    LunaOnlineFramedUsageObservation => 2
    LunaOnlineFramedCompletionObservation => 3
    LunaOnlineFramedCancellationObservation => 4
    LunaOnlineFramedDeadlineObservation => 5
    LunaOnlineFramedRequestFailureObservation => 6
    LunaOnlineFramedWorkerFailureObservation => 7
    LunaOnlineFramedWorkerRestartObservation => 8
    LunaOnlineFramedTokenObservation => 9
    LunaOnlineFramedWorkerRequestTerminalObservation => 10
  }
}

///|
fn LunaOnlineFramedService::publish_observation(
  self : LunaOnlineFramedService,
  kind : LunaOnlineFramedObservationKind,
  input_tokens : Int,
  cached_input_tokens : Int,
  output_tokens : Int,
  total_tokens : Int,
  latency? : LunaOnlineFramedTimingSample = no_luna_online_framed_timing_sample(),
) -> Unit raise LunaOnlineFramedCoordinatorError {
  if self.observation_kind != 0 || self.observation_taken {
    raise luna_coordinator_error(LunaOnlineFramedLifecycle)
  }
  if self.observation_epoch == 0xffffffffffffffffUL {
    self.begin_service_drain()
    raise luna_coordinator_error(LunaOnlineFramedOwnership)
  }
  self.observation_epoch += 1UL
  self.observation_kind = luna_online_observation_code(kind)
  self.observation_input_tokens = input_tokens
  self.observation_cached_input_tokens = cached_input_tokens
  self.observation_output_tokens = output_tokens
  self.observation_total_tokens = total_tokens
  self.observation_latency_kind = latency.kind
  self.observation_latency_millis = latency.millis
}

///|
fn LunaOnlineFramedService::sample_event_timing(
  self : LunaOnlineFramedService,
  token~ : Bool,
  terminal~ : Bool,
) -> LunaOnlineFramedTimingSample {
  if self.instance.multi.active_count > 0 {
    return self.sample_request_timing(
      self.instance.multi_event_ticket(),
      token~,
      terminal~,
    )
  }
  if self.tickets.is_empty() {
    return no_luna_online_framed_timing_sample()
  }
  self.sample_request_timing(self.ticket(), token~, terminal~)
}

///|
fn LunaOnlineFramedService::publish_event_observation(
  self : LunaOnlineFramedService,
  event : @luna_event.LunaEventView,
) -> Bool raise LunaOnlineFramedCoordinatorError {
  let kind = event.kind() catch {
    _ => raise luna_coordinator_error(LunaOnlineFramedEvent)
  }
  match kind {
    LunaAcceptedEventKind => {
      self.publish_observation(LunaOnlineFramedAdmissionObservation, 0, 0, 0, 0)
      true
    }
    LunaTokenEventKind => {
      let timing = self.sample_event_timing(token=true, terminal=false)
      self.publish_observation(
        LunaOnlineFramedTokenObservation,
        0,
        0,
        0,
        0,
        latency=timing,
      )
      true
    }
    LunaUsageEventKind => {
      let usage = event.usage() catch {
        _ => raise luna_coordinator_error(LunaOnlineFramedEvent)
      }
      self.publish_observation(
        LunaOnlineFramedUsageObservation,
        usage.input_tokens() catch {
          _ => raise luna_coordinator_error(LunaOnlineFramedEvent)
        },
        usage.cached_input_tokens() catch {
          _ => raise luna_coordinator_error(LunaOnlineFramedEvent)
        },
        usage.output_tokens() catch {
          _ => raise luna_coordinator_error(LunaOnlineFramedEvent)
        },
        usage.total_tokens() catch {
          _ => raise luna_coordinator_error(LunaOnlineFramedEvent)
        },
      )
      true
    }
    LunaCompletedEventKind => {
      let completed = event.completed() catch {
        _ => raise luna_coordinator_error(LunaOnlineFramedEvent)
      }
      let reason = completed.finish_reason() catch {
        _ => raise luna_coordinator_error(LunaOnlineFramedEvent)
      }
      let observation = match reason {
        Cancelled => LunaOnlineFramedCancellationObservation
        StopSequence | TokenLimit => LunaOnlineFramedCompletionObservation
      }
      let timing = self.sample_event_timing(token=false, terminal=true)
      self.publish_observation(observation, 0, 0, 0, 0, latency=timing)
      true
    }
    LunaFailedEventKind => {
      let failure_kind = if self.instance.multi.active_count > 0 {
        self.instance.multi_event_failure_kind()
      } else {
        self.instance.failure_kind
      }
      let observation = match failure_kind {
        DeadlineLatchedFailure => LunaOnlineFramedDeadlineObservation
        WorkerLatchedFailure => LunaOnlineFramedWorkerRequestTerminalObservation
        NoLatchedFailure | OutputLatchedFailure =>
          LunaOnlineFramedRequestFailureObservation
      }
      let timing = self.sample_event_timing(token=false, terminal=true)
      self.publish_observation(observation, 0, 0, 0, 0, latency=timing)
      true
    }
  }
}

///|
fn LunaOnlineFramedService::publish_worker_failure_if_needed(
  self : LunaOnlineFramedService,
) -> Bool raise LunaOnlineFramedCoordinatorError {
  let failed = if self.instance.multi.active_count > 0 {
    self.instance.multi_has_worker_failure()
  } else {
    self.instance.failure_kind == WorkerLatchedFailure
  }
  if failed && !self.worker_failure_observed {
    self.worker_failure_observed = true
    self.publish_observation(
      LunaOnlineFramedWorkerFailureObservation,
      0,
      0,
      0,
      0,
    )
    return true
  }
  false
}

///|
fn LunaOnlineFramedService::stream_take_observation(
  self : LunaOnlineFramedService,
) -> LunaOnlineFramedObservation raise LunaOnlineFramedCoordinatorError {
  if self.observation_kind == 0 || self.observation_taken {
    raise luna_coordinator_error(LunaOnlineFramedOwnership)
  }
  self.observation_taken = true
  {
    service: self,
    stream_epoch: self.stream_epoch,
    observation_epoch: self.observation_epoch,
  }
}

///|
fn LunaOnlineFramedObservation::require_owner(
  self : LunaOnlineFramedObservation,
) -> LunaOnlineFramedService raise LunaOnlineFramedCoordinatorError {
  let owner = self.service
  if self.stream_epoch == 0UL ||
    self.observation_epoch == 0UL ||
    owner.stream_epoch != self.stream_epoch ||
    (owner.stream_phase != 2 && owner.stream_phase != 3) ||
    owner.observation_epoch != self.observation_epoch ||
    owner.observation_kind == 0 ||
    !owner.observation_taken {
    raise luna_coordinator_error(LunaOnlineFramedOwnership)
  }
  owner
}

///|
pub fn LunaOnlineFramedObservation::kind(
  self : LunaOnlineFramedObservation,
) -> LunaOnlineFramedObservationKind raise LunaOnlineFramedCoordinatorError {
  match self.require_owner().observation_kind {
    1 => LunaOnlineFramedAdmissionObservation
    2 => LunaOnlineFramedUsageObservation
    3 => LunaOnlineFramedCompletionObservation
    4 => LunaOnlineFramedCancellationObservation
    5 => LunaOnlineFramedDeadlineObservation
    6 => LunaOnlineFramedRequestFailureObservation
    7 => LunaOnlineFramedWorkerFailureObservation
    8 => LunaOnlineFramedWorkerRestartObservation
    9 => LunaOnlineFramedTokenObservation
    10 => LunaOnlineFramedWorkerRequestTerminalObservation
    _ => abort("Luna online observation kind invariant")
  }
}

///|
pub fn LunaOnlineFramedObservation::latency_kind(
  self : LunaOnlineFramedObservation,
) -> LunaOnlineFramedLatencyKind raise LunaOnlineFramedCoordinatorError {
  match self.require_owner().observation_latency_kind {
    0 => LunaOnlineFramedNoLatency
    1 => LunaOnlineFramedRequestLatency
    2 => LunaOnlineFramedFirstTokenLatency
    3 => LunaOnlineFramedInterTokenLatency
    _ => abort("Luna online observation latency invariant")
  }
}

///|
pub fn LunaOnlineFramedObservation::latency_millis(
  self : LunaOnlineFramedObservation,
) -> Int raise LunaOnlineFramedCoordinatorError {
  let owner = self.require_owner()
  if owner.observation_latency_kind == 0 {
    raise luna_coordinator_error(LunaOnlineFramedLifecycle)
  }
  owner.observation_latency_millis
}

///|
fn LunaOnlineFramedObservation::require_usage_owner(
  self : LunaOnlineFramedObservation,
) -> LunaOnlineFramedService raise LunaOnlineFramedCoordinatorError {
  let owner = self.require_owner()
  if owner.observation_kind != 2 {
    raise luna_coordinator_error(LunaOnlineFramedLifecycle)
  }
  owner
}

///|
pub fn LunaOnlineFramedObservation::input_tokens(
  self : LunaOnlineFramedObservation,
) -> Int raise LunaOnlineFramedCoordinatorError {
  self.require_usage_owner().observation_input_tokens
}

///|
pub fn LunaOnlineFramedObservation::cached_input_tokens(
  self : LunaOnlineFramedObservation,
) -> Int raise LunaOnlineFramedCoordinatorError {
  self.require_usage_owner().observation_cached_input_tokens
}

///|
pub fn LunaOnlineFramedObservation::output_tokens(
  self : LunaOnlineFramedObservation,
) -> Int raise LunaOnlineFramedCoordinatorError {
  self.require_usage_owner().observation_output_tokens
}

///|
pub fn LunaOnlineFramedObservation::total_tokens(
  self : LunaOnlineFramedObservation,
) -> Int raise LunaOnlineFramedCoordinatorError {
  self.require_usage_owner().observation_total_tokens
}

///|
pub fn LunaOnlineFramedObservation::ack(
  self : LunaOnlineFramedObservation,
) -> Unit raise LunaOnlineFramedCoordinatorError {
  let owner = self.require_owner()
  owner.observation_kind = 0
  owner.observation_taken = false
  owner.observation_input_tokens = 0
  owner.observation_cached_input_tokens = 0
  owner.observation_output_tokens = 0
  owner.observation_total_tokens = 0
  owner.observation_latency_kind = 0
  owner.observation_latency_millis = 0
}
