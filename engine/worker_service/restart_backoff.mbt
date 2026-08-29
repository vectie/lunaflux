///|
/// Validates one explicit bounded restart policy. Delays fit the scalar wake
/// API and attempt count is finite so a crash loop deterministically fails
/// closed instead of wrapping.
pub fn WorkerRestartBackoffPolicy::new(
  initial_delay_millis~ : UInt64,
  maximum_delay_millis~ : UInt64,
  stable_success_millis~ : UInt64,
  maximum_attempts~ : Int,
) -> WorkerRestartBackoffPolicy raise WorkerServiceError {
  if initial_delay_millis == 0UL ||
    maximum_delay_millis < initial_delay_millis ||
    maximum_delay_millis > 0x7fffffffUL ||
    stable_success_millis == 0UL ||
    stable_success_millis > 0x7fffffffUL ||
    maximum_attempts <= 0 {
    raise Invalid(RestartBackoff)
  }
  {
    initial_delay_millis,
    maximum_delay_millis,
    stable_success_millis,
    maximum_attempts,
  }
}

///|
fn restart_checked_deadline(now : UInt64, delay : UInt64) -> UInt64? {
  if now > 0xffffffffffffffffUL - delay {
    None
  } else {
    Some(now + delay)
  }
}

///|
fn OnlineWorkerLease::forbid_restart_backoff(self : OnlineWorkerLease) -> Unit {
  self.restart_forbidden = true
  self.restart_phase = 4
  self.restart_deadline_millis = 0UL
  self.restart_cached_remaining_millis = 0
  self.restart_stability_generation = 0UL
  self.restart_stability_deadline_millis = 0UL
  self.restart_stability_predecessor = 0UL
}

///|
fn OnlineWorkerLease::observe_restart_clock(
  self : OnlineWorkerLease,
  now : UInt64,
) -> Bool {
  if self.restart_clock_initialized && now < self.restart_last_clock_millis {
    self.forbid_restart_backoff()
    return false
  }
  self.restart_clock_initialized = true
  self.restart_last_clock_millis = now
  true
}

///|
fn OnlineWorkerLease::sample_restart_clock(self : OnlineWorkerLease) -> UInt64? {
  let now = self.clock.now_millis() catch {
    _ => {
      self.forbid_restart_backoff()
      return None
    }
  }
  if !self.observe_restart_clock(now) {
    return None
  }
  Some(now)
}

///|
fn OnlineWorkerLease::next_restart_delay(self : OnlineWorkerLease) -> UInt64 {
  if self.restart_attempts == 0 {
    return self.restart_policy.initial_delay_millis
  }
  let current = self.restart_delay_millis
  let maximum = self.restart_policy.maximum_delay_millis
  if current >= maximum || current > maximum - current {
    maximum
  } else {
    current + current
  }
}

///|
fn OnlineWorkerLease::schedule_restart_backoff_at(
  self : OnlineWorkerLease,
  now : UInt64,
) -> Bool {
  if self.restart_forbidden ||
    self.restart_attempts >= self.restart_policy.maximum_attempts ||
    self.restart_generation == 0xffffffffffffffffUL {
    self.forbid_restart_backoff()
    return false
  }
  if !self.observe_restart_clock(now) {
    return false
  }
  let delay = self.next_restart_delay()
  guard restart_checked_deadline(now, delay) is Some(deadline) else {
    self.forbid_restart_backoff()
    return false
  }
  self.restart_attempts += 1
  self.restart_generation += 1UL
  self.restart_delay_millis = delay
  self.restart_deadline_millis = deadline
  self.restart_cached_remaining_millis = delay.to_int()
  self.restart_stability_generation = 0UL
  self.restart_stability_deadline_millis = 0UL
  self.restart_stability_predecessor = 0UL
  self.restart_phase = 1
  true
}

///|
fn OnlineWorkerLease::schedule_restart_backoff(
  self : OnlineWorkerLease,
) -> Bool {
  let now = self.clock.now_millis() catch {
    _ => {
      self.forbid_restart_backoff()
      return false
    }
  }
  self.schedule_restart_backoff_at(now)
}

///|
fn OnlineWorkerLease::restart_backoff_remaining_at(
  self : OnlineWorkerLease,
  now : UInt64,
) -> Int {
  if self.restart_forbidden || self.restart_phase != 1 {
    return 0
  }
  if !self.observe_restart_clock(now) {
    return 0
  }
  if now >= self.restart_deadline_millis {
    self.restart_phase = 2
    self.restart_cached_remaining_millis = 0
    return 0
  }
  let remaining = (self.restart_deadline_millis - now).to_int()
  self.restart_cached_remaining_millis = remaining
  remaining
}

///|
fn OnlineWorkerLease::restart_backoff_remaining_impl(
  self : OnlineWorkerLease,
) -> Int {
  let now = self.clock.now_millis() catch {
    _ => {
      self.forbid_restart_backoff()
      return 0
    }
  }
  self.restart_backoff_remaining_at(now)
}

///|
fn OnlineWorkerLease::record_replacement_ready_at(
  self : OnlineWorkerLease,
  now : UInt64,
  predecessor : UInt64,
) -> Unit {
  if !self.observe_restart_clock(now) {
    return
  }
  guard restart_checked_deadline(now, self.restart_policy.stable_success_millis)
    is Some(deadline) else {
    self.forbid_restart_backoff()
    return
  }
  self.restart_phase = 3
  self.restart_stability_generation = self.restart_generation
  self.restart_stability_deadline_millis = deadline
  self.restart_stability_predecessor = predecessor
}

///|
fn OnlineWorkerLease::record_replacement_ready(
  self : OnlineWorkerLease,
  service : WorkerService,
) -> Unit {
  guard self.sample_restart_clock() is Some(now) else { return }
  self.record_replacement_ready_at(
    now,
    service.scheduler.last_retired_plan_sequence(),
  )
}

///|
fn OnlineWorkerLease::record_stable_replacement_commit_at(
  self : OnlineWorkerLease,
  now : UInt64,
  sequence : UInt64,
  last_retired : UInt64,
  service_ready : Bool,
) -> Unit {
  if self.restart_phase != 3 ||
    self.restart_stability_generation == 0UL ||
    self.restart_stability_generation != self.restart_generation ||
    !service_ready ||
    sequence == 0UL ||
    sequence <= self.restart_stability_predecessor ||
    last_retired != sequence {
    return
  }
  if !self.observe_restart_clock(now) {
    return
  }
  if now < self.restart_stability_deadline_millis {
    return
  }
  self.restart_attempts = 0
  self.restart_delay_millis = 0UL
  self.restart_phase = 0
  self.restart_deadline_millis = 0UL
  self.restart_cached_remaining_millis = 0
  self.restart_stability_generation = 0UL
  self.restart_stability_deadline_millis = 0UL
  self.restart_stability_predecessor = 0UL
}

///|
fn OnlineWorkerLease::record_stable_replacement_commit(
  self : OnlineWorkerLease,
  service : WorkerService,
  sequence : UInt64,
) -> Unit {
  guard self.sample_restart_clock() is Some(now) else { return }
  self.record_stable_replacement_commit_at(
    now,
    sequence,
    service.scheduler.last_retired_plan_sequence(),
    service.lifecycle == ServiceReady,
  )
}

///|
fn OnlineWorkerLease::retry_after_clean_startup_failure(
  self : OnlineWorkerLease,
) -> OnlineWorkerRestartProgress {
  self.restart_phase = 0
  if !self.schedule_restart_backoff() {
    return OnlineWorkerRestartForbidden
  }
  let remaining = self.restart_backoff_remaining_impl()
  if self.restart_forbidden {
    OnlineWorkerRestartForbidden
  } else {
    OnlineWorkerRestartWaiting(remaining)
  }
}

///|
/// Advances only the delay/replacement transition. Old-child cleanup, flight
/// retirement, and device invalidation must already have completed.
pub fn OnlineWorkerLease::progress_restart_backoff(
  self : OnlineWorkerLease,
) -> OnlineWorkerRestartProgress raise WorkerServiceError {
  let service = self.require_current()
  service.require_no_maintenance()
  service.require_lifecycle(ServiceRestartReady)
  if self.restart_forbidden {
    return OnlineWorkerRestartForbidden
  }
  if self.restart_phase == 0 || self.restart_phase == 3 {
    if !self.schedule_restart_backoff() {
      return OnlineWorkerRestartForbidden
    }
  }
  if self.restart_phase == 1 {
    let remaining = self.restart_backoff_remaining_impl()
    if self.restart_forbidden {
      return OnlineWorkerRestartForbidden
    }
    if remaining > 0 {
      return OnlineWorkerRestartWaiting(remaining)
    }
  }
  if self.restart_phase != 2 {
    self.forbid_restart_backoff()
    return OnlineWorkerRestartForbidden
  }
  let result = try service.restart_impl() catch {
    PhysicalTransport(_) => {
      if service.lifecycle == ServiceRestartReady &&
        service.require_transport().is_recovering() {
        return self.retry_after_clean_startup_failure()
      }
      self.forbid_restart_backoff()
      return OnlineWorkerRestartForbidden
    }
    _ => {
      self.forbid_restart_backoff()
      return OnlineWorkerRestartForbidden
    }
  } noraise {
    value => value
  }
  match result {
    Restarted => {
      self.record_replacement_ready(service)
      OnlineWorkerRestarted
    }
    RestartCleanupRequired => {
      self.restart_phase = 0
      OnlineWorkerRestartCleanupRequired
    }
  }
}

///|
/// Returns the current bounded delay without sleeping or spawning.
pub fn OnlineWorkerLease::restart_backoff_remaining_millis(
  self : OnlineWorkerLease,
) -> Int raise WorkerServiceError {
  let service = self.require_current()
  service.require_no_maintenance()
  if service.lifecycle != ServiceRestartReady {
    return 0
  }
  if self.restart_forbidden || self.restart_phase != 1 {
    0
  } else {
    self.restart_cached_remaining_millis
  }
}
