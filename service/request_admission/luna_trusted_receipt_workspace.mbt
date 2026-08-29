///|
fn luna_receipt_clock_reader(
  clock : @monotonic_clock.MonotonicClock,
) -> () -> UInt64 raise RequestAdmissionError {
  () => clock.now_millis() catch { _ => raise ClockUnavailable }
}

///|
pub fn LunaRequestReceiptWorkspace::new(
  clock : @monotonic_clock.MonotonicClock,
  budget : @inference.DeadlineBudget,
) -> LunaRequestReceiptWorkspace raise RequestAdmissionError {
  new_luna_receipt_workspace_with_clock(
    luna_receipt_clock_reader(clock),
    budget,
  )
}

///|
fn new_luna_receipt_workspace_with_clock(
  clock_read : () -> UInt64 raise RequestAdmissionError,
  budget : @inference.DeadlineBudget,
) -> LunaRequestReceiptWorkspace raise RequestAdmissionError {
  let budget_millis = budget.milliseconds()
  if budget_millis == 0UL || budget_millis > 0x7fffffffUL {
    raise Invalid(Deadline)
  }
  {
    clock_read,
    budget_millis,
    epoch: 0UL,
    phase: LUNA_TRUSTED_RECEIPT_IDLE,
    received_at_millis: 0UL,
    deadline_at_millis: 0UL,
  }
}

///|
pub fn LunaRequestReceiptWorkspace::begin(
  self : LunaRequestReceiptWorkspace,
) -> LunaRequestReceipt raise RequestAdmissionError {
  if self.phase != LUNA_TRUSTED_RECEIPT_IDLE {
    raise Invalid(LunaPreparedLifecycle)
  }
  if self.epoch == 0xffffffffffffffffUL {
    raise Invalid(LunaPreparationPool)
  }
  let received = (self.clock_read)()
  if received > 0xffffffffffffffffUL - self.budget_millis {
    raise Invalid(ReceiptTime)
  }
  self.epoch += 1UL
  self.received_at_millis = received
  self.deadline_at_millis = received + self.budget_millis
  self.phase = LUNA_TRUSTED_RECEIPT_LIVE
  { workspace: self, epoch: self.epoch }
}

///|
fn LunaRequestReceipt::require_workspace(
  self : LunaRequestReceipt,
) -> LunaRequestReceiptWorkspace raise RequestAdmissionError {
  if self.epoch == 0UL ||
    self.workspace.epoch != self.epoch ||
    self.workspace.phase != LUNA_TRUSTED_RECEIPT_LIVE {
    raise Invalid(LunaPreparedLifecycle)
  }
  self.workspace
}

///|
fn LunaRequestReceipt::observe(
  self : LunaRequestReceipt,
) -> UInt64 raise RequestAdmissionError {
  let workspace = self.require_workspace()
  let observed = (workspace.clock_read)()
  if observed < workspace.received_at_millis {
    raise Invalid(ReceiptTime)
  }
  if observed >= workspace.deadline_at_millis {
    raise Invalid(Deadline)
  }
  observed
}

///|
pub fn LunaRequestReceipt::remaining_millis(
  self : LunaRequestReceipt,
) -> Int raise RequestAdmissionError {
  let workspace = self.require_workspace()
  let observed = (workspace.clock_read)()
  if observed < workspace.received_at_millis {
    raise Invalid(ReceiptTime)
  }
  if observed >= workspace.deadline_at_millis {
    return 0
  }
  let remaining = workspace.deadline_at_millis - observed
  if remaining > 0x7fffffffUL {
    raise Invalid(ReceiptTime)
  }
  remaining.to_int()
}

///|
fn LunaRequestReceipt::consume(
  self : LunaRequestReceipt,
) -> Unit raise RequestAdmissionError {
  let workspace = self.require_workspace()
  workspace.received_at_millis = 0UL
  workspace.deadline_at_millis = 0UL
  workspace.phase = LUNA_TRUSTED_RECEIPT_IDLE
}

///|
pub fn LunaRequestReceipt::abort(
  self : LunaRequestReceipt,
) -> Unit raise RequestAdmissionError {
  self.consume()
}
