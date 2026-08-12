///|
const MAX_STEP_TOKEN_BUDGET : Int = 1_048_576

///|
const MAX_ACTIVE_REQUESTS : Int = 65_536

///|
const MAX_WAITING_REQUESTS : Int = 1_000_000

///|
const MAX_REQUEST_SLOTS : Int = 1_048_576

///|
const MAX_PREFILL_CHUNK_TOKENS : Int = 262_144

///|
const MAX_EMERGENCY_DECODE_PAGE_RESERVE : Int = 65_536

///|
const MAX_WAITING_AGE_STEPS : Int = 1_000_000

///|
const MAX_OUTPUT_EVENT_CAPACITY : Int = 1_048_576

///|
const REQUIRED_PLAN_BUFFER_COUNT : Int = 2

///|
/// Fields accepted by the scheduler configuration constructor.
pub(all) enum SchedulerField {
  StepTokenBudget
  MaxActiveRequests
  MaxWaitingRequests
  PrefillChunkTokens
  EmergencyDecodePageReserve
  WaitingAgeThresholdSteps
  OutputEventCapacity
} derive(Debug, Eq)

///|
/// Bounded validation categories for scheduler settings.
pub(all) enum SchedulerIssue {
  BelowMinimum
  AboveMaximum
  ExceedsStepTokenBudget
  CombinedRequestCapacityExceeded
  BelowMaximumRowsPerStep
} derive(Debug, Eq)

///|
/// Payload-safe scheduler configuration validation failure.
pub(all) suberror SchedulerConfigError {
  Invalid(field~ : SchedulerField, issue~ : SchedulerIssue)
} derive(Debug, Eq)

///|
/// Immutable capacity and fairness policy for one scheduler owner.
pub struct SchedulerConfig {
  step_token_budget : Int
  max_active_requests : Int
  max_waiting_requests : Int
  prefill_chunk_tokens : Int
  emergency_decode_page_reserve : Int
  waiting_age_threshold_steps : Int
  output_event_capacity : Int
} derive(Debug, Eq)

///|
/// Constructs bounded scheduler settings.
///
/// The active and waiting capacities share a hard request-slot ceiling. The
/// subtraction-form check avoids overflowing while validating that sum.
pub fn new(
  step_token_budget~ : Int,
  max_active_requests~ : Int,
  max_waiting_requests~ : Int,
  prefill_chunk_tokens~ : Int,
  emergency_decode_page_reserve~ : Int,
  waiting_age_threshold_steps~ : Int,
  output_event_capacity~ : Int,
) -> SchedulerConfig raise SchedulerConfigError {
  require_positive_bounded(
    step_token_budget,
    MAX_STEP_TOKEN_BUDGET,
    StepTokenBudget,
  )
  require_positive_bounded(
    max_active_requests,
    MAX_ACTIVE_REQUESTS,
    MaxActiveRequests,
  )
  require_positive_bounded(
    max_waiting_requests,
    MAX_WAITING_REQUESTS,
    MaxWaitingRequests,
  )
  require_positive_bounded(
    prefill_chunk_tokens,
    MAX_PREFILL_CHUNK_TOKENS,
    PrefillChunkTokens,
  )
  require_positive_bounded(
    emergency_decode_page_reserve,
    MAX_EMERGENCY_DECODE_PAGE_RESERVE,
    EmergencyDecodePageReserve,
  )
  require_positive_bounded(
    waiting_age_threshold_steps,
    MAX_WAITING_AGE_STEPS,
    WaitingAgeThresholdSteps,
  )
  require_positive_bounded(
    output_event_capacity,
    MAX_OUTPUT_EVENT_CAPACITY,
    OutputEventCapacity,
  )
  if prefill_chunk_tokens > step_token_budget {
    raise Invalid(field=PrefillChunkTokens, issue=ExceedsStepTokenBudget)
  }
  if max_active_requests > MAX_REQUEST_SLOTS - max_waiting_requests {
    raise Invalid(
      field=MaxWaitingRequests,
      issue=CombinedRequestCapacityExceeded,
    )
  }
  let max_rows_per_step = if max_active_requests < step_token_budget {
    max_active_requests
  } else {
    step_token_budget
  }
  if output_event_capacity < max_rows_per_step {
    raise Invalid(field=OutputEventCapacity, issue=BelowMaximumRowsPerStep)
  }
  {
    step_token_budget,
    max_active_requests,
    max_waiting_requests,
    prefill_chunk_tokens,
    emergency_decode_page_reserve,
    waiting_age_threshold_steps,
    output_event_capacity,
  }
}

///|
fn require_positive_bounded(
  value : Int,
  maximum : Int,
  field : SchedulerField,
) -> Unit raise SchedulerConfigError {
  if value <= 0 {
    raise Invalid(field~, issue=BelowMinimum)
  }
  if value > maximum {
    raise Invalid(field~, issue=AboveMaximum)
  }
}

///|
pub fn SchedulerConfig::step_token_budget(self : SchedulerConfig) -> Int {
  self.step_token_budget
}

///|
pub fn SchedulerConfig::max_active_requests(self : SchedulerConfig) -> Int {
  self.max_active_requests
}

///|
pub fn SchedulerConfig::max_waiting_requests(self : SchedulerConfig) -> Int {
  self.max_waiting_requests
}

///|
pub fn SchedulerConfig::prefill_chunk_tokens(self : SchedulerConfig) -> Int {
  self.prefill_chunk_tokens
}

///|
pub fn SchedulerConfig::emergency_decode_page_reserve(
  self : SchedulerConfig,
) -> Int {
  self.emergency_decode_page_reserve
}

///|
pub fn SchedulerConfig::waiting_age_threshold_steps(
  self : SchedulerConfig,
) -> Int {
  self.waiting_age_threshold_steps
}

///|
/// Generated-token publication slots owned by the scheduler. This excludes
/// terminal notices and any downstream transport buffering.
pub fn SchedulerConfig::output_event_capacity(self : SchedulerConfig) -> Int {
  self.output_event_capacity
}

///|
/// Returns the fixed double-buffered plan count; this is not operator-selectable.
pub fn SchedulerConfig::plan_buffer_count(_self : SchedulerConfig) -> Int {
  REQUIRED_PLAN_BUFFER_COUNT
}
