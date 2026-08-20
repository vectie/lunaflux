///|
pub(all) enum RequestAdmissionRule {
  ModelIdentity
  TokenizerIdentity
  RequestContract
  FrameContract
  ReceiptTime
  Deadline
  Input
  Tokenization
  SchedulerContract
  OutputContract
  LunaPreparedLifecycle
  LunaPreparationPool
  LunaPreparationWork
  LunaPreparationStorage
} derive(Debug, Eq)

///|
/// Payload-safe admission failures. Request text, token IDs, stop strings,
/// cache scope, trace data, and tokenizer internals never enter the error.
pub(all) suberror RequestAdmissionError {
  Invalid(RequestAdmissionRule)
  ClockUnavailable
} derive(Debug, Eq)

///|
/// Monotonic timestamp captured inside the trusted frame-receipt operation.
#valtype
priv struct RequestReceipt {
  at_millis : UInt64
}

///|
/// One trusted incremental receiver. The first nonempty append captures its
/// monotonic receipt before the framed reader can copy any byte.
pub struct IncrementalRequestReceiver {
  priv reader : @framed_wire.IncrementalRequestReader
  priv clock : @monotonic_clock.MonotonicClock
  priv inference : @inference.InferenceLimits
  priv mut receipt_at_millis : UInt64
  priv mut has_receipt : Bool
  priv mut terminal : Bool
}

///|
/// One immutable request paired with the monotonic receipt and absolute
/// deadline derived during exact frame binding. The owner is admitted once.
pub struct ReceivedRequest {
  priv receipt : RequestReceipt
  priv request : @inference.GenerateRequest
  priv deadline : @inference.AdmissionDeadline
  priv inference : @inference.InferenceLimits
  priv mut consumed : Bool
}

///|
/// One revocable shell around a preallocated prepared-request claim. Retained
/// aliases observe `None` after the claim is taken, so scheduler and output
/// authority cannot be extracted twice. The type deliberately has no `Debug`.
pub struct LunaPreparedRequest {
  priv mut claim : LunaPreparedRequestClaim?
}

///|
/// Exact scheduler and incremental-output authority transferred from one
/// Luna-prepared shell. Receipt, deadline, tokenizer, and payload-bearing
/// state remain private and the type deliberately has no `Debug`.
pub struct LunaPreparedRequestClaim {
  priv mut scheduler_request : @core.TokenizedRequest?
  priv mut detached_output : @incremental_output.IncrementalOutput?
  priv mut pooled_output : @incremental_output.LunaIncrementalOutputLease?
  priv mut token_lease : @inference.LunaTokenBufferLease?
  priv mut pool : LunaRequestPreparationPool?
  priv lane : Int
  priv generation : UInt64
  priv tokenizer_digest : @tokenizer.TokenizerDigest
  priv inference_limits : @inference.InferenceLimits
  priv input_tokens : Int
}

///|
fn LunaPreparedRequest::require_claim(
  self : LunaPreparedRequest,
) -> LunaPreparedRequestClaim raise RequestAdmissionError {
  match self.claim {
    Some(claim) => {
      claim.require_prepared_authority()
      claim
    }
    None => raise Invalid(LunaPreparedLifecycle)
  }
}

///|
pub fn LunaPreparedRequest::take_claim(
  self : LunaPreparedRequest,
) -> LunaPreparedRequestClaim raise RequestAdmissionError {
  let claim = self.require_claim()
  claim.activate_claim()
  self.claim = None
  claim
}

///|
pub fn LunaPreparedRequest::discard(
  self : LunaPreparedRequest,
) -> Unit raise RequestAdmissionError {
  let claim = self.require_claim()
  claim.release_prepared()
  self.claim = None
}

///|
pub fn LunaPreparedRequest::input_token_count(
  self : LunaPreparedRequest,
) -> Int raise RequestAdmissionError {
  self.require_claim().input_tokens
}

///|
pub fn LunaPreparedRequest::tokenizer_digest(
  self : LunaPreparedRequest,
) -> @tokenizer.TokenizerDigest raise RequestAdmissionError {
  self.require_claim().tokenizer_digest
}

///|
pub fn LunaPreparedRequest::inference_limits(
  self : LunaPreparedRequest,
) -> @inference.InferenceLimits raise RequestAdmissionError {
  self.require_claim().inference_limits
}

///|
pub fn LunaPreparedRequest::model_identity(
  self : LunaPreparedRequest,
) -> @spec.ModelIdentity raise RequestAdmissionError {
  self.require_claim().require_scheduler_request().model_identity()
}

///|
pub fn LunaPreparedRequest::effective_limits(
  self : LunaPreparedRequest,
) -> @inference.EffectiveLimits raise RequestAdmissionError {
  self.require_claim().require_scheduler_request().effective_limits()
}

///|
pub fn LunaPreparedRequest::stream_preference(
  self : LunaPreparedRequest,
) -> @inference.StreamPreference raise RequestAdmissionError {
  self.require_claim().require_scheduler_request().stream_preference()
}

///|
fn LunaPreparedRequestClaim::require_scheduler_request(
  self : LunaPreparedRequestClaim,
) -> @core.TokenizedRequest raise RequestAdmissionError {
  match self.scheduler_request {
    Some(request) => request
    None => raise Invalid(LunaPreparedLifecycle)
  }
}

///|
fn LunaPreparedRequestClaim::require_prepared_authority(
  self : LunaPreparedRequestClaim,
) -> Unit raise RequestAdmissionError {
  self.require_scheduler_request() |> ignore
  match self.pool {
    Some(pool) => {
      if self.lane < 0 || self.lane >= pool.lanes.length() {
        raise Invalid(LunaPreparedLifecycle)
      }
      let lane = pool.lanes[self.lane]
      if self.generation == 0UL ||
        lane.generation != self.generation ||
        lane.phase != LUNA_PREPARATION_PREPARED {
        raise Invalid(LunaPreparedLifecycle)
      }
    }
    None => ()
  }
}

///|
fn LunaPreparedRequestClaim::activate_claim(
  self : LunaPreparedRequestClaim,
) -> Unit raise RequestAdmissionError {
  self.require_prepared_authority()
  match self.pool {
    Some(pool) => {
      let lane = pool.lanes[self.lane]
      validate_pool_time(
        lane.receipt_at_millis,
        lane.require_deadline(),
        (pool.clock_read)(),
      ) catch {
        error => {
          self.release_resources_only()
          lane.claim = None
          pool.release_lane(self.lane, self.generation)
          raise error
        }
      }
      lane.phase = LUNA_PREPARATION_CLAIMED
    }
    None => ()
  }
}

///|
pub fn LunaPreparedRequestClaim::scheduler_request(
  self : LunaPreparedRequestClaim,
) -> @core.TokenizedRequest raise RequestAdmissionError {
  self.require_scheduler_request()
}

///|
pub fn LunaPreparedRequestClaim::is_stop_token(
  self : LunaPreparedRequestClaim,
  token_id : Int,
) -> Bool {
  let request = try self.require_scheduler_request() catch {
    _ => return false
  } noraise {
    value => value
  }
  let candidates = request.stops().token_ids()
  for index in 0..<candidates.length() {
    if candidates[index] == token_id {
      return true
    }
  }
  false
}

///|
pub fn LunaPreparedRequestClaim::push_token_into_status(
  self : LunaPreparedRequestClaim,
  token_id : Int,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> @incremental_output.IncrementalOutputStatus {
  if self.is_stop_token(token_id) {
    return @incremental_output.IncrementalOutputStatus::rejected()
  }
  match self.pooled_output {
    Some(output) =>
      output.push_token_status(token_id, destination, destination_offset~)
    None =>
      match self.detached_output {
        Some(output) =>
          output.push_token_status(token_id, destination, destination_offset~)
        None => @incremental_output.IncrementalOutputStatus::rejected()
      }
  }
}

///|
pub fn LunaPreparedRequestClaim::finish_into_status(
  self : LunaPreparedRequestClaim,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> @incremental_output.IncrementalOutputStatus {
  match self.pooled_output {
    Some(output) => output.finish_into_status(destination, destination_offset~)
    None =>
      match self.detached_output {
        Some(output) =>
          output.finish_into_status(destination, destination_offset~)
        None => @incremental_output.IncrementalOutputStatus::rejected()
      }
  }
}

///|
fn LunaPreparedRequestClaim::release_resources_only(
  self : LunaPreparedRequestClaim,
) -> Unit {
  self.scheduler_request = None
  self.detached_output = None
  match self.pooled_output {
    Some(output) => try! output.release()
    None => ()
  }
  self.pooled_output = None
  match self.token_lease {
    Some(lease) => try! lease.release()
    None => ()
  }
  self.token_lease = None
  self.pool = None
}

///|
fn LunaPreparedRequestClaim::release_prepared(
  self : LunaPreparedRequestClaim,
) -> Unit raise RequestAdmissionError {
  self.require_prepared_authority()
  match self.pool {
    Some(pool) => {
      let lane = self.lane
      let generation = self.generation
      self.release_resources_only()
      pool.lanes[lane].claim = None
      pool.release_lane(lane, generation)
    }
    None => self.release_resources_only()
  }
}

///|
pub fn LunaPreparedRequestClaim::release(
  self : LunaPreparedRequestClaim,
) -> Unit raise RequestAdmissionError {
  self.require_scheduler_request() |> ignore
  match self.pool {
    Some(pool) => {
      if self.lane < 0 || self.lane >= pool.lanes.length() {
        raise Invalid(LunaPreparedLifecycle)
      }
      let lane = pool.lanes[self.lane]
      if self.generation == 0UL ||
        lane.generation != self.generation ||
        lane.phase != LUNA_PREPARATION_CLAIMED {
        raise Invalid(LunaPreparedLifecycle)
      }
      let lane_index = self.lane
      let generation = self.generation
      self.release_resources_only()
      lane.claim = None
      pool.release_lane(lane_index, generation)
    }
    None => self.release_resources_only()
  }
}
