///|
pub(all) suberror SerialGenerationError {
  InvalidLimit
  Busy
  Finished
  InvalidSeed
} derive(Eq, Debug)

///|
pub extend SerialGenerationError with Eq::{equal, not_equal}

///|
pub extend SerialGenerationError with @debug.Debug::{to_repr}

///|
pub(all) enum GenerationFinish {
  Running
  Length
  StopToken
  Cancelled
  Failed
} derive(Eq, Debug)

///|
pub extend GenerationFinish with Eq::{equal, not_equal}

///|
pub extend GenerationFinish with @debug.Debug::{to_repr}

///|
/// Model-neutral, one-request greedy continuation. The execution owner must
/// retire each step before accept(); this object owns no device or KV resource.
/// Two startup-owned frames preserve previous/next plan lifetimes, and generated
/// tokens have fixed capacity. Reserved pages and capabilities come from the
/// completed final-prefill plan, never from a model-specific synthetic recipe.
pub struct SerialGreedyGeneration {
  priv frames : FixedArray[@wire.PlanFrameBuffer]
  priv tokens : FixedArray[Int]
  priv stops : ReadOnlyArray[Int]
  priv mut current : Int
  priv mut count : Int
  priv mut pending : Bool
  priv mut reason : GenerationFinish
  priv mut completion : @wire.ValidatedCompletionFrame
}

///|
/// Start after the real final-prefill completion (which supplies token zero).
/// The full requested context must fit before another device submission.
pub fn SerialGreedyGeneration::new(
  final_prefill : @wire.ValidatedPlanFrame,
  completion : @wire.ValidatedCompletionFrame,
  limits : @protocol.WorkerProtocolLimits,
  inference : @inference.InferenceLimits,
  maximum_new_tokens~ : Int,
  maximum_context~ : Int,
  stop_tokens~ : ArrayView[Int],
) -> Self raise {
  guard maximum_new_tokens > 0 &&
    maximum_new_tokens <= inference.max_new_tokens() &&
    maximum_context > 0 &&
    maximum_context <= limits.max_sequence_tokens() &&
    maximum_context <= inference.max_context_tokens() &&
    stop_tokens.length() <= inference.max_stop_token_ids() else {
    raise SerialGenerationError::InvalidLimit
  }
  guard final_prefill.prefill_count() == 1 && final_prefill.decode_count() == 0 else {
    raise SerialGenerationError::InvalidSeed
  }
  let row = final_prefill.prefill_row(0)
  let prompt = row.sequence_token_start() + row.token_length()
  guard prompt <= maximum_context &&
    maximum_new_tokens <= maximum_context - prompt &&
    maximum_new_tokens - 1 <=
    limits.max_sequence_tokens() - 1 - row.sample_index() &&
    (maximum_new_tokens - 1).to_uint64() <=
    0xffffffffffffffffUL - final_prefill.sequence_value() else {
    raise SerialGenerationError::InvalidLimit
  }
  for index, token in stop_tokens {
    guard token >= 0 &&
      token <= limits.max_token_id() &&
      token <= inference.max_token_id() else {
      raise SerialGenerationError::InvalidLimit
    }
    for previous in 0..<index {
      guard stop_tokens[previous] != token else {
        raise SerialGenerationError::InvalidLimit
      }
    }
  }
  let token = completion.greedy_token_for(final_prefill)
  let frames = FixedArray::makei(2, _ => {
    @wire.PlanFrameBuffer::new(limits, inference)
  })
  let scratch = FixedArray::make(frames[0].capacity(), b'\x00')
  final_prefill.copy_to(scratch, destination_offset=0)
  ignore(frames[0].load(scratch, final_prefill.length()))
  let tokens = FixedArray::make(maximum_new_tokens, 0)
  tokens[0] = token
  {
    frames,
    tokens,
    stops: ReadOnlyArray::from_array(stop_tokens.to_owned()),
    current: 0,
    count: 1,
    pending: false,
    reason: if stop_tokens.contains(token) {
      StopToken
    } else if maximum_new_tokens == 1 {
      Length
    } else {
      Running
    },
    completion,
  }
}

///|
/// Prepare one next step only after its predecessor retired. The sampled
/// token is consumed exactly once; previous frame views become stale only
/// when their alternating owner is reused two steps later.
pub fn SerialGreedyGeneration::next_decode(
  self : Self,
) -> @wire.ValidatedPlanFrame raise {
  guard !self.pending else { raise SerialGenerationError::Busy }
  guard self.reason is Running else { raise SerialGenerationError::Finished }
  errdefer {
    self.reason = Failed
  }
  let next = 1 - self.current
  let plan = self.frames[next].encode_greedy_decode_after(
    self.frames[self.current].current(),
    self.completion,
  )
  self.current = next
  self.pending = true
  plan
}

///|
/// Accept only the current step's exact completion. A malformed/failed
/// completion poisons continuation rather than publishing a successful token.
/// Cancellation still requires the execution owner to drain and release KV.
pub fn SerialGreedyGeneration::accept(
  self : Self,
  completion : @wire.ValidatedCompletionFrame,
) -> Unit raise {
  guard self.pending && self.reason is (Running | Cancelled) else {
    raise SerialGenerationError::Busy
  }
  errdefer {
    self.reason = Failed
  }
  let token = completion.greedy_token_for(self.frames[self.current].current())
  self.completion = completion
  self.pending = false
  if self.reason is Cancelled {
    return
  }
  self.tokens[self.count] = token
  self.count += 1
  self.reason = if self.stops.contains(token) {
    StopToken
  } else if self.count == self.tokens.length() {
    Length
  } else {
    Running
  }
}

///|
pub fn SerialGreedyGeneration::cancel(self : Self) -> Unit {
  if self.reason is Running {
    self.reason = Cancelled
  }
}

///|
pub fn SerialGreedyGeneration::finish_reason(self : Self) -> GenerationFinish {
  self.reason
}

///|
pub fn SerialGreedyGeneration::token_count(self : Self) -> Int {
  self.count
}

///|
pub fn SerialGreedyGeneration::token_at(
  self : Self,
  index : Int,
) -> Int raise SerialGenerationError {
  guard index >= 0 && index < self.count else { raise InvalidLimit }
  self.tokens[index]
}
