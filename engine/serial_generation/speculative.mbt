///|
/// Model/device-neutral greedy speculative continuation. Drafts are proposals,
/// not output. Only the enclosing whole-model coordinator's committed prefix
/// may be recorded. Alternating prepared frames preserve the accepted frontier
/// after a rejected suffix; generated tokens have a fixed startup capacity.
pub struct SpeculativeGreedyGeneration {
  priv frames : FixedArray[@wire.PlanFrameBuffer]
  priv tokens : FixedArray[Int]
  priv maximum_context : Int
  priv mut current : Int
  priv mut committed_rows : Int
  priv mut count : Int
  priv mut pending : Bool
  priv mut reason : GenerationFinish
}

///|
pub fn SpeculativeGreedyGeneration::new(
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
    stop_tokens.length() <= inference.max_stop_token_ids() &&
    final_prefill.prefill_count() == 1 &&
    final_prefill.decode_count() == 0 else {
    raise SerialGenerationError::InvalidLimit
  }
  let row = final_prefill.prefill_row(0)
  let prompt = row.sequence_token_start() + row.token_length()
  guard prompt <= maximum_context &&
    maximum_new_tokens <= maximum_context - prompt else {
    raise SerialGenerationError::InvalidLimit
  }
  for token in stop_tokens {
    guard token >= 0 &&
      token <= limits.max_token_id() &&
      token <= inference.max_token_id() else {
      raise SerialGenerationError::InvalidLimit
    }
  }
  let seed = completion.greedy_token_for(final_prefill)
  let frames = FixedArray::makei(2, _ => {
    @wire.PlanFrameBuffer::new(limits, inference)
  })
  let storage = FixedArray::make(frames[0].capacity(), b'\x00')
  final_prefill.copy_to(storage, destination_offset=0)
  ignore(frames[0].load(storage, final_prefill.length()))
  let tokens = FixedArray::make(maximum_new_tokens, 0)
  tokens[0] = seed
  {
    frames,
    tokens,
    maximum_context,
    current: 0,
    committed_rows: row.token_length(),
    count: 1,
    pending: false,
    reason: if stop_tokens.contains(seed) {
      StopToken
    } else if maximum_new_tokens == 1 {
      Length
    } else {
      Running
    },
  }
}

///|
/// The returned plan consumes the seed exactly once. Oversized proposals are
/// clipped to the requested generation/context envelope, not submitted then
/// repaired by an unnecessary length rollback. sequence includes replay epochs.
pub fn SpeculativeGreedyGeneration::next_verification(
  self : Self,
  draft_inputs : ArrayView[Int],
  sequence_value~ : UInt64,
) -> @wire.ValidatedPlanFrame raise {
  guard !self.pending else { raise SerialGenerationError::Busy }
  guard self.reason is Running else { raise SerialGenerationError::Finished }
  guard !draft_inputs.is_empty() &&
    draft_inputs[0] == self.tokens[self.count - 1] else {
    raise SerialGenerationError::InvalidSeed
  }
  let previous = self.frames[self.current].current()
  let position = previous.prefill_row(0).sequence_token_start() +
    self.committed_rows
  let rows = draft_inputs
    .length()
    .min(self.tokens.length() - self.count)
    .min(self.maximum_context - position)
  guard rows > 0 else { raise SerialGenerationError::InvalidLimit }
  let next = 1 - self.current
  let plan = self.frames[next].encode_verification_after(
    previous,
    committed_input_rows=self.committed_rows,
    draft_inputs=draft_inputs[:rows],
    sequence_value~,
  )
  self.current = next
  self.pending = true
  plan
}

///|
/// Call only after target execution, any restore/replay, and both commits have
/// retired. The receipt's next seed is not already part of committed KV. This
/// object owns no GPU effects and does not run another verification or scan.
pub fn SpeculativeGreedyGeneration::record_commit(
  self : Self,
  prefix : @greedy.VerifiedGreedyPrefix,
  target_tokens : ArrayView[Int],
) -> Unit raise {
  guard self.pending && self.reason is (Running | Cancelled) else {
    raise SerialGenerationError::Busy
  }
  errdefer {
    self.reason = Failed
  }
  let rows = prefix.accepted_input_rows()
  guard rows > 0 &&
    rows <= self.frames[self.current].current().token_count() &&
    rows == prefix.emitted_token_count() &&
    target_tokens.length() == rows &&
    rows <= self.tokens.length() - self.count &&
    target_tokens[rows - 1] == prefix.next_seed() else {
    raise SerialGenerationError::InvalidSeed
  }
  self.committed_rows = rows
  self.pending = false
  if self.reason is Cancelled {
    return
  }
  for token in target_tokens {
    self.tokens[self.count] = token
    self.count += 1
  }
  self.reason = match prefix.finish_reason() {
    StopToken => StopToken
    Length => Length
    Continue =>
      if self.count == self.tokens.length() {
        Length
      } else {
        Running
      }
  }
}

///|
pub fn SpeculativeGreedyGeneration::remaining_new_tokens(self : Self) -> Int {
  self.tokens.length() - self.count
}

///|
pub fn SpeculativeGreedyGeneration::token_count(self : Self) -> Int {
  self.count
}

///|
pub fn SpeculativeGreedyGeneration::token_at(
  self : Self,
  index : Int,
) -> Int raise SerialGenerationError {
  guard index >= 0 && index < self.count else { raise InvalidLimit }
  self.tokens[index]
}

///|
pub fn SpeculativeGreedyGeneration::finish_reason(
  self : Self,
) -> GenerationFinish {
  self.reason
}

///|
pub fn SpeculativeGreedyGeneration::cancel(self : Self) -> Unit {
  if self.reason is Running {
    self.reason = Cancelled
  }
}
