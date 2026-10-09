///|
pub(all) suberror PolicyError {
  InvalidProfile
}

///|
/// One already qualified prefill shape. Construction is a startup operation.
#valtype
pub struct PrefillShape {
  tokens : Int
  workspace_bytes : Int64
}

///|
pub fn PrefillShape::new(
  tokens~ : Int,
  workspace_bytes~ : Int64,
) -> PrefillShape raise PolicyError {
  if tokens <= 0 || workspace_bytes < 0L {
    raise InvalidProfile
  }
  { tokens, workspace_bytes, }
}

///|
/// Returns an index into existing qualified shapes, or -1 under backpressure.
/// No new graph/shape is synthesized. Ties keep the first supplied candidate.
pub fn select_prefill(
  shapes : ArrayView[PrefillShape],
  remaining_tokens~ : Int,
  token_budget~ : Int,
  workspace_budget~ : Int64,
  available_pages~ : Int,
  decode_reserve~ : Int,
  tokens_per_page~ : Int,
  tail_free_tokens~ : Int,
) -> Int {
  if remaining_tokens <= 0 ||
    token_budget <= 0 ||
    workspace_budget < 0L ||
    available_pages < 0 ||
    decode_reserve < 0 ||
    tokens_per_page <= 0 ||
    tail_free_tokens < 0 ||
    tail_free_tokens >= tokens_per_page ||
    available_pages < decode_reserve {
    return -1
  }
  let page_tokens = (available_pages - decode_reserve).to_int64() *
    tokens_per_page.to_int64() +
    tail_free_tokens.to_int64()
  let budget = remaining_tokens.min(token_budget).to_int64().min(page_tokens)
  for i in 0..<shapes.length(); best = -1, best_tokens = 0 {
    let shape = shapes[i]
    if shape.tokens.to_int64() <= budget &&
      shape.workspace_bytes <= workspace_budget &&
      shape.tokens > best_tokens {
      continue i, shape.tokens
    }
    continue best, best_tokens
  } nobreak {
    best
  }
}

///|
/// Measured conservative costs for one exact startup-bound workload profile.
/// This value cannot itself establish that the measurements are authentic.
#valtype
pub struct RestoreCost {
  restore_ns : Int64
  recompute_ns : Int64
  minimum_saving_ns : Int64
}

///|
pub fn RestoreCost::new(
  restore_ns~ : Int64,
  recompute_ns~ : Int64,
  minimum_saving_ns~ : Int64,
) -> RestoreCost raise PolicyError {
  if restore_ns <= 0L || recompute_ns <= 0L || minimum_saving_ns < 0L {
    raise InvalidProfile
  }
  { restore_ns, recompute_ns, minimum_saving_ns, }
}

///|
/// Pure admission. Unknown costs are represented by absence at the caller and
/// select recomputation. Queue delay and deadline headroom are included.
pub fn RestoreCost::should_restore(
  self : RestoreCost,
  queued_ns~ : Int64,
  deadline_slack_ns~ : Int64,
  destination_reserved~ : Bool,
  transfer_credit~ : Bool,
) -> Bool {
  if !destination_reserved ||
    !transfer_credit ||
    queued_ns < 0L ||
    deadline_slack_ns < self.restore_ns ||
    queued_ns > deadline_slack_ns - self.restore_ns ||
    self.recompute_ns <= self.restore_ns {
    return false
  }
  let saving = self.recompute_ns - self.restore_ns
  queued_ns < saving && self.minimum_saving_ns <= saving - queued_ns
}
