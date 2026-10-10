///|
pub(all) suberror GreedyVerificationError {
  InvalidEnvelope
  InvalidToken
} derive(Debug, Eq)

///|
pub extend GreedyVerificationError with @debug.Debug::{to_repr}

///|
pub extend GreedyVerificationError with Eq::{equal, not_equal}

///|
pub(all) enum GreedyVerificationFinish {
  Continue
  Length
  StopToken
} derive(Debug, Eq)

///|
pub extend GreedyVerificationFinish with @debug.Debug::{to_repr}

///|
pub extend GreedyVerificationFinish with Eq::{equal, not_equal}

///|
/// Pure decision over a retired target-model verification. Input row zero is
/// the previously emitted seed, not another newly emitted token. Output row i
/// predicts the token following input row i. The final output is the next seed
/// and is not yet part of KV. The caller must restore and replay precisely the
/// accepted input rows when rollback_required() is true, and publish only after
/// both ranks retire that state transaction.
#valtype
pub struct VerifiedGreedyPrefix {
  priv accepted_rows : Int
  priv emitted : Int
  priv seed : Int
  priv rollback : Bool
  priv finish : GreedyVerificationFinish
} derive(Debug, Eq)

///|
pub extend VerifiedGreedyPrefix with @debug.Debug::{to_repr}

///|
pub extend VerifiedGreedyPrefix with Eq::{equal, not_equal}

///|
/// Greedy-only exact verification. This is not stochastic speculative rejection
/// sampling: proposal probabilities and correction distributions are absent.
/// Stop/length truncation can require rollback even if every inspected draft
/// matches. No device, model family, mutable cache or transport enters this law.
pub fn verify_greedy_prefix(
  draft_inputs : ArrayView[Int],
  target_outputs : ArrayView[Int],
  remaining_new_tokens~ : Int,
  vocabulary_size~ : Int,
  stop_tokens~ : ArrayView[Int],
) -> VerifiedGreedyPrefix raise GreedyVerificationError {
  guard !draft_inputs.is_empty() &&
    draft_inputs.length() == target_outputs.length() &&
    remaining_new_tokens > 0 &&
    vocabulary_size > 0 else {
    raise InvalidEnvelope
  }
  validate_tokens(draft_inputs, vocabulary_size)
  validate_tokens(target_outputs, vocabulary_size)
  validate_tokens(stop_tokens, vocabulary_size)
  let rows = draft_inputs.length()
  let limit = rows.min(remaining_new_tokens)
  for row in 0..<limit {
    let token = target_outputs[row]
    let count = row + 1
    let finish = if stop_tokens.contains(token) {
      StopToken
    } else if count == remaining_new_tokens {
      Length
    } else {
      Continue
    }
    if finish != Continue || count == rows || draft_inputs[count] != token {
      return {
        accepted_rows: count,
        emitted: count,
        seed: token,
        rollback: count < rows,
        finish,
      }
    }
  }
  // The final live row is always terminal above (length or row capacity).
  raise InvalidEnvelope
}

///|
fn validate_tokens(
  tokens : ArrayView[Int],
  vocabulary : Int,
) -> Unit raise GreedyVerificationError {
  for token in tokens {
    guard token >= 0 && token < vocabulary else { raise InvalidToken }
  }
}

///|
pub fn VerifiedGreedyPrefix::accepted_input_rows(self : Self) -> Int {
  self.accepted_rows
}

///|
pub fn VerifiedGreedyPrefix::accepted_draft_tokens(self : Self) -> Int {
  self.accepted_rows - 1
}

///|
/// Newly emitted tokens are target_outputs[0 .. emitted_token_count()).
pub fn VerifiedGreedyPrefix::emitted_token_count(self : Self) -> Int {
  self.emitted
}

///|
pub fn VerifiedGreedyPrefix::next_seed(self : Self) -> Int {
  self.seed
}

///|
pub fn VerifiedGreedyPrefix::rollback_required(self : Self) -> Bool {
  self.rollback
}

///|
pub fn VerifiedGreedyPrefix::finish_reason(
  self : Self,
) -> GreedyVerificationFinish {
  self.finish
}
