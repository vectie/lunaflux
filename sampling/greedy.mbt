///|
fn select_row(
  logits : ArrayView[Double],
  offset : Int,
  token_count : Int,
  row : Int,
) -> Selection raise SamplingError {
  let first = logits[offset]
  if first.is_nan() || first.is_inf() {
    raise NonFiniteLogit(row~, token_id=0)
  }
  let mut best_token = 0
  let mut best_logit = first
  for token_id in 1..<token_count {
    let logit = logits[offset + token_id]
    if logit.is_nan() || logit.is_inf() {
      raise NonFiniteLogit(row~, token_id~)
    }
    if logit > best_logit {
      best_token = token_id
      best_logit = logit
    }
  }
  { token_id: best_token, logit: best_logit }
}

///|
/// Selects the greatest finite logit from one borrowed row.
///
/// Equal logits choose the lowest token ID. The input view is never copied.
pub fn greedy(
  logits : ArrayView[Double],
  limits : SamplingLimits,
) -> Selection raise SamplingError {
  if logits.is_empty() {
    raise EmptyInput
  }
  if logits.length() > limits.max_tokens_per_row() {
    raise LimitExceeded(TokensPerRow)
  }
  select_row(logits, 0, logits.length(), 0)
}

///|
/// Selects the greatest finite binary32 logit from one borrowed row.
///
/// This avoids widening or copying reference-kernel output before selection.
/// Equal logits choose the lowest token ID; the returned logit is widened to
/// `Double` only after selection.
pub fn greedy_float(
  logits : ArrayView[Float],
  limits : SamplingLimits,
) -> Selection raise SamplingError {
  if logits.is_empty() {
    raise EmptyInput
  }
  if logits.length() > limits.max_tokens_per_row() {
    raise LimitExceeded(TokensPerRow)
  }
  let first = logits[0]
  if first.is_nan() || first.is_inf() {
    raise NonFiniteLogit(row=0, token_id=0)
  }
  let mut best_token = 0
  let mut best_logit = first
  for token_id in 1..<logits.length() {
    let logit = logits[token_id]
    if logit.is_nan() || logit.is_inf() {
      raise NonFiniteLogit(row=0, token_id~)
    }
    if logit > best_logit {
      best_token = token_id
      best_logit = logit
    }
  }
  { token_id: best_token, logit: best_logit.to_double() }
}

///|
/// Selects one token from each row of a borrowed flattened row-major matrix.
///
/// Results are bounded by `limits.max_rows`; logits remain borrowed and are
/// neither copied nor retained.
pub fn greedy_rows(
  logits : ArrayView[Double],
  row_count : Int,
  tokens_per_row : Int,
  limits : SamplingLimits,
) -> ReadOnlyArray[Selection] raise SamplingError {
  if logits.is_empty() {
    raise EmptyInput
  }
  if row_count <= 0 || tokens_per_row <= 0 {
    raise InvalidShape
  }
  if row_count > limits.max_rows() {
    raise LimitExceeded(Rows)
  }
  if tokens_per_row > limits.max_tokens_per_row() {
    raise LimitExceeded(TokensPerRow)
  }
  if logits.length() % row_count != 0 ||
    logits.length() / row_count != tokens_per_row {
    raise InvalidShape
  }
  let selections = []
  for row in 0..<row_count {
    selections.push(
      select_row(logits, row * tokens_per_row, tokens_per_row, row),
    )
  }
  ReadOnlyArray::from_array(selections)
}
