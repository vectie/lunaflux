///|
/// Tiny F32 oracle for the post-router sigmoid and correction boundary.
/// Group scoring, top-k, normalization, and scaling are deliberately absent.
pub fn moe_sigmoid_and_correction(
  raw_logits : ArrayView[Float],
  correction_bias : ArrayView[Float],
  row_count~ : Int,
  expert_count~ : Int,
  limits~ : GlmHybridReferenceLimits,
) -> GlmMoeSigmoidCorrectionResult raise GlmHybridReferenceError {
  limits.check_sequence(row_count)
  limits.check_channels(expert_count)
  if raw_logits.length() != row_count * expert_count ||
    correction_bias.length() != expert_count {
    raise InvalidShape
  }
  for value in raw_logits {
    if !oracle_float_finite(value) {
      raise NonFiniteInput
    }
  }
  for value in correction_bias {
    if !oracle_float_finite(value) {
      raise NonFiniteInput
    }
  }
  let scores = Array::make(raw_logits.length(), 0.0F)
  let scores_for_choice = Array::make(raw_logits.length(), 0.0F)
  for index in 0..<raw_logits.length() {
    let expert = index % expert_count
    let score = 1.0F / (1.0F + @math.expf(-raw_logits[index]))
    scores[index] = score
    scores_for_choice[index] = score + correction_bias[expert]
  }
  {
    row_count,
    expert_count,
    scores: ReadOnlyArray::from_array(scores),
    scores_for_choice: ReadOnlyArray::from_array(scores_for_choice),
  }
}

///|
pub fn GlmMoeSigmoidCorrectionResult::row_count(self : Self) -> Int {
  self.row_count
}

///|
pub fn GlmMoeSigmoidCorrectionResult::expert_count(self : Self) -> Int {
  self.expert_count
}

///|
pub fn GlmMoeSigmoidCorrectionResult::scores(
  self : Self,
) -> ReadOnlyArray[Float] {
  self.scores
}

///|
pub fn GlmMoeSigmoidCorrectionResult::scores_for_choice(
  self : Self,
) -> ReadOnlyArray[Float] {
  self.scores_for_choice
}
